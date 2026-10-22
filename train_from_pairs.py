"""
train_from_pairs.py

Learn what a REAL pedal does from a pair of recordings made at the same time:
    dry.wav   the guitar going into the pedal (from your splitter / re-amp)
    wet.wav   what came out of the pedal

Steps:
  1. Checks both files (clipping, levels, silence) and warns about problems.
  2. Aligns them in time: finds the delay between dry and wet by cross-correlation
     (digital pedals add a few ms; analogue ones add about none) and trims it off.
  3. Trains the same small tapped-delay network as neural_effect.py on the
     first 80% of the recording and tests it on the LAST 20% (a contiguous block
     the network never saw).
  4. Compares it against a trivial baseline (dry audio times the best constant gain)
     so you can tell whether the network has learned anything real.
  5. Saves neural/model_<name>.npz, which neural_effect.NeuralEffect can load, and
     writes listening files and a report plot.

Usage:
    python train_from_pairs.py --dry dry.wav --wet wet.wav --name friend_fuzz
    python train_from_pairs.py --dry d1.wav d2.wav --wet w1.wav w2.wav --name tubescreamer

To use the trained model as a streaming effect:
    NEURAL_MODEL=neural/model_friend_fuzz.npz python stream_effects.py clip.wav \
        --module neural_effect --chain NeuralEffect --check

Limits (be honest about these in the talk):
  * Only effects that depend on the last ~6 ms are modelled well: overdrive,
    distortion, fuzz, amp-like tone shaping. Delay, reverb, chorus, flanger and
    compressors are NOT, and the error numbers will show it.
  * The model is only valid for the playing levels and knob positions you recorded.
    Real pedals can't be re-rendered at other levels, so record soft to hard playing,
    and photograph the knobs for each recording.
"""
import argparse
from pathlib import Path

import numpy as np

import neural_effect as ne

SR = ne.SR
MAX_LAG_MS = 50.0          # search for a delay between dry and wet up to this long
TRAIN_FRACTION = 0.8
ROWS = 120000              # training rows sampled from the training part


def load_mono(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def check_signal(name, x):
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x ** 2))
    clipped = np.mean(np.abs(x) > 0.99) * 100
    msgs = []
    if clipped > 0.1:
        msgs.append(f"{clipped:.2f}% of samples clipped: turn the input gain down and re-record")
    if peak < 0.03:
        msgs.append("very quiet (peak below -30 dBFS): turn the input gain up")
    print(f"  {name}: {len(x)/SR:.1f} s, peak {20*np.log10(peak+1e-12):.1f} dBFS, "
          f"rms {20*np.log10(rms+1e-12):.1f} dBFS, clipped {clipped:.2f}%")
    for m in msgs:
        print(f"    WARNING: {m}")


def estimate_lag(dry, wet, max_lag):
    """Return (lag in samples, polarity, confidence). Positive lag = wet arrives later."""
    n = min(len(dry), len(wet))
    d, w = dry[:n] - dry[:n].mean(), wet[:n] - wet[:n].mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(w, size) * np.conj(np.fft.rfft(d, size)), size)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = cc[lags % size]
    k = int(np.argmax(np.abs(vals)))
    norm = np.sqrt(np.sum(d ** 2) * np.sum(w ** 2)) + 1e-12
    return int(lags[k]), (1 if vals[k] >= 0 else -1), float(np.abs(vals[k]) / norm)


def align(dry, wet):
    lag, pol, conf = estimate_lag(dry, wet, int(MAX_LAG_MS * SR / 1000))
    print(f"  alignment: wet arrives {lag/SR*1000:+.2f} ms after dry "
          f"({'inverted' if pol < 0 else 'normal'} polarity), correlation {conf:.2f}")
    if conf < 0.2:
        print("    WARNING: dry and wet barely correlate. Check the two files really are the same performance, "
              "recorded together. Heavy effects (fuzz, reverb) lower this number, but under 0.1 usually means a mismatch.")
    if lag > 0:
        dry, wet = dry[:len(dry) - lag], wet[lag:]
    elif lag < 0:
        dry, wet = dry[-lag:], wet[:len(wet) + lag]
    n = min(len(dry), len(wet))
    return dry[:n], wet[:n], lag


def baseline_esr(dry, wet):
    """Best a do-nothing-clever model can do: wet ~ g * dry."""
    g = np.dot(dry, wet) / (np.dot(dry, dry) + 1e-12)
    return ne.esr(wet, g * dry)


def verdict(esr_db, base_db):
    if esr_db < -25:
        return "close copy (should sound very similar)"
    if esr_db < -15:
        return "reasonable copy (audible differences, character should match)"
    if esr_db < base_db - 3:
        return "weak: better than a plain gain change, but this pedal probably needs memory the network lacks"
    return "FAILED: no better than a gain change. Likely a time-based effect, a mismatch in the recordings, or bad alignment"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", nargs="+", required=True)
    ap.add_argument("--wet", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="a label for the pedal, e.g. friend_fuzz")
    ap.add_argument("--hidden", nargs="+", type=int, default=[32, 16])
    args = ap.parse_args()
    if len(args.dry) != len(args.wet):
        raise SystemExit("Give the same number of --dry and --wet files, in matching order.")
    ne.OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(ne.SEED)

    train_X, train_Y, tests = [], [], []
    for dp, wp in zip(args.dry, args.wet):
        print(f"\n{Path(dp).name} / {Path(wp).name}")
        dry, wet = load_mono(dp), load_mono(wp)
        check_signal("dry", dry); check_signal("wet", wet)
        dry, wet, _ = align(dry, wet)
        cut = int(len(dry) * TRAIN_FRACTION)
        buf = np.concatenate([np.zeros(ne.HIST), dry])
        active = np.where(np.abs(dry[:cut]) > 0.2 * np.sqrt(np.mean(dry[:cut] ** 2)))[0]
        pos = np.concatenate([rng.integers(0, cut, ROWS // 2), rng.choice(active, ROWS - ROWS // 2)])
        train_X.append(ne.tap_rows(buf, pos)); train_Y.append(wet[pos])
        tests.append((Path(dp).stem, dry[cut:], wet[cut:]))

    X, Y = np.concatenate(train_X), np.concatenate(train_Y)
    print(f"\nTraining on {len(Y)} rows...")
    params, iters = ne.train(X, Y, args.hidden)
    model_path = ne.OUT / f"model_{args.name}.npz"
    ne.save_model(model_path, params, args.name)
    print(f"  {iters} epochs, saved {model_path}")

    print("\nHeld-out last 20% of each recording (error-to-signal ratio in dB, lower is better):")
    for stem, dry_t, wet_t in tests:
        pred, times = ne.stream_model(params, dry_t, 256)
        e, b = ne.esr(wet_t, pred), baseline_esr(dry_t, wet_t)
        print(f"  {stem}: network {e:.1f} dB vs plain-gain baseline {b:.1f} dB, {verdict(e, b)}")
        print(f"    worst block {times.max()*1000:.2f} ms (budget {256/SR*1000:.2f} ms)")

    stem, dry_t, wet_t = tests[0]
    pred, _ = ne.stream_model(params, dry_t, 256)
    gain = 0.8 / max(np.max(np.abs(dry_t)), np.max(np.abs(wet_t)), np.max(np.abs(pred)))
    for tag, sig in (("dry", dry_t), ("pedal", wet_t), ("network", pred)):
        sf.write(ne.OUT / f"{args.name}_{tag}.wav", np.clip(sig * gain, -1, 1), SR)

    loud = int(np.argmax(np.convolve(np.abs(dry_t), np.ones(2000), "same")))
    lo, hi = max(loud - 300, 0), loud + 300
    tt = np.arange(hi - lo) / SR * 1000
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(tt, dry_t[lo:hi] * (np.max(np.abs(wet_t)) / (np.max(np.abs(dry_t)) + 1e-12)), color="0.7", label="dry (scaled)")
    ax[0].plot(tt, wet_t[lo:hi], label="real pedal"); ax[0].plot(tt, pred[lo:hi], "--", label="network")
    ax[0].set_xlabel("time (ms)"); ax[0].set_title("zoom, loudest held-out part"); ax[0].legend(fontsize=8)
    ax[1].scatter(dry_t[::7], wet_t[::7], s=1, alpha=0.3, label="real pedal")
    ax[1].scatter(dry_t[::7], pred[::7], s=1, alpha=0.3, label="network")
    ax[1].set_xlabel("dry sample"); ax[1].set_ylabel("output sample"); ax[1].set_title("input vs output (shape of the clipping)")
    ax[1].legend(fontsize=8, markerscale=6)
    n = min(len(wet_t), 2 ** 17); f = np.fft.rfftfreq(n, 1 / SR)
    for sig, lab, ls in ((wet_t, "real pedal", "-"), (pred, "network", "--")):
        ax[2].plot(f, 20 * np.log10(np.abs(np.fft.rfft(sig[:n] * np.hanning(n))) + 1e-9), ls, lw=0.6, label=lab)
    ax[2].set_xscale("log"); ax[2].set_xlim(50, 20000); ax[2].set_title("spectrum"); ax[2].set_xlabel("Hz"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ne.OUT / f"{args.name}_report.png", dpi=115)
    print(f"\nSaved {args.name}_dry/pedal/network.wav and {args.name}_report.png in {ne.OUT}/")


if __name__ == "__main__":
    main()