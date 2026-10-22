"""
neural_effect.py

Trains a small neural network to REPRODUCE one of your effects from my_effects.py,
then runs the network on its own, as a streaming effect, with the original
effect nowhere in the signal path.

How it works (a "tapped delay line" network, a simple kind of black-box effect model):
  * input  : the current sample and ~35 past samples of the dry guitar, picked at
             spaced delays (dense for the last 24 samples, then sparser out to
             256 samples, about 5.8 ms)
  * network: a small multilayer perceptron (35 -> 32 -> 16 -> 1, tanh)
  * target : what the real effect would output at that moment
  * It is causal and needs only the last 256 samples of history, so streaming is
    just a ring buffer: the same idea as the Context wrapper in stream_effects.py.

What it can and can't learn:
  * Distortion, Overdrive, Fuzz: yes. They are mostly clipping plus short filters,
    which fit inside a 5.8 ms window.
  * Delay, Reverb, Chorus, Compressor: no. They depend on audio hundreds of ms
    old, far outside the window. Those need a recurrent network (GRU/LSTM) or
    much longer inputs, which is a bigger project.

Run from the my_pedal_dsp folder (after main.py has made data_trimmed/):
    python neural_effect.py                       # models Distortion
    python neural_effect.py --effect Overdrive
    python neural_effect.py --effect Fuzz --hidden 48 24

Outputs go to neural/:  model_<effect>.npz, compare_*.wav, neural_report.png
Use the trained model as a normal streaming effect:
    python stream_effects.py clip.wav --module neural_effect --chain NeuralEffect --check
(set the NEURAL_MODEL environment variable to pick another model file)
KEEP YOUR VOLUME LOW when playing the output.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

import my_effects as fx

# ---------------- settings you can tweak ----------------
from settings import SR
from settings import DATA_TRIMMED as DATA
OUT = Path("neural")
SKIP_CLASSES = ("silence",)

TAPS = np.array(sorted(set(list(range(0, 24)) + [28, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256])))
HIST = int(TAPS.max())
BLOCK = 4096                          # block size used to render training targets
GAINS_DB = (-12, -6, 0, 6)            # level augmentation, so it works at any playing level
ROWS_PER_CLIP_GAIN = 6000
INPUT_SCALE = 0.05                    # divide inputs by this before the network
SEED = 0
# ---------------------------------------------------------

EFFECTS = {"Distortion": fx.Distortion, "Overdrive": fx.Overdrive, "Fuzz": fx.Fuzz}


# ---------------------------------------------------------------- data
def tap_rows(buf, positions):
    """buf has HIST samples of history in front. positions index into the audio after it."""
    idx = HIST + np.asarray(positions)[:, None] - TAPS[None, :]
    return buf[idx]


def render(effect_cls, x):
    """Run the real effect over x, block by block."""
    e = effect_cls()
    e.prepare(SR, BLOCK)
    e.reset()
    return np.concatenate([e.process(x[i:i + BLOCK]) for i in range(0, len(x), BLOCK)])


def make_rows(clips, effect_cls, rng, gains_db=GAINS_DB, rows=ROWS_PER_CLIP_GAIN):
    X, Y = [], []
    for clip in clips:
        for g in gains_db:
            x = clip * 10 ** (g / 20)
            y = render(effect_cls, x)
            buf = np.concatenate([np.zeros(HIST), x])
            active = np.where(np.abs(x) > 0.2 * np.sqrt(np.mean(x ** 2)))[0]
            pos = np.concatenate([rng.integers(0, len(x), rows // 2),
                                  rng.choice(active, rows - rows // 2) if len(active) else
                                  rng.integers(0, len(x), rows - rows // 2)])
            X.append(tap_rows(buf, pos))
            Y.append(y[pos])
    return np.concatenate(X), np.concatenate(Y)


# ---------------------------------------------------------------- the network
def forward(params, X):
    """Plain numpy forward pass (float32): this is what runs at 'play time'."""
    h = X.astype(np.float32) / np.float32(INPUT_SCALE)
    n_layers = len([k for k in params if k.startswith("W")])
    for i in range(n_layers):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < n_layers - 1:
            h = np.tanh(h)
    return h[:, 0] * params["out_scale"]


def train(X, Y, hidden):
    from sklearn.neural_network import MLPRegressor
    out_scale = float(np.std(Y) + 1e-9)
    net = MLPRegressor(hidden_layer_sizes=tuple(hidden), activation="tanh", solver="adam",
                       learning_rate_init=3e-3, batch_size=512, max_iter=60,
                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
                       random_state=SEED)
    net.fit(X / INPUT_SCALE, Y / out_scale)
    params = {"out_scale": np.float32(out_scale)}
    for i, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
        params[f"W{i}"], params[f"b{i}"] = W.astype(np.float32), b.astype(np.float32)
    return params, net.n_iter_


def save_model(path, params, effect_name):
    np.savez(path, effect=effect_name, taps=TAPS, **params)


def load_model(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files if k.startswith("W") or k.startswith("b") or k == "out_scale"}


# ---------------------------------------------------------------- streaming effect
class NeuralEffect:
    """The trained network as a streaming effect (same interface as my_effects.py).
    Reads the model named by $NEURAL_MODEL (default neural/model_Distortion.npz)."""

    def __init__(self, path=None):
        path = path or os.environ.get("NEURAL_MODEL", "neural/model_Distortion.npz")
        self.name = f"Neural[{Path(path).stem}]"
        self.params = load_model(path)
        self.reset()

    def prepare(self, sr, block_size):
        if sr != SR:
            raise ValueError(f"Model was trained at {SR} Hz, got {sr} Hz")

    def reset(self):
        self.hist = np.zeros(HIST)

    def process(self, x):
        buf = np.concatenate([self.hist, x])
        out = forward(self.params, tap_rows(buf, np.arange(len(x))))
        self.hist = buf[-HIST:]
        return out.astype(np.float64)


# ---------------------------------------------------------------- evaluation
def esr(target, pred):
    """Error-to-signal ratio in dB (lower is better; -20 dB means the error has 1% of the energy)."""
    return 10 * np.log10(np.sum((target - pred) ** 2) / (np.sum(target ** 2) + 1e-12) + 1e-12)


def stream_model(params, x, block):
    m = NeuralEffect.__new__(NeuralEffect)
    m.params, m.name = params, "n"
    m.reset()
    out, times = [], []
    for i in range(0, len(x), block):
        t0 = time.perf_counter()
        out.append(m.process(x[i:i + block]))
        times.append(time.perf_counter() - t0)
    return np.concatenate(out), np.array(times)


def load_clips(held_out=True):
    import librosa
    train, test, names = [], [], []
    for cls_dir in sorted(p for p in DATA.iterdir() if p.is_dir() and p.name not in SKIP_CLASSES):
        files = sorted(cls_dir.glob("*.wav"))
        for j, f in enumerate(files):
            y, _ = librosa.load(f, sr=SR, mono=True)
            if held_out and j == len(files) - 1 and len(files) > 1:
                test.append(y); names.append(f"{cls_dir.name}/{f.stem}")
            else:
                train.append(y)
    return train, test, names


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--effect", default="Distortion", choices=list(EFFECTS))
    ap.add_argument("--hidden", nargs="+", type=int, default=[32, 16])
    args = ap.parse_args()
    cls = EFFECTS[args.effect]
    rng = np.random.default_rng(SEED)
    OUT.mkdir(exist_ok=True)

    if not DATA.exists():
        raise SystemExit("data_trimmed/ not found. Run main.py first, from the my_pedal_dsp folder.")
    train_clips, test_clips, test_names = load_clips()
    if not train_clips or not test_clips:
        raise SystemExit("Need clips in at least one class folder (and 2+ in one for a held-out test).")
    print(f"Modelling {args.effect} with network {len(TAPS)} -> {' -> '.join(map(str, args.hidden))} -> 1 "
          f"({len(train_clips)} training clips, {len(test_clips)} held out)")

    print("Rendering training targets with the real effect and training...")
    X, Y = make_rows(train_clips, cls, rng)
    t0 = time.time()
    params, iters = train(X, Y, args.hidden)
    n_params = sum(v.size for k, v in params.items() if k != "out_scale")
    print(f"  {len(Y)} training rows, {n_params} weights, {iters} epochs, {time.time()-t0:.0f} s")
    model_path = OUT / f"model_{args.effect}.npz"
    save_model(model_path, params, args.effect)

    print("\nHeld-out clips, never seen in training (error-to-signal ratio, dB: lower is better):")
    print(f"  {'clip':32s} " + "  ".join(f"{g:+3d} dB" for g in (-12, 0, 6)))
    all_esr = []
    for name, clip in zip(test_names, test_clips):
        row = []
        for g in (-12, 0, 6):
            x = clip * 10 ** (g / 20)
            pred, _ = stream_model(params, x, 256)
            row.append(esr(render(cls, x), pred))
        all_esr.append(row)
        print(f"  {name:32s} " + "  ".join(f"{v:6.1f}" for v in row))
    print("  rough guide: below -20 dB sounds close, below -30 dB is very close")

    # streaming checks on the first held-out clip, at its natural level
    x = test_clips[0]
    ref = render(cls, x)
    p256, t256 = stream_model(params, x, 256)
    p64, t64 = stream_model(params, x, 64)
    p_all, _ = stream_model(params, x, len(x))
    print("\nStreaming:")
    print(f"  output identical at block sizes 64 / 256 / whole file: "
          f"{np.max(np.abs(p64 - p_all)):.1e} / {np.max(np.abs(p256 - p_all)):.1e} max difference")
    e = cls(); e.prepare(SR, 256); e.reset()
    t_orig = np.array([(lambda t: (e.process(x[i:i + 256]), time.perf_counter() - t)[1])(time.perf_counter())
                       for i in range(0, len(x), 256)])
    print(f"  per 256-sample block (budget {256/SR*1000:.2f} ms): network mean {t256.mean()*1000:.3f} ms, "
          f"worst {t256.max()*1000:.3f} ms; original effect mean {t_orig.mean()*1000:.3f} ms")
    print(f"  late blocks: {int(np.sum(t256 > 256/SR))} of {len(t256)}")

    # audio and plots, all at one common gain
    gain = 0.8 / max(np.max(np.abs(x)), np.max(np.abs(ref)), np.max(np.abs(p256)))
    sf.write(OUT / "compare_dry.wav", x * gain, SR)
    sf.write(OUT / "compare_target.wav", np.clip(ref * gain, -1, 1), SR)
    sf.write(OUT / "compare_network.wav", np.clip(p256 * gain, -1, 1), SR)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    loud = int(np.argmax(np.convolve(np.abs(x), np.ones(2000), "same")))
    lo, hi = max(loud - 400, 0), loud + 400
    tt = np.arange(hi - lo) / SR * 1000
    ax[0].plot(tt, x[lo:hi] * 10, color="0.7", label="dry (x10)")
    ax[0].plot(tt, ref[lo:hi], label="real effect")
    ax[0].plot(tt, p256[lo:hi], "--", label="network")
    ax[0].set_xlabel("time (ms)"); ax[0].set_title("zoom on the loudest part"); ax[0].legend(fontsize=8)
    seg = slice(0, min(len(x), 2 ** 17))
    f = np.fft.rfftfreq(seg.stop - seg.start, 1 / SR)
    for sig, lab, ls in ((ref, "real effect", "-"), (p256, "network", "--")):
        S = 20 * np.log10(np.abs(np.fft.rfft(sig[seg] * np.hanning(seg.stop - seg.start))) + 1e-9)
        ax[1].plot(f, S, ls, lw=0.6, label=lab)
    ax[1].set_xscale("log"); ax[1].set_xlim(50, 20000); ax[1].set_xlabel("Hz"); ax[1].set_ylabel("dB")
    ax[1].set_title("spectrum"); ax[1].legend(fontsize=8)
    # static curve: feed a steady sine of each amplitude and take the peak output
    amps = np.linspace(0.005, 0.5, 40)
    t_s = np.arange(4096) / SR
    real_pk, net_pk = [], []
    for a in amps:
        s = a * np.sin(2 * np.pi * 440 * t_s)
        real_pk.append(np.max(np.abs(render(cls, s)[2048:])))
        net_pk.append(np.max(np.abs(stream_model(params, s, 256)[0][2048:])))
    ax[2].plot(amps, real_pk, label="real effect"); ax[2].plot(amps, net_pk, "--", label="network")
    ax[2].set_xlabel("input amplitude (440 Hz sine)"); ax[2].set_ylabel("output peak"); ax[2].legend(fontsize=8)
    ax[2].set_title("level response")
    fig.tight_layout(); fig.savefig(OUT / "neural_report.png", dpi=115)
    print(f"\nSaved {model_path}, compare_dry/target/network.wav and neural_report.png in {OUT}/")


if __name__ == "__main__":
    main()