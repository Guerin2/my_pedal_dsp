"""
section_recogniser.py

A "Shazam for your own playing": recognises which rehearsed section you are
playing from the DRY guitar signal, then fades in the effect preset you chose
for that section.

How it recognises (no fingerprint hashing, because your takes never repeat
exactly):
  * Chroma: 12 numbers per ~93 ms saying which pitch classes are sounding.
    It mostly ignores volume and tone, and it never hears your effects
    because it listens to the dry signal.
  * Subsequence DTW: the last W seconds of chroma are slid along each stored
    take of each section, allowing the tempo to differ by up to 2x.
    The lowest cost wins.
  * Rejection: if the best cost is above a threshold, or the runner-up is too
    close, the answer is "unknown" and no preset is applied.
  * Confirmation: a section must win CONFIRM decisions in a row before it is
    locked (the hysteresis idea again). Once locked it STAYS locked, through
    "unknown" moments and pauses, until a different section wins CONFIRM in a row.

Folder layout (you record these, dry, with the same phone/cable setup):
    sections/<name>/take_1.wav, take_2.wav, ...      at least 2 takes per section
    sections/_other/*.wav                            optional: playing NOT in the
                                                     library, used to test rejection
    sections/presets.json                            optional, e.g.
        {"riff_a": "Overdrive", "riff_b": "Compressor+Distortion+Delay",
         "solo": "neural:neural/model_friend_fuzz.npz"}

Commands (run from the my_pedal_dsp folder, venv active):
    python section_recogniser.py eval      leave-one-take-out accuracy, time-to-recognise
                                           against window length, rejection threshold
    python section_recogniser.py demo      stitch held-out takes together, recognise them
                                           as a stream, apply presets, save audio + plot

KEEP YOUR VOLUME LOW when playing the demo output.
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from settings import BLOCK, SECTIONS, SR

N_FFT, HOP, DS = 4096, 1024, 4          # chroma frame 93 ms, hop 23 ms, 4 frames averaged
FPS = SR / HOP / DS                     # matching frames per second (about 10.8)
SIL_RMS = 0.003                         # frames quieter than this count as silence
MIN_ACTIVE = 8                          # need at least this many non-silent frames to decide
STEP_PENALTY = 0.05                     # discourages cheating by skipping query frames
DECIDE_EVERY_S = 0.1
CONFIRM = 3                             # identical decisions needed before switching
MARGIN = 0.03                           # runner-up must be at least this much worse
WINDOWS_S = (1.0, 2.0, 3.0)
RAMP_S = 0.03
OUT = Path("sections_out")

_win = np.hanning(N_FFT)
_freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
_band = (_freqs >= 100) & (_freqs <= 2000)
_pc = np.round(12 * np.log2(_freqs[_band] / 440) + 69).astype(int) % 12
_onehot = np.zeros((int(_band.sum()), 12))
_onehot[np.arange(len(_pc)), _pc] = 1.0


# ---------------------------------------------------------------- chroma
def _frame_pc(frames):
    mag = np.abs(np.fft.rfft(frames * _win, axis=1))[:, _band]
    return np.sqrt(mag) @ _onehot


def _group(pc, rms):
    """Average DS frames, then make each unit length. Returns (features, silent)."""
    m = len(pc) // DS
    if m == 0:
        return np.zeros((0, 12)), np.zeros(0, bool)
    v = pc[:m * DS].reshape(m, DS, 12).mean(axis=1)
    r = rms[:m * DS].reshape(m, DS).mean(axis=1)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v, r < SIL_RMS


def chroma_features(y):
    """Offline: (M, 12) unit vectors and an (M,) silent flag."""
    if len(y) < N_FFT:
        return np.zeros((0, 12)), np.zeros(0, bool)
    frames = sliding_window_view(y, N_FFT)[::HOP]
    return _group(_frame_pc(frames), frames.std(axis=1))


class ChromaStream:
    """Same features as chroma_features(), produced block by block from live audio."""

    def __init__(self, keep_s=6.0):
        self.buf, self.base, self.k = np.zeros(0), 0, 0
        self.pc, self.rms = [], []
        self.feat = deque(maxlen=int(keep_s * FPS))     # (vector, silent)

    def push(self, x):
        self.buf = np.concatenate([self.buf, x])
        while True:
            s = self.k * HOP - self.base
            if s + N_FFT > len(self.buf):
                break
            fr = self.buf[s:s + N_FFT][None, :]
            self.pc.append(_frame_pc(fr)[0]); self.rms.append(fr.std())
            self.k += 1
            if len(self.pc) == DS:
                v, sil = _group(np.array(self.pc), np.array(self.rms))
                self.feat.append((v[0], bool(sil[0])))
                self.pc, self.rms = [], []
        drop = self.k * HOP - self.base
        if drop > 0:
            self.buf, self.base = self.buf[drop:], self.base + drop

    def query(self, window_s):
        n = int(round(window_s * FPS))
        items = list(self.feat)[-n:]
        act = [v for v, sil in items if not sil]
        return np.array(act) if len(act) >= MIN_ACTIVE else None


# ---------------------------------------------------------------- matching
def sdtw(Q, R):
    """Subsequence DTW cost (lower = better match) of query Q somewhere inside reference R."""
    q, r = len(Q), len(R)
    if r < 3:
        return 9.0
    C = 1.0 - Q @ R.T
    INF = 1e9
    D = np.full((q, r), INF)
    D[0] = C[0]
    for i in range(1, q):
        best = np.full(r, INF)
        best[1:] = D[i - 1, :-1]
        if i >= 2:
            best[1:] = np.minimum(best[1:], D[i - 2, :-1] + STEP_PENALTY)
        c3 = np.full(r, INF)
        c3[2:] = D[i - 1, :-2] + STEP_PENALTY
        D[i] = C[i] + np.minimum(best, c3)
    return float(D[-1].min() / q)


def rank_sections(Q, library):
    """library: {name: [take_features, ...]}. Returns [(cost, name), ...] best first."""
    out = [(min(sdtw(Q, t) for t in takes), name) for name, takes in library.items()]
    return sorted(out)


def confidences(ranked, threshold, temp_frac=0.2):
    """Turns match costs into rough confidences (they add up to 1) for each section plus
    "unknown". Not true probabilities: a softmax over -cost / T, where "unknown" competes at the
    lock threshold and T is a fraction of it. A section scores above "unknown" only when its cost
    is below the threshold. Returns [(name or None, confidence), ...] best first."""
    if not ranked:
        return []
    T = max(threshold * temp_frac, 1e-6)
    names = [n for _, n in ranked] + [None]
    costs = np.array([c for c, _ in ranked] + [threshold])
    z = np.exp(-(costs - costs.min()) / T)
    p = z / z.sum()
    return sorted(zip(names, p.tolist()), key=lambda x: -x[1])


class Tracker:
    """Turns noisy per-window rankings into a stable 'locked' section (None until the first lock).
    A section wins a decision if it matches better than the threshold and clearly beats the
    runner-up. CONFIRM wins in a row lock it; it then stays locked until a different section
    gets CONFIRM wins in a row. An "unknown" decision never unlocks, it only breaks a streak."""

    def __init__(self, threshold):
        self.thr, self.cand, self.count, self.locked = threshold, None, 0, None

    def update(self, ranked):
        if ranked is None:                   # not enough playing to decide: keep state
            return self.locked
        (c1, n1) = ranked[0]
        c2 = ranked[1][0] if len(ranked) > 1 else 9.0
        cand = n1 if (c1 < self.thr and c2 - c1 >= MARGIN) else None
        if cand is None:
            self.cand, self.count = None, 0
        else:
            self.count = self.count + 1 if cand == self.cand else 1
            self.cand = cand
            if self.count >= CONFIRM:
                self.locked = cand
        return self.locked

    def progress(self):
        """(challenger, wins in a row) while a different section is building a streak, else None."""
        if self.cand is not None and self.cand != self.locked:
            return self.cand, min(self.count, CONFIRM)
        return None


# ---------------------------------------------------------------- data
def load_audio(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def load_sections(root=SECTIONS):
    """Returns (audio dict {name: [arrays]}, other [arrays], presets dict)."""
    audio, other = {}, []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(d.glob("*.wav"))
        if d.name == "_other":
            other = [load_audio(f) for f in files]
        elif files:
            audio[d.name] = [load_audio(f) for f in files]
    presets = json.loads((root / "presets.json").read_text()) if (root / "presets.json").exists() else {}
    return audio, other, presets


def features_of(audio_lists):
    return {n: [chroma_features(a) for a in takes] for n, takes in audio_lists.items()}


def first_active_s(y):
    f = sliding_window_view(y, 1024)[::512].std(axis=1)
    idx = np.where(f > SIL_RMS)[0]
    return idx[0] * 512 / SR if len(idx) else 0.0


# ---------------------------------------------------------------- evaluation
def stream_decisions(y, library_feats, threshold, window_s):
    """Run the recogniser as a live stream. Returns (times, locked labels)."""
    cs, tr = ChromaStream(), Tracker(threshold)
    lib = {n: [f[0][~f[1]] for f in fs] for n, fs in library_feats.items()}   # silence removed once
    step = int(DECIDE_EVERY_S * SR)
    times, labels, costs = [], [], []
    for start in range(0, len(y) - step + 1, step):
        cs.push(y[start:start + step])
        Q = cs.query(window_s)
        t0 = time.perf_counter()
        ranked = None if Q is None else rank_sections(Q, lib)
        costs.append(time.perf_counter() - t0)
        times.append((start + step) / SR)
        labels.append(tr.update(ranked))
    return np.array(times), labels, np.array(costs)


def calibrate(audio, other, window_s, rng, per_take=10):
    """Leave-one-take-out: cost of the right section vs the best wrong section."""
    feats = features_of(audio)
    genuine, impostor = [], []
    n_win = int(round(window_s * FPS))
    for name, takes in feats.items():
        for ti, (F, sil) in enumerate(takes):
            lib = {n: [t[0][~t[1]] for k, t in enumerate(ts) if not (n == name and k == ti)]
                   for n, ts in feats.items()}
            lib = {n: v for n, v in lib.items() if v}
            for _ in range(per_take):
                if len(F) <= n_win:
                    break
                s = int(rng.integers(0, len(F) - n_win))
                Q = F[s:s + n_win][~sil[s:s + n_win]]
                if len(Q) < MIN_ACTIVE:
                    continue
                ranked = rank_sections(Q, lib)
                costs = dict((n, c) for c, n in ranked)
                if name in costs:
                    genuine.append(costs[name])
                    impostor.append(min(c for c, n in ranked if n != name) if len(ranked) > 1 else 9.0)
    for a in other:
        F, sil = chroma_features(a)
        for _ in range(per_take):
            if len(F) <= n_win:
                break
            s = int(rng.integers(0, len(F) - n_win))
            Q = F[s:s + n_win][~sil[s:s + n_win]]
            if len(Q) >= MIN_ACTIVE:
                impostor.append(rank_sections(Q, {n: [t[0][~t[1]] for t in ts] for n, ts in feats.items()})[0][0])
    g, im = np.array(genuine), np.array(impostor)
    thr = 0.5 * (np.percentile(g, 90) + np.percentile(im, 10)) if np.percentile(g, 90) < np.percentile(im, 10) \
        else float(np.percentile(g, 90))
    return float(thr), g, im


def evaluate(audio, other, windows=WINDOWS_S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    OUT.mkdir(exist_ok=True)
    feats_all = features_of(audio)
    rows = []
    print(f"{'window':>7} {'threshold':>9} {'right at end':>13} {'median time to lock':>20} "
          f"{'wrong locks':>11} {'false locks on other':>21} {'worst decision':>15}")
    for W in windows:
        thr, g, im = calibrate(audio, other, W, rng)
        overlap = "" if np.percentile(g, 90) < np.percentile(im, 10) else "  (genuine and impostor costs overlap)"
        right, total, delays, wrong, worst = 0, 0, [], 0, 0.0
        for name, takes in audio.items():
            for ti, take in enumerate(takes):
                lib = {n: [f for k, f in enumerate(fs) if not (n == name and k == ti)] for n, fs in feats_all.items()}
                lib = {n: v for n, v in lib.items() if v}
                if name not in lib:
                    continue
                t, labels, cost = stream_decisions(take, lib, thr, W)
                worst = max(worst, cost.max())
                total += 1
                right += labels[-1] == name
                t0 = first_active_s(take)
                ok = [tt for tt, l in zip(t, labels) if l == name]
                if ok:
                    delays.append(ok[0] - t0)
                wrong += sum(1 for l in labels if l not in (None, name)) > 0
        false_other = 0
        for a in other:
            _, labels, _ = stream_decisions(a, feats_all, thr, W)
            false_other += sum(1 for l in labels if l is not None) > 0
        med = float(np.median(delays)) if delays else float("nan")
        rows.append((W, right / max(total, 1), med))
        print(f"{W:6.1f}s {thr:9.3f} {right:>6}/{total:<6} {med:19.2f}s {wrong:>11} {false_other:>10}/{len(other):<10} "
              f"{worst*1000:12.1f} ms{overlap}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="median time to lock (s)")
    ax2 = ax.twinx()
    ax2.plot([r[0] for r in rows], [r[1] for r in rows], "s--", color="tab:red", label="correct at end")
    ax2.set_ylim(0, 1.05)
    ax.set_xlabel("query window (s)"); ax.set_ylabel("seconds after you start playing")
    ax2.set_ylabel("fraction of held-out takes recognised")
    ax.set_title("Recognise sooner vs recognise surely")
    fig.tight_layout(); fig.savefig(OUT / "recognition_vs_window.png", dpi=120)
    print(f"\nSaved {OUT}/recognition_vs_window.png")


# ---------------------------------------------------------------- applying presets
def make_chain(spec):
    import my_effects as fx
    if spec.startswith("neural:"):
        import neural_effect as ne
        return [ne.NeuralEffect(spec.split(":", 1)[1])]
    return [getattr(fx, n.strip())() for n in spec.split("+")]


def apply_presets(audio, times, labels, presets):
    """Every preset's chain runs on every block (warm state); weights glide to the locked section."""
    names = sorted(set(presets.values()))
    chains = {p: make_chain(p) for p in names}
    for ch in chains.values():
        for e in ch:
            e.prepare(SR, BLOCK); e.reset()
    w = {p: 0.0 for p in names}
    step = BLOCK / (RAMP_S * SR)
    out = np.empty(len(audio))
    for s in range(0, len(audio), BLOCK):
        blk = audio[s:s + BLOCK]
        i = np.searchsorted(times, s / SR, side="right") - 1
        lab = labels[i] if i >= 0 else None
        target = presets.get(lab)
        for p in names:
            w[p] = float(np.clip(w[p] + (step if p == target else -step), 0, 1))
        tot = sum(w.values())
        if tot > 1:
            w = {p: v / tot for p, v in w.items()}
        mix = (1.0 - sum(w.values())) * blk
        for p, ch in chains.items():
            y = blk
            for e in ch:
                y = e.process(y)
            mix = mix + w[p] * y
        out[s:s + len(blk)] = mix
    return out


def demo(audio, other, presets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    rng = np.random.default_rng(1)
    OUT.mkdir(exist_ok=True)
    names = sorted(audio)
    if any(len(audio[n]) < 2 for n in names):
        raise SystemExit("Each section needs at least 2 takes (one is held out for the demo).")
    default = ["Overdrive", "Fuzz", "Chorus+Delay+Reverb", "Distortion", "Tremolo"]
    presets = {n: presets.get(n, default[i % len(default)]) for i, n in enumerate(names)}
    held = {n: audio[n][-1] for n in names}
    lib_audio = {n: audio[n][:-1] for n in names}
    W = 2.0
    thr, _, _ = calibrate(lib_audio, other, W, rng) if all(len(v) >= 2 for v in lib_audio.values()) \
        else (0.25, None, None)
    feats = features_of(lib_audio)
    gap = np.zeros(SR)
    order = names + [names[0]]
    parts, spans, t = [], [], 0.0
    seg = int(min(12, min(len(a) / SR for a in held.values())) * SR)
    for i, n in enumerate(order):
        parts += [held[n][:seg], gap]
        spans.append((t, t + seg / SR, n)); t += seg / SR + 1.0
        if i == 0 and other:
            parts += [other[0][:seg], gap]
            spans.append((t, t + seg / SR, "(other)")); t += seg / SR + 1.0
    audio_in = np.concatenate(parts)
    times, labels, cost = stream_decisions(audio_in, feats, thr, W)
    wet = apply_presets(audio_in, times, labels, presets)
    gain = 0.8 / max(np.max(np.abs(audio_in)), np.max(np.abs(wet)))
    sf.write(OUT / "demo_dry.wav", audio_in * gain, SR)
    sf.write(OUT / "demo_processed.wav", np.clip(wet * gain, -1, 1), SR)
    print(f"Presets: {presets}")
    print(f"Decision cost: mean {cost.mean()*1000:.1f} ms, worst {cost.max()*1000:.1f} ms per decision")
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(np.arange(len(audio_in)) / SR, audio_in, lw=0.3, color="0.6")
    for a, b, n in spans:
        ax[0].axvspan(a, b, alpha=0.15, color="tab:blue" if n != "(other)" else "tab:orange")
        ax[0].text((a + b) / 2, ax[0].get_ylim()[1] * 0.8, n, ha="center", fontsize=8)
    idx = {n: i for i, n in enumerate(names)}
    ax[1].step(times, [idx.get(l, -1) for l in labels], where="post")
    ax[1].set_yticks([-1] + list(idx.values())); ax[1].set_yticklabels(["unknown"] + names)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("locked section")
    fig.tight_layout(); fig.savefig(OUT / "demo_timeline.png", dpi=120)
    for a, b, n in spans:
        got = [l for tt, l in zip(times, labels) if a <= tt <= b]
        first = next((tt - a for tt, l in zip(times, labels) if a <= tt <= b and l == n), None)
        print(f"  {n:12s} {a:5.1f}-{b:5.1f} s  locked: "
              f"{ 'never' if first is None and n != '(other)' else ('n/a' if n == '(other)' else f'{first:.2f} s after start') }"
              f"  (final label {got[-1] if got else None})")
    print(f"Saved demo_dry.wav, demo_processed.wav, demo_timeline.png in {OUT}/")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if not SECTIONS.exists():
        raise SystemExit("sections/ not found. Create sections/<name>/take_1.wav ... (see the top of this file).")
    audio, other, presets = load_sections()
    if len(audio) < 2:
        raise SystemExit("Need at least 2 sections, each with 2+ takes.")
    print("Sections: " + ", ".join(f"{n} ({len(t)} takes)" for n, t in audio.items())
          + f"; {len(other)} 'other' clips")
    if cmd == "eval":
        evaluate(audio, other)
    elif cmd == "demo":
        demo(audio, other, presets)
    else:
        raise SystemExit("Use: eval | demo")


if __name__ == "__main__":
    main()