"""
realtime_demo.py

Simulates the pedal on a test file with KNOWN change points, as if the audio
arrived live:

  1. Trains the logistic-regression model on your clips, holding out one soft
     and one loud clip so the demo is tested on playing the model never saw.
  2. Stitches the held-out clips into  soft | loud | soft | loud  (8 s each).
  3. Streams the file causally: every 50 ms it looks only at the last
     STREAM_WINDOW_S seconds of audio, computes the same features as training,
     and gets a "heaviness" score (probability the playing is loud/heavy).
  4. Turns scores into an effect on/off decision in three ways:
        raw + hard switch          threshold at 0.5, no smoothing
        hysteresis + hard switch   smoothing, two thresholds, minimum hold time
        hysteresis + crossfade     as above, plus a 30 ms wet/dry ramp
  5. Applies your effect chain (my_effects.py) block by block, like an audio callback:
     compressor + distortion + chorus faded in by the decision, delay + reverb tails that ring out.
  6. Reports switch delays, flip-flops and analysis cost, saves WAVs and a plot.

Outputs go to demo/ (demo_cable/ for the cable set):  stitched_*.wav, demo_summary.png, model.joblib

Run from the my_pedal_dsp folder (after main.py has made data_trimmed/):
    python realtime_demo.py
KEEP YOUR VOLUME LOW when playing the distorted files.
"""
import time
from pathlib import Path

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import my_effects as fx
from settings import BLOCK, OUT_SUFFIX, SILENCE_CLASS   # decisions are held during silence
from classify_windows import (DATA, FEATURE_NAMES, SR, TRAIN_CLASSES, build_windows,
                              frame_features_from_audio, load_clips, logistic,
                              window_vector)

# ---------------- settings you can tweak ----------------
STREAM_WINDOW_S = 0.5       # how much past audio each decision sees (= reaction time)
ANALYSIS_HOP_S = 0.05       # how often a new decision is made

HOLDOUT_CLASSES = ("single_soft", "chords_loud")   # last clip of each is kept out of training
SEGMENT_ORDER = ["single_soft", "chords_loud", "single_soft", "chords_loud"]
SEGMENT_S = 8.0

HEAVY_CLASSES = ("chords_loud", "palm_mute")       # classes that should switch the effect on

# Tuned on the cable set (5 folds of held-out soft / chords / palm-mute clips stitched together):
# 250 ms, alpha 0.5, 0.65/0.35 gave 33 flips/min (ideal ~11) and 23% of the time in the wrong state;
# these give 15 flips/min and 16% wrong, at the cost of switching ~0.5 s after a change instead of ~0.2 s.
ON_THRESH, OFF_THRESH = 0.8, 0.2     # hysteresis thresholds on the smoothed score
SMOOTH_ALPHA = 0.3                   # weight of the newest score in the smoothing
MIN_HOLD_S = 0.5                     # minimum time between switches
RAMP_S = 0.03                        # crossfade time for the "crossfade" mode

# Effect chains (edit freely; any class from my_effects.py works)
def make_heavy_chain():
    """Faded in by the classifier's decision (wet/dry crossfade)."""
    return [fx.Compressor(), fx.Distortion(gain=25.0, level=0.25), fx.Chorus(mix=0.3)]


def make_tail_chain():
    """Fed by the heavy signal but running all the time, so echoes and reverb ring out
    after the heavy effect switches off instead of being cut dead."""
    return [fx.Delay(delay_ms=320, feedback=0.35, mix=0.3), fx.Reverb(room=0.5, mix=0.2)]


BLOCK_CHAIN_LIMIT = 0.9              # soft ceiling on the output
OUT_DIR = Path("demo" + OUT_SUFFIX)
# ---------------------------------------------------------

MODES = {
    "raw + hard switch": dict(hysteresis=False, ramp_s=0.0),
    "hysteresis + hard switch": dict(hysteresis=True, ramp_s=0.0),
    "hysteresis + crossfade": dict(hysteresis=True, ramp_s=RAMP_S),
}


def slug(name):
    return name.replace(" + ", "_").replace(" ", "_")


# ---------- 1. training ----------
def pick_holdout(clips):
    held = set()
    for cls in HOLDOUT_CLASSES:
        names = sorted(c[1] for c in clips if c[0] == cls)
        if len(names) < 2:
            raise SystemExit(f"Need at least 2 clips in '{cls}' (one is held out for the demo).")
        held.add((cls, names[-1]))
    return held


def train_model(clips, held):
    train = [c for c in clips if (c[0], c[1]) not in held]
    X, y, _ = build_windows(train, STREAM_WINDOW_S)
    return logistic().fit(X, y)


# ---------- 2. test signal ----------
def make_test_signal(held):
    audios = {cls: librosa.load(DATA / cls / f"{name}.wav", sr=SR, mono=True)[0]
              for cls, name in held}
    seg_s = min(SEGMENT_S, np.floor(min(len(a) for a in audios.values()) / SR / 2 * 10) / 10)
    seg = int(seg_s * SR)
    fade = int(0.01 * SR)
    used, parts, labels, changes = {}, [], [], []
    t = 0.0
    for cls in SEGMENT_ORDER:
        k = used.get(cls, 0)
        used[cls] = k + 1
        a = audios[cls][k * seg:(k + 1) * seg].copy()
        ramp = np.linspace(0, 1, fade)
        a[:fade] *= ramp
        a[-fade:] *= ramp[::-1]
        parts.append(a)
        labels.append((t, t + seg_s, cls))
        if t > 0:
            changes.append((t, cls in HEAVY_CLASSES))
        t += seg_s
    return np.concatenate(parts), labels, changes


# ---------- 3. causal analysis ----------
def analyse(audio, model):
    """Look at the audio exactly as a ring buffer would: only the past window."""
    win, hop = int(STREAM_WINDOW_S * SR), int(ANALYSIS_HOP_S * SR)
    classes = list(model.classes_)
    heavy_idx = [classes.index(c) for c in HEAVY_CLASSES if c in classes]
    sil_idx = classes.index(SILENCE_CLASS) if SILENCE_CLASS in classes else None

    times, scores, silent, cost = [], [], [], []
    for end in range(win, len(audio) + 1, hop):
        t0 = time.perf_counter()
        b = frame_features_from_audio(audio[end - win:end])
        vec = np.array(window_vector(b, 0, len(b["rms_db"]))).reshape(1, -1)
        p = model.predict_proba(vec)[0]
        cost.append(time.perf_counter() - t0)

        p_sil = p[sil_idx] if sil_idx is not None else 0.0
        p_heavy = p[heavy_idx].sum()
        p_light = max(1.0 - p_sil - p_heavy, 0.0)
        times.append(end / SR)
        scores.append(p_heavy / (p_heavy + p_light + 1e-9))
        silent.append(p_sil > 0.5)
    return np.array(times), np.array(scores), np.array(silent), np.array(cost)


# ---------- 4. decision logic ----------
def decide(times, scores, silent, hysteresis):
    state, last_switch, smooth = 0, -1e9, None
    states = []
    for t, s, sil in zip(times, scores, silent):
        if not sil:
            if hysteresis:
                smooth = s if smooth is None else SMOOTH_ALPHA * s + (1 - SMOOTH_ALPHA) * smooth
                if t - last_switch >= MIN_HOLD_S:
                    if state == 0 and smooth > ON_THRESH:
                        state, last_switch = 1, t
                    elif state == 1 and smooth < OFF_THRESH:
                        state, last_switch = 0, t
            else:
                state = int(s > 0.5)
        states.append(state)
    return np.array(states)


# ---------- 5. effect, block by block ----------
def run_chain(effects, block):
    for e in effects:
        block = e.process(block)
    return block


def prepare_chain(effects):
    for e in effects:
        e.prepare(SR, BLOCK)
        e.reset()
    return effects


def calibrate_wet_gain(audio):
    """One constant so the heavy path is about as loud as the dry signal (otherwise the
    switch is mostly a volume jump). A calibration over the test file, not part of the live path."""
    chain = prepare_chain(make_heavy_chain())
    wet = np.concatenate([run_chain(chain, audio[i:i + BLOCK])
                          for i in range(0, len(audio), BLOCK)])
    return float(np.sqrt(np.mean(audio ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-12))


def apply_effect(audio, times, states, ramp_s, wet_gain):
    """Block by block, like an audio callback. The heavy chain runs on every block (so its
    internal state is always warm); the decision only sets the wet/dry mix. A tail chain
    (delay + reverb) is fed by the heavy path and keeps ringing after the mix returns to 0."""
    n = len(audio)
    heavy = prepare_chain(make_heavy_chain())
    tails = prepare_chain(make_tail_chain())
    out = np.empty(n)
    mix = 0.0
    step = 1.0 / (ramp_s * SR) if ramp_s > 0 else None
    for start in range(0, n, BLOCK):
        blk = audio[start:start + BLOCK]
        m = len(blk)
        idx = np.searchsorted(times, start / SR, side="right") - 1   # latest decision so far
        target = float(states[idx]) if idx >= 0 else 0.0
        if step is None:
            mixv = np.full(m, target)
            mix = target
        else:
            move = np.clip(target - mix, -step * m, step * m)
            mixv = mix + move * np.arange(1, m + 1) / m
            mix += move
        wet = run_chain(heavy, blk) * wet_gain
        blended = (1 - mixv) * blk + mixv * wet
        send = mixv * blended                                  # tails only hear the heavy part
        added = run_chain(tails, send) - send                  # just what the tails add
        out[start:start + m] = BLOCK_CHAIN_LIMIT * np.tanh((blended + added) / BLOCK_CHAIN_LIMIT)
    return out


# ---------- 6. metrics ----------
def switch_delays(times, states, changes):
    delays = []
    for t_c, to_heavy in changes:
        want = int(to_heavy)
        before = np.where(times < t_c)[0]
        if len(before) and states[before[-1]] == want:
            delays.append(float("nan"))        # already in that state: not a real detection
            continue
        later = np.where((times >= t_c) & (states == want))[0]
        delays.append(times[later[0]] - t_c if len(later) else float("nan"))
    return delays


def main():
    if not DATA.exists():
        raise SystemExit("data_trimmed/ not found. Run main.py first (from the my_pedal_dsp folder).")
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading clips and training...")
    clips = load_clips(TRAIN_CLASSES)
    held = pick_holdout(clips)
    print("Held out (never seen in training): " + ", ".join(f"{c}/{n}" for c, n in sorted(held)))
    model = train_model(clips, held)
    joblib.dump({"model": model, "window_s": STREAM_WINDOW_S,
                 "classes": list(model.classes_), "feature_names": FEATURE_NAMES},
                OUT_DIR / "model.joblib")

    audio, labels, changes = make_test_signal(held)
    print(f"\nTest file: {len(audio)/SR:.1f} s, change points at "
          + ", ".join(f"{t:.0f}s ({'->heavy' if h else '->soft'})" for t, h in changes))

    print(f"Streaming analysis (window {STREAM_WINDOW_S*1000:.0f} ms, new decision every "
          f"{ANALYSIS_HOP_S*1000:.0f} ms)...")
    times, scores, silent, cost = analyse(audio, model)

    # analysis cost vs the time budget
    ms = cost * 1000
    print(f"  analysis cost per decision: mean {ms.mean():.1f} ms, 95th pct {np.percentile(ms, 95):.1f} ms, "
          f"max {ms.max():.1f} ms  (budget: {ANALYSIS_HOP_S*1000:.0f} ms between decisions)")
    if ms.max() > ANALYSIS_HOP_S * 1000:
        print("  WARNING: some decisions took longer than the hop. A live version would fall behind "
              "(this is why real-time code needs a lean feature path).")

    wet_gain = calibrate_wet_gain(audio)
    print(f"  heavy chain level-matched to the dry signal (gain {wet_gain:.2f})")

    # decisions, effect, metrics
    outputs, all_states = {}, {}
    print("\nResults:")
    print(f"  {'mode':26s} {'switch delays (s)':32s} flips (ideal {len(changes)})")
    for name, cfg in MODES.items():
        states = decide(times, scores, silent, cfg["hysteresis"])
        all_states[name] = states
        outputs[name] = apply_effect(audio, times, states, cfg["ramp_s"], wet_gain)
        delays = switch_delays(times, states, changes)
        flips = int(np.sum(np.diff(states) != 0))
        d = ", ".join("n/a" if np.isnan(x) else f"{x:.2f}" for x in delays)
        print(f"  {name:26s} {d:32s} {flips}")

    # write audio, all at one common gain so loudness comparisons are fair
    peak = max(np.max(np.abs(audio)), *(np.max(np.abs(o)) for o in outputs.values()))
    gain = 0.8 / peak
    sf.write(OUT_DIR / "stitched_dry.wav", audio * gain, SR)
    for name, o in outputs.items():
        sf.write(OUT_DIR / f"stitched_{slug(name)}.wav", np.clip(o * gain, -1, 1), SR)

    # ---------- plot ----------
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.3, 1])
    t_audio = np.arange(len(audio)) / SR

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t_audio, audio, color="0.6", lw=0.4)
    for a, b, cls in labels:
        ax.text((a + b) / 2, ax.get_ylim()[1] * 0.85, cls, ha="center", fontsize=9)
    ax.set_ylabel("dry audio")
    ax.set_xlim(0, t_audio[-1])

    ax = fig.add_subplot(gs[1, :], sharex=ax)
    ax.plot(times, scores, color="tab:blue", lw=1, label="heaviness score (raw)")
    ax.axhline(ON_THRESH, color="grey", ls=":", lw=0.8)
    ax.axhline(OFF_THRESH, color="grey", ls=":", lw=0.8)
    styles = {"raw + hard switch": ("tab:red", "--"), "hysteresis + crossfade": ("tab:green", "-")}
    for name, (colour, ls) in styles.items():
        ax.step(times, all_states[name] * 0.9 + 0.05, where="post", color=colour, ls=ls,
                lw=1.5, label=f"effect on/off: {name}")
    for t_c, _ in changes:
        ax.axvline(t_c, color="k", ls="--", lw=0.8)
    ax.set_ylabel("score / effect state")
    ax.set_xlabel("time (s)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, loc="center right")

    first_on = np.where(all_states["hysteresis + hard switch"] == 1)[0]
    if len(first_on):
        t_s = times[first_on[0]]
        for col, name in enumerate(["hysteresis + hard switch", "hysteresis + crossfade"]):
            axz = fig.add_subplot(gs[2, col])
            lo, hi = int((t_s - 0.05) * SR), int((t_s + 0.15) * SR)
            axz.plot(t_audio[lo:hi], outputs[name][lo:hi], lw=0.6)
            axz.axvline(t_s, color="k", ls=":", lw=0.8)
            axz.set_title(f"zoom at first switch: {name}", fontsize=9)
            axz.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "demo_summary.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved to {OUT_DIR}/: stitched_dry.wav, one WAV per mode, demo_summary.png, model.joblib")


if __name__ == "__main__":
    main()