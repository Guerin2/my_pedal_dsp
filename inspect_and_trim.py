"""
inspect_and_trim.py

For every WAV in data/<class>/:
  1. Trims handling noise from the start and end:
       a) a fixed margin (removes the tap on "record" and "stop")
       b) for playing classes, an energy-based trim (removes dead air
          before you start playing and after you finish)
  2. Writes the trimmed clip to data_trimmed/<class>/ (originals untouched)
  3. Saves one plot per class to plots/, showing waveform + RMS (dB)
     with the removed regions shaded red.

Run from the my_pedal_dsp folder, with your venv active:
    python inspect_and_trim.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

from settings import DATA_RAW, DATA_TRIMMED, OUT_SUFFIX, SILENCE_CLASS   # silence gets fixed trim only, no energy trim

# ---------------- settings you can tweak ----------------
DATA_DIR = DATA_RAW
OUT_DIR = DATA_TRIMMED
PLOT_DIR = Path("plots" + OUT_SUFFIX)

# Fixed trim in seconds, per class ("default" covers any class not listed).
# Silence needs a long start trim because you're still handling the phone,
# and every clip needs a longer end trim to remove the reach for the stop button.
FIXED_TRIM_START_S = {"default": 0.7, "silence": 3.0}
FIXED_TRIM_END_S = {"default": 1.5, "silence": 1.5}
ENERGY_MARGIN_DB = 12      # "playing" = this many dB above the noise floor
PLAYING_DB = -45.0         # ...or simply louder than this (dBFS). Without it, a clip with no
                           # dead air (continuous strumming) has no real noise floor, and only
                           # its loudest few seconds would count as playing.
PAD_START_S = 0.15         # keep a little before the first note
PAD_END_S = 0.50           # keep the decay after the last note
FADE_S = 0.01              # 10 ms fade at each cut to avoid clicks
SILENCE_MAX_S = 15.0       # cap silence clips to this length (None = no cap)

FRAME_S, HOP_S = 0.025, 0.010
# ---------------------------------------------------------


def rms_db(y, sr):
    """Frame-wise RMS in dB. Returns (times, db)."""
    frame, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(y) < frame:
        return np.array([0.0]), np.array([-120.0])
    frames = sliding_window_view(y, frame)[::hop]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    t = (np.arange(len(rms)) * hop + frame / 2) / sr
    return t, 20 * np.log10(rms + 1e-12)


def find_trim(y, sr, cls):
    """Return (start_sample, end_sample) to keep."""
    n = len(y)
    start = int(FIXED_TRIM_START_S.get(cls, FIXED_TRIM_START_S["default"]) * sr)
    end = n - int(FIXED_TRIM_END_S.get(cls, FIXED_TRIM_END_S["default"]) * sr)
    if end <= start:  # clip too short to trim
        return 0, n

    if cls != SILENCE_CLASS:
        t, db = rms_db(y[start:end], sr)
        floor = np.percentile(db, 10)
        active = np.where(db > min(floor + ENERGY_MARGIN_DB, PLAYING_DB))[0]
        if len(active):
            first = start + int(max(t[active[0]] - PAD_START_S, 0) * sr)
            last = start + int((t[active[-1]] + PAD_END_S) * sr)
            start, end = first, min(last, end)
    elif SILENCE_MAX_S is not None:
        end = min(end, start + int(SILENCE_MAX_S * sr))
    return start, end


def fade_edges(y, sr):
    k = min(int(FADE_S * sr), len(y) // 2)
    if k > 0:
        ramp = np.linspace(0, 1, k)
        y = y.copy()
        y[:k] *= ramp
        y[-k:] *= ramp[::-1]
    return y


def main():
    classes = sorted(p.name for p in DATA_DIR.iterdir()
                     if p.is_dir() and p.name != "all_clips")
    PLOT_DIR.mkdir(exist_ok=True)

    for cls in classes:
        files = sorted((DATA_DIR / cls).glob("*.wav"))
        if not files:
            print(f"[{cls}] no clips, skipping")
            continue
        (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(len(files), 1, figsize=(12, 2.2 * len(files)),
                                 squeeze=False)
        for ax, path in zip(axes[:, 0], files):
            y, sr = sf.read(path)
            if y.ndim > 1:
                y = y.mean(axis=1)

            s, e = find_trim(y, sr, cls)
            trimmed = fade_edges(y[s:e], sr)
            sf.write(OUT_DIR / cls / path.name, trimmed, sr)

            peak_db = 20 * np.log10(np.max(np.abs(trimmed)) + 1e-12)
            print(f"[{cls}] {path.name}: {len(y)/sr:5.1f}s -> {len(trimmed)/sr:5.1f}s "
                  f"(cut {s/sr:.2f}s from start, {(len(y)-e)/sr:.2f}s from end), "
                  f"peak after trim {peak_db:.1f} dBFS")
            if cls == SILENCE_CLASS and peak_db > -30:
                print(f"    WARNING: {path.name} still has a loud event "
                      f"({peak_db:.1f} dBFS). Listen to it and consider re-recording.")

            # plot
            t_wav = np.arange(len(y)) / sr
            ax.plot(t_wav, y, color="0.7", lw=0.4)
            ax.set_ylim(-1, 1)
            ax.set_ylabel("amplitude")
            ax.set_title(path.name, fontsize=9, loc="left")
            ax2 = ax.twinx()
            t_db, db = rms_db(y, sr)
            ax2.plot(t_db, db, color="tab:blue", lw=1)
            ax2.set_ylim(-90, 0)
            ax2.set_ylabel("RMS dB")
            for a, b in [(0, s / sr), (e / sr, len(y) / sr)]:
                if b > a:
                    ax.axvspan(a, b, color="red", alpha=0.2)
            ax.set_xlim(0, len(y) / sr)

        axes[-1, 0].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{cls}.png", dpi=110)
        plt.close(fig)

    print(f"\nPlots saved in {PLOT_DIR}/, trimmed clips in {OUT_DIR}/")


if __name__ == "__main__":
    main()