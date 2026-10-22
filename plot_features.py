"""
plot_features.py

Plots RMS, spectral centroid, spectral flatness and spectral flux over time
for one clip from each class, side by side, and saves features.png.

Run from the my_pedal_dsp folder with your venv active:
    python plot_features.py                  # first clip of each class
    python plot_features.py 3                # third clip of each class
"""
import sys

import librosa
import matplotlib
matplotlib.use("Agg")  # save to a file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np

from settings import DATA_RAW, DATA_TRIMMED, HOP, OUT_SUFFIX, SR, TRAIN_CLASSES, WIN

# Use the trimmed clips if they exist, otherwise the originals
DATA = DATA_TRIMMED if DATA_TRIMMED.exists() else DATA_RAW
CLASSES = TRAIN_CLASSES


def features(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    # One STFT shared by every feature, so all of them have the same length
    S = np.abs(librosa.stft(y, n_fft=WIN, hop_length=HOP))
    rms = librosa.feature.rms(S=S, frame_length=WIN)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=SR)
    t = librosa.times_like(rms, sr=SR, hop_length=HOP, n_fft=WIN)
    return t, [("RMS", rms), ("Centroid (Hz)", cent), ("Flatness", flat), ("Flux", flux)]


def main():
    idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0

    chosen = []
    for cls in CLASSES:
        files = sorted((DATA / cls).glob("*.wav"))
        if len(files) > idx:
            chosen.append((cls, files[idx]))
        else:
            print(f"No clip number {idx + 1} in {cls}, skipping")
    if not chosen:
        sys.exit(f"No WAV files found under {DATA}/. Run this from the my_pedal_dsp folder.")

    fig, axes = plt.subplots(4, len(chosen), figsize=(4.5 * len(chosen), 8),
                             squeeze=False)
    for col, (cls, path) in enumerate(chosen):
        t, feats = features(path)
        axes[0, col].set_title(f"{cls}\n{path.name}", fontsize=9)
        for row, (name, v) in enumerate(feats):
            axes[row, col].plot(t, v, lw=0.8)
            if col == 0:
                axes[row, col].set_ylabel(name)
        axes[-1, col].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(f"features{OUT_SUFFIX}.png", dpi=110)
    print(f"Saved features{OUT_SUFFIX}.png using clips from {DATA}/")


if __name__ == "__main__":
    main()