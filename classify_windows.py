"""
classify_windows.py  (version 2)

1. Loads every clip in data_trimmed/<class>/
2. Computes frame-level features once per clip, then summarises them over
   analysis windows of different lengths (50 ms ... 2 s)
3. Trains baseline classifiers with leave-one-CLIP-out cross-validation
   (windows from the same clip never appear in both train and test)
4. Saves:
     window_length_vs_accuracy.png   accuracy vs window length, with chance line
     confusion_matrices.png          at REPORT_WINDOW_S
     ablation_window_length.png      which feature groups carry the result
   and prints per-clip accuracies, feature importances and, if you have
   recorded them, how the model labels the test-only edge cases.

Run from the my_pedal_dsp folder with your venv active:
    python classify_windows.py
"""

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from settings import DATA_SET, HOP, OUT_SUFFIX, SR, TEST_ONLY_CLASSES, TRAIN_CLASSES, WIN
from settings import DATA_TRIMMED as DATA

# ---------------- settings you can tweak ----------------
# (sample rate, STFT size, data folder and class lists live in settings.py)
WINDOW_LENGTHS_S = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
REPORT_WINDOW_S = 0.5                 # window used for the detailed reports
ONSET_THRESH = 5.0                    # flux height that counts as a note onset
MIN_CLIPS_WARNING = 4                 # warn if a class has fewer clips than this
DATASET_PREFIX = "idmt_"              # clips made by fetch_dataset.py
DECAY_S = 0.12                        # how long after a note starts its decay is measured
SUSTAIN_DB = 10.0                     # "still ringing" = within this many dB of the window's peak
CONTRAST_BANDS = 5                    # spectral contrast: peak-vs-valley in 5 octave bands from 100 Hz
                                      # (+1 for the top band); palm muting flattens the upper bands
# ---------------------------------------------------------

FRAME_S = HOP / SR
FEATURE_NAMES = ["rms_mean_db", "rms_max_db", "rms_std_db", "rms_range_db",
                 "centroid_mean", "flatness_mean", "flatness_max",
                 "flux_mean", "flux_max", "onsets_per_s",
                 "decay_db_per_s", "sustain_frac"] + [f"contrast_{i}" for i in range(CONTRAST_BANDS + 1)]

# Feature subsets for the ablation experiment
ALL_IDX = list(range(len(FEATURE_NAMES)))
NO_RMS_IDX = [i for i, n in enumerate(FEATURE_NAMES) if not n.startswith("rms_")]
RMS_ONLY_IDX = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("rms_")]
DENSITY_IDX = [FEATURE_NAMES.index(n) for n in ("flux_mean", "flux_max", "onsets_per_s")]
DECAY_IDX = [FEATURE_NAMES.index(n) for n in ("decay_db_per_s", "sustain_frac")]
CONTRAST_IDX = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("contrast_")]
SUBSETS = {
    "All features": ALL_IDX,
    "No loudness (RMS removed)": NO_RMS_IDX,
    "Loudness only (RMS)": RMS_ONLY_IDX,
    "Note density only (flux, onsets)": DENSITY_IDX,
    "Decay only (note length)": DECAY_IDX,
    "Spectral contrast only": CONTRAST_IDX,
    "Old features (no decay, no contrast)": [i for i in ALL_IDX if i not in DECAY_IDX + CONTRAST_IDX],
}


def frame_features(path):
    """Frame-level features for one clip file."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return frame_features_from_audio(y)


def frame_features_from_audio(y):
    """Frame-level features for an audio array (all returned arrays have the same length).
    Shared with realtime_demo.py so offline training and streaming use identical code."""
    S = np.abs(librosa.stft(y, n_fft=WIN, hop_length=HOP))
    rms = librosa.feature.rms(S=S, frame_length=WIN)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    flux = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S, ref=1.0, top_db=None), sr=SR, hop_length=HOP)
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR, n_bands=CONTRAST_BANDS, fmin=100)
    n = min(len(rms), len(cent), len(flat), len(flux), contrast.shape[1])
    flux = flux[:n]
    peaks, _ = find_peaks(flux, height=ONSET_THRESH, distance=3)
    onsets = np.zeros(n, dtype=bool)
    onsets[peaks] = True
    return {"rms_db": 20 * np.log10(rms[:n] + 1e-6), "cent": cent[:n],
            "flat": flat[:n], "flux": flux, "onsets": onsets,
            "contrast": contrast[:, :n].T}                   # (frames, bands)


def decay_rate(r, onsets):
    """Mean dB per second the level falls in the DECAY_S after each note start inside the
    window (palm mutes die fast, open chords ring). Only uses frames inside the window, so it
    stays causal. 0 if no note starts early enough in the window to measure."""
    k = max(2, int(round(DECAY_S / FRAME_S)))
    rates = [(r[p:p + 3].max() - r[p + k]) / (k * FRAME_S)
             for p in np.flatnonzero(onsets) if p + k < len(r)]
    return float(np.mean(rates)) if rates else 0.0


def window_vector(b, s, e):
    r, c, f, x = b["rms_db"][s:e], b["cent"][s:e], b["flat"][s:e], b["flux"][s:e]
    dur = (e - s) * FRAME_S
    return [r.mean(), r.max(), r.std(), np.percentile(r, 90) - np.percentile(r, 10),
            c.mean(), f.mean(), f.max(), x.mean(), x.max(), b["onsets"][s:e].sum() / dur,
            decay_rate(r, b["onsets"][s:e]), float(np.mean(r > r.max() - SUSTAIN_DB))] \
        + list(b["contrast"][s:e].mean(axis=0))


def build_windows(clips, window_s):
    """clips: list of (class, clip_name, frame_features). Returns X, y, groups."""
    w = max(2, int(round(window_s / FRAME_S)))
    step = max(1, w // 2)
    X, y, g = [], [], []
    for cls, name, b in clips:
        n = len(b["rms_db"])
        for s in range(0, n - w + 1, step):
            X.append(window_vector(b, s, s + w))
            y.append(cls)
            g.append(f"{cls}/{name}")
    return np.array(X), np.array(y), np.array(g)


def load_clips(classes):
    clips = []
    for cls in classes:
        files = sorted((DATA / cls).glob("*.wav"))
        for f in files:
            clips.append((cls, f.stem, frame_features(f)))
        if files:
            warn = "   <-- few clips, results will be noisy" if len(files) < MIN_CLIPS_WARNING else ""
            print(f"  {cls}: {len(files)} clips{warn}")
    return clips


def logistic():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def tree():
    return DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0)


def loo_predict(model, X, y, g):
    return cross_val_predict(model, X, y, groups=g, cv=LeaveOneGroupOut())


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA}/ not found. Run inspect_and_trim.py first, "
                         "and run this from the my_pedal_dsp folder.")

    print(f"Recording set: {DATA_SET} ({DATA}/)")
    print("Loading training clips and computing frame features...")
    clips = load_clips(TRAIN_CLASSES)
    labels = sorted({c[0] for c in clips})
    if len(labels) < 2:
        raise SystemExit("Need clips from at least two classes.")
    chance = 1 / len(labels)

    # Build every window set once and reuse it
    lengths = sorted(set(WINDOW_LENGTHS_S) | {REPORT_WINDOW_S})
    built = {W: build_windows(clips, W) for W in lengths}

    # ---------- window-length experiment ----------
    print("\nWindow-length experiment (leave-one-clip-out, balanced accuracy):")
    results = {"Logistic regression": [], "Decision tree (depth 4)": []}
    makers = {"Logistic regression": logistic, "Decision tree (depth 4)": tree}
    for W in WINDOW_LENGTHS_S:
        X, y, g = built[W]
        row = f"  {W*1000:6.0f} ms  ({len(y)} windows)  "
        for name, make in makers.items():
            acc = balanced_accuracy_score(y, loo_predict(make(), X, y, g))
            results[name].append(acc)
            row += f"{name.split(' ')[0]}={acc:.2f}  "
        print(row)
    print(f"  (chance level for {len(labels)} classes = {chance:.2f})")

    plt.figure(figsize=(7, 4.5))
    for name, accs in results.items():
        plt.plot(np.array(WINDOW_LENGTHS_S) * 1000, accs, marker="o", label=name)
    plt.axhline(chance, color="grey", ls="--", label=f"chance ({chance:.2f})")
    plt.xscale("log")
    plt.xlabel("analysis window length (ms)")
    plt.ylabel("balanced accuracy (leave-one-clip-out)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Deciding faster vs having more context")
    plt.tight_layout()
    plt.savefig(f"window_length_vs_accuracy{OUT_SUFFIX}.png", dpi=120)
    plt.close()

    # ---------- confusion matrices + per-clip accuracy at one window length ----------
    X, y, g = built[REPORT_WINDOW_S]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    preds = {}
    for ax, (name, make) in zip(axes, makers.items()):
        preds[name] = loo_predict(make(), X, y, g)
        cm = confusion_matrix(y, preds[name], labels=labels)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, colorbar=False, xticks_rotation=30)
        ax.set_title(f"{name}\n{REPORT_WINDOW_S*1000:.0f} ms windows")
    plt.tight_layout()
    plt.savefig(f"confusion_matrices{OUT_SUFFIX}.png", dpi=120)
    plt.close()

    print(f"\nPer-clip accuracy, logistic regression, {REPORT_WINDOW_S*1000:.0f} ms windows "
          f"(worst first):")
    pl = preds["Logistic regression"]
    per_clip = sorted(((np.mean(pl[g == grp] == y[g == grp]), grp, int(np.sum(g == grp)))
                       for grp in np.unique(g)))
    for acc, grp, n in per_clip:
        print(f"  {acc:5.0%}  {grp:30s} ({n} windows)")

    # ---------- which features matter? ----------
    t = tree().fit(X, y)
    print(f"\nTree feature importances at {REPORT_WINDOW_S*1000:.0f} ms windows:")
    for i in np.argsort(t.feature_importances_)[::-1][:5]:
        print(f"  {FEATURE_NAMES[i]:16s} {t.feature_importances_[i]:.2f}")

    # ---------- ablation: how much is just loudness? ----------
    print("\nAblation (logistic regression, balanced accuracy):")
    plt.figure(figsize=(7.5, 4.8))
    for label, idx in SUBSETS.items():
        accs = []
        for W in WINDOW_LENGTHS_S:
            Xw, yw, gw = built[W]
            accs.append(balanced_accuracy_score(
                yw, loo_predict(logistic(), Xw[:, idx], yw, gw)))
        plt.plot(np.array(WINDOW_LENGTHS_S) * 1000, accs, marker="o", label=label)
        at_report = accs[WINDOW_LENGTHS_S.index(REPORT_WINDOW_S)] \
            if REPORT_WINDOW_S in WINDOW_LENGTHS_S else float("nan")
        print(f"  {label:34s} {REPORT_WINDOW_S*1000:.0f} ms: {at_report:.2f}   "
              f"2 s: {accs[-1]:.2f}")
    plt.axhline(chance, color="grey", ls="--", label=f"chance ({chance:.2f})")
    plt.xscale("log")
    plt.xlabel("analysis window length (ms)")
    plt.ylabel("balanced accuracy (leave-one-clip-out)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.title("Which features carry the result?")
    plt.tight_layout()
    plt.savefig(f"ablation_window_length{OUT_SUFFIX}.png", dpi=120)
    plt.close()

    # ---------- dataset -> your own recordings ----------
    # Leave-one-clip-out mixes dataset and own clips in training, so it can't show
    # whether a model built from dataset clips works on your phone recordings.
    # Here: train only on the idmt_ clips, test only on your clips of those classes.
    is_ds = np.array([grp.split("/", 1)[1].startswith(DATASET_PREFIX) for grp in g])
    ds_classes = sorted(set(y[is_ds]))
    own_test = (~is_ds) & np.isin(y, ds_classes)
    if len(ds_classes) >= 2 and own_test.any():
        print(f"\nTrained on dataset clips only ({', '.join(ds_classes)}), "
              f"tested on your own clips, {REPORT_WINDOW_S*1000:.0f} ms windows:")
        for label, cols in (("all features", ALL_IDX), ("no RMS", NO_RMS_IDX)):
            m = logistic().fit(X[is_ds][:, cols], y[is_ds])
            p = m.predict(X[own_test][:, cols])
            yt = y[own_test]
            print(f"  [{label:12s}] accuracy {np.mean(p == yt):.0%} over {len(yt)} windows")
            for cls in sorted(set(yt)):
                dist = ", ".join(f"{c} {np.mean(p[yt == cls] == c):.0%}" for c in ds_classes)
                print(f"      your {cls:12s} -> {dist}")
        missing = [c for c in ds_classes if c not in set(y[own_test])]
        if missing:
            print(f"  (No own clips yet for {', '.join(missing)}, so that class isn't tested.)")

    # ---------- edge cases (test-only) ----------
    edge = [c for c in TEST_ONLY_CLASSES if list((DATA / c).glob("*.wav"))]
    if edge:
        print(f"\nEdge cases at {REPORT_WINDOW_S*1000:.0f} ms windows "
              f"(trained on main classes only):")
        full = logistic().fit(X, y)
        no_rms = logistic().fit(X[:, NO_RMS_IDX], y)
        Xe, ye, _ = build_windows(load_clips(edge), REPORT_WINDOW_S)
        for cls in edge:
            m = ye == cls
            for label, model, cols in (("all features", full, ALL_IDX),
                                       ("no RMS", no_rms, NO_RMS_IDX)):
                p = model.predict(Xe[m][:, cols])
                dist = ", ".join(f"{c} {np.mean(p == c):.0%}" for c in labels)
                print(f"  {cls:12s} [{label:12s}] {dist}")
    else:
        print(f"\n(No edge-case clips yet in {DATA}/chords_soft or single_loud.)")

    print(f"\nSaved window_length_vs_accuracy{OUT_SUFFIX}.png, confusion_matrices{OUT_SUFFIX}.png, "
          f"ablation_window_length{OUT_SUFFIX}.png")


if __name__ == "__main__":
    main()