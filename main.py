"""
main.py

Runs the whole pipeline in order:
    1. convert_clips.py     data/all_clips/*.m4a  ->  data/<class>/*.wav
    2. inspect_and_trim.py  data/<class>/*.wav    ->  data_trimmed/<class>/*.wav + plots/
    3. plot_features.py     data_trimmed/         ->  features.png
    4. classify_windows.py  data_trimmed/         ->  accuracy, confusion and ablation plots
    5. realtime_demo.py     data_trimmed/         ->  demo/ (stitched test audio, effect on/off, plot)

Put this file in the same folder as the other four scripts (my_pedal_dsp/)
and run it with your venv active:

    python main.py                  # run everything
    python main.py --skip convert   # skip a step (convert, trim, features, classify, demo)
    python main.py --only classify  # run just one step
    python main.py --overwrite      # reconvert all clips in step 1
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (short name, script, description)
STEPS = [
    ("convert", "convert_clips.py", "Convert and sort clips into class folders"),
    ("trim", "inspect_and_trim.py", "Trim start/end noise and plot each clip"),
    ("features", "plot_features.py", "Plot features for one clip per class"),
    ("classify", "classify_windows.py", "Window-length experiment and classifiers"),
    ("demo", "realtime_demo.py", "Stitched test file, causal analysis and effect demo"),
]
NAMES = [s[0] for s in STEPS]


def main():
    parser = argparse.ArgumentParser(description="Run the pedal prototype pipeline.")
    parser.add_argument("--skip", nargs="+", choices=NAMES, default=[],
                        help="steps to skip")
    parser.add_argument("--only", nargs="+", choices=NAMES, default=None,
                        help="run only these steps")
    parser.add_argument("--overwrite", action="store_true",
                        help="pass --overwrite to convert_clips.py")
    args = parser.parse_args()

    selected = [s for s in STEPS
                if (args.only is None or s[0] in args.only) and s[0] not in args.skip]
    if not selected:
        raise SystemExit("Nothing to run.")

    missing = [s[1] for s in selected if not (HERE / s[1]).exists()]
    if missing:
        raise SystemExit("Missing scripts next to main.py: " + ", ".join(missing))

    start = time.time()
    for i, (name, script, desc) in enumerate(selected, 1):
        print(f"\n{'=' * 70}\nStep {i}/{len(selected)}: {desc}  [{script}]\n{'=' * 70}", flush=True)
        cmd = [sys.executable, script]
        if name == "convert" and args.overwrite:
            cmd.append("--overwrite")

        t0 = time.time()
        result = subprocess.run(cmd, cwd=HERE)   # scripts use paths relative to this folder
        if result.returncode != 0:
            raise SystemExit(f"\nStep '{name}' failed (exit code {result.returncode}). "
                             f"Fix the error above and re-run. "
                             f"Steps before it already completed, so you can resume with "
                             f"--skip {' '.join(s[0] for s in selected[:i - 1]) or '<none>'}.")
        print(f"[{name} finished in {time.time() - t0:.1f}s]")

    print(f"\nAll done in {time.time() - start:.1f}s. Outputs in this folder:")
    print("  plots/                        per-class clip plots with trimmed regions")
    print("  features.png                  feature curves, one clip per class")
    print("  window_length_vs_accuracy.png")
    print("  confusion_matrices.png")
    print("  ablation_window_length.png")
    print("  demo/                         stitched WAVs per mode, demo_summary.png, model.joblib")


if __name__ == "__main__":
    main()