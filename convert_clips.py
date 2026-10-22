"""
convert_clips.py

Converts every audio file in data/all_clips/ (m4a, mp3, aac, wav, ...) to
mono 44.1 kHz WAV and sorts it into data/<class>/ based on the filename.

    chords_loud_3.m4a   ->  data/chords_loud/chords_loud_3.wav

Originals in all_clips/ are never modified or deleted.

Run from the my_pedal_dsp folder:
    python convert_clips.py              # skips clips already converted
    python convert_clips.py --overwrite  # reconvert everything

Needs ffmpeg:  sudo apt install ffmpeg
"""
import argparse
import shutil
import subprocess

from settings import DATA_RAW as DATA, SR, TEST_ONLY_CLASSES, TRAIN_CLASSES

SRC = DATA / "all_clips"
CLASSES = TRAIN_CLASSES + TEST_ONLY_CLASSES      # test-only edge cases are converted too

# Filename typos / old names -> correct class prefix
ALIASES = {"chords_heavy": "chords_loud", "sing_soft": "single_soft"}

EXTENSIONS = {".m4a", ".mp3", ".aac", ".wav", ".mp4", ".caf", ".ogg", ".flac", ".aiff"}


def class_for(stem):
    """Return (class, cleaned_stem) or (None, stem) if no class prefix matches."""
    s = stem.strip().lower().replace(" ", "_")
    for old, new in ALIASES.items():
        if s.startswith(old):
            s = new + s[len(old):]
    for cls in CLASSES:
        if s.startswith(cls):
            return cls, s
    return None, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="reconvert files that already exist")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Install it with: sudo apt install ffmpeg")
    if not SRC.exists():
        raise SystemExit(f"{SRC}/ not found. Run this from the my_pedal_dsp folder.")

    files = sorted(p for p in SRC.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files found in {SRC}/")

    converted = skipped = failed = 0
    unmatched = []

    for f in files:
        cls, name = class_for(f.stem)
        if cls is None:
            unmatched.append(f.name)
            continue

        out_dir = DATA / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{name}.wav"

        if out.exists() and not args.overwrite:
            skipped += 1
            continue

        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(f),
             "-ac", "1", "-ar", str(SR), str(out)],
            capture_output=True, text=True)
        if result.returncode == 0:
            print(f"OK    {f.name:28s} -> {out}")
            converted += 1
        else:
            print(f"FAIL  {f.name}: {result.stderr.strip()}")
            failed += 1

    print(f"\nConverted {converted}, skipped {skipped} (already done), failed {failed}.")
    if unmatched:
        print("\nNo class matched these files (rename them to start with a class name):")
        for n in unmatched:
            print(f"  {n}")
        print(f"Known classes: {', '.join(CLASSES)}")


if __name__ == "__main__":
    main()