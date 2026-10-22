"""
settings.py

Values shared by more than one script, kept in one place so they can't drift
apart. Settings that only one script uses stay at the top of that script.
"""
import os
from pathlib import Path

SR = 44100                      # sample rate used everywhere (Hz)
BLOCK = 256                     # audio callback block size in samples (~5.8 ms)

# STFT used by the classifier features and the feature plots
WIN, HOP = 1024, 512            # ~23 ms window, ~12 ms hop

# Folders. Two recording sets are kept apart so they can be compared:
#   phone  data/ and data_trimmed/              (the original phone recordings)
#   cable  data_cable/ and data_cable_trimmed/  (record_clips.py, 1/4" to USB-C)
# Pick one with the PEDAL_DATA environment variable, e.g.
#   PEDAL_DATA=cable python main.py --skip convert
DATA_SETS = {"phone": ("data", "data_trimmed"), "cable": ("data_cable", "data_cable_trimmed")}
DATA_SET = os.environ.get("PEDAL_DATA", "phone")
if DATA_SET not in DATA_SETS:
    raise SystemExit(f"PEDAL_DATA must be one of {', '.join(DATA_SETS)} (got {DATA_SET!r}).")
DATA_RAW = Path(DATA_SETS[DATA_SET][0])          # untrimmed clips, <root>/<class>/*.wav
DATA_TRIMMED = Path(DATA_SETS[DATA_SET][1])      # trimmed clips used for training
OUT_SUFFIX = "" if DATA_SET == "phone" else f"_{DATA_SET}"   # keeps each set's result plots apart
SECTIONS = Path("sections")             # section library for the recogniser and the app

# Playing-style classes
TRAIN_CLASSES = ["single_soft", "chords_loud", "palm_mute", "silence"]
TEST_ONLY_CLASSES = ["chords_soft", "single_loud"]     # edge cases, never trained on
SILENCE_CLASS = "silence"
