"""
record_clips.py

Records training clips straight from an audio input (e.g. the 1/4" to USB-C
converter), so you don't have to go through the phone and convert_clips.py.

Clips go into a SEPARATE recording set (data_cable/ by default, see settings.py),
so the phone clips stay untouched and the two sets can be compared:
    data_cable/<class>/cable_<class>_<k>.wav      44.1 kHz mono

Steps (run from the my_pedal_dsp folder, venv active):
    python record_clips.py --list                         which inputs exist
    python record_clips.py --device "USB Audio" --meter   set your guitar volume:
                                                          strum hard, stay out of the red
    python record_clips.py --device "USB Audio" --class palm_mute -n 5 -s 20
        5 clips of 20 s, with a 3 s countdown before each (k continues after
        any clips already there, so nothing is overwritten)

Then train and test on the cable set:
    PEDAL_DATA=cable python main.py --skip convert

Keep the input gain (guitar volume, laptop input level) the same for every
clip and for the live demo: loudness is one of the classifier's strongest features.
"""
import argparse
import sys
import time

import numpy as np
import soundfile as sf

from settings import DATA_SETS, SR, TEST_ONLY_CLASSES, TRAIN_CLASSES

CLASSES = TRAIN_CLASSES + TEST_ONLY_CLASSES
CLIP_DB = -1.0            # peaks at or above this count as clipping
QUIET_DB = -35.0          # a playing clip whose peak stays below this is probably too quiet
COUNTDOWN_S = 3
METER_WIDTH = 40


def find_device(name_or_index):
    import sounddevice as sd
    devs = sd.query_devices()
    if name_or_index is None:
        return sd.default.device[0]
    if str(name_or_index).isdigit():
        return int(name_or_index)
    hits = [i for i, d in enumerate(devs)
            if name_or_index.lower() in d["name"].lower() and d["max_input_channels"] > 0]
    if not hits:
        raise SystemExit(f"No input device matching {name_or_index!r}. Try: python record_clips.py --list")
    return hits[0]


def list_inputs():
    import sounddevice as sd
    print("Input devices (use the number or part of the name with --device):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i:3d}  {d['name']}  ({d['max_input_channels']} in, "
                  f"default {d['default_samplerate']:.0f} Hz)")


def device_rate(dev):
    """SR if the device takes it, otherwise its own default rate (resampled afterwards)."""
    import sounddevice as sd
    try:
        sd.check_input_settings(device=dev, samplerate=SR, channels=1)
        return SR
    except Exception:
        return int(sd.query_devices(dev)["default_samplerate"])


def db(x):
    return 20 * np.log10(x + 1e-9)


def meter_line(peak, label=""):
    p = db(peak)
    n = int(np.clip((p + 60) / 60, 0, 1) * METER_WIDTH)
    warn = "  CLIPPING, turn down" if p >= CLIP_DB else ("  quiet" if p < QUIET_DB else "")
    return f"\r{label}[{'#' * n}{' ' * (METER_WIDTH - n)}] {p:6.1f} dBFS{warn}   "


class Meter:
    """Input stream that keeps the latest block peak and, while recording, the audio."""

    def __init__(self, dev, rate):
        import sounddevice as sd
        self.peak, self.recording, self.blocks = 0.0, False, []
        self.stream = sd.InputStream(device=dev, samplerate=rate, channels=1, dtype="float32",
                                     blocksize=1024, callback=self._cb)

    def _cb(self, indata, frames, t, status):
        x = indata[:, 0]
        self.peak = max(float(np.abs(x).max()), self.peak * 0.85)    # falls back slowly, like a meter
        if self.recording:
            self.blocks.append(x.copy())

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *a):
        self.stream.stop(); self.stream.close()


def run_meter(dev, rate):
    print("Live input level. Play your loudest and softest; Ctrl-C to stop.")
    with Meter(dev, rate) as m:
        try:
            while True:
                sys.stdout.write(meter_line(m.peak)); sys.stdout.flush()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()


def next_index(folder, prefix):
    used = [int(p.stem.rsplit("_", 1)[1]) for p in folder.glob(f"{prefix}_*.wav")
            if p.stem.rsplit("_", 1)[1].isdigit()]
    return max(used, default=0) + 1


def check_clip(y, cls):
    """Warnings for one recorded clip."""
    out = []
    clipped = int(np.sum(np.abs(y) >= 10 ** (CLIP_DB / 20)))
    if clipped:
        out.append(f"{clipped} clipped samples: turn the guitar or input level down and redo it")
    if cls != "silence" and db(np.abs(y).max()) < QUIET_DB:
        out.append("very quiet: is the guitar volume up?")
    return out


def record(dev, rate, cls, n, seconds, root):
    folder = root / cls
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"cable_{cls}"
    k = next_index(folder, prefix)
    print(f"Recording {n} clip(s) of {seconds:.0f} s of '{cls}' into {folder}/ (Ctrl-C to stop early)")
    with Meter(dev, rate) as m:
        try:
            for i in range(n):
                end = time.time() + COUNTDOWN_S
                while time.time() < end:
                    sys.stdout.write(meter_line(m.peak, f"clip {i + 1}/{n} starts in {end - time.time():3.1f} s "))
                    sys.stdout.flush(); time.sleep(0.05)
                m.blocks, m.recording = [], True
                end = time.time() + seconds
                while time.time() < end:
                    sys.stdout.write(meter_line(m.peak, f"RECORDING {end - time.time():4.1f} s "))
                    sys.stdout.flush(); time.sleep(0.05)
                m.recording = False
                y = np.concatenate(m.blocks) if m.blocks else np.zeros(0, np.float32)
                if rate != SR:
                    from scipy.signal import resample_poly
                    y = resample_poly(y, SR, rate).astype(np.float32)
                path = folder / f"{prefix}_{k}.wav"
                sf.write(path, y, SR)
                k += 1
                print(f"\r  saved {path} ({len(y) / SR:.1f} s, peak {db(np.abs(y).max()):.1f} dBFS)" + " " * 30)
                for w in check_clip(y, cls):
                    print(f"    WARNING: {w}")
        except KeyboardInterrupt:
            m.recording = False
            print("\nStopped; the clip in progress was not saved.")


def main():
    ap = argparse.ArgumentParser(description="Record clips from an audio input into a recording set.")
    ap.add_argument("--list", action="store_true", help="list input devices and stop")
    ap.add_argument("--device", help="input device number or part of its name (default: system default)")
    ap.add_argument("--meter", action="store_true", help="show the live input level only")
    ap.add_argument("--class", dest="cls", choices=CLASSES, help="class to record")
    ap.add_argument("-n", type=int, default=5, help="number of clips (default 5)")
    ap.add_argument("-s", "--seconds", type=float, default=20.0, help="seconds per clip (default 20)")
    ap.add_argument("--set", default="cable", choices=[s for s in DATA_SETS if s != "phone"],
                    help="recording set to write into (default cable; the phone set is never written)")
    args = ap.parse_args()

    if args.list:
        return list_inputs()
    dev = find_device(args.device)
    rate = device_rate(dev)
    import sounddevice as sd
    print(f"Input: {sd.query_devices(dev)['name']} at {rate} Hz" + ("" if rate == SR else f" (resampled to {SR})"))
    if args.meter:
        return run_meter(dev, rate)
    if not args.cls:
        raise SystemExit("Give --class (one of: " + ", ".join(CLASSES) + "), or --meter / --list.")
    from pathlib import Path
    record(dev, rate, args.cls, args.n, args.seconds, Path(DATA_SETS[args.set][0]))


if __name__ == "__main__":
    main()
