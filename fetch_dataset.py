"""
fetch_dataset.py

Downloads the IDMT-SMT-Guitar dataset (Fraunhofer IDMT) and turns its muted
notes into extra palm_mute clips, plus normal picked/finger notes into extra
single_soft clips, so the dataset can't be told apart from your own clips just
by where it was recorded.

    Dataset: https://zenodo.org/records/7544110   (one 1.3 GB zip)
    Licence: CC BY-NC-ND 4.0. Fine for study and a non-commercial talk with
             credit; don't share the audio or clips made from it.
    Credit:  Kehling et al., "Automatic Tablature Transcription of Electric
             Guitar Recordings by Estimation of Score- and Instrument-Related
             Parameters", DAFx 2014.

How it works:
  1. Downloads the zip to external/ (skipped if already there) and checks its md5.
  2. Unzips it and reads every XML annotation. Each note event has an onset, an
     offset and a plucking ("excitation") style: muted, picked or finger.
  3. Keeps only dataset2 (the only part with muted notes), so both classes come
     from the same recording setup. Cuts out each note, joins notes from the same guitar into ~20 s clips, and
     sets their level to the average of your own playing clips (so loudness
     alone can't give the source away).
  4. Writes them to data_trimmed/<class>/idmt_*.wav. They are already clean, so
     they skip the trim step, and main.py leaves them alone.

Commands (run from the my_pedal_dsp folder, venv active):
    python fetch_dataset.py --inspect        download + unzip, then just report what it found
    python fetch_dataset.py                  make the clips (10 per class by default)
    python fetch_dataset.py --max-clips 5
    python fetch_dataset.py --remove         delete every idmt_*.wav clip again

Afterwards, run  python classify_windows.py  as usual.

A WARNING worth keeping in the talk: these are direct-input recordings, yours
are from a phone. Keep your own clips as the real test, and record at least a
couple of your own palm-mute clips too.
"""
import argparse
import hashlib
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from settings import DATA_TRIMMED, SILENCE_CLASS, SR

URL = "https://zenodo.org/records/7544110/files/IDMT-SMT-GUITAR_V2.zip?download=1"
MD5 = "06796e08731bccffaed6ae59361486e4"
EXTERNAL = Path("external")
ZIP = EXTERNAL / "IDMT-SMT-GUITAR_V2.zip"
UNZIPPED = EXTERNAL / "IDMT-SMT-GUITAR_V2"
PREFIX = "idmt_"

CLIP_S = 20.0          # length of each assembled clip
GAP_S = 0.05           # silence between joined notes
PAD_BEFORE_S = 0.01    # keep the pick attack
PAD_AFTER_S = 0.05     # and a little of the release
DEFAULT_NOTE_S = 1.0   # if an event has no offset
FADE_S = 0.005

# Plucking-style codes seen in IDMT annotations, and words that mean the same
MUTED = {"MU", "MUTED", "M", "PALM_MUTE", "PALMMUTE"}
NORMAL_PLUCK = {"PK", "PICKED", "FS", "FINGER", "FINGERSTYLE", "F", "P"}
NORMAL_EXPRESSION = {"", "NO", "NORMAL", "N"}

TARGETS = {"muted": "palm_mute", "normal": "single_soft"}

# Every muted note in IDMT is in dataset2, so both classes are cut from dataset2
# only. Otherwise palm_mute would be "dataset2 audio" and single_soft "dataset1
# audio" and the model could score by spotting the recording setup. (This also
# leaves out dataset1's Chords folders, which aren't single notes.)
SOURCE_FOLDER = "dataset2"


# ---------------------------------------------------------------- download
def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    EXTERNAL.mkdir(exist_ok=True)
    if ZIP.exists() and md5sum(ZIP) == MD5:
        print(f"Already downloaded: {ZIP}")
        return
    print(f"Downloading {URL}\n  -> {ZIP}  (1.3 GB, this takes a while)")
    tmp = ZIP.with_suffix(".part")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size / total, 1.0)
            sys.stdout.write(f"\r  {done:6.1%} of {total / 1e9:.2f} GB")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, tmp, progress)
    print()
    if md5sum(tmp) != MD5:
        raise SystemExit(f"Checksum mismatch: {tmp} is corrupt or incomplete. Delete it and try again.")
    tmp.rename(ZIP)


def unzip():
    if UNZIPPED.exists() and any(UNZIPPED.rglob("*.xml")):
        return
    print(f"Unzipping to {UNZIPPED} ...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(UNZIPPED)


# ---------------------------------------------------------------- annotations
def _child_text(el, *words):
    """Text of the first child whose tag contains any of the words (case-insensitive)."""
    for c in el:
        tag = c.tag.lower()
        if any(w in tag for w in words):
            return (c.text or "").strip()
    return None


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def kind_of(pluck, expression):
    p = (pluck or "").upper().replace("-", "_").replace(" ", "")
    e = (expression or "").upper()
    if p in MUTED or p.startswith("MUTE"):
        return "muted"
    if p in NORMAL_PLUCK and e in NORMAL_EXPRESSION:
        return "normal"
    return None


def read_events(xml_path):
    """Returns (audio file name or None, [(onset_s, offset_s, pluck, expression), ...])."""
    root = ET.parse(xml_path).getroot()
    audio_name = None
    for el in root.iter():
        if "audiofilename" in el.tag.lower() and el.text:
            audio_name = el.text.strip()
            break
    events = []
    for ev in root.iter():
        if ev.tag.lower() != "event":
            continue
        onset = _num(_child_text(ev, "onset"))
        if onset is None:
            continue
        offset = _num(_child_text(ev, "offset"))
        events.append((onset, offset, _child_text(ev, "excitation", "pluck"),
                       _child_text(ev, "expression")))
    return audio_name, events


def scan():
    """Pairs every annotation with its audio. Returns [(wav_path, guitar, events)]."""
    wavs = defaultdict(list)                       # the same file name can repeat across guitars
    for w in UNZIPPED.rglob("*.wav"):
        wavs[w.stem.lower()].append(w)

    def nearest(stem, xml):
        """Of the wavs with this name, the one sharing the most folders with the annotation."""
        cands = wavs.get(stem.lower(), [])
        return max(cands, key=lambda w: len(_common(w.parts, xml.parts)), default=None)

    found, missing = [], 0
    for x in sorted(UNZIPPED.rglob("*.xml")):
        try:
            name, events = read_events(x)
        except ET.ParseError:
            continue
        if not events:
            continue
        wav = nearest(Path(name).stem, x) if name else None
        wav = wav or nearest(x.stem, x)
        if wav is None:
            missing += 1
            continue
        found.append((wav, guitar_of(wav), events))
    if missing:
        print(f"  ({missing} annotation files had no matching audio and were skipped)")
    return found


def _common(a, b):
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def guitar_of(wav):
    """The guitar/setup folder a recording comes from (the folder above audio/ if present)."""
    parent = wav.parent
    if parent.name.lower() in ("audio", "wav", "wavs"):
        parent = parent.parent
    return re.sub(r"[^A-Za-z0-9]+", "_", parent.name).strip("_")[:40] or "guitar"


# ---------------------------------------------------------------- clips
def load_mono(path):
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        y = resample_poly(y, SR, sr)
    return y


def fade(y):
    k = min(int(FADE_S * SR), len(y) // 2)
    if k:
        r = np.linspace(0, 1, k)
        y[:k] *= r
        y[-k:] *= r[::-1]
    return y


def cut_notes(recordings):
    """Returns {kind: {guitar: [note arrays]}}."""
    notes = {k: defaultdict(list) for k in TARGETS}
    for wav, guitar, events in recordings:
        kinds = [kind_of(p, e) for _, _, p, e in events]
        if not any(kinds):
            continue
        y = load_mono(wav)
        for (on, off, _, _), kind in zip(events, kinds):
            if kind is None:
                continue
            end = off if off is not None and off > on else on + DEFAULT_NOTE_S
            a, b = int(max(on - PAD_BEFORE_S, 0) * SR), int(min(end + PAD_AFTER_S, len(y) / SR) * SR)
            if b - a > int(0.05 * SR):
                notes[kind][guitar].append(fade(y[a:b].copy()))
    return notes


def own_level():
    """Mean RMS of your own non-silent clips, used to match the dataset's level."""
    rms = []
    for d in DATA_TRIMMED.iterdir() if DATA_TRIMMED.exists() else []:
        if not d.is_dir() or d.name == SILENCE_CLASS:
            continue
        for f in d.glob("*.wav"):
            if not f.name.startswith(PREFIX):
                y, _ = sf.read(f, always_2d=True)
                rms.append(np.sqrt(np.mean(y.mean(axis=1) ** 2)))
    return float(np.mean(rms)) if rms else 0.05


def assemble(note_lists, max_clips, level, rng):
    """Joins notes into ~CLIP_S clips, spreading clips across guitars."""
    gap = np.zeros(int(GAP_S * SR))
    target = int(CLIP_S * SR)
    per_guitar = []
    for guitar, ns in sorted(note_lists.items()):
        ns = [ns[i] for i in rng.permutation(len(ns))]
        clips, cur, n = [], [], 0
        for x in ns:
            cur += [x, gap]
            n += len(x) + len(gap)
            if n >= target:
                clips.append(np.concatenate(cur)[:target])
                cur, n = [], 0
        if n >= target // 2:
            clips.append(np.concatenate(cur))
        per_guitar.append((guitar, clips))
    out, k = [], 0
    while len(out) < max_clips and any(k < len(c) for _, c in per_guitar):
        for guitar, clips in per_guitar:
            if k < len(clips) and len(out) < max_clips:
                y = clips[k]
                y = y * level / (np.sqrt(np.mean(y ** 2)) + 1e-12)
                out.append((f"{PREFIX}{guitar}_{k + 1}", np.clip(y, -0.99, 0.99)))
        k += 1
    return out


def remove():
    gone = list(DATA_TRIMMED.glob(f"*/{PREFIX}*.wav"))
    for f in gone:
        f.unlink()
    print(f"Removed {len(gone)} dataset clips from {DATA_TRIMMED}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inspect", action="store_true", help="only report what the dataset contains")
    ap.add_argument("--max-clips", type=int, default=10, help="clips per class (default 10)")
    ap.add_argument("--remove", action="store_true", help="delete all idmt_ clips and stop")
    args = ap.parse_args()

    if args.remove:
        return remove()

    download()
    unzip()
    print("Reading annotations...")
    recordings = scan()
    styles = Counter((p, e) for _, _, ev in recordings for _, _, p, e in ev)
    kinds = Counter()
    for (p, e), n in styles.items():
        kinds[kind_of(p, e)] += n
    print(f"  {len(recordings)} annotated recordings, {sum(styles.values())} note events")
    print("  plucking / expression styles found (count):")
    for (p, e), n in styles.most_common(15):
        print(f"    {str(p):10s} {str(e):10s} {n:6d}  -> {kind_of(p, e) or 'not used'}")
    print(f"  usable: {kinds['muted']} muted notes -> palm_mute, "
          f"{kinds['normal']} normal notes -> single_soft")
    if not kinds["muted"]:
        print("  No muted notes recognised. Check the styles listed above and add the code for\n"
              "  'muted' to MUTED at the top of this file.")
    if args.inspect:
        return

    recordings = [r for r in recordings if SOURCE_FOLDER in r[0].relative_to(UNZIPPED).parts]
    print(f"  using the {len(recordings)} recordings in {SOURCE_FOLDER}/ for both classes")
    level = own_level()
    rng = np.random.default_rng(0)
    notes = cut_notes(recordings)
    for kind, cls in TARGETS.items():
        clips = assemble(notes[kind], args.max_clips, level, rng)
        (DATA_TRIMMED / cls).mkdir(parents=True, exist_ok=True)
        for name, y in clips:
            sf.write(DATA_TRIMMED / cls / f"{name}.wav", y, SR)
        print(f"Wrote {len(clips)} clips of about {CLIP_S:.0f} s to {DATA_TRIMMED / cls}/")
    print("\nNext: python classify_windows.py   (undo with: python fetch_dataset.py --remove)")


if __name__ == "__main__":
    main()
