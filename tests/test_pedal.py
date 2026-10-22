"""
Quick checks that the pieces still behave after a change. Run from the
my_pedal_dsp folder with your venv active:
    python -m pytest tests
They use synthetic audio only, so they need no recordings.
"""
import numpy as np
import pytest

import my_effects as fx
import section_recogniser as sr
from classify_windows import FEATURE_NAMES, frame_features_from_audio, window_vector
from settings import BLOCK, SR
from stream_effects import Chain, compare, offline, stream

EFFECTS = ["Overdrive", "Distortion", "Fuzz", "Bitcrush", "Tremolo", "Chorus", "Delay", "Reverb",
           "Compressor", "Limiter", "Octave", "OctaveUp", "Wah", "AutoWah"]


def guitar_like(seconds=2.0, seed=0):
    """Plucked notes: decaying harmonics with a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for start, f in zip(np.arange(0, seconds, 0.25), rng.choice([82, 110, 147, 196, 247, 330], 8)):
        m = t >= start
        tt = t[m] - start
        y[m] += sum(np.sin(2 * np.pi * f * k * tt) / k for k in range(1, 5)) * np.exp(-4 * tt)
    return 0.1 * y + 0.001 * rng.standard_normal(len(t))


@pytest.mark.parametrize("name", EFFECTS)
def test_effect_streams_like_offline(name):
    """Block-by-block output must match whole-file output: the effect keeps its state
    between blocks and never looks at future audio."""
    x = guitar_like()
    streamed, _ = stream(x, SR, Chain([getattr(fx, name)]), BLOCK)
    reference = offline(x, SR, Chain([getattr(fx, name)]))
    _, rel = compare(streamed, reference)
    assert rel < 1e-6, f"{name}: streamed differs from offline by {rel:.2e} (relative)"


def test_classifier_features_line_up():
    b = frame_features_from_audio(guitar_like(seed=1))
    lengths = {k: len(v) for k, v in b.items()}
    assert len(set(lengths.values())) == 1, lengths
    assert len(window_vector(b, 0, 40)) == len(FEATURE_NAMES)


def section(notes, seconds=6.0, tempo=1.0, seed=0):
    """A repeating riff of sustained notes, optionally faster or slower."""
    rng = np.random.default_rng(seed)
    step = 0.4 / tempo
    t = np.arange(int(seconds * SR)) / SR
    idx = (t // step).astype(int) % len(notes)
    f = np.array(notes)[idx]
    phase = 2 * np.pi * np.cumsum(f) / SR
    return 0.1 * (np.sin(phase) + 0.5 * np.sin(2 * phase)) + 0.002 * rng.standard_normal(len(t))


def test_recogniser_locks_on_the_right_section():
    riffs = {"verse": [110, 147, 165, 196], "chorus": [220, 262, 330, 392]}
    library = {n: [sr.chroma_features(section(r, seed=k, tempo=1 + 0.1 * k))
                   for k in range(2)] for n, r in riffs.items()}
    probe = section(riffs["chorus"], seed=5, tempo=0.9)
    _, labels, _ = sr.stream_decisions(probe, library, threshold=0.3, window_s=2.0)
    assert labels[-1] == "chorus"
    assert "verse" not in labels


# ---------------------------------------------------------------- setlist mode
import queue
import threading

from scipy.io import wavfile

import setlist as sl
from recogniser_worker import worker_main

RIFFS = {"verse": [110, 147, 165, 196], "chorus": [220, 262, 330, 392], "bridge": [123, 185, 247, 294]}


def test_setlist_tracker_moves_only_to_the_next_section():
    t = sl.SetlistTracker(["verse", "chorus", "verse"], threshold=0.3)
    assert t.candidates() == ["verse", "chorus"]
    hear = lambda name: sorted([(0.05 if n == name else 0.5, n) for n in t.candidates()])
    for _ in range(sr.CONFIRM - 1):
        assert t.update(hear("chorus")) == "verse"            # not confirmed yet
    assert t.update(hear("chorus")) == "chorus" and t.pos == 1
    assert t.update([(0.6, "chorus"), (0.7, "verse")]) == "chorus"   # unknown: hold the section
    assert t.heard is None
    assert t.update(None) == "chorus"                          # silence: hold
    t.step(1)
    assert (t.pos, t.current, t.next) == (2, "verse", None)    # manual next, end of the song
    t.step(1)
    assert t.pos == 2


def test_setlist_resyncs_from_the_full_library_only_in_both_mode():
    for resync in (False, True):
        t = sl.SetlistTracker(["verse", "chorus", "bridge"], threshold=0.3)
        for _ in range(sl.RESYNC_CONFIRM):
            t.update([(0.6, "verse"), (0.7, "chorus")], library_locked="bridge", resync=resync)
        assert t.current == ("bridge" if resync else "verse")


def test_choose_section():
    song = {"current": "chorus"}
    assert sl.choose_section("library", "verse", song) == "verse"
    assert sl.choose_section("setlist", "verse", song) == "chorus"
    assert sl.choose_section("both", None, song) == "chorus"
    assert sl.choose_section("setlist", "verse", None) == "verse"    # no song loaded: full library


def test_worker_runs_library_and_setlist_together(tmp_path):
    paths = {}
    for name, riff in RIFFS.items():
        paths[name] = []
        for k in range(2):
            f = tmp_path / f"{name}_{k}.wav"
            wavfile.write(f, SR, section(riff, seed=k, tempo=1 + 0.1 * k).astype(np.float32))
            paths[name].append(str(f))
    cmd, res = queue.Queue(), queue.Queue()
    th = threading.Thread(target=worker_main, args=(cmd, res), daemon=True)
    th.start()
    assert res.get(timeout=10) == ("ready",)
    cmd.put(("load", 1, paths))
    while (msg := res.get(timeout=60))[0] != "reply":
        pass
    assert set(msg[2]["takes"]) == set(RIFFS)
    cmd.put(("setlist", ["verse", "chorus", "bridge"], "both"))
    assert res.get(timeout=10)[0] == "setlist_decision"

    play = np.concatenate([section(RIFFS["verse"], seed=7, tempo=0.95),
                           section(RIFFS["chorus"], seed=8, tempo=1.05)])
    step = int(0.1 * SR)
    for s in range(0, len(play) - step + 1, step):
        cmd.put(("audio", play[s:s + step].astype(np.float32)))
    cmd.put(("stop",))
    th.join(timeout=120)
    msgs = []
    while not res.empty():
        msgs.append(res.get())
    library = [m[1] for m in msgs if m[0] == "decision"]
    songs = [m[1] for m in msgs if m[0] == "setlist_decision"]
    assert len(library) == len(songs) == len(play) // step       # one answer from each per step
    assert library[-1] == "chorus" and "bridge" not in library
    assert songs[-1]["current"] == "chorus" and songs[-1]["pos"] == 1
    assert songs[-1]["next"] == "bridge"
    assert all(s["current"] in ("verse", "chorus") for s in songs)

    # manual control, and switching the setlist off leaves only the library decisions
    th = threading.Thread(target=worker_main, args=(cmd, res), daemon=True)
    th.start()
    res.get(timeout=10)
    cmd.put(("setlist", ["verse", "chorus"], "setlist"))
    cmd.put(("setlist_step", 1))
    cmd.put(("setlist", None))
    cmd.put(("audio", play[:step].astype(np.float32)))
    cmd.put(("stop",))
    th.join(timeout=30)
    msgs = []
    while not res.empty():
        msgs.append(res.get())
    assert [m[1]["pos"] for m in msgs if m[0] == "setlist_decision"] == [0, 1]
    assert [m[0] for m in msgs][-1] == "decision"


# ---------------------------------------------------------------- live input with one channel
class _FakeStream:
    """Stands in for sounddevice.Stream: keeps the callback so the test can drive it."""
    last = None

    def __init__(self, channels, callback, **kw):
        self.channels, self.callback = channels, callback
        _FakeStream.last = self

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class _FakeSD:
    Stream = _FakeStream

    @staticmethod
    def query_devices(dev=None, kind=None):
        return {"max_input_channels": 1, "name": "USB Audio"}


def test_app_teaches_from_a_one_input_device_with_a_standin_pedal(tmp_path, monkeypatch):
    import sys
    import pedal_app
    monkeypatch.chdir(tmp_path)                       # the app's sections/ goes in the temp folder
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD)
    monkeypatch.setattr(pedal_app, "SECTIONS", tmp_path / "sections")
    eng = pedal_app.Engine()
    try:
        assert eng.start({"type": "device", "input": 5, "output": None}) is None
        assert eng.can_teach and "recognition only" in eng.teach_hint     # no pedal signal: recognition only
        assert eng.start({"type": "device", "input": 5, "output": None, "pedal_preset": "Fuzz"}) is None
        assert eng.can_teach and _FakeStream.last.channels == (1, 1)
        assert eng.teach_start("riff") is None
        x = guitar_like(seconds=0.5).astype(np.float32)
        out = np.zeros((BLOCK, 1), np.float32)
        for s in range(0, len(x) - BLOCK + 1, BLOCK):
            _FakeStream.last.callback(x[s:s + BLOCK, None], out, BLOCK, None, None)
        dry, wet = np.concatenate(eng.teaching["dry"]), np.concatenate(eng.teaching["wet"])
        assert len(dry) == len(wet) > 0
        assert np.max(np.abs(wet - dry)) > 1e-3       # the stand-in pedal really changed the sound
    finally:
        eng.close()


def test_app_teaches_recognition_only_without_a_pedal(tmp_path, monkeypatch):
    import sys
    import pedal_app
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD)
    monkeypatch.setattr(pedal_app, "SECTIONS", tmp_path / "sections")
    eng = pedal_app.Engine()
    try:
        assert eng.start({"type": "device", "input": 5, "output": None, "pedal_preset": ""}) is None
        assert eng.teach_start("verse") is None
        x = section(RIFFS["verse"], seconds=4.0).astype(np.float32)
        out = np.zeros((BLOCK, 1), np.float32)
        for s in range(0, len(x) - BLOCK + 1, BLOCK):
            _FakeStream.last.callback(x[s:s + BLOCK, None], out, BLOCK, None, None)
        assert eng.teach_stop() is None, eng.message
        d = tmp_path / "sections" / "verse"
        assert (d / "take_1.wav").exists() and not (d / "take_1_wet.wav").exists()
        assert eng.meta["verse"] == 1 and eng.presets["verse"] == "Clean"
        assert "verse" not in eng.effect_info                    # no pedal training started
    finally:
        eng.close()


def test_confidences_add_up_and_respect_the_threshold():
    conf = dict(sr.confidences([(0.05, "verse"), (0.20, "chorus")], threshold=0.15))
    assert abs(sum(conf.values()) - 1) < 1e-9
    assert conf["verse"] > conf[None] > conf["chorus"]          # below threshold beats unknown
    conf = dict(sr.confidences([(0.30, "verse")], threshold=0.15))
    assert conf[None] > conf["verse"]                           # nothing matches well: unknown wins
    assert sr.confidences(None, 0.15) == []


@pytest.mark.parametrize("f0", [82.4, 196.0, 330.0])
def test_octave_shifts_the_pitch(f0):
    """Pitch by autocorrelation (what the ear hears), on a note with harmonics, streamed in blocks."""
    t = np.arange(2 * SR) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * 0.1 * np.exp(-1.0 * t)

    def pitch(y):
        y = y[SR // 2:SR // 2 + 8192]
        c = np.correlate(y - y.mean(), y - y.mean(), "full")[len(y) - 1:]
        lo = int(np.argmax(c < 0))                  # search after the first dip, not the lag-0 hump
        return SR / (lo + np.argmax(c[lo:int(SR / 30)]))

    for effect, want in ((fx.Octave(down=1, up=0, dry=0), f0 / 2), (fx.OctaveUp(dry=0), 2 * f0)):
        y, _ = stream(x, SR, Chain([effect]), BLOCK)
        got = pitch(y)
        assert abs(got - want) < 0.05 * want, (effect.name, got, want)


def test_tracker_switches_after_confirm_identical_answers():
    t = sr.Tracker(threshold=0.3)
    win = lambda n: [(0.05, n), (0.5, "other_" + n)]
    unknown = [(0.6, "verse"), (0.7, "chorus")]
    for _ in range(sr.CONFIRM):
        t.update(win("verse"))
    assert t.locked == "verse" and t.progress() is None
    for k in range(1, sr.CONFIRM):
        assert t.update(unknown) == "verse"                  # not yet
        assert t.progress() == (None, k)
    assert t.update(unknown) is None                          # CONFIRM unknowns in a row: unlocked
    for _ in range(sr.CONFIRM):
        t.update(win("chorus"))
    assert t.locked == "chorus"



def test_wah_opens_on_the_pick_and_closes_as_the_note_fades():
    t = np.arange(SR) / SR
    x = sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 12)) * 0.2 * np.exp(-3 * t)
    wah = fx.Wah()
    wah.prepare(SR, BLOCK)
    fc = np.concatenate([(wah.process(x[s:s + BLOCK]), wah.last_fc)[1] for s in range(0, len(x), BLOCK)])
    at = lambda ms: fc[int(ms / 1000 * SR)]
    assert at(50) > 1500 and at(900) < 0.5 * at(50)        # "jow": up on the attack, down on the decay
    assert np.all(np.isfinite(fc))


def test_unknown_needs_a_higher_confidence_than_the_best_section():
    close_call = [(0.05, "verse"), (0.06, "chorus")]          # within MARGIN of each other
    no_match = [(0.40, "verse"), (0.45, "chorus")]            # above the threshold
    t = sr.Tracker(threshold=0.3)
    for _ in range(sr.CONFIRM):
        t.update(close_call)
    assert t.locked == "verse"                                 # best beats unknown: it gets the vote
    for _ in range(sr.CONFIRM):
        t.update(no_match)
    assert t.locked is None                                    # unknown beats the best: unknown
