"""
recogniser_worker.py

The recogniser as its own PROCESS. pedal_app.py starts it and talks to it
through two queues, so the matching code (which holds Python's global
interpreter lock for long stretches) can never stall the audio callback or the
web page. It owns the section library: the audio side only sends it audio.

Messages in  (pedal_app -> worker):
    ("audio", float32 array)            about 0.1 s of dry guitar
    ("reset",)                          a new stream started
    ("config", {"threshold_scale"?, "strictness"?, "window_s"?})
    ("load", rid, {name: [wav paths]})  build the library from saved takes
    ("add_take", rid, name, float32 array)
    ("delete", rid, name)
    ("setlist", [section names] or None, drive)   load a song for setlist mode (None = off)
    ("setlist_goto", index) / ("setlist_step", +1 or -1)   manual position control
    ("drive", "library" | "setlist" | "both")
    ("stop",)
Messages out (worker -> pedal_app):
    ("ready",)
    ("decision", locked or None, [(cost, name), ...] or None, milliseconds,
                 (challenger, wins in a row) or None)
    ("setlist_decision", state dict)    sent before each "decision" while a song is loaded
                                        (see setlist.SetlistTracker.state)
    ("threshold", value)                recalibrated after a config change
    ("reply", rid, payload)             answer to load / add_take / delete

The full-library recogniser and the setlist tracker (setlist.py) both run on the
same chroma window, so both answers are available at once; the "decision"
message is exactly what it was before setlist mode existed.

This module imports only numpy, scipy, section_recogniser and setlist.
"""
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

import section_recogniser as sr
from setlist import SetlistTracker
from settings import SR


def read_wav(path):
    rate, y = wavfile.read(path)
    if y.dtype.kind == "i":
        y = y.astype(np.float64) / float(np.iinfo(y.dtype).max)
    elif y.dtype.kind == "u":
        y = (y.astype(np.float64) - 128.0) / 128.0
    else:
        y = y.astype(np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if rate != SR:
        y = resample_poly(y, SR, rate)
    return y


def calibrate(library, window_s):
    """Leave-one-take-out on the stored chroma: the cost of the right section against the best
    wrong one. Returns the lock threshold (before the user's strictness scaling)."""
    rng = np.random.default_rng(0)
    n_win = int(round(window_s * sr.FPS))
    genuine, impostor = [], []
    for name, takes in library.items():
        for ti, F in enumerate(takes):
            lib = {n: [t for k, t in enumerate(ts) if not (n == name and k == ti)]
                   for n, ts in library.items()}
            lib = {n: v for n, v in lib.items() if v}
            if len(F) <= n_win or name not in lib:
                continue
            for _ in range(8):
                s = int(rng.integers(0, len(F) - n_win))
                ranked = sr.rank_sections(F[s:s + n_win], lib)
                costs = {n: c for c, n in ranked}
                genuine.append(costs[name])
                impostor.append(min((c for c, n in ranked if n != name), default=9.0))
    if not genuine:
        return 0.15
    g, im = np.array(genuine), np.array(impostor)
    if np.all(im >= 9.0):                          # only one section: nothing to compare against
        return float(np.clip(1.5 * np.percentile(g, 90), 0.03, 0.4))
    if np.percentile(g, 90) < np.percentile(im, 10):
        thr = 0.5 * (np.percentile(g, 90) + np.percentile(im, 10))
    else:
        thr = 1.5 * np.percentile(g, 90)
    return float(np.clip(thr, 0.03, 0.4))


def worker_main(cmd_q, res_q):
    library = {}                                   # name -> [active chroma frames per take]
    cfg = {"strictness": 1.0, "window_s": 2.0}
    base_thr = 0.15
    cs, tracker = sr.ChromaStream(), sr.Tracker(base_thr)
    song, drive = None, "library"                  # song: SetlistTracker while a setlist is loaded
    res_q.put(("ready",))

    def send_setlist():
        if song is not None:
            res_q.put(("setlist_decision", song.state()))

    while True:
        msg = cmd_q.get()
        kind = msg[0]
        try:
            if kind == "audio":
                t0 = time.perf_counter()
                cs.push(msg[1].astype(np.float64))
                Q = cs.query(cfg["window_s"])
                ranked = None if (Q is None or not library) else sr.rank_sections(Q, library)
                tracker.thr = base_thr * cfg["strictness"]
                locked = tracker.update(ranked)
                if song is not None:
                    pair = {n: library[n] for n in song.candidates() if n in library}
                    pair_ranked = None if (Q is None or not pair) else sr.rank_sections(Q, pair)
                    song.thr = base_thr * cfg["strictness"]
                    song.update(pair_ranked, library_locked=locked, resync=drive == "both")
                    send_setlist()
                res_q.put(("decision", locked, ranked, (time.perf_counter() - t0) * 1000, tracker.progress()))
            elif kind == "reset":
                cs, tracker = sr.ChromaStream(), sr.Tracker(base_thr * cfg["strictness"])
                if song is not None:
                    song.soft_reset()
            elif kind == "setlist":
                song = SetlistTracker(msg[1], base_thr * cfg["strictness"]) if msg[1] else None
                if len(msg) > 2:
                    drive = msg[2]
                send_setlist()
            elif kind in ("setlist_goto", "setlist_step"):
                if song is not None:
                    (song.goto if kind == "setlist_goto" else song.step)(msg[1])
                    send_setlist()
            elif kind == "drive":
                drive = msg[1]
            elif kind == "config":
                changed_window = "window_s" in msg[1] and msg[1]["window_s"] != cfg["window_s"]
                cfg.update(msg[1])
                if changed_window and library:
                    base_thr = calibrate(library, cfg["window_s"])
                    res_q.put(("threshold", base_thr))
            elif kind == "load":
                _, rid, paths = msg
                library = {}
                for name, files in paths.items():
                    takes = []
                    for f in files:
                        F, sil = sr.chroma_features(read_wav(f))
                        if (~sil).sum() >= sr.MIN_ACTIVE:
                            takes.append(F[~sil])
                    if takes:
                        library[name] = takes
                base_thr = calibrate(library, cfg["window_s"]) if library else 0.15
                tracker = sr.Tracker(base_thr * cfg["strictness"])
                res_q.put(("reply", rid, {"takes": {n: len(t) for n, t in library.items()},
                                          "threshold": base_thr}))
            elif kind == "add_take":
                _, rid, name, audio = msg
                F, sil = sr.chroma_features(audio.astype(np.float64))
                active = F[~sil]
                if len(active) < 2 * sr.MIN_ACTIVE:
                    res_q.put(("reply", rid, {"error": "Not enough playing in that take (too quiet?). Try again."}))
                    continue
                library.setdefault(name, []).append(active)
                base_thr = calibrate(library, cfg["window_s"])
                tracker = sr.Tracker(base_thr * cfg["strictness"])
                res_q.put(("reply", rid, {"takes": len(library[name]), "threshold": base_thr}))
            elif kind == "delete":
                _, rid, name = msg
                library.pop(name, None)
                base_thr = calibrate(library, cfg["window_s"]) if library else 0.15
                tracker = sr.Tracker(base_thr * cfg["strictness"])
                res_q.put(("reply", rid, {"threshold": base_thr}))
            elif kind == "stop":
                break
        except Exception as e:                      # never let one bad message kill the recogniser
            if kind in ("load", "add_take", "delete"):
                res_q.put(("reply", msg[1], {"error": f"Recogniser error: {e}"}))
            else:
                res_q.put(("decision", None, None, 0.0, None))