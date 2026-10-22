"""
pedal_app.py

The laptop demo: one program, one browser page, no files to move around.

    python pedal_app.py            # opens http://127.0.0.1:8765 in your browser

HOW IT WORKS
  TEACH a section by playing it while the app hears TWO signals at once:
      the clean guitar, and what your pedal does to it.
  From that one take the app learns
      1. WHAT THE PART IS   (a fingerprint of the clean playing), and
      2. THE EFFECT         (a small neural network that copies the pedal's sound).
  Later, when you play that part, the app recognises it and applies the learned
  effect by itself. There is nothing to choose or configure per section.

THE TWO SIGNALS
  * Live: a 2-input audio interface. Input 1 = clean guitar (from a splitter),
    input 2 = the pedal's output.
  * No hardware: play a recorded clip as the stand-in guitar and either
      - pick a second recording of the same clip with the pedal on, or
      - pick a "stand-in pedal": one of the built-in effects, which the app treats
        as a black box (it learns it from audio exactly as it would a real pedal).

Architecture: audio callback + web page in this process; the recogniser in its own
process (recogniser_worker.py); pedal learning in short-lived processes
(effect_trainer.py). Heavy work can never stall the audio.

Needs: numpy, scipy, scikit-learn, sounddevice (+ libportaudio2). No librosa.
KEEP THE VOLUME LOW. Master volume starts at 30%.

Options:
    python pedal_app.py --port 8765 --no-browser
"""
import argparse
import itertools
import json
import multiprocessing as mp
import queue
import re
import shutil
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import effect_trainer
import my_effects as fx
import section_recogniser as sr
import setlist as sl
from recogniser_worker import read_wav, worker_main
from settings import BLOCK, SECTIONS, SR

RAMP_S = 0.05
CHUNK = int(0.1 * SR)                      # audio sent to the recogniser in chunks of about 0.1 s
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
BUILTIN_PRESETS = ["Clean", "Overdrive", "Distortion", "Fuzz", "Tremolo", "Chorus", "Delay", "Reverb",
                   "Chorus+Delay+Reverb", "Compressor+Distortion+Delay", "Overdrive+Delay",
                   "Octave", "OctaveUp+Fuzz", "Octave+Overdrive",
                   "Wah", "Wah+Overdrive", "Fuzz+Wah", "AutoWah"]
STANDIN_PEDALS = [p for p in BUILTIN_PRESETS if p != "Clean"]
DEFAULT_SETTINGS = {"strictness": 1.0, "master": 0.3, "bypass": False, "window_s": 2.0,
                    "drive": "library"}       # what sets the effect: "library", "setlist" or "both"
MAX_SETLIST = 64


# ---------------------------------------------------------------- helpers
def write_wav(path, y):
    wavfile.write(path, SR, np.asarray(y, dtype=np.float32))


def learned_path(name):
    return (SECTIONS / name / "effect.npz").as_posix()


def available_presets():
    neural = [f"neural:{p.as_posix()}" for p in sorted(Path("neural").glob("model_*.npz"))]
    learned = [f"neural:{p.as_posix()}" for p in sorted(SECTIONS.glob("*/effect.npz"))]
    return BUILTIN_PRESETS + learned + neural


def make_chain(preset):
    if preset == "Clean":
        return []
    return sr.make_chain(preset)


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.meta = {}                     # section name -> number of takes
        self.presets = {}                  # section name -> preset applied when it is recognised
        self.effect_info = {}              # section name -> {"status", "verdict", "esr", "baseline"}
        self.settings = dict(DEFAULT_SETTINGS)
        self.threshold = 0.15
        self.chains, self.weights, self.target = {}, {}, None
        self.limiter = fx.Limiter()
        self.limiter.prepare(SR, BLOCK)
        self.locked, self.ranked, self.challenger = None, [], None
        self.setlists = sl.load_setlists()     # {"songs": {song: [sections]}, "active": song or None}
        self.setlist_state = None              # latest SetlistTracker.state() from the worker
        self.teaching = None
        self.can_teach, self.teach_hint = False, "Press Start first."
        self.sim_chain = None
        self.message = "Choose a source and press Start."
        self.stream, self.source_desc, self.running = None, None, False
        self.cb_times, self.analysis_times = deque(maxlen=400), deque(maxlen=100)
        self.late = self.xruns = self.dropped = 0
        self.sent = self.received = 0
        self.level_db = -120.0
        self._file_pos = 0
        self._batch, self._batch_n = [], 0
        self._ids, self._pending = itertools.count(1), {}
        self._train_gen, self._trainers = {}, []
        self._closing = False
        self.ready = False
        self.proc = self.cmd_q = self.res_q = None
        self.train_q = mp.get_context("spawn").Queue()
        self._start_worker()
        threading.Thread(target=self._listen, daemon=True).start()
        threading.Thread(target=self.load_state, daemon=True).start()

    # ---- the recogniser process ----
    def _start_worker(self):
        ctx = mp.get_context("spawn")
        self.cmd_q, self.res_q = ctx.Queue(maxsize=200), ctx.Queue()
        self.ready = False
        self.proc = ctx.Process(target=worker_main, args=(self.cmd_q, self.res_q), daemon=True)
        self.proc.start()

    def _request(self, kind, *args, timeout=120):
        rid = next(self._ids)
        ev = threading.Event()
        self._pending[rid] = [ev, None]
        self.cmd_q.put((kind, rid) + args)
        if not ev.wait(timeout):
            self._pending.pop(rid, None)
            return {"error": "The recogniser did not answer in time."}
        return self._pending.pop(rid)[1]

    def _listen(self):
        """Reads answers from the recogniser and the trainers. Builds effect chains here, never in the callback."""
        while not self._closing:
            try:
                msg = self.train_q.get_nowait()
                self._on_trained(msg)
            except (queue.Empty, OSError, EOFError):
                pass
            try:
                msg = self.res_q.get(timeout=0.1)
            except (queue.Empty, OSError, EOFError):
                continue
            kind = msg[0]
            if kind == "decision":
                _, locked, ranked, ms, self.challenger = msg
                self.received += 1
                self.locked = locked
                if ranked is not None:
                    self.ranked = ranked
                self.analysis_times.append(ms / 1000.0)
                self._apply_decision()
            elif kind == "setlist_decision":
                if self.setlists["active"]:
                    self.setlist_state = msg[1]
                    if not self.running:
                        self._apply_decision()
            elif kind == "reply" and msg[1] in self._pending:
                self._pending[msg[1]][1] = msg[2]
                self._pending[msg[1]][0].set()
            elif kind == "threshold":
                self.threshold = msg[1]
            elif kind == "ready":
                self.ready = True

    def driving_section(self):
        """The section whose preset should play, from whichever recogniser drives."""
        song = self.setlist_state if self.setlists["active"] else None
        return sl.choose_section(self.settings["drive"], self.locked, song)

    def _apply_decision(self):
        try:
            if self.teaching:
                self._set_target(None)                       # teaching: hear the clean sound
            else:
                section = self.driving_section()
                self._set_target(self.presets.get(section) if section else None)
        except Exception as e:
            self.message = f"Could not load that effect: {e}"

    def _send_setlist(self):
        active = self.setlists["active"]
        if not active:
            self.setlist_state = None
        self.cmd_q.put(("setlist", self.setlists["songs"].get(active) if active else None,
                        self.settings["drive"]))

    def _send_config(self):
        self.cmd_q.put(("config", {"strictness": self.settings["strictness"],
                                   "window_s": self.settings["window_s"]}))

    # ---- learning the pedal ----
    def _launch_training(self, name):
        d = SECTIONS / name
        pairs = [(str(f), str(f.with_name(f.stem + "_wet.wav"))) for f in sorted(d.glob("take_*.wav"))
                 if not f.stem.endswith("_wet") and f.with_name(f.stem + "_wet.wav").exists()]
        if not pairs:
            return
        gen = self._train_gen.get(name, 0) + 1
        self._train_gen[name] = gen
        self.effect_info[name] = {"status": "learning", "verdict": "learning the pedal's sound...",
                                  "esr": None, "baseline": None}
        p = mp.get_context("spawn").Process(target=effect_trainer.train_job,
                                            args=(name, pairs, d.as_posix(), gen, self.train_q), daemon=True)
        p.start()
        self._trainers = [t for t in self._trainers if t.is_alive()] + [p]

    def _on_trained(self, msg):
        _, name, gen, info = msg
        if name not in self.meta or gen != self._train_gen.get(name):
            return                                                     # section deleted or superseded
        self.effect_info[name] = {"status": "ready" if info["ok"] else "failed", "verdict": info["verdict"],
                                  "esr": info["esr"], "baseline": info["baseline"]}
        if info["path"]:
            key = f"neural:{info['path']}"
            self.chains.pop(key, None)                                 # reload the new weights
            self.weights.pop(key, None)
        self.presets[name] = f"neural:{info['path']}" if info["ok"] else "Clean"
        self.save_meta()
        self.message = (f"'{name}': learned the pedal ({info['esr']} dB error, a plain gain change would give "
                        f"{info['baseline']} dB). {info['verdict']}." if info["path"] else
                        f"'{name}': {info['verdict']}")

    # ---- persistence ----
    def load_state(self):
        SECTIONS.mkdir(exist_ok=True)
        presets_file = SECTIONS / "presets.json"
        saved = json.loads(presets_file.read_text()) if presets_file.exists() else {}
        settings_file = SECTIONS / "app_settings.json"
        if settings_file.exists():
            self.settings.update(json.loads(settings_file.read_text()))
        if self.settings.get("drive") not in sl.DRIVES:
            self.settings["drive"] = "library"
        self._send_config()
        self._send_setlist()
        paths = {}
        for d in sorted(p for p in SECTIONS.iterdir() if p.is_dir() and not p.name.startswith("_")):
            files = [str(f) for f in sorted(d.glob("take_*.wav")) if not f.stem.endswith("_wet")]
            if files:
                paths[d.name] = files
        if not paths:
            return
        self.message = "Loading saved sections..."
        reply = self._request("load", paths)
        if "error" in reply:
            self.message = reply["error"]
            return
        self.meta = dict(reply["takes"])
        self.threshold = reply["threshold"]
        for n in self.meta:
            info_file = SECTIONS / n / "effect.json"
            if info_file.exists() and (SECTIONS / n / "effect.npz").exists():
                info = json.loads(info_file.read_text())
                self.effect_info[n] = {"status": "ready" if info["ok"] else "failed", "verdict": info["verdict"],
                                       "esr": info["esr"], "baseline": info["baseline"]}
                self.presets[n] = saved.get(n, f"neural:{learned_path(n)}" if info["ok"] else "Clean")
            else:
                self.presets[n] = saved.get(n, "Clean")
        self.message = f"Loaded {len(self.meta)} section(s) from {SECTIONS}/"

    def save_meta(self):
        SECTIONS.mkdir(exist_ok=True)
        (SECTIONS / "presets.json").write_text(json.dumps(self.presets, indent=2))
        (SECTIONS / "app_settings.json").write_text(json.dumps(self.settings, indent=2))

    # ---- audio callback path (must stay light, never block) ----
    def process_block(self, x, wet=None):
        t0 = time.perf_counter()
        x32 = x.astype(np.float32)
        t = self.teaching
        if t is not None:
            t["dry"].append(x32)
            if wet is not None:
                t["wet"].append(wet.astype(np.float32))
        self._batch.append(x32)
        self._batch_n += len(x)
        if self._batch_n >= CHUNK:
            chunk = np.concatenate(self._batch)
            self._batch, self._batch_n = [], 0
            try:
                self.cmd_q.put_nowait(("audio", chunk))
                self.sent += 1
            except queue.Full:
                self.dropped += 1
        self.level_db = 20 * np.log10(float(np.max(np.abs(x))) + 1e-9)
        if self.settings["bypass"]:
            y = x.astype(np.float64)
        else:
            y = self._mix(x.astype(np.float64))
        y = self.limiter.process(y * self.settings["master"] * 2.0)
        dt = time.perf_counter() - t0
        self.cb_times.append(dt)
        if dt > len(x) / SR:
            self.late += 1
        return y

    def _mix(self, x):
        target = self.target
        step = len(x) / (RAMP_S * SR)
        total, wet = 0.0, np.zeros_like(x)
        for p in list(set(self.weights) | ({target} if target else set())):
            w = float(np.clip(self.weights.get(p, 0.0) + (step if p == target else -step), 0.0, 1.0))
            chain = self.chains.get(p)
            if w <= 0.0 and p != target or chain is None:
                self.weights.pop(p, None)
                for e in (chain or []):
                    e.reset()
                continue
            self.weights[p] = w
            y = x
            for e in chain:
                y = e.process(y)
            wet += w * y
            total += w
        if total > 1.0:
            wet, total = wet / total, 1.0
        return (1.0 - total) * x + wet

    def _set_target(self, preset):
        if preset in (None, "Clean"):
            self.target = None
            return
        if preset not in self.chains:
            chain = make_chain(preset)
            for e in chain:
                e.prepare(SR, BLOCK)
                e.reset()
            self.chains[preset] = chain
        self.target = preset

    # ---- teaching (needs the clean AND the pedalled signal) ----
    def teach_start(self, name):
        if not self.running:
            return "Press Start first, so there is audio to learn from."
        if not self.can_teach:
            return "Press Start first, so there is audio to learn from."
        if not NAME_RE.match(name or ""):
            return "Section names: letters, digits, _ or - (up to 32 characters)."
        if self.teaching:
            return "Already recording."
        if self.sim_chain:
            for e in self.sim_chain:
                e.reset()
        self.teaching = {"name": name, "dry": [], "wet": [], "t0": time.time()}
        self.message = f"Recording a take of '{name}': clean guitar and pedal together. Play it, then press Stop."
        return None

    def teach_stop(self):
        t = self.teaching
        if not t:
            return "Not recording."
        self.teaching = None
        dry = np.concatenate(t["dry"]) if t["dry"] else np.zeros(0, np.float32)
        wet = np.concatenate(t["wet"]) if t["wet"] else None              # None: recognition only
        if len(dry) < 2 * SR:
            return "That take was under 2 seconds. Play the section through and try again."
        if wet is not None and float(np.sqrt(np.mean(wet ** 2))) < 1e-4:
            self.message = "The pedal signal was silent. Check input 2 / the pedalled recording."
            return self.message
        self.message = "Analysing the take..."
        reply = self._request("add_take", t["name"], dry, timeout=60)     # learns what the part is
        if "error" in reply:
            self.message = reply["error"]
            return reply["error"]
        d = SECTIONS / t["name"]
        d.mkdir(parents=True, exist_ok=True)
        k = len([f for f in d.glob("take_*.wav") if not f.stem.endswith("_wet")]) + 1
        write_wav(d / f"take_{k}.wav", dry)
        if wet is not None:
            write_wav(d / f"take_{k}_wet.wav", wet)
        self.meta[t["name"]] = reply["takes"]
        self.threshold = reply["threshold"]
        self.presets.setdefault(t["name"], "Clean")
        if wet is not None:
            self._launch_training(t["name"])                              # learns what the pedal does
        self.save_meta()
        self.message = (f"Saved take {k} of '{t['name']}' ({len(dry)/SR:.1f} s). "
                        + ("Learning the pedal's sound" if wet is not None else "Recognition only (no pedal sound)")
                        + ("; a second take would make the recogniser threshold more reliable." if reply["takes"] < 2
                           else "."))
        return None

    def delete_section(self, name):
        if not NAME_RE.match(name or "") or name not in self.meta:
            return "No such section."
        reply = self._request("delete", name, timeout=60)
        if "error" in reply:
            return reply["error"]
        self._train_gen[name] = self._train_gen.get(name, 0) + 1          # ignore any result still coming
        key = f"neural:{learned_path(name)}"
        self.chains.pop(key, None); self.weights.pop(key, None)
        shutil.rmtree(SECTIONS / name, ignore_errors=True)
        self.meta.pop(name, None)
        self.presets.pop(name, None)
        self.effect_info.pop(name, None)
        self.threshold = reply["threshold"]
        self.save_meta()
        self.message = f"Deleted '{name}'."
        return None

    def set_preset(self, name, preset):
        if name not in self.meta or preset not in available_presets():
            return "Unknown section or preset."
        self.presets[name] = preset
        self.save_meta()
        return None

    # ---- setlist mode (setlist.py) ----
    def setlist_save(self, song, sections):
        if not NAME_RE.match(song or ""):
            return "Song names: letters, digits, _ or - (up to 32 characters)."
        if not isinstance(sections, list) or not sections or len(sections) > MAX_SETLIST \
                or not all(isinstance(n, str) and NAME_RE.match(n) for n in sections):
            return f"A setlist is 1 to {MAX_SETLIST} section names (letters, digits, _ or -)."
        self.setlists["songs"][song] = list(sections)
        sl.save_setlists(self.setlists)
        if self.setlists["active"] == song:
            self._send_setlist()                              # edited the loaded song: restart it
        unknown = [n for n in dict.fromkeys(sections) if n not in self.meta]
        self.message = (f"Saved setlist '{song}'." + (f" Not taught yet, so only the manual Next control "
                        f"can move past: {', '.join(unknown)}." if unknown else ""))
        return None

    def setlist_delete(self, song):
        if song not in self.setlists["songs"]:
            return "No such song."
        del self.setlists["songs"][song]
        if self.setlists["active"] == song:
            self.setlists["active"] = None
            self._send_setlist()
        sl.save_setlists(self.setlists)
        return None

    def setlist_select(self, song):
        if song and song not in self.setlists["songs"]:
            return "No such song."
        self.setlists["active"] = song or None
        sl.save_setlists(self.setlists)
        self._send_setlist()
        self.message = f"Setlist mode: '{song}' from the top." if song else "Setlist mode off."
        return None

    def setlist_move(self, step=None, goto=None):
        if not self.setlists["active"]:
            return "No song loaded."
        self.cmd_q.put(("setlist_goto", int(goto)) if goto is not None else ("setlist_step", 1 if step > 0 else -1))
        return None

    def set_setting(self, key, value):
        if key not in DEFAULT_SETTINGS:
            return "Unknown setting."
        if key == "drive":
            if value not in sl.DRIVES:
                return "Drive must be library, setlist or both."
            self.settings["drive"] = value
            self.cmd_q.put(("drive", value))
            self._apply_decision()
            self.save_meta()
            return None
        self.settings[key] = bool(value) if key == "bypass" else float(value)
        if key in ("strictness", "window_s"):
            self._send_config()
        self.save_meta()
        return None

    # ---- audio streams ----
    def _ensure_worker(self):
        if self.proc is None or not self.proc.is_alive():
            self._start_worker()
            threading.Thread(target=self.load_state, daemon=True).start()

    def start(self, source):
        if self.running:
            self.stop()
        self._ensure_worker()
        self.cmd_q.put(("reset",))
        self.locked, self.ranked, self.target, self.weights, self.challenger = None, [], None, {}, None
        self.late = self.xruns = self.dropped = 0
        self.sent = self.received = 0
        self._batch, self._batch_n = [], 0
        self.cb_times.clear(); self.analysis_times.clear()
        self.can_teach, self.sim_chain = False, None
        try:
            import sounddevice as sd
        except Exception as e:
            return f"sounddevice is not available ({e}). Install libportaudio2 and sounddevice."
        try:
            if source.get("type") == "file":
                path = Path(source["path"])
                if not path.exists():
                    return f"File not found: {path}"
                data = read_wav(path)
                wet_data, how = None, ""
                if source.get("wet_path"):
                    wet_data = read_wav(Path(source["wet_path"]))
                    n = min(len(data), len(wet_data))
                    data, wet_data = data[:n], wet_data[:n]
                    how = f"pedalled recording {Path(source['wet_path']).name}"
                elif source.get("pedal_preset") in STANDIN_PEDALS:
                    self.sim_chain = make_chain(source["pedal_preset"])
                    for e in self.sim_chain:
                        e.prepare(SR, BLOCK); e.reset()
                    how = f"stand-in pedal '{source['pedal_preset']}'"
                if len(data) < BLOCK * 4:
                    return "That file is too short."
                self.can_teach = True
                self.teach_hint = ("" if wet_data is not None or self.sim_chain is not None else
                                   "No pedal signal, so takes teach recognition only: the app learns the part, and you can pick its effect from the list. To learn a pedal's sound too, pick a pedalled recording or a stand-in pedal.")
                self._file_pos = 0
                out_dev = source.get("output")

                def cb(outdata, frames, tinfo, status):
                    idx = (self._file_pos + np.arange(frames)) % len(data)
                    self._file_pos = (self._file_pos + frames) % len(data)
                    x = data[idx]
                    wet = None
                    if self.teaching is not None:
                        wet = wet_data[idx] if wet_data is not None else self._standin(x)
                    y = self.process_block(x, wet)
                    outdata[:, :] = y.astype(np.float32)[:, None]
                    if status:
                        self.xruns += 1
                try:
                    self.stream = sd.OutputStream(samplerate=SR, blocksize=BLOCK, device=out_dev,
                                                  channels=1, dtype="float32", callback=cb)
                except Exception:
                    def cb2(o, f, t, s):
                        cb(o[:, :1], f, t, s)
                        o[:, 1:] = o[:, :1]
                    self.stream = sd.OutputStream(samplerate=SR, blocksize=BLOCK, device=out_dev,
                                                  channels=2, dtype="float32", callback=cb2)
                self.source_desc = f"file: {path.name} (looping)" + (f" with {how}" if how else "")
            else:
                inp, out = source.get("input"), source.get("output")
                n_in = min(2, int(sd.query_devices(inp, "input")["max_input_channels"]))
                how = ""
                if n_in < 2 and source.get("pedal_preset") in STANDIN_PEDALS:
                    # one input (e.g. a 1/4" to USB-C cable): the clean guitar is live and a
                    # stand-in pedal makes the "pedal" signal, exactly as with a recorded clip
                    self.sim_chain = make_chain(source["pedal_preset"])
                    for e in self.sim_chain:
                        e.prepare(SR, BLOCK); e.reset()
                    how = f" with stand-in pedal '{source['pedal_preset']}'"
                self.can_teach = True
                self.teach_hint = ("" if n_in >= 2 or self.sim_chain is not None else
                                   "No pedal signal, so takes teach recognition only: the app learns the part, and you can pick its effect from the list. To learn a pedal's sound too, pick a stand-in pedal or use a 2-input interface.")

                def cb(indata, outdata, frames, tinfo, status):
                    x = indata[:, 0].astype(np.float64)
                    wet = None
                    if n_in >= 2:
                        wet = indata[:, 1].astype(np.float64)
                    elif self.teaching is not None and self.sim_chain is not None:
                        wet = self._standin(x)
                    y = self.process_block(x, wet)
                    outdata[:, :] = y.astype(np.float32)[:, None]
                    if status:
                        self.xruns += 1
                try:
                    self.stream = sd.Stream(samplerate=SR, blocksize=BLOCK, device=(inp, out), channels=(n_in, 1),
                                            dtype="float32", latency="low", callback=cb)
                except Exception:                                   # some outputs only open in stereo
                    def cb2(i, o, f, t, s):
                        cb(i, o[:, :1], f, t, s)
                        o[:, 1:] = o[:, :1]
                    self.stream = sd.Stream(samplerate=SR, blocksize=BLOCK, device=(inp, out), channels=(n_in, 2),
                                            dtype="float32", latency="low", callback=cb2)
                self.source_desc = f"live input ({n_in} channel{'s' if n_in > 1 else ''}){how}"
            self.running = True
            self.stream.start()
            self.message = f"Running: {self.source_desc}. Keep the volume low."
            return None
        except Exception as e:
            self.running = False
            self.stream = None
            return f"Could not start audio: {e}"

    def _standin(self, x):
        y = x
        for e in self.sim_chain:
            y = e.process(y)
        return y

    def stop(self):
        self.running = False
        self.teaching = None
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
        self.stream = None
        self.target = None
        self.message = "Stopped."

    def close(self):
        self.stop()
        self._closing = True
        try:
            self.cmd_q.put(("stop",))
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.terminate()
            for t in self._trainers:
                if t.is_alive():
                    t.terminate()
        except Exception:
            pass

    # ---- headless driving (testing, or rendering without a sound card) ----
    def feed(self, audio, wet=None, speed=8.0, wait=True):
        """Push audio through exactly as the audio callback would. Returns the processed audio."""
        self.running = True
        out = []
        for s in range(0, len(audio) - BLOCK + 1, BLOCK):
            w = wet[s:s + BLOCK] if wet is not None else None
            out.append(self.process_block(audio[s:s + BLOCK], w))
            time.sleep(BLOCK / SR / speed)
        if wait:
            t0 = time.time()
            while self.received < self.sent and time.time() - t0 < 30:
                time.sleep(0.01)
        return np.concatenate(out) if out else np.zeros(0)

    # ---- state for the page ----
    def state(self):
        cb = np.array(self.cb_times) * 1000 if self.cb_times else np.zeros(1)
        an = np.array(self.analysis_times) * 1000 if self.analysis_times else np.zeros(1)
        alive = self.proc is not None and self.proc.is_alive()
        return {
            "running": self.running, "source": self.source_desc, "message": self.message,
            "locked": self.locked, "target": self.target, "driving": self.driving_section(),
            "challenger": list(self.challenger) if self.challenger else None, "confirm": sr.CONFIRM,
            "setlist": {"songs": self.setlists["songs"], "active": self.setlists["active"],
                        "state": self.setlist_state if self.setlists["active"] else None},
            "ranked": [[n, round(c, 3)] for c, n in self.ranked],
            "guesses": [[n, round(100 * p, 1)] for n, p in
                        sr.confidences(self.ranked, self.threshold * self.settings["strictness"])],
            "threshold": round(self.threshold * self.settings["strictness"], 3),
            "level_db": round(float(self.level_db), 1),
            "can_teach": self.can_teach, "teach_hint": self.teach_hint,
            "teaching": ({"name": self.teaching["name"], "seconds": round(time.time() - self.teaching["t0"], 1)}
                         if self.teaching else None),
            "sections": [{"name": n, "takes": k, "preset": self.presets.get(n, "Clean"),
                          "effect": self.effect_info.get(n)} for n, k in sorted(self.meta.items())],
            "settings": self.settings,
            "recogniser": (f"separate process, pid {self.proc.pid}" if alive and self.ready else
                           "starting..." if alive else "STOPPED (restarts on Start)"),
            "stats": {"cb_mean_ms": round(float(cb.mean()), 3), "cb_max_ms": round(float(cb.max()), 3),
                      "deadline_ms": round(BLOCK / SR * 1000, 2), "late": self.late, "xruns": self.xruns,
                      "dropped": self.dropped, "analysis_mean_ms": round(float(an.mean()), 2),
                      "analysis_max_ms": round(float(an.max()), 2)},
            "presets": available_presets(), "standin": STANDIN_PEDALS,
            "devices": list_devices(), "files": list_files(),
        }


_cache = {}


def _cached(key, fn, ttl=5.0):
    now = time.time()
    if key not in _cache or now - _cache[key][0] > ttl:
        _cache[key] = (now, fn())
    return _cache[key][1]


def list_devices():
    return _cached("devices", _list_devices)


def list_files():
    return _cached("files", _list_files)


def _list_devices():
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        return {"input": [[i, f"{d['name']} ({d['max_input_channels']} in)"] for i, d in enumerate(devs)
                          if d["max_input_channels"] > 0],
                "output": [[i, d["name"]] for i, d in enumerate(devs) if d["max_output_channels"] > 0]}
    except Exception:
        return {"input": [], "output": []}


def _list_files():
    found = []
    for root in ("recordings", "data_cable_trimmed", "data_trimmed", "sections", "demo", "pedal_captures", "neural"):
        found += sorted(Path(root).glob("**/*.wav"))
    return [p.as_posix() for p in found[:300]]


# ---------------------------------------------------------------- web page
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pedal demo</title>
<style>
:root{--bg:#14161a;--panel:#1d2026;--line:#2c313a;--text:#e8eaed;--dim:#9aa3b0;--acc:#4cc38a;--warn:#e5a23c;--bad:#e5645a}
@media(prefers-color-scheme:light){:root{--bg:#f4f5f7;--panel:#fff;--line:#d9dde3;--text:#1b1f24;--dim:#5b6573;--acc:#12805a;--warn:#a96a00;--bad:#b3261e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px;display:grid;gap:14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
h2{margin:0 0 10px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim)}
.big{font-size:40px;font-weight:650;line-height:1.1}.dim{color:var(--dim)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button,select,input{font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:7px 11px;max-width:100%}
button{cursor:pointer}button.primary{background:var(--acc);border-color:var(--acc);color:#04130c;font-weight:600}
button.danger{border-color:var(--bad);color:var(--bad)}button:disabled{opacity:.45;cursor:default}
table{width:100%;border-collapse:collapse}td,th{padding:6px 4px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--dim);font-weight:500}
.bar{height:9px;border-radius:5px;background:var(--line);overflow:hidden;min-width:100px}.bar>i{display:block;height:100%;background:var(--acc)}
.msg{color:var(--warn)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat b{display:block;font-size:20px}.stat span{color:var(--dim);font-size:12px}.bad{color:var(--bad)}.ok{color:var(--acc)}
input[type=range]{padding:0;width:150px}label{color:var(--dim);font-size:13px}.small{font-size:12px}
</style></head><body><main>
<div class="panel"><div class="row" style="justify-content:space-between">
 <div><div class="dim">Recognised section <span id="drivetag" class="small"></span></div><div class="big" id="locked">-</div>
 <div class="dim" id="target"></div><div class="dim small" id="both"></div>
 <div id="guesses" style="margin-top:8px;min-width:260px"></div></div>
 <div style="min-width:220px"><div class="dim">Input level <span id="lvl"></span></div><div class="bar"><i id="lvlbar"></i></div></div>
</div><p class="msg" id="msg"></p></div>

<div class="panel"><h2>Audio source</h2><div class="row">
 <select id="srctype"><option value="file">Recorded clip (loops)</option><option value="device">Live input (2-input interface)</option></select>
 <span id="filebox" class="row">
  <label>Clean clip <select id="file"></select></label>
  <label>Pedalled version (optional) <select id="wetfile"></select></label></span>
 <label><span id="pedallbl">or stand-in pedal</span> <select id="pedal"></select></label>
 <select id="indev" style="display:none"></select><select id="outdev"></select>
 <button class="primary" id="start">Start</button><button id="stop">Stop</button>
 <label><input type="checkbox" id="bypass"> Bypass</label>
 <label>Volume <input type="range" id="master" min="0" max="1" step="0.02"></label>
</div>
<p class="dim small" style="margin-bottom:0">Teaching needs two signals: the clean guitar and the pedal's output. Live: input 1 = clean, input 2 = pedal;
 with a 1-input device (e.g. a 1/4" to USB-C cable) the stand-in pedal plays the pedal's part.
 Recorded clip: use a pedalled version of the clip, or a stand-in pedal that the app treats as a black box.</p></div>

<div class="panel"><h2>Setlist</h2>
 <p class="dim" style="margin-top:0">Give a song's sections in order. With a song loaded, a second recogniser only decides
  "current section or next one?" and runs alongside the full-library one. Choose which sets the effect.</p>
 <div class="row"><label>Effect follows <select id="drive"><option value="library">full library</option>
  <option value="setlist">setlist</option><option value="both">setlist, full library re-syncs it</option></select></label>
  <label>Song <select id="song"></select></label>
  <button id="sprev">&#9664; Prev</button><button class="primary" id="snext">Next section &#9654;</button></div>
 <div id="sline" class="row" style="margin-top:10px"></div>
 <div class="row" style="margin-top:10px"><input id="sname" placeholder="song name" size="14">
  <input id="slist" placeholder="intro, verse, chorus, verse, chorus, solo" size="44">
  <button id="ssave">Save setlist</button><button class="danger" id="sdel">Delete song</button></div></div>

<div class="panel"><h2>Teach a section</h2>
 <p class="dim" style="margin-top:0">Type a name, press Record, play the section, press Stop. The app learns what the part is <b>and</b> what the pedal does to it. Do 2 or 3 takes.</p>
 <div class="row"><input id="tname" placeholder="section name, e.g. verse_riff" size="24">
 <button class="primary" id="rec">Record take</button><span id="recinfo" class="dim"></span></div>
 <p class="msg small" id="hint"></p></div>

<div class="panel"><h2>Sections</h2><table><thead><tr><th>Name</th><th>Takes</th><th>Learned pedal sound</th><th>Effect used</th><th>Confidence (match cost)</th><th></th></tr></thead><tbody id="secs"></tbody></table>
 <p class="msg small" id="fewsecs"></p>
 <div class="row" style="margin-top:10px"><label>Strictness <input type="range" id="strict" min="0.5" max="2" step="0.05"></label>
 <span class="dim" id="thr"></span></div></div>

<div class="panel"><h2>Real-time numbers</h2><div class="grid" id="stats"></div></div>
</main><script>
const $=id=>document.getElementById(id);let S=null;
async function api(a,b){const r=await fetch('/api/'+a,{method:'POST',body:JSON.stringify(b||{})});return r.json()}
function fill(sel,items,keep){const key=JSON.stringify(items);if(sel.dataset.key===key)return;   // unchanged: leave it alone so an open list stays open
 sel.dataset.key=key;const cur=sel.value;sel.innerHTML='';items.forEach(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;sel.appendChild(o)});if(keep&&[...sel.options].some(o=>o.value==cur))sel.value=cur}
function esc(t){return String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function render(s){S=s;const sls=s.setlist.state,lead=sls&&s.settings.drive!='library';
 $('locked').textContent=(lead?s.driving:s.locked)||(s.running?'unknown':'-');
 $('drivetag').textContent=lead?'(setlist'+(s.settings.drive=='both'?' + library re-sync':'')+')':'(full library)';
 $('both').textContent=sls?('full library: '+(s.locked||'unknown')+'  |  setlist: '+(sls.current||'-')+' ('+(sls.pos+1)+'/'+sls.sections.length+'), hears '+(sls.heard||'unknown')):'';
 if(document.activeElement!==$('drive'))$('drive').value=s.settings.drive;
 if(document.activeElement!==$('song')){fill($('song'),[['','(no song: full library only)'],...Object.keys(s.setlist.songs).map(n=>[n,n])],false);$('song').value=s.setlist.active||''}
 $('snext').disabled=$('sprev').disabled=!sls;
 const skey=sls?JSON.stringify([sls.sections,sls.pos]):'';
 if(skey!==$('sline').dataset.key){$('sline').dataset.key=skey;      // rebuild only on change, so clicks aren't lost
  $('sline').innerHTML=sls?sls.sections.map((n,i)=>`<button data-i="${i}" class="${i==sls.pos?'primary':''}">${i+1}. ${esc(n)}</button>`).join(''):'';
  $('sline').querySelectorAll('button').forEach(b=>b.onclick=()=>api('setlist_goto',{pos:+b.dataset.i}))}
 $('target').textContent=s.running?('effect: '+(s.target?s.target.replace(/^neural:/,'learned pedal '):'clean')):'';$('msg').textContent=s.message||'';
 const lv=Math.max(0,Math.min(100,(s.level_db+60)/60*100));$('lvlbar').style.width=lv+'%';$('lvl').textContent=s.level_db+' dB';
 fill($('file'),s.files.map(f=>[f,f]),true);fill($('wetfile'),[['','(none)'],...s.files.map(f=>[f,f])],true);
 if($('pedal').options.length!=s.standin.length+1)fill($('pedal'),[['','none (recognition only)'],...s.standin.map(p=>[p,p])],false);
 fill($('indev'),s.devices.input.map(d=>[d[0],d[1]]),true);fill($('outdev'),[['','default output'],...s.devices.output.map(d=>[d[0],d[1]])],true);
 $('start').disabled=s.running;$('stop').disabled=!s.running;
 if(document.activeElement!==$('master'))$('master').value=s.settings.master;if(document.activeElement!==$('strict'))$('strict').value=s.settings.strictness;
 $('bypass').checked=s.settings.bypass;$('thr').textContent='lock threshold '+s.threshold;
 $('rec').textContent=s.teaching?'Stop and save take':'Record take';$('rec').className=s.teaching?'danger':'primary';
 $('rec').disabled=!s.teaching&&(!s.running||!s.can_teach);
 $('hint').textContent=!s.running?'Press Start to enable teaching.':(s.teach_hint||'');
 $('recinfo').textContent=s.teaching?('recording '+s.teaching.name+' ... '+s.teaching.seconds+' s'):'';
 const pre=s.presets.map(p=>[p,p.replace(/^neural:/,'neural: ')]);
 const cost=Object.fromEntries(s.ranked);const conf=Object.fromEntries(s.guesses.filter(g=>g[0]));const tb=$('secs');
 $('guesses').innerHTML=(s.challenger?`<div class="small" style="color:var(--warn)">switching to ${esc(s.challenger[0])}? ${s.challenger[1]}/${s.confirm} in a row</div>`:
  (s.locked&&s.running?`<div class="dim small">locked: needs another section ${s.confirm}/${s.confirm} in a row to switch</div>`:''))+
  (s.guesses.length?'<div class="dim small">Guesses (confidence from match cost)</div>'+s.guesses.map(([n,p])=>
  `<div class="row small" style="gap:6px;flex-wrap:nowrap"><span style="width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${n?esc(n):'<i>unknown</i>'}</span><div class="bar" style="flex:1"><i style="width:${p}%;${n&&n==s.locked?'':'background:var(--dim)'}"></i></div><span style="width:48px;text-align:right">${p.toFixed(0)}%</span></div>`).join(''):'');
 $('fewsecs').textContent=s.sections.length==1?'Only one section taught: anything you play can only be compared with it. Teach at least two (the more different, the better) so it has something to choose between.':'';
 const tkey=JSON.stringify([s.sections,s.presets]);
 if(tkey!==tb.dataset.key&&!tb.contains(document.activeElement)){tb.dataset.key=tkey;tb.innerHTML='';   // rebuild rows only when they change
 s.sections.forEach(x=>{const tr=document.createElement('tr');tr.dataset.name=x.name;
  const e=x.effect;const eff=!e?'<span class="dim">no pedal data</span>':(e.status=='learning'?'<span class="dim">learning...</span>':
   `<span class="${e.status=='ready'?'ok':'bad'}">${e.status=='ready'?'learned':'not usable'}</span> <span class="dim small">${e.esr} dB vs ${e.baseline} dB (plain gain)</span><div class="dim small">${esc(e.verdict)}</div>`);
  tr.innerHTML=`<td><b>${esc(x.name)}</b></td><td>${x.takes}</td><td>${eff}</td><td></td><td class="conf"></td><td><button class="danger">Delete</button></td>`;
  const sel=document.createElement('select');fill(sel,pre,false);sel.value=x.preset;sel.onchange=()=>api('set_preset',{name:x.name,preset:sel.value}).then(r=>r.error&&alert(r.error));tr.children[3].appendChild(sel);
  tr.querySelector('button').onclick=()=>{if(confirm('Delete section '+x.name+' and its takes?'))api('delete_section',{name:x.name})};tb.appendChild(tr)})}
 tb.querySelectorAll('tr').forEach(tr=>{const n=tr.dataset.name,c=cost[n],w=conf[n]==null?0:conf[n];   // live numbers, every poll
  tr.querySelector('.conf').innerHTML=`<div class="bar"><i style="width:${w}%;${s.locked==n?'':'background:var(--dim)'}"></i></div><span>${conf[n]==null?'':conf[n].toFixed(0)+'%'}</span> <span class="dim small">${c==null?'':'('+c+')'}</span>`});
 const st=s.stats;$('stats').innerHTML=[['Audio callback (mean)',st.cb_mean_ms+' ms'],['Audio callback (worst)',st.cb_max_ms+' ms of '+st.deadline_ms+' ms'],
  ['Late blocks',st.late,st.late>0],['Dropouts (xruns)',st.xruns,st.xruns>0],['Dropped analysis blocks',st.dropped,st.dropped>0],
  ['Recogniser (mean)',st.analysis_mean_ms+' ms per 100 ms'],['Recogniser (worst)',st.analysis_max_ms+' ms'],['Recogniser runs in',s.recogniser]].map(a=>`<div class="stat"><b class="${a[2]?'bad':''}">${a[1]}</b><span>${a[0]}</span></div>`).join('')}
async function poll(){try{render(await (await fetch('/api/state')).json())}catch(e){}setTimeout(poll,250)}
$('srctype').onchange=()=>{const f=$('srctype').value=='file';$('filebox').style.display=f?'':'none';$('indev').style.display=f?'none':'';
 $('pedallbl').textContent=f?'or stand-in pedal':'stand-in pedal (1-input devices)'};
$('start').onclick=async()=>{const t=$('srctype').value;const out=$('outdev').value;const o=out===''?null:+out;
 const r=await api('start',t=='file'?{type:'file',path:$('file').value,wet_path:$('wetfile').value||null,pedal_preset:$('pedal').value,output:o}:{type:'device',input:+$('indev').value,output:o,pedal_preset:$('pedal').value});if(r.error)alert(r.error)};
$('stop').onclick=()=>api('stop');
$('rec').onclick=async()=>{const r=S&&S.teaching?await api('teach_stop'):await api('teach_start',{name:$('tname').value.trim()});if(r.error)alert(r.error)};
$('master').oninput=()=>api('setting',{key:'master',value:+$('master').value});$('strict').oninput=()=>api('setting',{key:'strictness',value:+$('strict').value});
$('bypass').onchange=()=>api('setting',{key:'bypass',value:$('bypass').checked});
$('drive').onchange=()=>api('setting',{key:'drive',value:$('drive').value});
$('song').onchange=()=>{const n=$('song').value;api('setlist_select',{song:n||null});if(n&&S){$('sname').value=n;$('slist').value=S.setlist.songs[n].join(', ')}};
$('snext').onclick=()=>api('setlist_step',{step:1});$('sprev').onclick=()=>api('setlist_step',{step:-1});
$('ssave').onclick=async()=>{const r=await api('setlist_save',{song:$('sname').value.trim(),sections:$('slist').value.split(',').map(x=>x.trim()).filter(x=>x)});if(r.error)alert(r.error)};
$('sdel').onclick=async()=>{const n=$('sname').value.trim();if(n&&confirm('Delete setlist '+n+'?')){const r=await api('setlist_delete',{song:n});if(r.error)alert(r.error)}};
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    engine = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(self.engine.state()))
        else:
            self._send(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, json.dumps({"error": "bad request"}))
        e, a = self.engine, self.path.rsplit("/", 1)[-1]
        with e.lock:
            if a == "start":
                err = e.start(body)
            elif a == "stop":
                e.stop(); err = None
            elif a == "teach_start":
                err = e.teach_start(body.get("name"))
            elif a == "teach_stop":
                err = e.teach_stop()
            elif a == "set_preset":
                err = e.set_preset(body.get("name"), body.get("preset"))
            elif a == "delete_section":
                err = e.delete_section(body.get("name"))
            elif a == "setting":
                err = e.set_setting(body.get("key"), body.get("value"))
            elif a == "setlist_save":
                err = e.setlist_save(body.get("song"), body.get("sections"))
            elif a == "setlist_delete":
                err = e.setlist_delete(body.get("song"))
            elif a == "setlist_select":
                err = e.setlist_select(body.get("song"))
            elif a == "setlist_step":
                err = e.setlist_move(step=int(body.get("step") or 1))
            elif a == "setlist_goto":
                err = e.setlist_move(goto=int(body.get("pos") or 0))
            else:
                return self._send(404, "{}")
        self._send(200, json.dumps({"error": err} if err else {"ok": True}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    Handler.engine = Engine()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)    # this computer only
    url = f"http://127.0.0.1:{args.port}"
    print(f"Pedal demo running at {url}  (Ctrl-C to quit)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.engine.close()
        server.server_close()


if __name__ == "__main__":
    main()