"""
stream_effects.py

Runs your own effect scripts on audio *as if it were streaming*: the file is
cut into small blocks and each block is passed through the effect chain one at
a time, exactly like a sound-card callback would. Effects never see the whole
file or any future audio.

An effect can be any of these:

  1. A class with  process(block) -> block   (best: it can keep state, e.g. a
     delay line or filter memory). Optional methods:
         prepare(sr, block_size)   called once before streaming
         reset()                   called before each run
  2. A plain function  f(block) -> block  or  f(block, sr) -> block
     (fine for stateless effects like distortion or gain)
  3. A function that needs past audio (FIR filter, convolution, FFT-based):
     wrap it with  Context(f, context_samples=N)  and it will receive the
     last N samples plus the new block, and only the new block is kept.

Every effect must return the SAME number of samples it was given.

Usage:
    python stream_effects.py clip.wav --module my_effects --chain Distortion Delay
    python stream_effects.py clip.wav --module my_effects --chain Distortion --block 128 --check

    --check   also runs the effect on the whole file at once and compares.
              A difference means the effect is not streamable as written
              (it uses future audio, whole-file statistics, or loses state
              between blocks). This is the most useful test to run.

Outputs go to streamed/: <name>_dry.wav, <name>_wet.wav and streamed_report.png
"""
import argparse
import importlib
import inspect
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- adapters
class FunctionEffect:
    """Wraps a stateless function f(block) or f(block, sr)."""

    def __init__(self, fn):
        self.fn, self.sr = fn, 44100
        self.takes_sr = len(inspect.signature(fn).parameters) >= 2
        self.name = getattr(fn, "__name__", "function")

    def prepare(self, sr, block_size):
        self.sr = sr

    def process(self, block):
        return self.fn(block, self.sr) if self.takes_sr else self.fn(block)


class Context:
    """Gives a function the last `context_samples` of audio plus the new block
    (overlap-save). Only the part corresponding to the new block is returned."""

    def __init__(self, fn, context_samples):
        self.fn, self.n = FunctionEffect(fn), int(context_samples)
        self.name = f"{self.fn.name}[context {self.n}]"
        self.hist = np.zeros(self.n)

    def prepare(self, sr, block_size):
        self.fn.prepare(sr, block_size)

    def reset(self):
        self.hist = np.zeros(self.n)

    def process(self, block):
        buf = np.concatenate([self.hist, block])
        out = np.asarray(self.fn.process(buf))
        self.hist = buf[-self.n:] if self.n else self.hist
        return out[-len(block):]


def as_effect(obj):
    """Turn a class, instance or function into something with .process()."""
    if inspect.isclass(obj):
        obj = obj()
    if hasattr(obj, "process"):
        if not hasattr(obj, "name"):
            obj.name = type(obj).__name__
        return obj
    if callable(obj):
        return FunctionEffect(obj)
    raise TypeError(f"Don't know how to use {obj!r} as an effect")


class Chain:
    def __init__(self, effects):
        self.effects = [as_effect(e) for e in effects]

    def prepare(self, sr, block_size):
        for e in self.effects:
            if hasattr(e, "prepare"):
                e.prepare(sr, block_size)

    def reset(self):
        for e in self.effects:
            if hasattr(e, "reset"):
                e.reset()

    def process(self, block):
        for e in self.effects:
            out = np.asarray(e.process(block), dtype=np.float64)
            if out.shape != block.shape:
                raise ValueError(f"Effect '{e.name}' returned {out.shape[0]} samples for a "
                                 f"block of {block.shape[0]}. Streaming effects must return "
                                 f"exactly as many samples as they receive.")
            block = out
        return block


# ---------------------------------------------------------------- streaming
def stream(audio, sr, chain, block_size):
    """Feed `audio` through `chain` block by block. Returns (output, block_times_s)."""
    chain.prepare(sr, block_size)
    chain.reset()
    out = np.empty(len(audio))
    times = []
    for start in range(0, len(audio), block_size):
        blk = audio[start:start + block_size].copy()
        t0 = time.perf_counter()
        out[start:start + len(blk)] = chain.process(blk)
        times.append(time.perf_counter() - t0)
    return out, np.array(times)


def offline(audio, sr, chain):
    """Same chain on the whole file in one go (the reference answer)."""
    chain.prepare(sr, len(audio))
    chain.reset()
    return chain.process(audio.copy())


def report_timing(times, block_size, sr):
    budget = block_size / sr
    ms = times * 1000
    over = int(np.sum(times > budget))
    print(f"  block = {block_size} samples = {budget*1000:.2f} ms of audio")
    print(f"  processing time: mean {ms.mean():.3f} ms, 99th pct {np.percentile(ms, 99):.3f} ms, "
          f"max {ms.max():.3f} ms")
    print(f"  blocks slower than real time (would cause a dropout): {over} of {len(times)}")
    return over


def compare(streamed, reference):
    n = min(len(streamed), len(reference))
    diff = np.max(np.abs(streamed[:n] - reference[:n]))
    scale = max(np.max(np.abs(reference[:n])), 1e-12)
    return diff, diff / scale


# ---------------------------------------------------------------- CLI
def load_chain(module_name, names):
    sys.path.insert(0, str(Path.cwd()))
    mod = importlib.import_module(module_name)
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        public = [n for n in dir(mod) if not n.startswith("_")]
        raise SystemExit(f"{module_name} has no {missing}. It contains: {public}")
    return Chain([getattr(mod, n) for n in names])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--module", required=True, help="python module holding your effects")
    ap.add_argument("--chain", nargs="+", required=True, help="effect names, in order")
    ap.add_argument("--block", type=int, default=256, help="block size in samples")
    ap.add_argument("--check", action="store_true", help="compare against whole-file processing")
    args = ap.parse_args()

    audio, sr = sf.read(args.audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    chain = load_chain(args.module, args.chain)

    print(f"Streaming {len(audio)/sr:.1f} s of audio through: " +
          " -> ".join(e.name for e in chain.effects))
    wet, times = stream(audio, sr, chain, args.block)
    report_timing(times, args.block, sr)

    if args.check:
        ref = offline(audio, sr, Chain(chain.effects))
        diff, rel = compare(wet, ref)
        verdict = "matches whole-file processing" if rel < 1e-6 else \
            "DIFFERS from whole-file processing, so the effect is not streamable as written"
        print(f"  streamed vs whole-file: max difference {diff:.2e} ({rel:.2%} of peak): {verdict}")

    out = Path("streamed"); out.mkdir(exist_ok=True)
    stem = Path(args.audio).stem
    gain = 0.8 / max(np.max(np.abs(audio)), np.max(np.abs(wet)), 1e-12)   # one gain for both
    sf.write(out / f"{stem}_dry.wav", audio * gain, sr)
    sf.write(out / f"{stem}_wet.wav", np.clip(wet * gain, -1, 1), sr)

    t = np.arange(len(audio)) / sr
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax[0].plot(t, audio, lw=0.4); ax[0].set_ylabel("dry")
    ax[1].plot(t, wet, lw=0.4, color="tab:red"); ax[1].set_ylabel("streamed output")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(out / "streamed_report.png", dpi=110)
    print(f"Saved {out}/{stem}_dry.wav, {stem}_wet.wav and streamed_report.png")


if __name__ == "__main__":
    main()