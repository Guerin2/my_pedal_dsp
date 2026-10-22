"""
effect_trainer.py

Learns the pedal's sound for one section from the takes recorded in Teach.
pedal_app.py runs train_job() in its own short-lived PROCESS, so training (which
is heavy) can never disturb the audio or the recogniser.

For each take there is a pair of recordings made at the same moment:
    take_k.wav       the clean guitar
    take_k_wet.wav   what the pedal did to it
The job aligns each pair in time, trains the small tapped-delay network from
neural_effect.py on the first 80% of every take, tests it on the last 20% of every
take, compares it with a plain gain change, and saves:
    sections/<name>/effect.npz    the model (loadable by neural_effect.NeuralEffect)
    sections/<name>/effect.json   the scores and a verdict

Result message put on res_q:
    ("trained", name, generation, {"ok", "verdict", "esr", "baseline", "path", "takes", "seconds"})
"""
import json
import time
from pathlib import Path

import numpy as np

import neural_effect as ne
import train_from_pairs as tp
from recogniser_worker import read_wav

SR = ne.SR
MIN_TOTAL_S = 6.0          # less audio than this cannot teach the network anything useful
TRAIN_FRACTION = 0.8


def train_job(name, pairs, out_dir, gen, res_q, hidden=(32, 16)):
    t0 = time.time()
    try:
        rng = np.random.default_rng(0)
        rows = max(20000, 120000 // max(len(pairs), 1))
        Xs, Ys, tests, total = [], [], [], 0
        for dry_path, wet_path in pairs:
            dry, wet = read_wav(dry_path), read_wav(wet_path)
            n = min(len(dry), len(wet))
            dry, wet = dry[:n], wet[:n]
            lag, _, _ = tp.estimate_lag(dry, wet, int(tp.MAX_LAG_MS * SR / 1000))
            if lag > 0:
                dry, wet = dry[:len(dry) - lag], wet[lag:]
            elif lag < 0:
                dry, wet = dry[-lag:], wet[:len(wet) + lag]
            n = min(len(dry), len(wet))
            dry, wet = dry[:n], wet[:n]
            total += n
            cut = int(n * TRAIN_FRACTION)
            if cut < SR:
                continue
            buf = np.concatenate([np.zeros(ne.HIST), dry])
            active = np.where(np.abs(dry[:cut]) > 0.2 * np.sqrt(np.mean(dry[:cut] ** 2)))[0]
            pos = np.concatenate([rng.integers(0, cut, rows // 2),
                                  rng.choice(active, rows - rows // 2) if len(active)
                                  else rng.integers(0, cut, rows - rows // 2)])
            Xs.append(ne.tap_rows(buf, pos))
            Ys.append(wet[pos])
            tests.append((dry[cut:], wet[cut:]))
        if total / SR < MIN_TOTAL_S or not Xs:
            raise ValueError(f"only {total / SR:.1f} s of paired audio. Record at least {MIN_TOTAL_S:.0f} s "
                             f"in total (2 or 3 takes of the section).")
        params, _ = ne.train(np.concatenate(Xs), np.concatenate(Ys), list(hidden))

        test_dry = np.concatenate([d for d, _ in tests])
        test_wet = np.concatenate([w for _, w in tests])
        pred = np.concatenate([ne.stream_model(params, d, 256)[0] for d, _ in tests])
        esr = float(ne.esr(test_wet, pred))
        base = float(tp.baseline_esr(test_dry, test_wet))
        verdict = tp.verdict(esr, base)
        ok = not verdict.startswith("FAILED")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ne.save_model(out / "effect.npz", params, name)
        info = {"ok": ok, "verdict": verdict, "esr": round(esr, 1), "baseline": round(base, 1),
                "path": (out / "effect.npz").as_posix(), "takes": len(pairs),
                "seconds": round(time.time() - t0, 1)}
        (out / "effect.json").write_text(json.dumps(info, indent=2))
        res_q.put(("trained", name, gen, info))
    except Exception as e:
        res_q.put(("trained", name, gen, {"ok": False, "verdict": f"training error: {e}", "esr": 0.0,
                                          "baseline": 0.0, "path": None, "takes": len(pairs),
                                          "seconds": round(time.time() - t0, 1)}))