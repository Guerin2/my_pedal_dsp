import numpy as np, soundfile as sf

from settings import DATA_RAW

db = lambda x: 20 * np.log10(x + 1e-12)

for path in sorted(DATA_RAW.glob("*/*.wav")):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame = int(0.025 * sr)                      # 25 ms frames
    n = len(y) // frame
    fr = np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1))
    print(f"{path.parent.name:12s} {path.name:20s} "
          f"{len(y)/sr:5.1f}s  sr={sr}  "
          f"peak {db(np.max(np.abs(y))):6.1f} dBFS  "
          f"rms {db(np.sqrt(np.mean(y**2))):6.1f}  "
          f"floor {db(np.percentile(fr, 10)):6.1f}  "
          f"clipped {np.mean(np.abs(y) > 0.99) * 100:.2f}%")