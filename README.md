# my_pedal_dsp

A guitar pedal prototype for live performance: it recognises what you are playing
and changes the effect by itself.

- **Playing-style classifier** (`classify_windows.py`, `realtime_demo.py`): soft notes / loud chords /
  palm mute / silence from short audio windows, with window-length, ablation and per-clip results.
- **Section recogniser** (`section_recogniser.py`, `recogniser_worker.py`): a "Shazam for your own
  playing" using chroma features and subsequence DTW, with rejection ("unknown") and sticky locking.
- **Setlist mode** (`setlist.py`): follows a song's section order, running alongside the full library.
- **Pedal app** (`pedal_app.py`): browser UI to teach sections, pick effects, and watch it run live.
- **Effects** (`my_effects.py`): streaming-safe overdrive, distortion, fuzz, octave, wah, chorus,
  delay, reverb, compressor and more; `neural_effect.py` / `train_from_pairs.py` learn a real pedal.

## Setup
```
python -m venv venv
venv/bin/pip install -r requirements.txt
sudo apt install libportaudio2 ffmpeg
venv/bin/python -m pytest -q tests
```

## Typical use
```
venv/bin/python record_clips.py --device "USB Audio" --meter      # set input level
venv/bin/python record_clips.py --device "USB Audio" --class palm_mute -n 5
PEDAL_DATA=cable venv/bin/python main.py --skip convert            # trim, features, classify, demo
venv/bin/python pedal_app.py                                       # live app at http://127.0.0.1:8765
```

Recordings, the dataset and generated audio are not in the repository (see `.gitignore`).

## Credit
Optional extra training data: IDMT-SMT-Guitar (Kehling et al., DAFx 2014), CC BY-NC-ND 4.0,
downloaded by `fetch_dataset.py`. It is not redistributed here.
