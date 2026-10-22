"""
my_effects.py

Streaming-safe guitar effects for stream_effects.py. Each effect is a class
with process(block) -> block of the same length, keeps its own state between
blocks, and gives the same output whatever the block size.

    python stream_effects.py clip.wav --module my_effects --chain Compressor Overdrive Chorus Delay Reverb --check

Effects (all parameters are constructor arguments):
    Overdrive   soft clipping with a tone control
    Distortion  harder tanh clipping with pre/post filtering
    Fuzz        asymmetric hard clipping
    Bitcrush    lo-fi sample rounding
    Octave      adds an octave below and/or above (analog style, best on single notes)
    OctaveUp    octave up only (Octavia); try OctaveUp then Fuzz
    Wah         touch wah: each pick sweeps a resonant filter up, then it falls back
                as the note fades ("jow" / "joo")
    AutoWah     the same filter swept by a slow wobble instead of your picking
    Tremolo     amplitude wobble
    Chorus      short modulated delay mixed with the dry signal
    Delay       echo with feedback
    Reverb      small Schroeder reverb (parallel combs + series all-passes)
    Compressor  smooths level differences
    Limiter     keeps peaks below a ceiling (put last in a chain)
"""
import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------- helpers
class _OnePole:
    """Streaming one-pole low-pass: y[n] = (1-a) x[n] + a y[n-1]."""

    def __init__(self, a):
        self.b, self.a = np.array([1.0 - a]), np.array([1.0, -a])
        self.reset()

    def reset(self):
        self.zi = np.zeros(1)

    def process(self, x):
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


def _cutoff_to_coeff(fc, sr):
    return float(np.exp(-2 * np.pi * fc / sr))



class _Loop:
    """Feedback delay loop: w[n] = x[n] + g w[n-D], using a history of the last D values.
    Much cheaper than a D-tap lfilter, which is too slow for a 5 ms block deadline."""

    def __init__(self, D, g):
        self.D, self.g = max(int(D), 1), g
        self.reset()

    def reset(self):
        self.hist = np.zeros(self.D)

    def run(self, x):
        """Returns (w, w_delayed_by_D) for the block."""
        w_out, d_out = np.empty_like(x), np.empty_like(x)
        pos = 0
        while pos < len(x):
            m = min(self.D, len(x) - pos)
            past = self.hist[:m]
            w = x[pos:pos + m] + self.g * past
            w_out[pos:pos + m], d_out[pos:pos + m] = w, past
            self.hist = np.concatenate([self.hist[m:], w])
            pos += m
        return w_out, d_out


class _Base:
    sr = 44100

    def prepare(self, sr, block_size):
        self.sr = sr
        self._setup()
        self.reset()

    def _setup(self):
        pass

    def reset(self):
        pass


# ---------------------------------------------------------------- distortion family
class Overdrive(_Base):
    """Gentle soft clipping. drive 1-10, tone_hz = low-pass after clipping."""

    def __init__(self, drive=4.0, tone_hz=4500, level=0.7, mix=1.0):
        self.name, self.drive, self.tone_hz, self.level, self.mix = "Overdrive", drive, tone_hz, level, mix
        self._setup(); self.reset()

    def _setup(self):
        self.lp = _OnePole(_cutoff_to_coeff(self.tone_hz, self.sr))

    def reset(self):
        self.lp.reset()

    def process(self, x):
        wet = self.lp.process(np.arctan(self.drive * x) / (np.pi / 2)) * self.level
        return (1 - self.mix) * x + self.mix * wet


class Distortion(_Base):
    """Heavier clipping. A pre-filter removes low-end mud before the clipper."""

    def __init__(self, gain=30.0, pre_hp_hz=120, tone_hz=3500, level=0.25, mix=1.0):
        self.name = "Distortion"
        self.gain, self.pre_hp_hz, self.tone_hz, self.level, self.mix = gain, pre_hp_hz, tone_hz, level, mix
        self._setup(); self.reset()

    def _setup(self):
        self.hp_lp = _OnePole(_cutoff_to_coeff(self.pre_hp_hz, self.sr))   # high-pass = x - lowpass(x)
        self.lp = _OnePole(_cutoff_to_coeff(self.tone_hz, self.sr))

    def reset(self):
        self.hp_lp.reset(); self.lp.reset()

    def process(self, x):
        hp = x - self.hp_lp.process(x)
        wet = self.lp.process(np.tanh(self.gain * hp)) * self.level
        return (1 - self.mix) * x + self.mix * wet


class Fuzz(_Base):
    """Asymmetric hard clipping (different positive and negative thresholds)."""

    def __init__(self, gain=40.0, pos=0.5, neg=0.25, tone_hz=3000, level=0.5):
        self.name, self.gain, self.pos, self.neg = "Fuzz", gain, pos, neg
        self.tone_hz, self.level = tone_hz, level
        self._setup(); self.reset()

    def _setup(self):
        self.lp = _OnePole(_cutoff_to_coeff(self.tone_hz, self.sr))

    def reset(self):
        self.lp.reset()

    def process(self, x):
        y = np.clip(self.gain * x, -self.neg, self.pos)
        return self.lp.process(y) * self.level


class Bitcrush(_Base):
    """Lo-fi: rounds every sample to 2**(bits-1) levels. Stateless, so it streams trivially."""

    def __init__(self, bits=4, mix=1.0):
        self.name, self.bits, self.mix = "Bitcrush", bits, mix

    def process(self, x):
        shift = 2.0 ** (self.bits - 1)
        return (1 - self.mix) * x + self.mix * np.round(x * shift) / shift


# ---------------------------------------------------------------- pitch
class Octave(_Base):
    """Analog-style octaver, like an OC-2 (down) plus an Octavia (up).

    Octave up:   a "doppler" pitch shifter: two read heads sweep through a short delay
                 line at twice playback speed, crossfaded so each jump back is silent.
                 Shifts everything up an octave (chords too) with a slight warble.
                 (Rectifying |x| like an analog Octavia only works high up the neck.)
    Octave down: the input is low-passed so mostly the fundamental is left, a Schmitt
                 trigger turns it into a square wave (the hysteresis, a fraction of the
                 playing level, ignores noise and small wiggles), and a flip-flop toggles on
                 every second upward crossing: a square wave at half the frequency. That is
                 shaped by the playing envelope and smoothed into a sub-bass tone.
    Like the real pedals the octave down tracks single notes well and glitches on chords."""

    def __init__(self, down=0.8, up=0.0, dry=0.7, track_hz=900, down_tone_hz=600, up_tone_hz=5000,
                 hysteresis=0.2, up_window_ms=80.0, level=1.0):
        self.name = "Octave"
        self.down, self.up, self.dry, self.level = down, up, dry, level
        self.track_hz, self.down_tone_hz, self.up_tone_hz, self.hyst = track_hz, down_tone_hz, up_tone_hz, hysteresis
        self.up_window_ms = up_window_ms
        self._setup(); self.reset()

    def _setup(self):
        a = _cutoff_to_coeff(self.track_hz, self.sr)
        self.track1, self.track2 = _OnePole(a), _OnePole(a)           # isolate the fundamental
        self.env = _OnePole(float(np.exp(-1.0 / (0.01 * self.sr))))  # 10 ms envelope follower
        self.down_lp1 = _OnePole(_cutoff_to_coeff(self.down_tone_hz, self.sr))
        self.down_lp2 = _OnePole(_cutoff_to_coeff(self.down_tone_hz, self.sr))
        self.up_lp = _OnePole(_cutoff_to_coeff(self.up_tone_hz, self.sr))
        self.D = max(4, int(self.up_window_ms * self.sr / 1000))

    def reset(self):
        for f in (self.track1, self.track2, self.env, self.down_lp1, self.down_lp2, self.up_lp):
            f.reset()
        self.zone, self.flip = -1.0, 1.0          # Schmitt trigger state, flip-flop output
        self.hist, self.n = np.zeros(self.D), 0   # octave-up delay line and sample counter

    def _octave_down(self, x):
        t = self.track2.process(self.track1.process(x))
        env = self.env.process(np.abs(t))
        h = self.hyst * env
        zone = np.where(t > h, 1.0, np.where(t < -h, -1.0, 0.0))
        # hold the last definite zone through the dead band (forward fill, carried across blocks)
        idx = np.where(zone != 0, np.arange(len(zone)), -1)
        idx = np.maximum.accumulate(idx)
        held = np.where(idx >= 0, zone[np.maximum(idx, 0)], self.zone)
        prev = np.concatenate([[self.zone], held[:-1]])
        rising = (held > 0) & (prev < 0)
        flips = np.cumsum(rising)
        sq = self.flip * np.where(flips % 2 == 0, 1.0, -1.0)
        if len(x):
            self.zone, self.flip = float(held[-1]), float(sq[-1])
        # 2x: the low-passed tracker signal is quieter than the input
        return self.down_lp2.process(self.down_lp1.process(sq * 2.0 * env))

    def _octave_up(self, x):
        D, m = self.D, len(x)
        buf = np.concatenate([self.hist, x])            # buf[D + i] is x[i]
        i = np.arange(m)
        out = np.zeros(m)
        for offset in (0.0, 0.5):
            ph = ((self.n + i) / D + offset) % 1.0      # 0..1 along this head's sweep
            d = np.round(D * (1.0 - ph)).astype(int)    # delay shrinks 1 sample per sample: 2x speed
            w = 1.0 - np.abs(2.0 * ph - 1.0)            # silent at the jump, the two heads sum to 1
            out += w * buf[D + i - d]
        self.hist = buf[-D:]
        self.n = (self.n + m) % (2 * D)
        return self.up_lp.process(out)

    def process(self, x):
        out = self.dry * x
        if self.down:
            out = out + self.down * self._octave_down(x)
        if self.up:
            out = out + self.up * self._octave_up(x)
        return out * self.level


class OctaveUp(Octave):
    """Octave up only (the Octavia sound); put a Fuzz after it."""

    def __init__(self, up=1.0, dry=0.6, **kw):
        super().__init__(down=0.0, up=up, dry=dry, **kw)
        self.name = "OctaveUp"


# ---------------------------------------------------------------- filter
class Wah(_Base):
    """Wah: a resonant band-pass filter whose centre frequency moves between low_hz ("oo")
    and high_hz ("ah"). A real wah pedal moves it with your foot; this one is a touch wah,
    so your picking moves it: an envelope follower (fast attack, slower release) reads how
    hard you play, so each note opens the filter and it closes again as the note fades.
    That down-sweep is the "jow" / "joo" sound.

    sensitivity_db: playing this many dB above floor_db opens the filter fully.
    q: resonance (higher = more vocal and peaky). release_ms: how slowly it closes.
    The filter is a TPT state-variable filter, stable while its frequency changes every sample."""

    def __init__(self, low_hz=350.0, high_hz=2200.0, q=5.0, floor_db=-45.0, sensitivity_db=30.0,
                 attack_ms=4.0, release_ms=180.0, level=1.8, mix=1.0):
        self.name = "Wah"
        self.low_hz, self.high_hz, self.q = low_hz, high_hz, q
        self.floor_db, self.sens_db, self.attack_ms, self.release_ms = floor_db, sensitivity_db, attack_ms, release_ms
        self.level, self.mix = level, mix
        self._setup(); self.reset()

    def _setup(self):
        self.att = float(np.exp(-1.0 / (self.attack_ms * self.sr / 1000)))
        self.rel = float(np.exp(-1.0 / (self.release_ms * self.sr / 1000)))
        self.pre = _OnePole(_cutoff_to_coeff(60, self.sr))       # the follower ignores low rumble

    def reset(self):
        self.pre.reset()
        self.env, self.ic1, self.ic2 = 0.0, 0.0, 0.0
        self.last_fc = np.zeros(0)

    def _position(self, x):
        """0..1 sweep position per sample (0 = closed/"oo", 1 = open/"ah")."""
        hp = np.abs(x - self.pre.process(x))
        env = np.empty_like(hp)
        e, att, rel = self.env, self.att, self.rel
        for i, v in enumerate(hp):
            e = (att if v > e else rel) * (e - v) + v
            env[i] = e
        self.env = e
        return np.clip((20 * np.log10(env + 1e-9) - self.floor_db) / self.sens_db, 0.0, 1.0)

    def process(self, x):
        pos = self._position(x)
        fc = self.low_hz * (self.high_hz / self.low_hz) ** pos          # sweep in octaves, like the ear
        self.last_fc = fc
        g = np.tan(np.pi * np.minimum(fc, 0.45 * self.sr) / self.sr)
        k = 1.0 / self.q
        a1 = 1.0 / (1.0 + g * (g + k))
        a2, a3 = g * a1, g * g * a1
        bp = np.empty_like(x)
        ic1, ic2 = self.ic1, self.ic2
        for i in range(len(x)):
            v3 = x[i] - ic2
            v1 = a1[i] * ic1 + a2[i] * v3
            v2 = ic2 + a2[i] * ic1 + a3[i] * v3
            ic1, ic2 = 2.0 * v1 - ic1, 2.0 * v2 - ic2
            bp[i] = v1
        self.ic1, self.ic2 = ic1, ic2
        wet = bp * k * self.q ** 0.5 * self.level     # resonant peak, but not Q times louder
        return (1 - self.mix) * x + self.mix * wet


class AutoWah(Wah):
    """The wah filter swept by a slow sine wobble instead of your picking."""

    def __init__(self, rate_hz=1.5, depth=1.0, **kw):
        self.rate, self.depth = rate_hz, depth
        super().__init__(**kw)
        self.name = "AutoWah"

    def reset(self):
        super().reset()
        self.phase = 0.0

    def _position(self, x):
        ph = self.phase + 2 * np.pi * self.rate * np.arange(len(x)) / self.sr
        self.phase = float((self.phase + 2 * np.pi * self.rate * len(x) / self.sr) % (2 * np.pi))
        return 0.5 + 0.5 * self.depth * np.sin(ph)


# ---------------------------------------------------------------- modulation
class Tremolo(_Base):
    def __init__(self, rate_hz=5.0, depth=0.6):
        self.name, self.rate, self.depth = "Tremolo", rate_hz, depth
        self.reset()

    def reset(self):
        self.n = 0

    def process(self, x):
        t = (self.n + np.arange(len(x))) / self.sr
        self.n += len(x)
        lfo = 0.5 * (1 + np.sin(2 * np.pi * self.rate * t))        # 0..1
        return x * (1 - self.depth * lfo)


class Chorus(_Base):
    """Modulated delay (about 15 ms +/- depth) mixed with the dry signal."""

    def __init__(self, rate_hz=0.8, base_ms=15.0, depth_ms=4.0, mix=0.5):
        self.name, self.rate, self.base_ms, self.depth_ms, self.mix = "Chorus", rate_hz, base_ms, depth_ms, mix
        self._setup(); self.reset()

    def _setup(self):
        self.base = self.base_ms * self.sr / 1000
        self.depth = self.depth_ms * self.sr / 1000
        self.L = int(np.ceil(self.base + self.depth)) + 3

    def reset(self):
        self.hist = np.zeros(self.L)
        self.n = 0

    def process(self, x):
        n = len(x)
        buf = np.concatenate([self.hist, x])
        t = (self.n + np.arange(n)) / self.sr
        self.n += n
        d = self.base + self.depth * np.sin(2 * np.pi * self.rate * t)
        pos = self.L + np.arange(n) - d                              # fractional read position
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        delayed = (1 - frac) * buf[i0] + frac * buf[i0 + 1]
        self.hist = buf[-self.L:]
        return (1 - self.mix) * x + self.mix * delayed


# ---------------------------------------------------------------- time based
class Delay(_Base):
    """Echo: delay_ms between repeats, feedback sets how many repeats you hear."""

    def __init__(self, delay_ms=350.0, feedback=0.4, mix=0.35, damp_hz=3500):
        self.name = "Delay"
        self.delay_ms, self.feedback, self.mix, self.damp_hz = delay_ms, feedback, mix, damp_hz
        self._setup(); self.reset()

    def _setup(self):
        self.loop = _Loop(self.sr * self.delay_ms / 1000, self.feedback)
        self.damp = _OnePole(_cutoff_to_coeff(self.damp_hz, self.sr))

    def reset(self):
        self.loop.reset(); self.damp.reset()

    def process(self, x):
        _, echo = self.loop.run(x)            # echo[n] = x[n-D] + fb * echo[n-D]
        return x + self.mix * self.damp.process(echo)


class Reverb(_Base):
    """Small Schroeder reverb. room 0-1 sets decay length."""

    COMBS_MS = (29.7, 37.1, 41.1, 43.7)
    ALLPASS_MS = (5.0, 1.7)

    def __init__(self, room=0.6, damp_hz=5000, mix=0.25):
        self.name, self.room, self.damp_hz, self.mix = "Reverb", room, damp_hz, mix
        self._setup(); self.reset()

    def _setup(self):
        fb = 0.70 + 0.25 * self.room
        self.combs = [_Loop(self.sr * ms / 1000, fb) for ms in self.COMBS_MS]
        self.aps = [_Loop(self.sr * ms / 1000, 0.5) for ms in self.ALLPASS_MS]
        self.damp = _OnePole(_cutoff_to_coeff(self.damp_hz, self.sr))

    def reset(self):
        for l in self.combs + self.aps:
            l.reset()
        self.damp.reset()

    def process(self, x):
        wet = np.zeros_like(x)
        for c in self.combs:
            wet += c.run(x)[1]
        wet = self.damp.process(wet / len(self.combs))
        for ap in self.aps:                    # Schroeder all-pass: y = -g w[n] + w[n-D]
            w, wd = ap.run(wet)
            wet = -ap.g * w + wd
        return (1 - self.mix) * x + self.mix * wet * 1.5


# ---------------------------------------------------------------- dynamics
class Compressor(_Base):
    """Level detector with a single smoothing time, then gain reduction above the threshold."""

    def __init__(self, threshold_db=-24.0, ratio=4.0, time_ms=15.0, makeup_db=6.0):
        self.name = "Compressor"
        self.threshold_db, self.ratio, self.time_ms, self.makeup_db = threshold_db, ratio, time_ms, makeup_db
        self._setup(); self.reset()

    def _setup(self):
        self.det = _OnePole(float(np.exp(-1.0 / (self.time_ms * self.sr / 1000))))
        self.makeup = 10 ** (self.makeup_db / 20)

    def reset(self):
        self.det.reset()

    def process(self, x):
        level = self.det.process(np.abs(x)) + 1e-9
        over_db = np.maximum(20 * np.log10(level) - self.threshold_db, 0.0)
        gain_db = -over_db * (1 - 1 / self.ratio)
        return x * 10 ** (gain_db / 20) * self.makeup


class Limiter(_Base):
    """Soft ceiling so the end of a chain can't clip."""

    def __init__(self, ceiling=0.9):
        self.name, self.ceiling = "Limiter", ceiling

    def process(self, x):
        return self.ceiling * np.tanh(x / self.ceiling)