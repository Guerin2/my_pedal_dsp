"""
setlist.py

Setlist mode: you tell the pedal the order of sections in a song, and the
recogniser only has to answer "still the current section, or has the next one
started?" instead of "which of everything I know is this?". Two candidates
instead of the whole library means fewer wrong locks and less work per decision.

It is separate from the full-library recogniser and runs NEXT TO it on the same
audio (recogniser_worker.py feeds both), so the page can show both answers and
you choose which one sets the effect:

    drive = "library"   the full-library recogniser, exactly as before
    drive = "setlist"   the setlist position sets the effect
    drive = "both"      the setlist leads; the full library is the cross-check:
                        if the setlist hears neither the current nor the next
                        section while the full library is sure of another
                        section in this song, the setlist jumps there
                        (you skipped or repeated a part)

With no setlist loaded, every mode behaves like "library".

While a song is loaded the effect HOLDS the current section's preset, even
through "unknown" moments (a pause, a fluffed bar): live, a dropout to clean
is worse than staying put. A manual next/previous control moves the position
too, for when the recogniser can't (e.g. the same section twice in a row).

File: sections/setlists.json
    {"songs": {"my_song": ["intro", "verse", "chorus", "verse", "chorus", "solo"]},
     "active": "my_song"}
"""
import json

from section_recogniser import CONFIRM, MARGIN
from settings import SECTIONS

SETLISTS_FILE = SECTIONS / "setlists.json"
RESYNC_CONFIRM = 2 * CONFIRM            # the full library must agree this long before a jump
DRIVES = ("library", "setlist", "both")


def load_setlists(path=SETLISTS_FILE):
    if not path.exists():
        return {"songs": {}, "active": None}
    data = json.loads(path.read_text())
    songs = {str(k): [str(s) for s in v] for k, v in data.get("songs", {}).items()}
    active = data.get("active") if data.get("active") in songs else None
    return {"songs": songs, "active": active}


def save_setlists(data, path=SETLISTS_FILE):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


class SetlistTracker:
    """Follows one song's section order. Feed it the chroma query each decision step."""

    def __init__(self, sections, threshold):
        self.sections = list(sections)
        self.thr = threshold
        self.pos = 0
        self.heard = None                    # what the last decision heard: a section name or None
        self.cand, self.count = None, 0      # confirmation for "the next section started"
        self.resync_cand, self.resync_count = None, 0
        self.last_costs = {}

    # ---- position ----
    @property
    def current(self):
        return self.sections[self.pos] if self.sections else None

    @property
    def next(self):
        return self.sections[self.pos + 1] if self.pos + 1 < len(self.sections) else None

    def goto(self, pos):
        self.pos = max(0, min(int(pos), len(self.sections) - 1)) if self.sections else 0
        self.cand, self.count = None, 0
        self.resync_cand, self.resync_count = None, 0

    def step(self, delta):
        self.goto(self.pos + delta)

    def soft_reset(self):
        """A new audio stream: forget half-made decisions, keep the position."""
        self.heard, self.cand, self.count = None, None, 0
        self.resync_cand, self.resync_count = None, 0

    # ---- recognising ----
    def candidates(self):
        """The (at most two) section names this tracker chooses between."""
        return [n for n in dict.fromkeys([self.current, self.next]) if n is not None]

    def update(self, ranked, library_locked=None, resync=False):
        """ranked: [(cost, name), ...] for candidates() only, or None when there is not
        enough playing to decide. library_locked: the full-library answer, used to re-sync
        when resync is True. Returns the section whose preset should play now."""
        if not self.sections:
            return None
        if ranked is None:
            return self.current
        self.last_costs = {n: c for c, n in ranked}
        (c1, n1) = ranked[0]
        c2 = ranked[1][0] if len(ranked) > 1 else 9.0
        if c1 >= self.thr:
            heard = None
        elif len(ranked) == 1 or c2 - c1 >= MARGIN:
            heard = n1
        else:
            heard = self.current                    # too close to call: assume we haven't moved
        self.heard = heard

        nxt = self.next
        if nxt is not None and nxt != self.current and heard == nxt:
            self.count = self.count + 1 if self.cand == nxt else 1
            self.cand = nxt
            if self.count >= CONFIRM:
                self.goto(self.pos + 1)
                self.heard = self.current
        else:
            self.cand, self.count = None, 0

        if resync:
            self._resync(heard, library_locked)
        return self.current

    def _resync(self, heard, library_locked):
        """The setlist hears neither candidate but the full library is sure of another
        section of this song: after RESYNC_CONFIRM agreeing steps, jump to it."""
        target = library_locked
        if heard is not None or target is None or target in self.candidates() or target not in self.sections:
            self.resync_cand, self.resync_count = None, 0
            return
        self.resync_count = self.resync_count + 1 if self.resync_cand == target else 1
        self.resync_cand = target
        if self.resync_count >= RESYNC_CONFIRM:
            later = [i for i, n in enumerate(self.sections) if n == target and i > self.pos]
            self.goto(later[0] if later else self.sections.index(target))
            self.heard = self.current

    def state(self):
        return {"sections": self.sections, "pos": self.pos, "current": self.current, "next": self.next,
                "heard": self.heard, "costs": {n: round(c, 3) for n, c in self.last_costs.items()}}


def choose_section(drive, library_locked, setlist_state):
    """Which section's preset plays. setlist_state is SetlistTracker.state() or None (no song loaded)."""
    if drive in ("setlist", "both") and setlist_state is not None:
        return setlist_state["current"]
    return library_locked
