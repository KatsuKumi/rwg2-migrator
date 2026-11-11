"""How well one RWG2 config fits the kept area: the seam at the core border, the look by ring, the land rule (POC Evaluator)."""

import hashlib
from dataclasses import dataclass, fields

import numpy as np

from . import appearance, area

CHUNK = 16
SEA = 63
ZONE = 256
NEAR = 128
FAR = 512
GRID_FULL = 256
FAR_STRIDE = 2
LAND_SAMPLES = (4, 12)
EDGE = "edge"
BANDS = (("1-32", 0, 32, 4.0), ("33-128", 32, 128, 2.0), ("129-256", 128, 256, 1.0), ("257-512", 256, 512, 0.5))
BAND_NAMES = (EDGE, *(b[0] for b in BANDS))
WEIGHTS = {EDGE: 4.0, **{b[0]: b[3] for b in BANDS}}
TOLERANCE = {EDGE: 0.15, "1-32": 0.15, "33-128": 0.15, "129-256": 0.25, "257-512": 0.35}
EXCESS = 4.0
STEPS = ((10, 10.0), (30, 30.0))
P90 = 90
BELOW_SEA_MAX = 0.02
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
TAINT = "tainted land"
WATER = appearance.LOOKS.index("water")


def surface(h):
    """RWG2 fills stone up to (int) h, so the top of the ground is floor(h) + 1."""
    return np.floor(np.asarray(h, np.float64)) + 1


@dataclass
class Site:
    """Where a config is sampled: edge pairs across the core border, a chunk grid by ring, land samples near kept."""

    h_in: np.ndarray
    inner: np.ndarray
    edge: np.ndarray
    edge_idx: np.ndarray
    grid: np.ndarray
    grid_d: np.ndarray
    land: np.ndarray
    core: np.ndarray
    kept: np.ndarray

    @property
    def main(self):
        return np.r_[self.edge, self.grid]

    def bands(self):
        n = len(self.edge)
        out = {EDGE: np.arange(n)}
        for name, lo, hi, _ in BANDS:
            out[name] = n + np.flatnonzero((self.grid_d > lo) & (self.grid_d <= hi))
        return out

    def near(self):
        return np.r_[np.arange(len(self.edge)), len(self.edge) + np.flatnonzero(self.grid_d <= NEAR)]

    def key(self):
        """A hash of what the site holds: two sites with the same key sample and score every config alike."""
        h = hashlib.sha256()
        for f in fields(self):
            a = np.ascontiguousarray(getattr(self, f.name))
            h.update(f"{f.name} {a.dtype} {a.shape}".encode() + a.tobytes())
        return h.hexdigest()[:16]

    def save(self, path):
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(**{f.name: z[f.name] for f in fields(cls)})


def edge_pairs(core):
    """(inner core column, outer column) for every column step that leaves core, as two (n, 2) arrays."""
    inner, outer = [], []
    side = np.arange(CHUNK)
    for cx, cz in sorted(core):
        for dx, dz in DIRS:
            if (cx + dx, cz + dz) in core:
                continue
            fixed = CHUNK - 1 if dx + dz > 0 else 0
            xs = np.full(CHUNK, fixed) if dx else side
            zs = np.full(CHUNK, fixed) if dz else side
            cols = np.c_[cx * CHUNK + xs, cz * CHUNK + zs]
            inner.append(cols)
            outer.append(cols + (dx, dz))
    return np.concatenate(inner), np.concatenate(outer)


def grid_points(core):
    """Chunk origins outside core: every chunk within GRID_FULL blocks, every FAR_STRIDE-th chunk out to FAR; with distances."""
    cs = np.array(sorted(core))
    lo, hi = cs.min(0) - FAR // CHUNK - 1, cs.max(0) + FAR // CHUNK + 2
    gx, gz = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
    c = np.c_[gx.ravel(), gz.ravel()]
    c = c[np.array([tuple(v) not in core for v in c.tolist()], bool)]
    d = area.distance(c * CHUNK, core)
    keep = (d <= FAR) & ((d <= GRID_FULL) | ((c[:, 0] % FAR_STRIDE == 0) & (c[:, 1] % FAR_STRIDE == 0)))
    return c[keep] * CHUNK, d[keep]


def land_points(kept, ground_at):
    """LAND_SAMPLES x LAND_SAMPLES columns per chunk within ZONE blocks of kept where the real ground is at sea level or above."""
    zone = np.array(sorted(area.near(kept, ZONE)))
    local = np.array([(a, b) for a in LAND_SAMPLES for b in LAND_SAMPLES])
    pts = (zone[:, None, :] * CHUNK + local[None]).reshape(-1, 2)
    return pts[ground_at(pts) >= SEA]


def build_site(ground_at, core, kept):
    """ground_at(points) gives the real ground height (NaN where unknown) at block columns (n, 2)."""
    core, kept = set(map(tuple, core)), set(map(tuple, kept))
    inner, outer = edge_pairs(core)
    h_in = ground_at(inner)
    good = np.isfinite(h_in)
    edge, idx = np.unique(outer[good], axis=0, return_inverse=True)
    grid, grid_d = grid_points(core)
    return Site(
        h_in[good].astype(np.float64),
        inner[good],
        edge,
        idx.reshape(-1),
        grid,
        grid_d,
        land_points(kept, ground_at),
        np.array(sorted(core)),
        np.array(sorted(kept)),
    )


def seam(h_in, top):
    """Steps between the real ground inside and RWG2's top of ground outside: median, p90, shares over 10 and 30, score."""
    step = np.abs(h_in - top)
    r = {"pairs": int(len(step)), "median": float(np.median(step)), "p90": float(np.percentile(step, P90)), "max": float(step.max())}
    for t, _ in STEPS:
        r[f"over{t}"] = float(np.mean(step > t))
    r["seam"] = r["median"] + sum(w * r[f"over{t}"] for t, w in STEPS)
    return r


def penalty(inc, scale=1.0):
    return EXCESS * sum(WEIGHTS[b] * max(0.0, inc[b] - TOLERANCE[b] * scale) for b in BAND_NAMES)


def incs(r, prefix="inc_"):
    return {b: r[prefix + b] for b in BAND_NAMES}


def total(r, scale=1.0):
    """seam score + biome penalty; an edge-only record uses its estimated shares beyond the edge."""
    inc = incs(r) if r["level"] == "main" else {EDGE: r["inc_edge"]} | {b: r["est_" + b] for b in BAND_NAMES[1:]}
    return r["seam"] + penalty(inc, scale)


class Judge:
    """Per sampler palette (a list of biome names): look index, incompatibility and 'same look as the base' arrays."""

    def __init__(self, ref, bans, tab):
        self.ref, self.bans, self.tab = ref, bans, tab
        self.banned = np.array([look in bans for look in appearance.LOOKS])
        self.memo = {}

    def lut(self, names):
        key = tuple(names)
        if key not in self.memo:
            looks = [appearance.judge(n, self.ref, self.bans, self.tab) for n in names]
            self.memo[key] = (
                np.array([appearance.LOOKS.index(x.look) for x in looks], np.int8),
                np.array([x.incompat for x in looks]),
                np.array([x.same for x in looks]),
            )
        return self.memo[key]

    def columns(self, rec, names):
        """(look index, incompatibility, same look) per sampled column; ocean flag or meta 0 makes it water."""
        look, inc, same = self.lut(names)
        water = (rec["ocean"] == 1) | (rec["meta"] == 0)
        c = np.where(water, WATER, look[rec["palette"]])
        return c, np.where(self.banned[c], 1.0, inc[rec["palette"]]), same[rec["palette"]] & ~water


def evaluate_edge(site, sample, judge):
    """Seam and edge look from the sample (records, palette names) of the edge points."""
    rec, names = sample
    r = seam(site.h_in, surface(rec["h"])[site.edge_idx])
    c, inc, same = judge.columns(rec, names)
    looks = {"inc_edge": round(float(inc.mean()), 4), "same_edge": round(float(same.mean()), 4), "hard_edge": int(judge.banned[c].sum())}
    return r | {"level": "edge"} | looks


def evaluate_grid(site, sample, judge, edge):
    """The rings beyond the edge from the sample of the grid points; edge is the config's edge result."""
    c, inc, _ = judge.columns(*sample)
    n = len(site.edge)
    shares = {EDGE: edge["inc_edge"]} | {b: float(inc[idx - n].mean()) for b, idx in site.bands().items() if b != EDGE}
    near = site.near()
    return {
        "level": "main",
        **{"inc_" + b: round(v, 4) for b, v in shares.items() if b != EDGE},
        "biome": round(sum(WEIGHTS[b] * v for b, v in shares.items()), 4),
        "hard": edge["hard_edge"] + int(judge.banned[c[near[near >= n] - n]].sum()),
    }


def evaluate_land(rec):
    """The land rule over site.land records: no ocean and at most BELOW_SEA_MAX of the real land below sea level."""
    if not len(rec):
        return {"ocean": 0.0, "below_sea": 0.0, "feasible": True}
    ocean = float(np.mean((rec["ocean"] == 1) | (rec["meta"] == 0)))
    below = float(np.mean(surface(rec["h"]) < SEA))
    return {"ocean": ocean, "below_sea": round(below, 6), "feasible": ocean == 0 and below <= BELOW_SEA_MAX}
