"""The best-match search (POC rwg2_match.search, Smooth only): prefilter, coarse grid, refine, landmass, alternatives."""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from . import Refusal, __version__, appearance, area, score, verdicts
from .generator import RECORD
from .jobs import RUNNING, append_line, key_of, read_json, read_lines
from .region import write_atomic

CONFIG_FIELDS = ("bx", "bz", "lx", "lz")
BASELINE = (0, 0, 0, 0)
EDGE, MAIN, LAND = ("edge",), ("edge", "grid"), ("edge", "grid", "land")
CHUNK = 16
LMAP_BPP = 16
LMAP_ROUND = CHUNK * LMAP_BPP
CHUNK_BUDGET = 40_000
CACHE_FILE = "configs.jsonl"
STATE_FILE = "state.json"
INPUTS_FILE = "inputs.json"
CANDIDATES = ("candidates", "candidates.json")
TAINT_RADIUS = 2048
TILE = 8192
MID_BPP = 4
TOP_BIOMES = 4
PCT = 100
SIDE_NAMES = ("east", "north", "west", "south")
QUARTER = 90
EPS = 1e-9
MANY = 2
STAGES = (
    ("baseline", "Measuring the new terrain without any shift", 0.02),
    ("coarse-land", "Checking that the best places stay on land", 0.3),
    ("coarse-full", "Looking closer at the best places", 0.22),
    ("coarse", "Looking over the whole range", 0.1),
    ("frontier-coarse", "Looking for greener alternatives", 0.35),
    ("refine-biome", "Refining the best places", 0.5),
    ("landmass", "Moving the coastlines around them", 0.62),
    ("refine-landmass", "Fine-tuning the coastlines", 0.78),
    ("refine-joint", "Fine-tuning both together", 0.86),
    ("frontier", "Looking for greener alternatives", 0.93),
)


def stage_label(stage):
    """(plain label, rough share of the search done) of a search stage."""
    return next(((label, share) for prefix, label, share in STAGES if stage.startswith(prefix)), ("Searching", None))


@dataclass(frozen=True)
class Plan:
    """Search stages and their sizes; the defaults are the POC's."""

    b_range: int = 24576
    coarse_grid: int = 64
    coarse_max: int = 8000
    coarse_full: int = 600
    coarse_land: int = 300
    frontier: tuple = (1.5, 2, 3, 4, 6)
    frontier_check: int = 20
    top: int = 10
    sep: int = 1024
    refine_half: int = 2
    refine_land: int = 5
    marginal: float = 0.05
    b_steps: tuple = (64, 32, 16, 8)
    land_basins: int = 5
    l_range: int = 16384
    l_step: int = 2048
    landmass_full: int = 289
    land_seeds: int = 3
    l_steps: tuple = (512, 256, 128, 64, 32, 16, 8)
    picks: int = 3
    tolerance: float = 1.0


@dataclass(frozen=True)
class Walk:
    """One refine: which two offset axes move, by which steps, ranked how."""

    axes: tuple
    steps: tuple
    rank: object = None


class Cancelled(Exception):
    pass


def key(r):
    return tuple(int(r[f]) for f in CONFIG_FIELDS)


def shifted(r, axes, delta):
    c = dict(zip(CONFIG_FIELDS, key(r), strict=True))
    c[axes[0]] += delta[0]
    c[axes[1]] += delta[1]
    return key(c)


def feasible(r):
    return bool(r.get("feasible"))


def main_level(r):
    return r.get("level") == "main"


def distinct(results, n, sep, rank):
    """The best n by rank whose biome offsets are at least sep blocks apart (Chebyshev)."""
    picked = []
    for r in sorted(results, key=rank):
        if all(max(abs(r["bx"] - p["bx"]), abs(r["bz"] - p["bz"])) >= sep for p in picked):
            picked.append(r)
        if len(picked) == n:
            break
    return picked


def greenest(cap, scale):
    """Rank for the alternatives: smallest biome penalty among configs whose seam score is at most cap."""
    return lambda r: (score.penalty(score.incs(r), scale), r["biome"]) if r["seam"] <= cap else (math.inf, r["seam"])


def estimate(r, bands, scale):
    """Total of an edge-only record with the shares beyond the edge taken from bands (predicted, or a measured neighbour)."""
    return r["seam"] + score.penalty({score.EDGE: r["inc_edge"]} | {b: bands[b] for b in score.BAND_NAMES[1:]}, scale)


def chunk_count(points):
    return len(np.unique(np.asarray(points) // CHUNK, axis=0))


class ProbeSampler:
    """Samples and renders through a running generator (sampler boot)."""

    def __init__(self, gen, seed, threads):
        self.gen, self.seed, self.threads = gen, str(seed), int(threads)

    def sample(self, configs, points):
        r = self.gen.call("sample", seed=self.seed, continental=True, configs=[list(map(int, c)) for c in configs],
                          points=np.asarray(points).tolist(), threads=self.threads)  # fmt: skip
        return np.frombuffer(r["data"], RECORD).reshape(len(configs), len(points)), r["palette"]

    def render(self, config, window, bpp, continental=True):
        x0, z0, size = window
        r = self.gen.call("render", seed=self.seed, continental=continental, config=list(map(int, config)), x0=int(x0), z0=int(z0),
                          size=int(size), bpp=int(bpp), threads=self.threads)  # fmt: skip
        return np.frombuffer(r["data"], RECORD).reshape(r["res"], r["res"]), r["palette"]


class Lazy:
    """Starts the sampler only when a step really needs to sample."""

    def __init__(self, factory):
        self.factory, self.real = factory, None

    def __getattr__(self, name):
        if self.real is None:
            self.real = self.factory()
        return getattr(self.real, name)


def load_cache(path):
    """{config: record} from the append-only cache; a line torn by a crash is ignored (its batch was never cached)."""
    return {key(r): r for batch in read_lines(path) for r in batch}


class Evaluator:
    """Samples configs at the site's points; every evaluated config is appended to the cache and never sampled again."""

    def __init__(self, site, judge, sampler, folder):
        self.site, self.judge, self.sampler = site, judge, sampler
        self.job, self.plan = None, Plan()
        self.path = os.path.join(folder, CACHE_FILE)
        self.cache = load_cache(self.path)
        self.seen = {}
        self.sampled = 0
        self.started = time.monotonic()
        self.excellent_at = None
        self.best = None
        self.points = {"edge": site.edge, "grid": site.grid, "land": site.land}
        self.chunks = {k: chunk_count(v) for k, v in self.points.items()}

    def total(self, r):
        return score.total(r, self.plan.tolerance)

    def emit(self, event):
        if self.job:
            self.job.emit({"step": "search"} | event)

    def needs(self, c, need):
        """The parts (edge, grid, land) of need that this config has not been sampled for yet."""
        r = self.cache.get(c) or {}
        done = {"edge": "seam" in r, "grid": main_level(r), "land": "feasible" in r}
        return tuple(p for p in need if not done[p])

    def _evaluate(self, parts, rec, names, r):
        at = 0
        for part in parts:
            n = len(self.points[part])
            part_rec, at = rec[at : at + n], at + n
            if part == "edge":
                r = r | score.evaluate_edge(self.site, (part_rec, names), self.judge)
            elif part == "grid":
                r = r | score.evaluate_grid(self.site, (part_rec, names), self.judge, r)
            else:
                r = r | score.evaluate_land(part_rec)
        return r

    def _batch(self, todo, parts, stage):
        if self.job and self.job.cancel_requested():
            raise Cancelled
        recs, names = self.sampler.sample(todo, np.concatenate([self.points[p] for p in parts]))
        batch = []
        for c, rec in zip(todo, recs, strict=True):
            r = self._evaluate(parts, rec, names, self.cache.get(c, {"stage": stage, **dict(zip(CONFIG_FIELDS, c, strict=True))}))
            self.cache[c] = r
            batch.append(r)
        append_line(self.path, batch)
        self.sampled += len(batch)
        self.emit({"event": "batch", "stage": stage, "sampled": len(batch), "parts": list(parts), "cache": len(self.cache)})
        self._watch(batch)
        self._best()

    def _watch(self, batch):
        hits = [r for r in batch if main_level(r) and verdicts.excellent(r)]
        if self.excellent_at is None and hits:
            self.excellent_at = round(time.monotonic() - self.started, 1)
            best = min(hits, key=self.total)
            self.emit({
                "event": "excellent",
                "seconds": self.excellent_at,
                "config": list(key(best)),
                "total": round(self.total(best), 3),
                "job": self.job and self.job.id,
            })

    def _best(self):
        """A 'best' event whenever the top candidates so far (land-checked, at least plan.sep apart) change."""
        ok = [r for r in map(self.cache.get, self.seen) if r and main_level(r) and feasible(r)]
        top = distinct(ok, self.plan.picks, self.plan.sep, self.total)
        if [key(r) for r in top] != self.best:
            self.best = [key(r) for r in top]
            self.emit({"event": "best", "top": [self.brief(r) for r in top]})

    def brief(self, r):
        """What a card can say before the search ends: the border, the look at the edge, the numbers behind them."""
        out = {k: r[k] for k in ("median", "p90", "over10", "over30", "max", "seam", "same_edge", "below_sea")}
        lines = [verdicts.border(r)[1], verdicts.looks(r["same_edge"], self.judge.ref.name, None), verdicts.PENDING]
        return out | {"config": list(key(r)), "total": round(self.total(r), 4), "inc": score.incs(r), "lines": lines,
                      "excellent": verdicts.excellent(r)}  # fmt: skip

    def run(self, configs, stage, need=MAIN):
        """Sample what these configs still lack; only configs run so far count for later picks, so a resumed search, whose
        cache already holds what later stages sampled, takes the same path as an uninterrupted one."""
        configs = list(dict.fromkeys(tuple(int(v) for v in c) for c in configs))
        self.seen.update(dict.fromkeys(configs))
        groups = {}
        for c in configs:
            parts = self.needs(c, need)
            if parts:
                groups.setdefault(parts, []).append(c)
        t0 = time.monotonic()
        for parts, todo in groups.items():
            per = max(1, CHUNK_BUDGET // sum(self.chunks[p] for p in parts))
            for i in range(0, len(todo), per):
                self._batch(todo[i : i + per], parts, stage)
        done = {"+".join(k): len(v) for k, v in groups.items()}
        self.emit({"event": "stage", "stage": stage, "configs": len(configs), "sampled": done, "seconds": round(time.monotonic() - t0, 1)})
        return [self.cache[c] for c in configs]

    def screen(self, records, stage, top):
        best = sorted(records, key=self.total)[:top]
        return [r for r in self.run([key(r) for r in best], stage, LAND) if feasible(r)]

    def refine(self, seeds, stage, walk):
        """Move each seed over a (2 half + 1)^2 neighbourhood per step; stop a seed once a step improves its rank too little."""
        best, active, plan = list(seeds), set(range(len(seeds))), self.plan
        rank = walk.rank or self.total
        span = range(-plan.refine_half, plan.refine_half + 1)
        for s in walk.steps:
            if not active:
                break
            cands = {i: [shifted(best[i], walk.axes, (a * s, b * s)) for a in span for b in span] for i in sorted(active)}
            self.run([c for cs in cands.values() for c in cs], f"{stage}-{s}")
            check = [key(r) for cs in cands.values() for r in sorted((self.cache[c] for c in cs), key=rank)[: plan.refine_land]]
            self.run(check, f"{stage}-{s}-land", LAND)
            for i, cs in cands.items():
                new = min([self.cache[c] for c in cs if feasible(self.cache[c])] + [best[i]], key=rank)
                if s != walk.steps[0] and not _lead(rank(best[i])) - _lead(rank(new)) >= plan.marginal:
                    active.discard(i)
                best[i] = new
        return best

    def frontier(self, stage):
        picks = {}
        for cap in self.plan.frontier:
            rank = greenest(cap, self.plan.tolerance)
            ranked = sorted((r for r in map(self.cache.get, self.seen) if main_level(r) and math.isfinite(rank(r)[0])), key=rank)
            res = [r for r in self.run([key(r) for r in ranked[: self.plan.frontier_check]], f"{stage}-{cap:g}", LAND) if feasible(r)]
            picks[cap] = min(res, key=rank) if res else None
        return picks


def _lead(v):
    return v[0] if isinstance(v, tuple) else v


def centre_of(core):
    cs = np.array(sorted(core))
    return (cs.min(0) + cs.max(0) + 1) * CHUNK // 2


def lmap_window(core, plan):
    """A square chunk-aligned window centred on core, wide enough that the prefilter's correlation never wraps around."""
    cs = np.array(sorted(core))
    reach = plan.b_range + score.FAR + int((cs.max(0) - cs.min(0) + 2).max()) * CHUNK // 2
    half = -(-reach // LMAP_ROUND) * LMAP_ROUND
    x0, z0 = (centre_of(core) // LMAP_ROUND) * LMAP_ROUND - half
    return int(x0), int(z0), int(2 * half)


def taint_ids(names):
    return [i for i, n in enumerate(names) if n.casefold() == score.TAINT]


def prefilter(site, judge, lmap, plan):
    """Predicted look shares per ring for every biome offset on the coarse grid, by FFT correlation over the land map at offset 0.

    lmap = (palette indices per chunk, names, (x0, z0, size)); biome offset b shows at column p what offset 0 shows at p + b.
    """
    pal, names, (x0, z0, _) = lmap
    look, inc, _ = judge.lut(names)
    spectrum = np.fft.rfft2(np.where(judge.banned[look[pal]], 1.0, inc[pal]))
    r = plan.b_range // CHUNK
    origin = np.array([x0, z0]) // CHUNK
    main = site.main // CHUNK
    step = plan.coarse_grid // CHUNK
    shares = {}
    for band, idx in site.bands().items():
        k = np.zeros(pal.shape)
        np.add.at(k, (main[idx, 1] - origin[1], main[idx, 0] - origin[0]), 1.0 / len(idx))
        out = np.fft.irfft2(spectrum * np.conj(np.fft.rfft2(k)), s=pal.shape)
        shares[band] = np.roll(out, (r, r), (0, 1))[: 2 * r + 1 : step, : 2 * r + 1 : step]
    pen = score.EXCESS * sum(score.WEIGHTS[b] * np.maximum(0.0, shares[b] - score.TOLERANCE[b]) for b in score.BAND_NAMES)
    term = sum(score.WEIGHTS[b] * shares[b] for b in score.BAND_NAMES)
    order = np.lexsort((term.ravel(), pen.ravel()))[: plan.coarse_max]
    zi, xi = np.unravel_index(order, pen.shape)
    cands = [(int((x * step - r) * CHUNK), int((z * step - r) * CHUNK), 0, 0) for z, x in zip(zi, xi, strict=True)]
    pred = {c: {b: float(shares[b][z, x]) for b in score.BAND_NAMES} for c, z, x in zip(cands, zi, xi, strict=True)}
    return cands, pred, {"grid_offsets": int(pen.size), "picked": len(cands), "pred_penalty_zero": int((pen < EPS).sum())}


def run(ev, lmap):
    """All search stages; returns the state the candidates are drawn from. With a warm cache nothing is sampled twice."""
    plan = ev.plan
    cands, pred, summary = prefilter(ev.site, ev.judge, lmap, plan)
    ev.emit({"event": "prefilter"} | summary)
    ev.run([BASELINE], "baseline", LAND)
    coarse = ev.run(cands, "coarse", EDGE)
    by_est = sorted(coarse, key=lambda r: estimate(r, pred[key(r)], plan.tolerance))[: plan.coarse_full]
    for cap in plan.frontier:
        low = [r for r in coarse if r["seam"] <= cap]
        by_est += sorted(low, key=lambda r: estimate(r, pred[key(r)], plan.tolerance) - r["seam"])[: plan.frontier_check]
    full = ev.run([key(r) for r in by_est], "coarse-full")
    feas = ev.screen(full, "coarse-land", plan.coarse_land)
    front = ev.frontier("frontier-coarse")
    seeds = distinct(feas + [p for p in front.values() if p], plan.top + len(plan.frontier), plan.sep // 4, ev.total)
    best = ev.refine(seeds, "refine-biome", Walk(("bx", "bz"), plan.b_steps))
    lm_seeds = [s for b in distinct(best, plan.land_basins, plan.sep, ev.total) for s in landmass(ev, b)]
    lm_best = ev.refine(lm_seeds, "refine-landmass", Walk(("lx", "lz"), plan.l_steps))
    ev.refine(lm_best, "refine-joint", Walk(("bx", "bz"), plan.b_steps[-3:]))
    for cap, p in ev.frontier("frontier").items():
        if p:
            ev.refine([p], f"frontier-refine-{cap:g}", Walk(("bx", "bz"), plan.b_steps[-3:], greenest(cap, plan.tolerance)))
    front = ev.frontier("frontier-final")
    return {"prefilter": summary, "frontier": {f"{cap:g}": list(key(p)) if p else None for cap, p in front.items()}}


def landmass(ev, b):
    """Landmass offsets around one biome basin: the grid is sampled at the edge, then fully wherever it could still beat the basin.

    The landmass offset moves continents, so it changes the outer rings far more than the edge (on the fixture's best basin,
    135 of 288 offsets leave the edge exactly as it was); only a full sample can tell whether one improves the rings."""
    plan = ev.plan
    span = range(-plan.l_range, plan.l_range + 1, plan.l_step)
    recs = ev.run([(b["bx"], b["bz"], lx, lz) for lx in span for lz in span], "landmass", EDGE)
    cut = ev.total(b) - EPS

    def floor(r):
        return estimate(r, dict.fromkeys(score.BAND_NAMES, 0.0), plan.tolerance)

    better = sorted((r for r in recs if floor(r) < cut), key=floor)[: plan.landmass_full]
    full = [r for r in ev.run([key(r) for r in better], "landmass-full") if ev.total(r) < cut]
    res = [r for r in ev.run([key(r) for r in sorted(full, key=ev.total)], "landmass-land", LAND) if feasible(r)]
    return sorted(res, key=ev.total)[: plan.land_seeds]


def picks(cache, state, plan):
    """The top candidates (at least plan.sep apart) and the alternatives (best look at each seam cap), as config tuples."""
    ok = [r for r in cache.values() if main_level(r) and feasible(r)]
    top = [key(r) for r in distinct(ok, plan.picks, plan.sep, lambda r: score.total(r, plan.tolerance))]
    alts = [tuple(c) for c in (state.get("frontier") or {}).values() if c and tuple(c) not in top]
    return top, list(dict.fromkeys(alts))


def _ceil(a, b):
    return -(-a // b)


def find_windows(kept):
    """Square chunk-aligned render windows, at most TILE blocks wide, covering every part of kept grown by verdicts.REACH (parts
    whose grown boxes overlap share them): everything within that reach of a kept chunk is on some window."""
    unit = CHUNK * MID_BPP
    out = []
    for x0, z0, x1, z1 in area.windows(kept, verdicts.REACH):
        nx, nz = _ceil(x1 - x0, TILE), _ceil(z1 - z0, TILE)
        side = _ceil(max(_ceil(x1 - x0, nx), _ceil(z1 - z0, nz)), unit) * unit
        out += [(x0 + i * side, z0 + j * side, side) for i in range(nx) for j in range(nz)]
    return out


def _inside(points, windows):
    hit = np.zeros(len(points), bool)
    for x0, z0, size in windows:
        hit |= (points[:, 0] >= x0) & (points[:, 0] < x0 + size) & (points[:, 1] >= z0) & (points[:, 1] < z0 + size)
    return hit


def _nearest(points, core):
    d = area.distance(points, core) if len(points) else np.zeros(0)
    points, d = points[d <= verdicts.REACH], d[d <= verdicts.REACH]
    if not len(d):
        return None
    i = int(np.argmin(d))
    return {"blocks": round(float(d[i]), 1), "point": points[i].tolist(), "within_2km": int((d <= TAINT_RADIUS).sum())}


def found_near(renders, core):
    """Nearest Tainted Land and ocean within verdicts.REACH blocks of core over renders [(rec, names, window, bpp)]: blocks from
    core and the point; None when there is none that close (a farther one may lie off the renders)."""
    found, seen = {"taint": [], "ocean": []}, []
    for rec, names, window, bpp in renders:
        masks = {"taint": np.isin(rec["palette"], taint_ids(names)), "ocean": (rec["ocean"] == 1) | (rec["meta"] == 0)}
        for name, mask in masks.items():
            zz, xx = np.nonzero(mask)
            pts = np.c_[window[0] + xx * bpp, window[1] + zz * bpp]
            found[name].append(pts[~_inside(pts, seen)])
        seen.append(window)
    return {name: _nearest(np.concatenate(pts).reshape(-1, 2), core) for name, pts in found.items()}


def with_direction(found, core):
    if found is None:
        return None
    dx, dz = np.asarray(found["point"]) - area.nearest_point(found["point"], core)
    return found | {"direction": verdicts.compass(dx, dz)}


def off_side(points, look, same, centre):
    """The side (east/north/west/south of centre) with the most ground nearby that does not look like the base.

    points are the grid columns within NEAR blocks (one per chunk, so each stands for the same area); the edge itself is a
    one-block strip and would outweigh what a player sees around the base. centre is one point, or one per column (the centre
    of the part of the kept area it lies next to)."""
    off = ~same
    if not off.any():
        return None
    d = points - centre
    side = np.round(np.degrees(np.arctan2(-d[:, 1], d[:, 0])) / QUARTER).astype(int) % len(SIDE_NAMES)
    counts = np.bincount(side[off], minlength=len(SIDE_NAMES))
    worst = int(np.argmax(counts))
    kinds = np.bincount(look[off & (side == worst)], minlength=len(appearance.LOOKS))
    return {"side": SIDE_NAMES[worst], "look": appearance.LOOKS[int(np.argmax(kinds))], "columns": int(counts[worst])}


def part_centres(points, kept):
    """The centre of the map window (area.windows: a kept part grown by FAR, near parts merged) holding each column."""
    boxes = np.array(area.windows(kept))
    x, z = points[:, :1], points[:, 1:]
    inside = (x >= boxes[:, 0]) & (x < boxes[:, 2]) & (z >= boxes[:, 1]) & (z < boxes[:, 3])
    which = np.argmax(inside, axis=1)
    return (boxes[which, :2] + boxes[which, 2:]) // 2


def part_seams(site, top):
    """When kept has several parts: each part's chunks, centre block and border (seam stats of its own pairs), largest first, so a
    card can say which part fits badly."""
    groups = area.parts(set(map(tuple, site.kept.tolist())))
    if len(groups) < MANY:
        return []
    which = {c: i for i, part in enumerate(groups) for c in part}
    label = np.array([which[(x // CHUNK, z // CHUNK)] for x, z in site.inner.tolist()])
    out = []
    for i, part in enumerate(groups):
        mine = label == i
        if mine.any():
            centre = (np.array(sorted(part)).mean(0) * CHUNK + CHUNK // 2).round().astype(int).tolist()
            out.append({"chunks": len(part), "centre": centre} | score.seam(site.h_in[mine], top[mine]))
    return out


def top_biomes(rec, names):
    v, n = np.unique(rec["palette"], return_counts=True)
    return {names[int(v[i])]: round(PCT * float(n[i]) / float(n.sum()), 1) for i in np.argsort(-n)[:TOP_BIOMES]}


def no_candidates(job):
    """The refusal for a search job without candidates yet, with what to do in its state."""
    if job.state() == RUNNING:
        return Refusal(f"search job {job.id} has no candidates yet", "Wait for it to finish")
    return Refusal(f"search job {job.id} has no candidates yet ({job.state()})", f"Resume it: rwg2-migrator jobs resume {job.id}")


def describe(ev, configs, ref):
    """Everything a candidate card shows, for each config (resampled at the site, and rendered around every kept part)."""
    site, core, kept = ev.site, set(map(tuple, ev.site.core.tolist())), set(map(tuple, ev.site.kept.tolist()))
    recs, names = ev.sampler.sample(configs, site.main)
    windows = find_windows(kept)
    near = site.grid_d <= score.NEAR
    centres = part_centres(site.grid[near], kept)
    out = []
    for c, rec in zip(configs, recs, strict=True):
        r = ev.cache[c]
        edge = rec[: len(site.edge)]
        _, _, same = ev.judge.columns(edge, names)
        look_near, _, same_near = ev.judge.columns(rec[len(site.edge) :][near], names)
        found = found_near(((*ev.sampler.render(c, w, MID_BPP), w, MID_BPP) for w in windows), core)
        m = {k: r[k] for k in ("pairs", "median", "p90", "over10", "over30", "max", "seam", "below_sea", "feasible", "hard")}
        m |= {
            "config": list(c),
            "total": round(ev.total(r), 4),
            "penalty": round(score.penalty(score.incs(r), ev.plan.tolerance), 4),
            "inc": score.incs(r),
            "same_edge": round(float(same.mean()), 4),
            "off": off_side(site.grid[near], look_near, same_near, centres),
            "edge_biomes": top_biomes(edge, names),
            "base": ref.name,
            "ocean": found["ocean"]["blocks"] if found["ocean"] else None,
            "ocean_point": found["ocean"]["point"] if found["ocean"] else None,
            "taint": with_direction(found["taint"], core),
            "parts": part_seams(site, score.surface(edge["h"])[site.edge_idx]),
        }
        out.append(m)
    return out


def finalize(ev, state, ref):
    """The top candidates with their cards, and the alternatives."""
    top, alts = picks(ev.cache, state, ev.plan)
    described = describe(ev, top + alts, ref) if top or alts else []
    best = verdicts.recommended(described[: len(top)])
    for i, m in enumerate(described):
        m["card"] = verdicts.card(m, i + 1 if i < len(top) else 0, i == best and i < len(top))
    return described[: len(top)], described[len(top) :]


@dataclass
class Inputs:
    """What a search job works from: the world scan, the area, the preferences, the release, the seed, the sampler factory, the
    plan, and the key of the user's config folder the sampler boots on."""

    scan: object
    area: dict
    prefs: dict
    release: str
    seed: int
    sampler: object
    plan: Plan = field(default_factory=Plan)
    config: str | None = None


def scan_key(sc):
    """The scan's cached regions as (file, size, mtime): any rescan of the world changes it."""
    return key_of(sorted((os.path.basename(p), os.path.getsize(p), os.path.getmtime(p)) for p in sc.regions.values()))


def dump(path, value):
    write_atomic(path, (json.dumps(value, indent=1, default=float) + "\n").encode())


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _site_step(job, inp):
    core, kept = set(map(tuple, inp.area["core"])), set(map(tuple, inp.area["kept"]))
    keys = {"core": key_of(sorted(core)), "kept": key_of(sorted(kept)), "release": inp.release, "scan": scan_key(inp.scan)}
    keys["base"] = inp.prefs.get("base")
    folder = job.step_dir("site")
    if not job.step_valid("site", keys):
        score.build_site(lambda p: inp.scan.at("ground", p), core, kept).save(os.path.join(folder, "site.npz"))
        dump(os.path.join(folder, "reference.json"), asdict(appearance.reference(inp.scan, core, inp.release, inp.prefs.get("base"))))
        job.finish_step("site", ["site.npz", "reference.json"], keys)
    ref = appearance.Reference.from_dict(load(os.path.join(folder, "reference.json")))
    return score.Site.load(os.path.join(folder, "site.npz")), ref


def _lmap_step(job, inp, core, sampler):
    window = lmap_window(core, inp.plan)
    keys = {"seed": str(inp.seed), "window": list(window), "release": inp.release, "config": inp.config}
    path = os.path.join(job.step_dir("lmap"), "lmap.npz")
    if not job.step_valid("lmap", keys):
        t0 = time.monotonic()
        rec, names = sampler.render(BASELINE, window, LMAP_BPP, continental=False)
        np.savez_compressed(path, palette=rec["palette"].astype(np.uint16), names=np.array(names))
        job.finish_step("lmap", ["lmap.npz"], keys)
        job.emit({"step": "lmap", "event": "rendered", "window": list(window), "seconds": round(time.monotonic() - t0, 1)})
    with np.load(path) as z:
        return (z["palette"], [str(n) for n in z["names"]], window), keys


def own_cache(job, keys):
    """The cache holds samples of these inputs only. When the ground at the border, the base's look, the generator's inputs
    (seed, release, the user's config) or the settings changed since it was written, it is dropped and the search starts again."""
    folder = job.step_dir("search")
    path = os.path.join(folder, INPUTS_FILE)
    was = read_json(path)
    if was is not None and key_of(was) == key_of(keys):
        return
    outputs = [os.path.join(folder, CACHE_FILE), os.path.join(folder, STATE_FILE), os.path.join(job.folder, *CANDIDATES)]
    stale = [f for f in outputs if os.path.exists(f)]
    for f in stale:
        os.remove(f)
    job.drop_step("search")
    job.drop_step(CANDIDATES[0])
    if stale:
        changed = sorted(k for k in keys if not isinstance(was, dict) or key_of(was.get(k)) != key_of(keys[k]))
        job.emit({"step": "search", "event": "reset", "changed": changed})
    dump(path, keys)


def sampled_config(job):
    """The key of the config folder a search job's sampler booted on; False for a job that never recorded one."""
    keys = read_json(os.path.join(job.folder, "search", INPUTS_FILE)) or {}
    return keys.get("lmap", {}).get("config", False)


def _search_step(job, ev, lmap, keys):
    state_path = os.path.join(job.step_dir("search"), STATE_FILE)
    if job.step_valid("search", keys):
        return load(state_path)
    try:
        state = run(ev, lmap)
    except Cancelled:
        job.emit({"step": "search", "event": "cancelled", "configs": len(ev.cache)})
        return {"frontier": {}, "cancelled": True}
    dump(state_path, state)
    job.finish_step("search", [CACHE_FILE, STATE_FILE], keys)
    return state


def run_job(job, inp):
    """Steps site, lmap, search, candidates; each one is skipped when its manifest still matches its inputs."""
    t0 = time.monotonic()
    sampler = Lazy(inp.sampler)
    site, ref = _site_step(job, inp)
    bans = appearance.bans_for(ref, inp.prefs)
    judge = score.Judge(ref, bans, appearance.table(inp.release))
    lmap, lmap_keys = _lmap_step(job, inp, set(map(tuple, site.core.tolist())), sampler)
    plan = replace(inp.plan, tolerance=float(inp.prefs.get("tolerance") or 1.0))
    keys = {"site": site.key(), "reference": key_of(asdict(ref)), "lmap": lmap_keys, "prefs": inp.prefs, "plan": asdict(plan)}
    keys |= {"bans": sorted(bans), "version": __version__}
    own_cache(job, keys)
    ev = Evaluator(site, judge, sampler, job.step_dir("search"))
    ev.job, ev.plan = job, plan
    cached_before = len(ev.cache)
    state = _search_step(job, ev, lmap, keys)
    top, alts = finalize(ev, state, ref)
    summary = {
        "configs": len(ev.cache),
        "cached_before": cached_before,
        "sampled_now": ev.sampled,
        "seconds": round(time.monotonic() - t0, 1),
        "excellent_at": ev.excellent_at,
        "cancelled": bool(state.get("cancelled")),
    }
    result = {
        "reference": asdict(ref),
        "bans": sorted(bans),
        "prefs": inp.prefs,
        "search": summary,
        "candidates": top,
        "alternatives": alts,
    }
    dump(os.path.join(job.step_dir(CANDIDATES[0]), CANDIDATES[1]), result)
    job.finish_step(CANDIDATES[0], [CANDIDATES[1]], {"search": keys, "state": key_of(state)})
    job.emit({"step": "search", "event": "done"} | summary)
    return result
