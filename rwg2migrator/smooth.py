"""Border smoothing: a ramp across the kept border from the old ground to RWG2's, written into the margin band and the first
ring of staging. Core chunks are never touched [POC rwg2_smooth]."""

import hashlib
import os
import re
import struct
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import ISSUES, Refusal, area, blend, chunks, generator, jobs, nbt, region, transplant, world
from .ground import ID_SPACE, column_ground, ground_ids
from .region import write_atomic

REACH = 16
MIN_DIFF = 2.0
SPREAD = 8
PROFILE = 4
HEADROOM = 2
PASSES = 2
SEA_TOP = 62
LEAF_REACH = 4
FLOOD_STEPS = 96
TREE_HEADROOM = 48
NO_LEVEL = -1
HALO = 2
LAYOUT_CHUNKS = 2
PACKED_3D = 4
CHUNK_BYTES = 1_100_000
COLUMN_BYTES = 160
VOXEL_BYTES = 10
HEIGHT = chunks.HEIGHT
SIDE = chunks.SIDE
COLUMNS = SIDE * SIDE
OPAQUE, LEAF_OPACITY, FLUID_OPACITY = 255, 1, 3
FULL_SKY = 15
AIR, PLANT, FLUID, WATER, LEAVES, LOG, PLAIN, ORE, OTHER = range(9)
AIR_ID = 0
CARVABLE = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], bool)
FILLABLE = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0], bool)
PLANTS = np.arange(OTHER + 1) == PLANT
FLUIDS = (WATER, FLUID)
SOLID_FLUID = "ice"
PLANT_RE = re.compile(
    r"tallgrass|double_plant|flower|foliage|^plants$|bush|garden|lily|^(red|brown)_mushroom$|bluebells|sapling|"
    r"reeds|vine|fern|coral|lichen|crop|^carrots$|^potatoes$|^wheat$|^cactus$|snow_layer|longgrass|grass\.tall|"
    r"^web$|^pumpkin$|^melon_block$|^fire$",
    re.I,
)
FLUID_RE = re.compile(r"^(lava|flowing_lava|ice|blockOil|poison|blockFuel|honey)$")
LEAVES_RE = re.compile(r"leaves", re.I)
LOG_RE = re.compile(
    r"^(log2?|logs\d|blockRubWood|blockMagicalLog|tree|Rare Tree|Dark Tree|redwood|willow|bloodwood|tile\.TFLog|tile\.TFMagicLog)$"
)
PLAIN_RE = re.compile(
    r"^(stone|dirt|grass|sand|gravel|clay|sandstone|mycelium|hardened_clay|stained_hardened_clay|snow|packed_ice|cobblestone|"
    r"mossy_cobblestone|farmland|mud|bopGrass|newBopGrass|newBopDirt|hardDirt|driedDirt|hardSand|coarse_dirt|GrassBlock|"
    r"gt\.blockstones|gt\.blockgranites|deepslate|tuff|old_gravel|oldGravel|stoneFormations)$"
)
NOT_SURFACE_RE = re.compile(r"^(cobblestone|mossy_cobblestone|farmland|stoneFormations)$")
GRASSY_RE = re.compile(r"^(grass|bopGrass|newBopGrass|GrassBlock|mycelium)$")
ORE_RE = re.compile(r"ore", re.I)
NOT_ORE_RE = re.compile(r"core|corpo", re.I)
TRANSPARENT_RE = re.compile(
    r"torch|glass|fence|rail|pane|sign|ladder|lever|button|plate|carpet|flower_pot|skull|door|wire|repeater|comparator|web|bars|node|airy|lichen",
    re.I,
)
WATER_NAMES = ("minecraft:water", "minecraft:flowing_water")
LAVA_NAMES = ("lava", "flowing_lava")
BASIC = ("stone", "dirt", "grass", "water")
SMOOTH, RECORD = "smooth", "record.json"
_Y = np.arange(HEIGHT)[:, None, None]


@dataclass
class Kinds:
    cat: np.ndarray
    opac: np.ndarray
    surface: np.ndarray
    grassy: np.ndarray
    vine: np.ndarray
    liquid: np.ndarray
    ok: np.ndarray
    block: dict


def lookup(table, ids):
    """table[ids], where an id beyond the registry reads the table's last entry (OTHER, opaque)."""
    return table[np.minimum(ids, table.size - 1)]


def _category(name, ok):
    local = name.split(":", 1)[-1]
    rules = (
        (name in WATER_NAMES, WATER),
        (PLANT_RE.search(local), PLANT),
        (FLUID_RE.match(local), FLUID),
        (LEAVES_RE.search(local), LEAVES),
        (LOG_RE.match(local), LOG),
        (PLAIN_RE.match(local), PLAIN),
        (ok and ORE_RE.search(local) and not NOT_ORE_RE.search(local), ORE),
    )
    return next((cat for hit, cat in rules if hit), OTHER)


def _opacity(cat, local):
    if cat == PLANT or (cat == OTHER and TRANSPARENT_RE.search(local)) or local in LAVA_NAMES:
        return 0
    return {LEAVES: LEAF_OPACITY, WATER: FLUID_OPACITY, FLUID: FLUID_OPACITY}.get(cat, OPAQUE)


def classify(reg):
    """Per block id: smoothing category, light opacity, and whether it may form the surface, is grassy, or is a vine."""
    ok = ground_ids(reg)
    n = max(ID_SPACE, max(reg, default=0) + 1) + 1
    cat, opac = np.full(n, OTHER, np.uint8), np.full(n, OPAQUE, np.uint8)
    surface, grassy, vine, liquid = (np.zeros(n, bool) for _ in range(4))
    cat[AIR_ID], opac[AIR_ID] = AIR, 0
    for i, name in reg.items():
        if i == AIR_ID:
            continue
        local = name.split(":", 1)[-1]
        cat[i] = _category(name, ok[i])
        surface[i] = cat[i] == PLAIN and not NOT_SURFACE_RE.match(local)
        grassy[i] = bool(GRASSY_RE.match(local))
        vine[i] = cat[i] == PLANT and "vine" in local.lower()
        liquid[i] = cat[i] in FLUIDS and local != SOLID_FLUID
        opac[i] = _opacity(cat[i], local)
    named = {n: i for i, n in reg.items()}
    missing = [b for b in BASIC if "minecraft:" + b not in named]
    if missing:
        raise Refusal(
            f"the world's block registry has no minecraft:{missing[0]}", "Open the world once in GTNH 2.9, then run preview again"
        )
    return Kinds(cat, opac, surface, grassy, vine, liquid, np.append(ok, False), {b: named["minecraft:" + b] for b in BASIC})


def heightmap(ids, opac):
    o = lookup(opac, ids) > 0
    return np.where(o.any(0), HEIGHT - np.argmax(o[::-1], 0), 0).astype(np.int32)


def skylight(ids, opac):
    op = lookup(opac, ids).astype(np.int32)[::-1]
    hit = np.maximum.accumulate(op > 0, axis=0)
    light = FULL_SKY - np.cumsum(np.where(hit, np.maximum(op, 1), 0), axis=0)
    return np.clip(light, 0, FULL_SKY).astype(np.uint8)[::-1]


def smoothstep(t):
    return t * t * (3 - 2 * t)


def shift(a, d, ax, fill=False):
    out = np.full_like(a, fill)
    src, dst = [slice(None)] * a.ndim, [slice(None)] * a.ndim
    src[ax], dst[ax] = (slice(0, -d), slice(d, None)) if d > 0 else (slice(-d, None), slice(0, d))
    out[tuple(dst)] = a[tuple(src)]
    return out


class Layout:
    """Chunks packed side by side: a 2D field is (n, 16, 16) as (chunk, z, x) and a 3D one (y, n, 16, 16). A shift crosses into a
    chunk's four neighbours, so smoothing costs memory by the length of the band around kept, never by the part's bounding box.
    Logical axes are those of a dense field: (z, x), or (y, z, x)."""

    def __init__(self, cells):
        self.cells = sorted(cells)
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.nb = {
            (lat, d): np.array([self.index.get((x + d * dx, z + d * dz), -1) for x, z in self.cells], np.int64)
            for lat, (dx, dz) in enumerate(((0, 1), (1, 0)))
            for d in (1, -1)
        }

    @property
    def n(self):
        return len(self.cells)

    def ix(self, c):
        return self.index[c]

    def full(self, value, dtype):
        return np.full((self.n, SIDE, SIDE), value, dtype)

    def mask(self, cells):
        m = self.full(False, bool)
        m[[self.index[c] for c in cells if c in self.index]] = True
        return m

    def gather(self, fn):
        """A 2D field from fn(box) per chunk, box = (x0, z0, x1, z1) in blocks."""
        return np.stack([fn((x * SIDE, z * SIDE, (x + 1) * SIDE, (z + 1) * SIDE)) for x, z in self.cells])

    def shift(self, a, d, ax, fill=False):
        """a moved by d cells (|d| <= 16) along logical axis ax; cells with nothing behind them get fill."""
        if a.ndim == PACKED_3D and ax == 0:
            return shift(a, d, 0, fill)
        p = ax + 1
        out = shift(a, d, p, fill)
        src_nb = self.nb[(p - (a.ndim - 2), -1 if d > 0 else 1)]
        have = np.flatnonzero(src_nb >= 0)
        if have.size:
            dst, src = [slice(None)] * a.ndim, [slice(None)] * a.ndim
            dst[a.ndim - 3], src[a.ndim - 3] = have, src_nb[have]
            dst[p], src[p] = (slice(0, d), slice(SIDE - d, SIDE)) if d > 0 else (slice(SIDE + d, SIDE), slice(0, -d))
            out[tuple(dst)] = a[tuple(src)]
        return out

    def grow(self, m, full):
        """One step of growth: through all 26 (full) or the 6 face neighbours."""
        axes = range(m.ndim - 1)
        if full:
            for ax in axes:
                m = m | self.shift(m, 1, ax) | self.shift(m, -1, ax)
            return m
        out = m.copy()
        for ax in axes:
            out |= self.shift(m, 1, ax) | self.shift(m, -1, ax)
        return out

    def flood(self, seed, allowed, full, steps=FLOOD_STEPS):
        cur = seed & allowed
        n = np.count_nonzero(cur)
        for _ in range(steps):
            cur = self.grow(cur, full) & allowed
            m = np.count_nonzero(cur)
            if m == n:
                break
            n = m
        return cur

    def _box_sum(self, a, r, ax):
        out, lo, hi = a.copy(), a, a
        for _ in range(r):
            lo, hi = self.shift(lo, 1, ax, 0), self.shift(hi, -1, ax, 0)
            out += lo + hi
        return out

    def box_mean(self, a, w, r):
        """Mean of a over the cells where w holds in a (2r+1)^2 box around each cell; NaN where none."""
        s, n = np.where(w, a, 0.0), w.astype(float)
        for ax in (0, 1):
            s, n = self._box_sum(s, r, ax), self._box_sum(n, r, ax)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    def fill_nan(self, a, where, steps=2 * REACH):
        a = a.copy()
        for _ in range(steps):
            miss = where & np.isnan(a)
            if not miss.any():
                break
            a = np.where(miss, self.box_mean(a, np.isfinite(a), 1), a)
        return a

    def chebyshev(self, m, limit):
        d = np.full(m.shape, np.inf)
        d[m] = 0
        cur = m
        for k in range(1, limit + 1):
            nxt = self.grow(cur, True)
            d[nxt & ~cur] = k
            cur = nxt
        return d

    def near(self, m):
        return np.any([self.shift(m, d, ax, False) for ax in (0, 1) for d in (1, -1)], 0)


@dataclass
class Fields:
    """The 2D inputs of the plan on one layout: ground now, old and new ground, kept and zone masks, top of standing water."""

    cur: np.ndarray
    old: np.ndarray
    new: np.ndarray
    kept: np.ndarray
    zone: np.ndarray
    wtop: np.ndarray
    layout: Layout


def _ramp(f):
    """The target ground before water rules: a smoothstep across the kept border, as wide as the local height difference."""
    lay = f.layout
    u = np.where(f.kept, -(lay.chebyshev(~f.kept, REACH + 1) - 0.5), lay.chebyshev(f.kept, REACH + 1) - 0.5)
    delta = f.new - f.old
    peak = np.where(f.zone & np.isfinite(delta), np.abs(delta), 0)
    for _ in range(SPREAD):
        peak = np.maximum.reduce([peak] + [lay.shift(peak, d, ax, 0) for ax in (0, 1) for d in (1, -1)])
    spread = np.nan_to_num(lay.box_mean(peak, f.zone, SPREAD))
    h = np.where(spread >= MIN_DIFF, np.minimum(spread, REACH), 0)
    w = np.where(h > 0, smoothstep(np.clip((u + h) / np.maximum(2 * h, 1e-9), 0, 1)), (u > 0).astype(float))
    t = np.where(f.zone, f.old + w * delta, f.cur)
    alpha = np.where(f.zone & (h > 0), 4 * w * (1 - w), 0)
    for _ in range(PASSES):
        t = np.where(alpha > 0, t + alpha * (lay.box_mean(t, np.isfinite(t), 1) - t), t)
    return {"u": u, "h": h, "w": w, "delta": delta, "t": np.rint(t)}


def _sea_fill(f, change, target):
    """Columns lowered to sea level or below get sea water, unless they touch dry low ground that would leave a water wall."""
    wtop = f.wtop
    existing = np.where(change, np.where(wtop >= target, wtop, NO_LEVEL), wtop)
    ground = np.where(np.isfinite(target), target, np.inf)
    fill = change & (wtop == NO_LEVEL) & (target <= SEA_TOP)
    while True:
        dry_low = (ground <= SEA_TOP) & (np.where(fill, SEA_TOP, existing) == NO_LEVEL)
        keep = fill & ~f.layout.near(dry_low)
        if np.array_equal(keep, fill):
            return fill, np.where(fill, SEA_TOP, existing)
        fill = keep


def plan(f):
    """Per column: whether its ground changes, its target height and the water level above it, as in the POC."""
    r = _ramp(f)
    target, lake = r["t"], f.wtop > SEA_TOP
    change = f.zone & (r["h"] > 0) & np.isfinite(f.cur) & np.isfinite(target) & (target != f.cur) & ~(lake & (target <= f.wtop))
    target = np.where(change, target, f.cur)
    fill, wet = _sea_fill(f, change, target)
    high = np.max([f.layout.shift(wet, d, ax, NO_LEVEL) for ax in (0, 1) for d in (1, -1)], 0)
    shore = change & (wet == NO_LEVEL) & (high > NO_LEVEL) & (high >= target)
    target = np.where(shore, high + 1, target)
    change &= target != f.cur
    counts = {
        "sea_filled_cols": int((fill & change).sum()),
        "lake_cols_kept": int((f.zone & lake & ~change).sum()),
        "lake_cols_filled": int((f.zone & lake & change).sum()),
        "shore_cols": int(shore.sum()),
    }
    return r | {"target": np.where(change, target, f.cur), "change": change, "level": np.where(change, wet, NO_LEVEL)} | counts


def profile(col, g, k):
    """The top PROFILE surface blocks under the old ground g, as (id, meta); grass over dirt when there are none."""
    ids, meta = col
    out = []
    for y in range(g - 1, max(g - 1 - 2 * PROFILE, 0), -1):
        if lookup(k.surface, ids[y]):
            out.append((int(ids[y]), int(meta[y])))
        if len(out) == PROFILE:
            break
    if not out:
        out = [(k.block["grass"], 0)] + [(k.block["dirt"], 0)] * (PROFILE - 1)
    return out + [out[-1]] * (PROFILE - len(out))


def _put(col, y, block):
    col[0][y], col[1][y] = block


def _fill(col, span, prof, k):
    """Fill the fillable blocks of col over span = (y0, t) with stone, then the surface profile on top."""
    y0, t = span
    for y in range(y0, t):
        if FILLABLE[lookup(k.cat, col[0][y])]:
            _put(col, y, (k.block["stone"], 0) if y < t - PROFILE else prof[t - 1 - y])


def _clear(col, span, k, which):
    """Blocks of col over span = (y0, y1, water level) whose category passes which become water up to the level, air above."""
    y0, y1, level = span
    for y in range(y0, y1):
        if which[lookup(k.cat, col[0][y])]:
            _put(col, y, (k.block["water"], 0) if y <= level else (0, 0))


def rebuild(col, heights, k):
    """Move one column's ground from g to t, with the old surface blocks on top, water up to level; non-terrain blocks stay."""
    g, t, level = heights
    prof = profile(col, g, k)
    if level >= t and lookup(k.grassy, prof[0][0]):
        prof[0] = prof[1] if not lookup(k.grassy, prof[1][0]) else (k.block["dirt"], 0)
    if t > g:
        _fill(col, (max(g - PROFILE, 1), t), prof, k)
    else:
        _clear(col, (t, min(g + HEADROOM + 1, HEIGHT), level), k, CARVABLE)
        _fill(col, (max(t - PROFILE, 1), t), prof, k)
    _clear(col, (t, min(max(g, t) + HEADROOM + 1, HEIGHT), level), k, PLANTS)
    for y in range(t, min(level + 1, HEIGHT)):
        if col[0][y] == 0:
            _put(col, y, (k.block["water"], 0))


class Chunk:
    """One decoded chunk: ids and meta as (y, z, x), sky and block light, its HeightMap; re-encoded in its own format."""

    def __init__(self, pos, rec, ts):
        self.pos, self.ts = pos, ts
        raw = region.inflate(rec)
        self.name, self.root = nbt.parse(raw)
        if nbt.write(self.name, self.root) != raw:
            raise Refusal(f"chunk {pos[0]},{pos[1]} does not re-encode byte for byte", f"Report this chunk position at {ISSUES}")
        self.lvl = self.root["Level"][1]
        if (self.lvl["xPos"][1], self.lvl["zPos"][1]) != tuple(pos):
            raise Refusal(
                f"chunk {pos[0]},{pos[1]} is stored at the wrong slot", "Restore its region file from a backup, then run preview again"
            )
        self.fmt = chunks.chunk_format(self.lvl)
        self.secs = {s["Y"][1]: s for s in chunks.sections(self.lvl)}
        self.ids = np.zeros((HEIGHT, SIDE, SIDE), np.uint32)
        self.meta = np.zeros((HEIGHT, SIDE, SIDE), np.uint16)
        self.sky = np.full((HEIGHT, SIDE, SIDE), FULL_SKY, np.uint8)
        self.light = np.zeros((HEIGHT, SIDE, SIDE), np.uint8)
        for y, sec in self.secs.items():
            v, sl = chunks.decode_section(sec), slice(y * chunks.SECTION, (y + 1) * chunks.SECTION)
            for arr, part in zip((self.ids, self.meta, self.sky, self.light), (v.ids, v.meta, v.sky, v.light), strict=True):
                arr[sl] = part.reshape(chunks.SECTION, SIDE, SIDE)
        self.hm = chunks.heightmap(self.lvl).copy()
        got = chunks.biomes(self.lvl)
        self.biomes = None if got is None else got.copy()
        self.orig = None

    def snapshot(self):
        self.orig = (self.ids.copy(), self.meta.copy())

    def modified(self):
        return (self.ids != self.orig[0]) | (self.meta != self.orig[1])

    def biomes_changed(self):
        return self.biomes is not None and not np.array_equal(self.biomes, chunks.biomes(self.lvl))

    def _sections(self):
        out = []
        for y in range(HEIGHT // chunks.SECTION):
            sl = slice(y * chunks.SECTION, (y + 1) * chunks.SECTION)
            if y not in self.secs and not self.ids[sl].any():
                continue
            v = chunks.Voxels(*(a[sl].reshape(-1) for a in (self.ids, self.meta, self.sky, self.light)))
            out.append(chunks.encode_section(self.fmt, y, v, like=self.secs.get(y, ())))
        return out

    def encode(self, k, cols):
        """(record, voxels changed, removed tile entities, ticks and entities); cols = (changed columns, lowest changed y)."""
        if self.fmt is None:
            raise Refusal(f"chunk {self.pos[0]},{self.pos[1]} mixes chunk formats", f"Report this chunk position at {ISSUES}")
        mod = self.modified()
        colmod = mod.any(0)
        if self.biomes_changed():
            self.lvl[chunks.BIOME_KEY[self.fmt]] = [nbt.BYTE_ARRAY, chunks.encode_biomes(self.fmt, self.biomes)]
        if colmod.any():
            self.light[mod] = 0
            hm = np.where(colmod, heightmap(self.ids, k.opac), self.hm)
            ylo = np.where(colmod, np.minimum(cols[1], hm - 1), HEIGHT)
            self.sky = np.where(ylo[None] <= _Y, skylight(self.ids, k.opac), self.sky)
            self.lvl["HeightMap"][1] = hm.astype(">i4").tobytes()
            self.lvl["LightPopulated"] = [nbt.BYTE, 0]
            self.lvl["Sections"] = [nbt.LIST, [nbt.COMPOUND, self._sections()]]
        dropped = {key: self._drop(key, mod, cols) for key in _lists(self.lvl)}
        return region.deflate(nbt.write(self.name, self.root)), int(mod.sum()), dropped

    def _drop(self, key, mod, cols):
        items = self.lvl[key][1][1]
        keep = [e for e in items if not self._hit(key, e, mod, cols)]
        self.lvl[key][1][1] = keep
        return len(items) - len(keep)

    def _local(self, x, z):
        return int(np.floor(x)) - self.pos[0] * SIDE, int(np.floor(z)) - self.pos[1] * SIDE

    def _hit(self, key, e, mod, cols):
        if key != "Entities":
            (lx, lz), y = self._local(e["x"][1], e["z"][1]), e["y"][1]
            return 0 <= y < HEIGHT and 0 <= lx < SIDE and 0 <= lz < SIDE and bool(mod[y, lz, lx])
        x, y, z = (struct.unpack(">d", v)[0] for v in e["Pos"][1][1])
        (lx, lz), ly = self._local(x, z), int(np.floor(y))
        if not (0 <= lx < SIDE and 0 <= lz < SIDE):
            return False
        if 0 <= ly < HEIGHT and mod[ly, lz, lx]:
            return True
        return bool(cols[0][lz, lx]) and ly >= cols[1][lz, lx] - 1


def _lists(level):
    return [k for k in ("TileEntities", "TileTicks", "Entities") if k in level and level[k][1][0] == nbt.COMPOUND]


def load_chunks(world_dir, cells):
    """{pos: Chunk} of the cells present in world_dir's overworld region files."""
    out, wanted = {}, {(x // region.SIDE, z // region.SIDE) for x, z in cells}
    for path in transplant.region_files(world_dir):
        rx, rz = region.coords(path)
        if (rx, rz) not in wanted:
            continue
        with open(path, "rb") as f:
            recs = region.records(f.read())
        for i, (rec, ts) in recs.items():
            pos = region.slot_pos(rx, rz, i)
            if pos in cells and rec is not None:
                out[pos] = Chunk(pos, rec, ts)
    return out


def water_top(ch, ground, k):
    w = (lookup(k.cat, ch.ids) == WATER) & (ch.meta == 0) & (np.nan_to_num(ground, nan=HEIGHT)[None] <= _Y)
    return np.where(w.any(0), HEIGHT - 1 - np.argmax(w[::-1], 0), NO_LEVEL)


@dataclass
class Scene:
    """One connected part of the smoothing zone: its chunks, layout, the y range the 3D fields cover, and the block kinds."""

    chunks: dict
    layout: Layout
    k: Kinds
    ys: tuple = (0, HEIGHT)
    sets: dict = field(default_factory=dict)

    def dense(self, fn, before=False):
        y0, y1 = self.ys
        out = np.zeros((y1 - y0, self.layout.n, SIDE, SIDE), bool)
        for c, ch in self.chunks.items():
            ids = ch.orig[0] if before and ch.orig is not None else ch.ids
            out[:, self.layout.ix(c)] = fn(ids[y0:y1])
        return out

    def of(self, *cats, before=False):
        return self.dense(lambda a: np.isin(lookup(self.k.cat, a), cats), before)


def trees(s, mod_cols, change, low):
    """Logs standing in changed columns, the leaves no remaining log holds, and the vines hanging from either. Each 3D field
    is dropped once used, as they set the tool's peak memory."""
    lay = s.layout
    log1 = s.of(LOG)
    seeds = log1 & change[None] & (np.arange(*s.ys)[:, None, None, None] >= (low - 1)[None])
    logs = lay.flood(seeds, log1, True)
    del seeds
    logs &= ~lay.flood(log1 & ~mod_cols[None], log1, True)
    log1 &= ~logs
    leaf1 = s.of(LEAVES)
    held = lay.flood(log1, log1 | leaf1, False, LEAF_REACH)
    del log1
    orphan = leaf1 & ~held & (s.sets.pop("leaves_held_before") | change[None]) & mod_cols[None]
    del leaf1, held
    vine1 = s.dense(lambda a: lookup(s.k.vine, a))
    vines = lay.flood(lay.grow(logs | orphan, False) & vine1, vine1, False) & mod_cols[None]
    return logs, orphan, vines


def fluid_walls(s):
    """Air that was solid before, next to a source of water, lava or another fluid: a wall of fluid the game would let flow."""
    src = s.dense(lambda a: lookup(s.k.liquid, a)) & _sources(s)
    walls = s.layout.grow(src, False) & ~src
    del src
    walls &= s.dense(lambda a: a == 0)
    walls &= s.dense(lambda a: lookup(s.k.opac, a) == OPAQUE, before=True)
    return int(walls.sum())


def _sources(s):
    y0, y1 = s.ys
    out = np.zeros((y1 - y0, s.layout.n, SIDE, SIDE), bool)
    for c, ch in s.chunks.items():
        out[:, s.layout.ix(c)] = ch.meta[y0:y1] == 0
    return out


def _fields(s, cells, ground):
    """Fields of the part: the old ground is the snapshot's (kept and first ring), the new one RWG2's under kept, the generated
    terrain outside; cells = (kept, ring, mod)."""
    lay, k = s.layout, s.k
    cur, wtop = lay.full(np.nan, np.float32), lay.full(NO_LEVEL, np.int32)
    for c, ch in s.chunks.items():
        i = lay.ix(c)
        cur[i] = column_ground(ch.ids, ch.hm, k.ok)
        wtop[i] = water_top(ch, cur[i], k)
    kept_m, ring_m = lay.mask(cells[0]), lay.mask(cells[1])
    zone = lay.mask(cells[2]) & np.isfinite(cur)
    old_in = lay.gather(lambda box: ground[0].grid("ground", box))
    old = np.where(kept_m, cur, np.where(ring_m, old_in, np.nan))
    new = np.where(kept_m, lay.gather(ground[1].grid), cur)
    missing = int(np.sum(ring_m & np.isnan(old)))
    return Fields(cur, lay.fill_nan(old, zone), lay.fill_nan(new, zone), kept_m, zone, wtop, lay), missing


def _carve(s, p, mod):
    """Rebuild every changed column; returns the non-terrain voxels kept in place, by id."""
    preserved = Counter()
    cur, target, level = s.sets["cur"], p["target"], p["level"]
    for c in sorted(mod):
        ch, i = s.chunks[c], s.layout.ix(c)
        for lz, lx in zip(*np.nonzero(p["change"][i]), strict=True):
            g, t = int(cur[i, lz, lx]), int(target[i, lz, lx])
            rebuild((ch.ids[:, lz, lx], ch.meta[:, lz, lx]), (g, t, int(level[i, lz, lx])), s.k)
            lo, hi = max(min(g, t) - PROFILE, 1), min(max(g, t) + HEADROOM + 1, HEIGHT)
            col = ch.ids[lo:hi, lz, lx]
            preserved.update(col[lookup(s.k.cat, col) == OTHER].tolist())
    return preserved


def _cut_trees(s, p, mod, low):
    """Remove the trees of the changed columns; returns the counts and the log voxels removed per chunk."""
    logs, orphan, vines = trees(s, s.layout.mask(mod), p["change"], low)
    per_chunk = {c: int(logs[:, s.layout.ix(c)].sum()) for c in mod}
    counts = {
        "tree_log_voxels_removed": int(logs.sum()),
        "orphan_leaf_voxels_removed": int(orphan.sum()),
        "vine_voxels_removed": int(vines.sum()),
    }
    logs |= orphan
    logs |= vines
    del orphan, vines
    y0, y1 = s.ys
    for c in mod:
        gone, ch = logs[:, s.layout.ix(c)], s.chunks[c]
        ch.ids[y0:y1][gone], ch.meta[y0:y1][gone] = 0, 0
    return counts, per_chunk


def _ys(f, target):
    lo = np.nanmin(np.where(f.zone, np.fmin(f.cur, target), np.inf))
    hi = np.nanmax(np.where(f.zone, np.fmax(f.cur, target), -np.inf))
    if not np.isfinite(lo):
        return 1, 1
    return max(int(lo) - PROFILE - 2, 1), min(int(hi) + TREE_HEADROOM, HEIGHT)


def _encode(s, p, mod, low):
    out, counts, dropped = {}, Counter(), Counter()
    for c in sorted(mod):
        ch = s.chunks[c]
        m = ch.modified()
        if not m.any() and not ch.biomes_changed():
            continue
        i = s.layout.ix(c)
        low_c = np.where(p["change"][i], low[i], np.argmax(m, 0)).astype(np.int32) - 1
        rec, n, drop = ch.encode(s.k, (p["change"][i], low_c))
        jobs.checkpoint(SMOOTH)
        out[c] = (rec, ch.ts)
        counts[c] = n
        dropped.update(drop)
    return out, counts, dropped


class LowMemory(Refusal):
    """Too little free memory to smooth; the caller names the job to resume."""


def _require_ram(need):
    """Smoothing runs while the user's server may run on this machine: refuse rather than let the system run out of memory."""
    free, want = generator.free_ram(), need + generator.RAM_SPARE
    if free < want:
        raise LowMemory(
            f"Smoothing the border needs {want / generator.GB:.1f} GB free; {max(free, 0) / generator.GB:.1f} GB is free now",
            "Close GTNH (or other large programs), then continue the preview",
        )


@dataclass
class Inputs:
    """What smoothing reads besides the cells: staging, the block kinds, the old ground (the snapshot), the candidate's render,
    and the grass colours when the biomes are blended (None: no blend)."""

    stage: str
    k: Kinds
    old: object
    render: object
    colours: object = None


def _biome_field(s):
    field = s.layout.full(blend.NO_BIOME, np.int64)
    for c, ch in s.chunks.items():
        if ch.biomes is not None:
            field[s.layout.ix(c)] = ch.biomes
    return field


def _xz(lay):
    span = np.arange(SIDE)
    x = lay.gather(lambda b: np.broadcast_to(b[0] + span, (SIDE, SIDE)))
    z = lay.gather(lambda b: np.broadcast_to((b[1] + span)[:, None], (SIDE, SIDE)))
    return x, z


def _sampler(lay, fields):
    """at(xz) -> the values of each packed field at global block columns (NaN or -1 off the layout)."""

    def at(xz):
        xz = np.asarray(xz, np.int64).reshape(-1, 2)
        i = np.array([lay.index.get((x // SIDE, z // SIDE), -1) for x, z in xz.tolist()], np.int64)
        lz, lx, ok = xz[:, 1] % SIDE, xz[:, 0] % SIDE, i >= 0
        return [np.where(ok, f[np.maximum(i, 0), lz, lx], np.nan if f.dtype.kind == "f" else -1) for f in fields]

    return at


def _blend(s, cells, inputs, ground):
    """Blend the band's biomes into RWG2's; cells = (core, kept, ring, mod), ground = the ground after smoothing."""
    core, kept, ring, mod = cells
    lay = s.layout
    old = _biome_field(s)
    names = lay.gather(inputs.render.biome_grid)
    masks = (lay.mask(kept), lay.mask(core), lay.mask(set(s.chunks) - kept))
    take, new, unmapped = blend.blended(old, (lay, masks, names, _xz(lay), blend.depth(core, kept)))
    changed = take & (new != old)
    for c in sorted(kept - core):
        if c in s.chunks and s.chunks[c].biomes is not None:
            s.chunks[c].biomes = new[lay.ix(c)].astype(np.uint16)
    blend.refuse_bytes({c: (s.chunks[c].fmt, s.chunks[c].biomes) for c in kept - core if c in s.chunks}, inputs.colours.names)
    zone = lay.mask(mod)
    steps = [blend.border_steps(kept, _sampler(lay, (f, ground)), inputs.colours) for f in (old, new)]
    zones = [blend.zone_steps(lay, f, ground, (zone, inputs.colours)) for f in (old, new)]
    return {
        "blend_columns_changed": int(changed.sum()),
        "blend_chunks_changed": int(sum(bool(changed[lay.ix(c)].any()) for c in kept - core)),
        "blend_unmapped_columns": unmapped,
        "blend_border_pairs": steps[0][0],
        "blend_border_over_before": steps[0][1],
        "blend_border_over_after": steps[1][1],
        "blend_zone_pairs": zones[0][0],
        "blend_zone_over_before": zones[0][1],
        "blend_zone_over_after": zones[1][1],
    }


def _strips(s, p, mod, removed):
    """{pos: (columns reshaped, log voxels removed)} of the chunks smoothing reshaped or took logs from: where the plant step
    replants."""
    out = {c: (p["change"][s.layout.ix(c)].copy(), removed.get(c, 0)) for c in sorted(mod)}
    return {c: v for c, v in out.items() if v[0].any() or v[1]}


def _part(cells, inputs):
    """Smooth one connected part on a layout of the band and 2 chunks around it, so memory follows the length of the border.
    cells = (core, kept) of the part."""
    core, kept = cells
    ring = area.dilate(kept, 1) - kept
    mod = (kept - core) | ring
    halo = area.dilate(mod, 1) - mod
    lay = Layout(area.dilate(mod, LAYOUT_CHUNKS))
    _require_ram(len(mod | halo) * CHUNK_BYTES + lay.n * COLUMNS * COLUMN_BYTES)
    loaded = load_chunks(inputs.stage, mod | halo)
    missing = mod - set(loaded)
    if missing:
        raise Refusal(
            f"{jobs.count(len(missing), 'chunk')} of the smoothing band {'is' if len(missing) == 1 else 'are'} not in the working copy, "
            f"the first {min(missing)}",
            "Run preview again; the new terrain around the kept area is generated first",
        )
    s = Scene(loaded, lay, inputs.k)
    f, legacy_missing = _fields(s, (kept, ring, mod), (inputs.old, inputs.render))
    p = plan(f)
    s.sets["cur"] = f.cur
    low = np.where(p["change"], np.minimum(f.cur, p["target"]) - PROFILE, HEIGHT).astype(np.int32)
    s.ys = _ys(f, p["target"])
    _require_ram(lay.n * COLUMNS * (s.ys[1] - s.ys[0]) * VOXEL_BYTES)
    s.sets["leaves_held_before"] = s.layout.flood(s.of(LOG), s.of(LOG, LEAVES), False, LEAF_REACH) & s.of(LEAVES)
    for c in mod:
        s.chunks[c].snapshot()
    preserved = _carve(s, p, mod)
    removed, per_chunk = _cut_trees(s, p, mod, low)
    jobs.checkpoint(SMOOTH)
    walls = fluid_walls(s)
    blended = _blend(s, (core, kept, ring, mod), inputs, p["target"]) if inputs.colours else {}
    out, counts, dropped = _encode(s, p, mod, low)
    change, target, cur = p["change"], p["target"], f.cur
    stats = (
        {
            "chunks_loaded": len(loaded),
            "layout_chunks": lay.n,
            "band_chunks": len(kept - core),
            "ring_chunks": len(ring),
            "legacy_missing_ring_columns": legacy_missing,
            "columns_planned": int(change.sum()),
            "columns_raised": int((change & (target > cur)).sum()),
            "columns_lowered": int((change & (target < cur)).sum()),
            "max_abs_change": int(np.nanmax(np.abs(target - cur)[change])) if change.any() else 0,
            "water_filled_columns": int((change & (p["level"] >= target)).sum()),
            "new_fluid_walls": walls,
            "voxels_changed": int(sum(counts.values())),
        }
        | {k_: p[k_] for k_ in ("sea_filled_cols", "lake_cols_kept", "lake_cols_filled", "shore_cols")}
        | removed
        | blended
    )
    before = {c: f.cur[s.layout.ix(c)].copy() for c in s.chunks}
    return out, stats, (dropped, preserved, before, _strips(s, p, mod, per_chunk))


def write(stage, out, step=SMOOTH):
    """Put the rewritten chunks into staging's region files (their timestamps kept); returns {rel: sha256}."""
    by_region, changed = {}, {}
    for pos, rec in out.items():
        by_region.setdefault((pos[0] // region.SIDE, pos[1] // region.SIDE), {})[region.slot_index(*pos)] = rec
    for (rx, rz), new in sorted(by_region.items()):
        name = f"r.{rx}.{rz}.mca"
        path = os.path.join(stage, "region", name)
        with open(path, "rb") as f:
            slots = region.records(f.read())
        slots |= new
        data = region.build(slots)
        problem = region.check(data, slots)
        if problem:
            raise AssertionError(f"region/{name}: {problem}")
        write_atomic(path, data)
        with open(path, "rb") as f:
            if f.read() != data:
                raise AssertionError(f"region/{name} does not read back as written")
        changed[f"region/{name}"] = hashlib.sha256(data).hexdigest()
        jobs.checkpoint(step)
    return changed


def run(stage, a, ground, colours=None):
    """Smooth staging in place; ground = (snapshot ground with .grid(field, box), render with .grid(box) and .biome_grid(box)),
    colours = blend.Colours to blend the band's biomes too. Returns the record, whose 'changed' maps each region file it wrote to
    its sha256, {(cx, cz): 16x16 ground before smoothing} and {(cx, cz): (columns reshaped, log voxels removed)}."""
    core, kept = ({tuple(c) for c in a[key]} for key in ("core", "kept"))
    if not area.dilate(core, 1) <= kept:
        raise Refusal(
            f"{jobs.count(len(area.dilate(core, 1) - kept), 'protected chunk')} touch the new terrain, so smoothing has no band to work in",
            "Make the area again with a margin of at least 1 chunk",
        )
    reg = world.registry(world.read_level(stage))
    inputs = Inputs(stage, classify(reg), *ground, colours)
    out, stats, removed, preserved, before, strips = {}, Counter(), Counter(), Counter(), {}, {}
    for part in area.parts(area.dilate(kept, HALO)):
        got, st, (drop, kept_ids, ground_before, part_strips) = _part((core & part, kept & part), inputs)
        out |= got
        before |= ground_before
        strips |= part_strips
        stats.update({key: v for key, v in st.items() if key != "max_abs_change"})
        stats["max_abs_change"] = max(stats["max_abs_change"], st["max_abs_change"])
        removed.update(drop)
        preserved.update(kept_ids)
    changed = write(stage, out)
    if set(out) & core:
        raise AssertionError(f"smoothing changed protected chunks, the first {min(set(out) & core)}")
    record = dict(stats) | {
        "params": {
            "reach": REACH,
            "min_diff": MIN_DIFF,
            "spread": SPREAD,
            "profile": PROFILE,
            "headroom": HEADROOM,
            "passes": PASSES,
            "sea_top": SEA_TOP,
            "leaf_reach": LEAF_REACH,
            "blend": colours is not None,
        },  # fmt: skip
        "chunks_changed": sorted(out),
        "chunks_changed_in_core": len(set(out) & core),
        "removed": dict(removed),
        "preserved_non_terrain_voxels": {reg.get(b, str(b)): n for b, n in preserved.most_common()},
        "changed": changed,
    }
    return record, before, strips
