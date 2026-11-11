import errno
import hashlib
import io
import multiprocessing
import os
import re
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from . import Refusal, chunks, nbt, provision, region, world
from .ground import column_ground, column_top, cover_kinds, ground_ids, top_kinds, water_ids

SCAN_VERSION = 4
CHUNKS = region.SIDE
BLOCKS = CHUNKS * chunks.SIDE
ABSENT, OK, UNREADABLE = 0, 1, 2
OTHER_FORMAT = 3
FORMAT_CODE = {chunks.ENDLESSIDS: 1, chunks.NEID: 2}
FORMAT_NAME = {v: k for k, v in FORMAT_CODE.items()} | {OTHER_FORMAT: "other"}
PORTAL = "minecraft:portal"
MAX_WORKERS = 6
SPARE_CORES = 2
READ_ERRORS = (ValueError, KeyError, TypeError, IndexError, EOFError)
CHUNK_FIELDS = ("status", "format", "populated", "inhabited", "portals")
COLUMN_FIELDS = ("ground", "biome", "top", "water")
EMPTY = {"ground": np.nan, "biome": 0, "top": 0, "water": 0}
APP = "rwg2-migrator"
OWNER_KEY = re.compile(r"owner|placer|player", re.I)
GRAVE = "openblocks_grave"
XYZ = ("x", "y", "z")
GRAVE_FIELDS = (*XYZ, "stacks")


@dataclass(frozen=True)
class Rule:
    ok: np.ndarray
    kinds: np.ndarray
    cover: np.ndarray
    wet: np.ndarray
    portal: int
    digest: str


def rule_for(registry):
    ok, kinds, cover, wet = ground_ids(registry), top_kinds(registry), cover_kinds(registry), water_ids(registry)
    portal = next((i for i, n in registry.items() if n == PORTAL), -1)
    parts = (np.packbits(ok).tobytes(), kinds.tobytes(), cover.tobytes(), np.packbits(wet).tobytes(), str((portal, SCAN_VERSION)).encode())
    return Rule(ok, kinds, cover, wet, portal, hashlib.sha256(b"".join(parts)).hexdigest()[:16])


def cache_root(env=None):
    env = os.environ if env is None else env
    if env.get("RWG2_CACHE"):
        return env["RWG2_CACHE"]
    if os.name == "nt":
        return os.path.join(env.get("LOCALAPPDATA", os.path.expanduser("~")), APP, "cache")
    mac = os.path.expanduser("~/Library/Caches")
    if os.path.isdir(mac):
        return os.path.join(mac, APP)
    return os.path.join(env.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), APP)


def world_cache(world, root):
    return os.path.join(root, "scan", hashlib.sha1(os.path.realpath(world).encode()).hexdigest()[:16])


def _empty():
    return {
        "status": np.zeros((CHUNKS, CHUNKS), np.uint8),
        "format": np.zeros((CHUNKS, CHUNKS), np.uint8),
        "populated": np.zeros((CHUNKS, CHUNKS), bool),
        "inhabited": np.zeros((CHUNKS, CHUNKS), np.int64),
        "portals": np.zeros((CHUNKS, CHUNKS), np.int32),
        "ground": np.full((BLOCKS, BLOCKS), np.nan, np.float32),
        "biome": np.zeros((BLOCKS, BLOCKS), np.uint16),
        "top": np.zeros((BLOCKS, BLOCKS), np.uint8),
        "water": np.zeros((BLOCKS, BLOCKS), np.uint8),
    }


def _level(rec, pos):
    if rec is None:
        raise region.RegionError("bad region header entry")
    level = nbt.parse(region.inflate(rec))[1]["Level"][1]
    if (level["xPos"][1], level["zPos"][1]) != pos:
        raise region.RegionError("chunk stored at the wrong slot")
    return level


def _columns(ids, level, rule):
    if "HeightMap" not in level:
        return {"ground": None, "top": None, "water": None}
    hm = chunks.heightmap(level)
    top, water = column_top(ids, hm, rule)
    return {"ground": column_ground(ids, hm, rule.ok), "top": top, "water": water}


def owners(te):
    """The names in a tile entity's owner tags (GregTech mOwnerName and mOwnerUuid, Railcraft owner and ownerId, BuildCraft owner,
    Thaumcraft recipeplayer, ...): who placed or used it, a player or a mod's own name for world generation."""
    return {v[1].strip() for k, v in te.items() if v[0] == nbt.STRING and OWNER_KEY.search(k) and v[1].strip()}


def grave(te):
    """An OpenBlocks grave: the dead player's name, where it lies and how many stacks it holds."""
    items = te.get("Items", [nbt.LIST, [nbt.END, []]])[1][1]
    return {"name": te.get("perishedUsername", [nbt.STRING, "?"])[1], "stacks": len(items)} | {k: te[k][1] for k in XYZ}


def _tile_entities(level):
    found = level["TileEntities"][1][1] if "TileEntities" in level else []
    return {
        "tes": Counter(t.get("id", [0, "?"])[1] for t in found),
        "owners": sorted({(t.get("id", [0, "?"])[1], name) for t in found for name in owners(t)}),
        "graves": [grave(t) for t in found if t.get("id", [0, ""])[1] == GRAVE],
    }


def _chunk(level, rule):
    ids = chunks.block_ids(level)
    return (
        _columns(ids, level, rule)
        | _tile_entities(level)
        | {
            "format": FORMAT_CODE.get(chunks.chunk_format(level), OTHER_FORMAT),
            "populated": bool(level.get("TerrainPopulated", [0, 0])[1]),
            "inhabited": level.get("InhabitedTime", [0, 0])[1],
            "portals": int(np.count_nonzero(ids == rule.portal)) if rule.portal >= 0 else 0,
            "biome": chunks.biomes(level),
        }
    )


def _read(path):
    with open(path, "rb") as f:
        return region.records(f.read())


def _decoded(path, i, recs, rule):
    pos = region.slot_pos(*region.coords(path), i)
    try:
        return _chunk(_level(recs[0][i][0], pos), rule)
    except READ_ERRORS:
        if recs[1] is None:
            recs[1] = _read(path)
        try:
            return _chunk(_level(recs[1].get(i, (None, 0))[0], pos), rule)
        except READ_ERRORS:
            return None


def scan_region(path, rule):
    """Per-chunk fields and per-column ground and biome of one region file; a chunk that fails twice is UNREADABLE."""
    out, te, own, graves = _empty(), [], [], []
    recs = [_read(path), None]
    for i in recs[0]:
        lz, lx = divmod(i, CHUNKS)
        c = _decoded(path, i, recs, rule)
        out["status"][lz, lx] = UNREADABLE if c is None else OK
        if c is None:
            continue
        for k in CHUNK_FIELDS[1:]:
            out[k][lz, lx] = c[k]
        win = (slice(lz * chunks.SIDE, (lz + 1) * chunks.SIDE), slice(lx * chunks.SIDE, (lx + 1) * chunks.SIDE))
        for k in COLUMN_FIELDS:
            if c[k] is not None:
                out[k][win] = c[k]
        te += [(i, name, n) for name, n in c["tes"].items()]
        own += [(i, *o) for o in c["owners"]]
        graves += [(i, g) for g in c["graves"]]
    out["te_slot"] = np.array([t[0] for t in te], np.int16)
    out["te_name"] = np.array([t[1] for t in te], str)
    out["te_count"] = np.array([t[2] for t in te], np.int32)
    out["owner_slot"] = np.array([o[0] for o in own], np.int16)
    out["owner_type"] = np.array([o[1] for o in own], str)
    out["owner_name"] = np.array([o[2] for o in own], str)
    out["grave_slot"] = np.array([g[0] for g in graves], np.int16)
    out["grave_name"] = np.array([g[1]["name"] for g in graves], str)
    out["grave_xyzn"] = np.array([[g[1][k] for k in GRAVE_FIELDS] for g in graves], np.int32).reshape(-1, len(GRAVE_FIELDS))
    return out


def _key(path, rule):
    st = os.stat(path)
    return np.array([st.st_mtime_ns, st.st_size, SCAN_VERSION], np.int64), rule.digest


def _fresh(npz, path, rule):
    if not os.path.isfile(npz):
        return False
    key, digest = _key(path, rule)
    try:
        with np.load(npz) as z:
            return np.array_equal(z["key"], key) and str(z["rule"]) == digest
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def _save(npz, arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    region.write_atomic(npz, buf.getvalue())


def _scan_one(path, npz, rule):
    key, digest = _key(path, rule)
    _save(npz, scan_region(path, rule) | {"key": key, "rule": np.array(digest)})
    return path


@dataclass
class Scan:
    world: str
    regions: dict
    seconds: float
    read: int
    cached: int

    def chunk_table(self):
        """Present chunks as columns cx, cz, status, format, populated, inhabited, portals."""
        parts = []
        for (rx, rz), npz in sorted(self.regions.items()):
            with np.load(npz) as z:
                f = {k: z[k] for k in CHUNK_FIELDS}
            lz, lx = np.nonzero(f["status"] != ABSENT)
            parts.append({"cx": rx * CHUNKS + lx, "cz": rz * CHUNKS + lz} | {k: f[k][lz, lx] for k in CHUNK_FIELDS})
        keys = ("cx", "cz", *CHUNK_FIELDS)
        return {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0, np.int64) for k in keys}

    def tile_entities(self):
        """Tile entities as (cx, cz, id, count) rows."""
        rows = []
        for (rx, rz), npz in sorted(self.regions.items()):
            with np.load(npz) as z:
                slot, name, count = z["te_slot"], z["te_name"], z["te_count"]
            rows += [(*region.slot_pos(rx, rz, int(s)), str(n), int(c)) for s, n, c in zip(slot, name, count, strict=True)]
        return rows

    def owners(self):
        """Owner tags as (cx, cz, tile-entity type, name) rows."""
        rows = []
        for (rx, rz), npz in sorted(self.regions.items()):
            with np.load(npz) as z:
                found = zip(z["owner_slot"], z["owner_type"], z["owner_name"], strict=True)
            rows += [(*region.slot_pos(rx, rz, int(s)), str(t), str(name)) for s, t, name in found]
        return rows

    def graves(self):
        """OpenBlocks graves: {chunk, name, x, y, z, stacks}."""
        rows = []
        for (rx, rz), npz in sorted(self.regions.items()):
            with np.load(npz) as z:
                found = zip(z["grave_slot"], z["grave_name"], z["grave_xyzn"].tolist(), strict=True)
            for s, name, xyzn in found:
                rows.append({"chunk": region.slot_pos(rx, rz, int(s)), "name": str(name)} | dict(zip(GRAVE_FIELDS, xyzn, strict=True)))
        return rows

    def grid(self, field, box, bpp=1):
        """The column field over the block box (x0, z0, x1, z1), x1 and z1 excluded, every bpp-th column (the box a multiple of
        bpp, as region edges are); EMPTY where no chunk exists."""
        x0, z0, x1, z1 = box
        out = np.full(((z1 - z0) // bpp, (x1 - x0) // bpp), EMPTY[field], np.float32 if field == "ground" else np.uint16)
        for (rx, rz), npz in self.regions.items():
            bx, bz = rx * BLOCKS, rz * BLOCKS
            ax0, ax1, az0, az1 = max(x0, bx), min(x1, bx + BLOCKS), max(z0, bz), min(z1, bz + BLOCKS)
            if ax0 < ax1 and az0 < az1:
                with np.load(npz) as z:
                    part = z[field][az0 - bz : az1 - bz : bpp, ax0 - bx : ax1 - bx : bpp]
                out[(az0 - z0) // bpp : (az1 - z0) // bpp, (ax0 - x0) // bpp : (ax1 - x0) // bpp] = part
        return out

    def at(self, field, xz):
        """The column field at block columns xz (n, 2); EMPTY where no chunk exists."""
        xz = np.asarray(xz, np.int64).reshape(-1, 2)
        out = np.full(len(xz), EMPTY[field], np.float32 if field == "ground" else np.uint16)
        reg = xz // BLOCKS
        for key in {tuple(r) for r in reg.tolist()} & set(self.regions):
            sel = (reg[:, 0] == key[0]) & (reg[:, 1] == key[1])
            with np.load(self.regions[key]) as z:
                out[sel] = z[field][xz[sel, 1] - key[1] * BLOCKS, xz[sel, 0] - key[0] * BLOCKS]
        return out

    def summary(self):
        t = self.chunk_table()
        ok = t["status"] == OK
        formats = Counter(FORMAT_NAME.get(int(f), "none") for f in t["format"][ok])
        return {
            "chunks": int(len(t["cx"])),
            "formats": dict(sorted(formats.items())),
            "unpopulated": int(np.sum(ok & ~t["populated"])),
            "unreadable": int(np.sum(t["status"] == UNREADABLE)),
            "portal_blocks": int(t["portals"].sum()),
            "regions": len(self.regions),
            "regions_read": self.read,
            "regions_cached": self.cached,
            "seconds": round(self.seconds, 1),
        }


def _quiet(done, total):
    pass


def default_workers():
    return max(1, min((os.cpu_count() or 1) - SPARE_CORES, MAX_WORKERS))


def _paths(world_dir):
    return [p for p in world.files(os.path.join(world_dir, "region"), "r.*.*.mca") if region.NAME.fullmatch(os.path.basename(p))]


def _npz(out_dir, path):
    return os.path.join(out_dir, os.path.basename(path)[: -len(".mca")] + ".npz")


def adopt(world_dir, moved_from, cache):
    """Take the cached scans of world_dir's region files from the cache of moved_from, the folder they were renamed from (Apply
    moves the world aside as its backup), wherever they are still the same files (same size and time). The number taken."""
    rule = rule_for(world.registry(world.read_level(world_dir)))
    out_dir, src = (world_cache(w, provision.claim_cache(cache)) for w in (world_dir, moved_from))
    os.makedirs(out_dir, exist_ok=True)
    taken = 0
    for p in _paths(world_dir):
        npz, seed = _npz(out_dir, p), _npz(src, p)
        if _fresh(npz, p, rule) or not _fresh(seed, p, rule):
            continue
        try:
            os.link(seed, npz + ".tmp")
        except OSError:
            shutil.copy2(seed, npz + ".tmp")
        os.replace(npz + ".tmp", npz)
        taken += 1
    return taken


def _scan_all(todo, npz, rule, run):
    workers, progress = run
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers or default_workers(), mp_context=multiprocessing.get_context("spawn")) as pool:
        for n, f in enumerate(as_completed([pool.submit(_scan_one, p, npz[p], rule) for p in todo]), 1):
            f.result()
            (progress or _quiet)(n, len(todo))


def scan(world_dir, cache, workers=None, progress=None):
    """Scan every overworld region of world_dir, reusing cached regions whose file and block registry are unchanged."""
    t0 = time.monotonic()
    rule = rule_for(world.registry(world.read_level(world_dir)))
    out_dir = world_cache(world_dir, provision.claim_cache(cache))
    os.makedirs(out_dir, exist_ok=True)
    paths = _paths(world_dir)
    npz = {p: _npz(out_dir, p) for p in paths}
    todo = [p for p in paths if not _fresh(npz[p], p, rule)]
    try:
        _scan_all(todo, npz, rule, (workers, progress))
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        raise Refusal(
            f"the drive of the cache {cache} filled up while scanning",
            "Free some space on that drive, then run this again",
        ) from e
    regions = {region.coords(p): npz[p] for p in paths}
    return Scan(world_dir, regions, time.monotonic() - t0, len(todo), len(paths) - len(todo))
