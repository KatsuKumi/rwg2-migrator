"""Tree replanting on the strips smoothing reshaped: spots with a log budget per chunk, a plant boot on a stripped
copy of staging that grows each spot's biome tree, and a voxel merge of only the new log and leaf voxels, only into air. The
merge list is kept in the job, so a refresh replays it without a boot while the spots, the ground under them and the terrain
outside kept around them hold."""

import hashlib
import io
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import Refusal, area, blend, chunks, generator, genworld, jobs, provision, region, smooth, transplant, world
from .ground import column_ground, ground_ids
from .region import rmtree, write_atomic

PLANT, TREES = "plant", "trees"
MERGE_FILE, RECORD, STRIPS = "merge.npz", "record.json", "strips.npz"
CLEARANCE = 8
SPACING = 4
PICK_OCTAVE, ORDER_OCTAVE = 7, 8
SIDE = chunks.SIDE
GROWN = (smooth.LOG, smooth.LEAVES)
REPLY = ("spots", "attempts", "trees", "logs", "missingChunks", "unknownBiome")
PHASES = ("build", "boot", "plant", "save_stop", "diff")


def save_strips(path, strips):
    """strips = {(cx, cz): (16x16 columns reshaped, log voxels removed)}, as smoothing returns them."""
    pos = sorted(strips)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        pos=np.array(pos, np.int64).reshape(-1, 2),
        change=np.array([strips[p][0] for p in pos], bool).reshape(-1, SIDE, SIDE),
        logs=np.array([strips[p][1] for p in pos], np.int64),
    )
    write_atomic(path, buf.getvalue())


def load_strips(path):
    with np.load(path) as z:
        return {tuple(p): (c, int(n)) for p, c, n in zip(z["pos"].tolist(), z["change"], z["logs"], strict=True)}


def _records(world_dir, rxz):
    path = os.path.join(world_dir, "region", f"r.{rxz[0]}.{rxz[1]}.mca")
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as f:
        return region.records(f.read())


def _level(recs, pos):
    rec = recs.get(region.slot_index(*pos), (None, 0))[0]
    got = transplant.decode(rec, pos) if rec is not None else None
    return (rec, got[1]["Level"][1]) if got is not None else (None, None)


def by_region(cells):
    """{(rx, rz): sorted cells in that region file}."""
    out = {}
    for c in sorted(cells):
        out.setdefault((c[0] // region.SIDE, c[1] // region.SIDE), []).append(c)
    return out


def levels(world_dir, cells):
    """(pos, record, Level compound) of the cells present in world_dir's overworld region files, one region file at a time."""
    for rxz, mine in sorted(by_region(cells).items()):
        recs = _records(world_dir, rxz)
        for pos in mine:
            rec, level = _level(recs, pos)
            if level is not None:
                yield pos, rec, level


def clearance(pos, core):
    """Blocks (Chebyshev) from each column of chunk pos to the nearest core column, capped above CLEARANCE."""
    span = np.arange(SIDE)
    x, z = np.meshgrid(pos[0] * SIDE + span, pos[1] * SIDE + span)
    out = np.full((SIDE, SIDE), CLEARANCE + 1)
    for cx, cz in area.dilate({pos}, 1) & core:
        dx = np.maximum(np.maximum(cx * SIDE - x, x - (cx * SIDE + SIDE - 1)), 0)
        dz = np.maximum(np.maximum(cz * SIDE - z, z - (cz * SIDE + SIDE - 1)), 0)
        out = np.minimum(out, np.maximum(dx, dz))
    return out


def candidates(pos, ok):
    """At most one column of ok per SPACING x SPACING cell of the chunk, in a fixed pseudo-random order, as (x, z) blocks."""
    span = np.arange(SIDE)
    x, z = np.meshgrid(pos[0] * SIDE + span, pos[1] * SIDE + span)
    rank = np.where(ok, blend._lattice(x, z, PICK_OCTAVE), -1.0)
    picked = []
    for i in range(0, SIDE, SPACING):
        for j in range(0, SIDE, SPACING):
            cell = rank[i : i + SPACING, j : j + SPACING]
            if cell.max() >= 0:
                lz, lx = np.unravel_index(np.argmax(cell), cell.shape)
                picked.append((int(x[i + lz, j + lx]), int(z[i + lz, j + lx])))
    order = blend._lattice(np.array([c[0] for c in picked], np.int64), np.array([c[1] for c in picked], np.int64), ORDER_OCTAVE)
    return [picked[i] for i in np.argsort(order, kind="stable")]


@dataclass
class Plan:
    """What the plant boot is asked: spots [x, z, biome, group], a log budget per group, and each group's chunk; plus the ground
    height under each spot, which the boot is not sent."""

    spots: list
    budgets: list
    groups: list
    ground: list = field(default_factory=list)

    def key(self):
        """The spots, their chunks and the ground each tree stands on. The budgets are left out: they count the logs smoothing
        removed, the player's own trees in the margin among them, so cutting one there between preview and Apply must not plant
        everything again."""
        return jobs.key_of([self.spots, self.groups, self.ground])

    def reach(self, core):
        """The chunks the trees of these spots can write into, core left out (it is never changed)."""
        return area.dilate({tuple(g) for g in self.groups}, 1) - core


def _columns(level, ok):
    """(biome, ground height) per column of a chunk, or None without biomes."""
    b = chunks.biomes(level)
    return None if b is None else (b, column_ground(chunks.block_ids(level), chunks.heightmap(level), ok))


def plan(stage, a, strips):
    """Spots in the reshaped columns of every chunk that lost logs, at least CLEARANCE blocks from core; each chunk is a group
    whose budget is the log voxels smoothing removed there, and each spot carries its column's biome and ground in staging."""
    core = {tuple(c) for c in a["core"]}
    wanted = {p for p, (change, logs) in strips.items() if logs > 0 and change.any()}
    ok = ground_ids(world.registry(world.read_level(stage)))
    found = {pos: _columns(level, ok) for pos, _, level in levels(stage, wanted)}
    p = Plan([], [], [])
    for pos in sorted(wanted):
        change, logs = strips[pos]
        picked = candidates(pos, change & (clearance(pos, core) >= CLEARANCE)) if found.get(pos) is not None else []
        if not picked:
            continue
        biomes, ground = found[pos]
        cols = [(z - pos[1] * SIDE, x - pos[0] * SIDE) for x, z in picked]
        p.spots += [[x, z, int(biomes[c]), len(p.budgets)] for (x, z), c in zip(picked, cols, strict=True)]
        p.ground += [float(ground[c]) for c in cols]
        p.budgets.append(logs)
        p.groups.append(list(pos))
    return p


def key(stage, a, p, settings):
    """What the merge list depends on: the spots with the ground under them, the release and probe, and the blocks and biomes of
    every chunk outside kept the trees can write into, as staging holds them before the merge. Kept chunks count only through the
    spots: play re-saves them (InhabitedTime, entities) and changes their blocks, and the replay writes only where staging still
    holds air."""
    h = hashlib.sha256()
    for pos, _, level in levels(stage, p.reach({tuple(c) for c in a["kept"]})):
        h.update(json.dumps(pos).encode() + _terrain(level))
    return {"spots": p.key(), "terrain": h.hexdigest(), "release": (settings or {}).get("release"), "probe": generator.PROBE_VERSION}


def _terrain(level):
    ids, meta = _voxels(level)
    b = chunks.biomes(level)
    return ids.tobytes() + meta.tobytes() + (b.tobytes() if b is not None else b"")


def _voxels(level):
    out = [np.zeros((chunks.HEIGHT, SIDE, SIDE), np.uint32) for _ in range(2)]
    for sec in chunks.sections(level):
        v, y = chunks.decode_section(sec), sec["Y"][1] * chunks.SECTION
        out[0][y : y + chunks.SECTION] = v.ids.reshape(chunks.SECTION, SIDE, SIDE)
        out[1][y : y + chunks.SECTION] = v.meta.reshape(chunks.SECTION, SIDE, SIDE)
    return out


def diff(stage, plant_dir, cells, k):
    """The merge list: per chunk of cells, the voxels the plant boot made log or leaves where staging holds air."""
    out = {}
    for rxz, mine in sorted(by_region(cells).items()):
        before, after = _records(stage, rxz), _records(plant_dir, rxz)
        for pos in mine:
            (_, was), (_, now) = _level(before, pos), _level(after, pos)
            jobs.checkpoint(PLANT)
            if was is None or now is None:
                continue
            ids, meta = _voxels(now)
            grown = np.isin(smooth.lookup(k.cat, ids), GROWN) & (_voxels(was)[0] == smooth.AIR_ID)
            idx = np.flatnonzero(grown)
            if idx.size:
                out[pos] = (idx, ids.reshape(-1)[idx], meta.reshape(-1)[idx])
    return out


MERGE_ARRAYS = ("idx", "ids", "meta")


def save_merge(path, merge):
    pos = sorted(merge)
    arrays = {name: np.concatenate([merge[p][i] for p in pos] or [np.zeros(0)]).astype(np.uint32) for i, name in enumerate(MERGE_ARRAYS)}
    buf = io.BytesIO()
    np.savez_compressed(
        buf, pos=np.array(pos, np.int64).reshape(-1, 2), counts=np.array([merge[p][0].size for p in pos], np.int64), **arrays
    )
    write_atomic(path, buf.getvalue())


def load_merge(path):
    with np.load(path) as z:
        ends = np.cumsum(z["counts"])
        parts = [np.split(z[name], ends[:-1]) for name in MERGE_ARRAYS]
        return {tuple(p): tuple(part[i] for part in parts) for i, p in enumerate(z["pos"].tolist())}


def boot(job, gens, a, p):
    """Build the plant world from staging (stripped, every chunk marked decorated so the boot decorates nothing), boot on it, plant,
    save and stop; the merge list counts only after the JVM exited with code 0. Returns the boot's record."""
    ctx, settings = gens.context()
    gens.stop()
    generator.stop_running(gens.cache, job.emit)
    stage = job.data["args"]["staging"]
    dest = genworld.folder(gens.cache, settings["release"], job.id, genworld.PLANT_PREFIX)
    t = [time.monotonic()]
    built = genworld.build(stage, dest, {tuple(c) for c in a["kept"]}, populated=True)
    jobs.checkpoint(PLANT)
    t.append(time.monotonic())
    gen = genworld.boot(gens, os.path.basename(dest), job.data["args"]["config"], built["map_features"])
    memory = generator.PeakMemory(gen.record["pid"])
    t.append(time.monotonic())
    reply = gen.call("plant", spots=p.spots, budgets=p.budgets)
    t.append(time.monotonic())
    gen.call("save")
    code = gens.stop()
    memory.stop()
    t.append(time.monotonic())
    if code != 0:
        raise Refusal(
            f"the generator exited with code {code} after planting the trees, so the planted chunks may be half saved",
            f"Run it again: rwg2-migrator jobs resume {job.id} (the trees are planted again from the start)",
        )
    genworld.registry_check(stage, dest)
    k = smooth.classify(world.registry(world.read_level(stage)))
    merge = diff(stage, dest, p.reach({tuple(c) for c in a["core"]}), k)
    rmtree(dest)
    t.append(time.monotonic())
    seconds = {name: round(t[i + 1] - t[i], 1) for i, name in enumerate(PHASES)}
    return merge, _booted(p, ctx.release, reply) | {"seconds": seconds, "peak_working_set_mb": round(memory.peak / provision.MB)}


def _booted(p, release, reply):
    return {"release": release, "groups": len(p.budgets), "budget": int(sum(p.budgets))} | {k: reply.get(k, 0) for k in REPLY}


def replay(stage, a, merge, strips):
    """Write the merge list into staging, only where staging holds air; returns the record with the region files changed."""
    core = {tuple(c) for c in a["core"]}
    if set(merge) & core:
        raise AssertionError(f"the merge list would change protected chunks, the first {min(set(merge) & core)}")
    k = smooth.classify(world.registry(world.read_level(stage)))
    loaded = smooth.load_chunks(stage, set(merge))
    out, counts = {}, Counter(log=0, leaves=0, log_in_strips=0, skipped_not_air=0)
    for pos in sorted(loaded):
        rec, n = _apply(loaded[pos], merge[pos], strips.get(pos), k)
        jobs.checkpoint(TREES)
        if rec is not None:
            out[pos] = rec
        counts.update(n)
    changed = smooth.write(stage, out, TREES)
    return dict(counts) | {"chunks": len(out), "chunks_changed": sorted(out), "changed": changed}


def _apply(ch, voxels, strip, k):
    idx, ids, meta = voxels
    ch.snapshot()
    flat_ids, flat_meta = ch.ids.reshape(-1), ch.meta.reshape(-1)
    free = flat_ids[idx] == smooth.AIR_ID
    flat_ids[idx[free]], flat_meta[idx[free]] = ids[free], meta[free]
    cat = smooth.lookup(k.cat, ids[free])
    cols = idx[free] % (SIDE * SIDE)
    in_strip = strip[0].reshape(-1)[cols] if strip is not None else np.zeros(cols.size, bool)
    n = {
        "log": int(np.sum(cat == smooth.LOG)),
        "leaves": int(np.sum(cat == smooth.LEAVES)),
        "log_in_strips": int(np.sum((cat == smooth.LOG) & in_strip)),
        "skipped_not_air": int(np.sum(~free)),
    }
    if not free.any():
        return None, n
    mod = ch.modified()
    low = np.where(mod.any(0), np.argmax(mod, 0), chunks.HEIGHT).astype(np.int32)
    rec, _, dropped = ch.encode(k, (np.zeros_like(mod[0]), low))
    return (rec, ch.ts), n | {f"dropped_{key.lower()}": v for key, v in dropped.items()}


@dataclass
class Step:
    """What the plant steps need from the preview: the job, its generator, the area and the generator settings."""

    job: object
    gens: object
    a: dict
    settings: dict


def run(st, strips_path, say=print):
    """Plant (reusing the cached merge list while its key holds) and merge into staging; returns the trees record."""
    job, a = st.job, st.a
    stage = job.data["args"]["staging"]
    strips = load_strips(strips_path)
    p = plan(stage, a, strips)
    want = key(stage, a, p, st.settings)
    path = os.path.join(job.step_dir(PLANT), MERGE_FILE)
    replayed = job.step_valid(PLANT, want)
    if not replayed:
        job.drop_step(PLANT)
        if p.spots:
            say("preview: replanting trees on the reshaped strips (a third start of the generator, about 1 to 2.5 minutes)")
        merge, booted = boot(job, st.gens, a, p) if p.spots else ({}, _booted(p, None, {}))
        save_merge(path, merge)
        write_atomic(os.path.join(job.step_dir(PLANT), RECORD), json.dumps(booted).encode())
        job.finish_step(PLANT, [MERGE_FILE, RECORD], want)
    booted = jobs.read_json(os.path.join(job.folder, PLANT, RECORD))
    merged = replay(stage, a, load_merge(path), strips)
    removed = int(sum(n for _, n in strips.values()))
    return merged | {"boot": booted, "spots": len(p.spots), "log_removed": removed, "replayed": replayed}
