"""The generated surroundings: a stripped gen-world built from staging, the gen boot that generates the ring around kept, the
cached render of the chosen config, and the harvest of what the boot generated into staging."""

import gzip
import hashlib
import io
import json
import os
import time
from collections import Counter

import numpy as np

from . import ISSUES, Refusal, area, chunks, generator, jobs, nbt, postcheck, provision, region, smooth, transplant, world
from .ground import column_ground
from .maps import render_window
from .region import copy_tree, rmtree, write_atomic

GEN_PREFIX, PLANT_PREFIX = "gen-", "plant-"
RING = 8
MIN_RING = 3
COVERAGE = 2
EMPTY_LISTS = ("Entities", "TileEntities", "TileTicks")
GENERATE, RENDER, HARVEST = "generate", "render", "harvest"
GENERATED, GROUND, RECORD = "generated.json", "ground.npz", "record.json"
RENDER_FILE = "render-{}.npz"
BPP = 1
VERIFY_REACH = 4
SPAWN_CHUNKS = 12


def folder(cache, release, job_id, prefix=GEN_PREFIX):
    return os.path.join(provision.pack_dir(cache, release), prefix + job_id)


def centre(cells):
    """The block column at the middle of the kept chunk nearest to the mean of cells."""
    mx, mz = np.mean(np.array(sorted(cells), float), axis=0)
    x, z = min(cells, key=lambda c: ((c[0] - mx) ** 2 + (c[1] - mz) ** 2, c))
    return x * chunks.SIDE + chunks.SIDE // 2, z * chunks.SIDE + chunks.SIDE // 2


def spawn_area(kept):
    """The chunks the gen boot loads and decorates around its spawn before any request: vanilla's -192..192 blocks."""
    x, z = centre(kept)
    cx, cz = x // chunks.SIDE, z // chunks.SIDE
    return area.dilate({(cx, cz)}, SPAWN_CHUNKS)


def _level(src, dest, spawn):
    """level.dat of the gen-world: staging's, with the spawn at the centre of kept and no single-player player. Returns MapFeatures."""
    with open(src, "rb") as f:
        name, root = nbt.parse(nbt.unpack(f.read()))
    data = root["Data"][1]
    data["SpawnX"], data["SpawnZ"] = [nbt.INT, int(spawn[0])], [nbt.INT, int(spawn[1])]
    data.pop("Player", None)
    write_atomic(dest, gzip.compress(nbt.write(name, root), mtime=0))
    return bool(data.get("MapFeatures", [nbt.BYTE, 1])[1])


def _air_at_tile_entities(level):
    """Blocks that carry a tile entity become air; returns how many."""
    spots = {}
    for te in level.get("TileEntities", [nbt.LIST, [nbt.END, []]])[1][1]:
        x, y, z = (te[k][1] for k in ("x", "y", "z"))
        spots.setdefault(y >> 4, set()).add(((y & 15) << 8) | ((z & 15) << 4) | (x & 15))
    cleared = 0
    secs = chunks.sections(level)
    for i, sec in enumerate(secs):
        idx = sorted(spots.get(sec["Y"][1], ()))
        if idx:
            v = chunks.decode_section(sec)
            v.ids[idx], v.meta[idx] = 0, 0
            secs[i] = chunks.encode_section(chunks.section_format(sec), sec["Y"][1], v, like=sec)
            cleared += len(idx)
    return cleared


def strip_chunk(rec, populated=False):
    """(record, tile-entity blocks cleared): tile-entity blocks become air, entities and ticks are emptied, so nothing in a kept
    chunk can act or wander into the ring while the gen boot runs; with populated, the chunk is also marked decorated."""
    name, root = nbt.parse(region.inflate(rec))
    level = root["Level"][1]
    cleared = _air_at_tile_entities(level)
    for key in EMPTY_LISTS:
        if key in level:
            level[key] = [nbt.LIST, [nbt.END, []]]
    if populated:
        level["TerrainPopulated"] = [nbt.BYTE, 1]
    return region.deflate(nbt.write(name, root)), cleared


def build(stage, dest, kept, populated=False):
    """The gen-world: level.dat, data/ and the overworld chunks of staging, stripped (with populated, all marked decorated, so a
    boot on it decorates nothing). Other dimensions, playerdata and forcedchunks.dat stay out, so chunk loaders, machines and
    other dimensions never load or tick."""
    rmtree(dest)
    os.makedirs(os.path.join(dest, "region"))
    spawn = centre(kept)
    features = _level(os.path.join(stage, "level.dat"), os.path.join(dest, "level.dat"), spawn)
    if os.path.isdir(os.path.join(stage, "data")):
        copy_tree(os.path.join(stage, "data"), os.path.join(dest, "data"))
    stats = Counter()
    for path in transplant.region_files(stage):
        with open(path, "rb") as f:
            recs = region.records(f.read())
        out = {}
        for i, (rec, ts) in recs.items():
            out[i], n = strip_chunk(rec, populated)
            out[i] = (out[i], ts)
            stats.update(chunks=1, tile_entity_blocks_cleared=n)
        write_atomic(os.path.join(dest, "region", os.path.basename(path)), region.build(out))
        stats.update(regions=1)
        jobs.checkpoint(GENERATE)
    return {"spawn": list(spawn), "map_features": features} | dict(stats)


def world_files(dest):
    """{relative path: sha256} of the gen-world files the harvest reads."""
    rels = ["level.dat", *(f"region/{os.path.basename(p)}" for p in transplant.region_files(dest))]
    return {rel: jobs.file_hash(os.path.join(dest, *rel.split("/"))) for rel in rels if os.path.isfile(os.path.join(dest, *rel.split("/")))}


def generate_key(job, a, settings):
    """What the gen-world depends on: core and kept, offsets, the user's config folder, release, probe, ring, structures, noise."""
    args = job.data["args"]
    return {
        "core": jobs.key_of(sorted(map(tuple, a["core"]))),
        "kept": jobs.key_of(sorted(map(tuple, a["kept"]))),
        "offsets": list(args["config"]),
        "config": jobs.tree_key(args.get("config_dir")),
        "release": (settings or {}).get("release"),
        "probe": generator.PROBE_VERSION,
        "ring": args["ring"],
        "rwg_data": args.get("rwg_data"),
    }


def generated(job):
    return jobs.read_json(os.path.join(job.folder, GENERATE, GENERATED))


def gen_dir(job, rec):
    """The gen-world of a job, under the cache the job names now (not the path recorded at generate time), so a job moved to
    another machine with its cache folder finds it there."""
    return folder(job.data["args"]["cache"], rec["release"], job.id)


def gen_valid(job, key):
    """The gen-world counts only after a clean exit (its manifest is written then) and while its files are as the boot left them."""
    rec = generated(job)
    return bool(rec) and job.step_valid(GENERATE, key) and world_files(gen_dir(job, rec)) == rec["files"]


def boot(gens, name, offsets, features):
    boot = generator.Boot(kind=generator.GEN, world=name, offsets=tuple(offsets), properties={"generate-structures": str(features).lower()})
    return gens.start(boot)


def _progress(job):
    def emit(event):
        job.emit(event)
        jobs.checkpoint(GENERATE)

    return emit


def run_generate(job, gens, a, key):
    """Build the gen-world from staging, boot on it, generate dilate(kept, ring), save, stop. Its output counts only after the
    JVM exited with code 0; any earlier stop (crash, kill, power cut) leaves no manifest, so the next run builds it again. key()
    gives the step's input keys once the generator is set up."""
    job.drop_step(GENERATE)
    ctx, settings = gens.context()
    gens.stop()
    generator.stop_running(gens.cache, job.emit)
    dest = folder(gens.cache, settings["release"], job.id)
    kept = {tuple(c) for c in a["kept"]}
    t0 = time.monotonic()
    built = build(job.data["args"]["staging"], dest, kept)
    t1 = time.monotonic()
    gen = boot(gens, os.path.basename(dest), job.data["args"]["config"], built["map_features"])
    memory = generator.PeakMemory(gen.record["pid"])
    t2 = time.monotonic()
    cells = area.dilate(kept, job.data["args"]["ring"])
    totals = generator.generate(gen, job.data["args"]["config"], cells, _progress(job))
    t3 = time.monotonic()
    status = gen.call("status")
    gen.call("save")
    t4 = time.monotonic()
    code = gens.stop()
    memory.stop()
    seconds = {"build": t1 - t0, "boot": t2 - t1, "generate": t3 - t2, "save": t4 - t3, "stop": time.monotonic() - t4}
    if code != 0:
        raise Refusal(
            f"the generator exited with code {code} after generating the new terrain, so the generated chunks may be half saved",
            f"Run it again: rwg2-migrator jobs resume {job.id} (the new terrain is generated again from the start)",
        )
    rec = {
        "folder": dest,
        "release": ctx.release,
        "built": built,
        "requested": len(cells),
        "totals": totals,
        "exit_code": code,
        "noise": status.get("noise"),
        "heap": gen.record["heap"],
        "heap_used_mb": status.get("usedHeapMB"),
        "peak_working_set_mb": round(memory.peak / provision.MB),
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "files": world_files(dest),
    }
    write_atomic(os.path.join(job.step_dir(GENERATE), GENERATED), json.dumps(rec, indent=1).encode())
    job.finish_step(GENERATE, [GENERATED], key())
    job.emit({"step": GENERATE, "event": "done"} | {k: rec[k] for k in ("requested", "totals", "noise", "peak_working_set_mb", "seconds")})
    return rec


def render_boxes(a, ring):
    """One render window per connected part of the ring and the gen boot's spawn area grown by VERIFY_REACH, at 1 block per
    pixel: past what the harvest takes too, where the user's game generates first after Apply, so verify can compare those
    chunks with it."""
    kept = {tuple(c) for c in a["kept"]}
    cells = area.dilate(area.dilate(kept, ring) | spawn_area(kept), VERIFY_REACH)
    return [render_window(b, BPP) for b in area.windows(cells, ring=0)]


def render_key(job, a, settings):
    args = job.data["args"]
    return {
        "config": list(args["config"]),
        "windows": [list(w) for w in render_boxes(a, args["ring"])],
        "seed": str(args["seed"]),
        "user_config": jobs.tree_key(args.get("config_dir")),
        "release": (settings or {}).get("release"),
        "probe": generator.PROBE_VERSION,
    }


def run_render(job, gens, key):
    """Render the chosen config over every window; cached in the job, so a refresh never boots the sampler again. key() gives
    the step's input keys (config and windows among them) once the generator is set up."""
    job.drop_step(RENDER)
    folder_ = job.step_dir(RENDER)
    sampler = gens.sampler(job.data["args"]["seed"], gens.threads)()
    keys, names = key(), []
    for i, win in enumerate(keys["windows"]):
        rec, palette = sampler.render(keys["config"], tuple(win), BPP)
        name = RENDER_FILE.format(i)
        np.savez_compressed(os.path.join(folder_, name), h=rec["h"], palette=rec["palette"], names=np.array(palette), window=np.array(win))
        names.append(name)
    gens.stop()
    job.finish_step(RENDER, names, key())
    job.emit({"step": RENDER, "event": "done", "windows": len(names)})


class Render:
    """RWG2's top of ground (floor(h) + 1) and biome name of the chosen config, looked up at block columns from the cached
    renders. Biomes are indices into names, -1 off the renders."""

    def __init__(self, job):
        self.parts, self.names = [], []
        for name in sorted(os.listdir(os.path.join(job.folder, RENDER))):
            if name.startswith("render-"):
                with np.load(os.path.join(job.folder, RENDER, name)) as z:
                    names = [str(n) for n in z["names"]]
                    index = np.array([self._name(n) for n in names] or [-1], np.int32)
                    biome = index[np.minimum(z["palette"].astype(np.int64), index.size - 1)]
                    self.parts.append((tuple(int(v) for v in z["window"]), np.floor(z["h"].astype(np.float64)) + 1, biome))

    def _name(self, name):
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name)

    def _lookup(self, xz, layer, fill):
        xz = np.asarray(xz, np.int64).reshape(-1, 2)
        out = np.full(len(xz), fill, np.float64 if layer == 1 else np.int32)
        for part in self.parts:
            (x0, z0, size), values = part[0], part[layer]
            px, pz = xz[:, 0] - x0, xz[:, 1] - z0
            sel = (px >= 0) & (pz >= 0) & (px < size) & (pz < size)
            out[sel] = values[pz[sel], px[sel]]
        return out

    def at(self, xz):
        return self._lookup(xz, 1, np.nan)

    def biome_at(self, xz):
        return self._lookup(xz, 2, -1)

    def _grid(self, box, fn, bpp):
        x0, z0, x1, z1 = box
        zz, xx = np.mgrid[z0:z1:bpp, x0:x1:bpp]
        return fn(np.c_[xx.ravel(), zz.ravel()]).reshape(zz.shape)

    def grid(self, box, bpp=1):
        return self._grid(box, self.at, bpp)

    def biome_grid(self, box, bpp=1):
        return self._grid(box, self.biome_at, bpp)


def _item_map(level):
    return {e["K"]: e["V"] for e in level["FML"]["ItemData"]}


def _fluids(level):
    return set(level.get("Forge", {}).get("DefaultFluidList", []))


def _ids_action(level):
    """What to do about ids that differ: the generator always runs the whole pack, so a world last opened without some of the
    pack's mods is the usual cause, and those mods are named."""
    try:
        mods = world.mods(level)
        missing = world.mod_diff(mods, world.release_for(mods, world.releases())[1])["not_in_world"]
    except (Refusal, KeyError, TypeError):
        missing = []
    if not missing:
        return "Install exactly the GTNH release your world uses (rwg2-migrator generator setup), then run preview again"
    return (
        f"Your world was last opened without these mods of the GTNH pack: {jobs.listed(missing)}. Open it once with the full "
        "pack, close it, then run preview again"
    )


def registry_check(stage, gen_dir):
    """The gen-world's block, item and fluid registries must agree with the user's on every entry they share, and hold none the
    user's lacks: chunks copied from it then mean the same blocks in the user's world, and so do items in its chests."""
    user, gen = (nbt.read_file(os.path.join(d, "level.dat")) for d in (stage, gen_dir))
    u, g = _item_map(user), _item_map(gen)
    bad = sorted(k for k, v in g.items() if u.get(k) != v)
    fluids = sorted(_fluids(gen) - _fluids(user))
    if bad or fluids:
        first = bad[0] if bad else fluids[0]
        raise Refusal(
            f"the generator's block and item ids differ from your world's in {len(bad)} entries and {len(fluids)} fluids "
            f"(the first {first[1:] if bad else first!r}), so its chunks would hold the wrong blocks in your world",
            _ids_action(user),
        )
    return {"entries": len(g), "shared": len(set(g) & set(u)), "fluids": len(_fluids(gen))}


def _take(gen_dir, skip, ok):
    """{pos: (record, timestamp)} of every chunk of the gen-world outside skip, decorated or not, the decorated ones with their
    ground, and the entity count. An undecorated chunk at the edge already holds what its decorated neighbours spilled into it;
    once the user's game decorates it, it writes only outward."""
    taken, ground, entities = {}, {}, 0
    for path in transplant.region_files(gen_dir):
        rx, rz = region.coords(path)
        with open(path, "rb") as f:
            recs = region.records(f.read())
        for i, (rec, ts) in recs.items():
            pos = region.slot_pos(rx, rz, i)
            got = transplant.decode(rec, pos) if pos not in skip and rec is not None else None
            if got is None:
                continue
            level = got[1]["Level"][1]
            taken[pos] = (rec, ts)
            if level.get("TerrainPopulated", [nbt.BYTE, 0])[1]:
                ground[pos] = column_ground(chunks.block_ids(level), chunks.heightmap(level), ok)
            entities += len(level.get("Entities", [nbt.LIST, [nbt.END, []]])[1][1])
    return taken, ground, entities


def _write(stage, taken):
    """Add the taken chunks to staging's overworld region files; returns {rel: sha256} of every file written."""
    by_region = {}
    for pos, rec in taken.items():
        by_region.setdefault((pos[0] // region.SIDE, pos[1] // region.SIDE), {})[region.slot_index(*pos)] = rec
    changed = {}
    for (rx, rz), add in sorted(by_region.items()):
        rel = f"region/r.{rx}.{rz}.mca"
        path = os.path.join(stage, "region", f"r.{rx}.{rz}.mca")
        have = {}
        if os.path.exists(path):
            with open(path, "rb") as f:
                have = region.records(f.read())
        if set(have) & set(add):
            raise AssertionError(f"{rel}: the harvest would replace chunks staging already holds")
        slots = have | add
        data = region.build(slots)
        problem = region.check(data, slots)
        if problem:
            raise AssertionError(f"{rel}: {problem}")
        write_atomic(path, data)
        changed[rel] = hashlib.sha256(data).hexdigest()
        jobs.checkpoint(HARVEST)
    return changed


def _coverage(a, never, decorated):
    """Every chunk within COVERAGE chunks of kept must be in staging and decorated, or RWG2 would decorate it into kept."""
    kept = {tuple(c) for c in a["kept"]}
    need = area.dilate(kept, COVERAGE) - (kept - never)
    missing = need - set(decorated)
    if missing:
        raise Refusal(
            f"{jobs.count(len(missing), 'chunk')} within {COVERAGE} chunks of the kept area {'was' if len(missing) == 1 else 'were'} "
            "not generated and decorated: "
            f"{jobs.listed_cells(missing)}",
            f"Run preview again with a larger --ring (at least 3); if it repeats, report it with the job folder at {ISSUES}",
        )
    return len(need)


def _block_ids(world_dir, cells):
    """{pos: block ids} of the cells present in world_dir, one region file at a time."""
    for path in transplant.region_files(world_dir):
        rx, rz = region.coords(path)
        mine = {c for c in cells if (c[0] // region.SIDE, c[1] // region.SIDE) == (rx, rz)}
        if not mine:
            continue
        with open(path, "rb") as f:
            recs = region.records(f.read())
        for pos in sorted(mine):
            rec = recs.get(region.slot_index(*pos), (None, 0))[0]
            got = transplant.decode(rec, pos) if rec is not None else None
            if got is not None:
                yield pos, chunks.block_ids(got[1]["Level"][1])


def _clipped(stage, gen_dir, cells, k):
    """Log and leaf voxels that the ring's decoration put into kept chunks of the gen-world: trees cut at the kept border,
    because kept chunks are never copied from it."""
    out = Counter(logs=0, leaves=0, chunks=0)
    gen = dict(_block_ids(gen_dir, cells))
    for pos, have in _block_ids(stage, cells):
        if pos not in gen:
            continue
        cat = smooth.lookup(k.cat, gen[pos])
        new = gen[pos] != have
        logs, leaves = int(np.sum(new & (cat == smooth.LOG))), int(np.sum(new & (cat == smooth.LEAVES)))
        out.update(logs=logs, leaves=leaves, chunks=int(logs + leaves > 0))
    return dict(out)


def harvest(job, a, gen_rec, never):
    """Copy every chunk the gen boot made outside kept, decorated or not, and the kept chunks the world never had, from the
    gen-world into staging, after the registry guard; then check the coverage. That is the ring, plus whatever of the boot's
    spawn area and of the decoration's spill lies past it: a decorated chunk left out would be decorated a second time by the
    user's game, and the chunks it decorated into would lose that decoration. Kept chunks the world has are never copied."""
    stage, gen = job.data["args"]["staging"], gen_dir(job, gen_rec)
    reg = registry_check(stage, gen)
    kept = {tuple(c) for c in a["kept"]}
    never = {tuple(c) for c in never}
    ring = (area.dilate(kept, job.data["args"]["ring"]) - kept) | never
    ok = postcheck.registry_ok(stage)
    taken, ground, entities = _take(gen, kept - never, ok)
    covered = _coverage(a, never, ground)
    k = smooth.classify(world.registry(world.read_level(stage)))
    clipped = _clipped(stage, gen, area.boundary(kept) - never, k)
    changed = _write(stage, taken)
    buf = io.BytesIO()
    pos = np.array(sorted(ground), np.int64).reshape(-1, 2)
    np.savez_compressed(buf, pos=pos, ground=np.array([ground[tuple(p)] for p in pos.tolist()], np.float32).reshape(-1, 16, 16))
    write_atomic(os.path.join(job.step_dir(HARVEST), GROUND), buf.getvalue())
    return {
        "registry": reg,
        "harvested": len(taken),
        "never_generated_harvested": sorted(set(taken) & never),
        "ring_requested": len(ring),
        "beyond_ring": len(set(taken) - ring),
        "beyond_ring_decorated": len(set(ground) - ring),
        "not_decorated": len(set(taken) - set(ground)),
        "not_generated": len(ring - set(taken)),
        "coverage_checked": covered,
        "entities": entities,
        "trees_clipped": clipped,
        "chunks": sorted(taken),
        "changed": changed,
    }


def harvest_ground(job):
    """{(cx, cz): 16x16 ground} of the decorated harvested chunks as the gen boot left them (before smoothing). The undecorated
    ones at the edge are left out: their ground is not final until the user's game decorates them."""
    with np.load(os.path.join(job.folder, HARVEST, GROUND)) as z:
        return {tuple(p): g for p, g in zip(z["pos"].tolist(), z["ground"], strict=True)}


def discard(job):
    """Delete a preview's gen-world and any plant world a crash left next to it (its staging is deleted with it)."""
    rec = generated(job)
    if rec:
        rmtree(gen_dir(job, rec))
        rmtree(os.path.join(os.path.dirname(gen_dir(job, rec)), PLANT_PREFIX + job.id))
