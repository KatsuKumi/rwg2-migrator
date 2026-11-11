"""Keep only the kept overworld chunks of a staging copy and switch it to RWG2 Smooth [POC rwg2_transplant]."""

import gzip
import hashlib
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import Refusal, chunks, jobs, nbt, region, world
from .region import remove_file, write_atomic

GENERATOR = "RWG2_CONTINENTS"
FIELDS = {"generatorName": (nbt.STRING, GENERATOR), "generatorVersion": (nbt.INT, 0), "generatorOptions": (nbt.STRING, "")}
LEVEL_FILES = ("level.dat", "level.dat_old")
PLACED_BY_BIOME, PLACED_BY_SEED = "biome", "seed"
RWG2_STRUCTURES = {"Village": PLACED_BY_BIOME, "Stronghold": PLACED_BY_BIOME, "Mineshaft": PLACED_BY_SEED}
DROP_DATA = tuple(f"{name}.dat" for name, placed in RWG2_STRUCTURES.items() if placed == PLACED_BY_BIOME)
RWG_DATA = world.RWG_DATA.replace(os.sep, "/")
READ_ERRORS = (ValueError, KeyError, TypeError, IndexError, EOFError)
CHUNK = chunks.SIDE
CENTRE = CHUNK // 2
STAND = 0.5
EMPTY_COLUMN_TOP = 64


@dataclass(frozen=True)
class Options:
    set_populated: bool = False
    move_players: bool = True


@dataclass
class Area:
    core: set
    kept: set
    teams: list = field(default_factory=list)
    erased: set = field(default_factory=set)

    @classmethod
    def of(cls, a):
        cells = lambda key: {tuple(c) for c in a.get(key, [])}  # noqa: E731
        return cls(cells("core"), cells("kept"), list(a.get("teams", [])), cells("erased_claims"))


@dataclass
class RegionPlan:
    kept: dict = field(default_factory=dict)
    roots: dict = field(default_factory=dict)
    unreadable: list = field(default_factory=list)
    unpopulated: list = field(default_factory=list)
    dropped: int = 0


def region_files(world_dir):
    return [p for p in world.files(os.path.join(world_dir, "region"), "r.*.*.mca") if region.NAME.fullmatch(os.path.basename(p))]


def decode(rec, pos):
    """(name, root) of a chunk record stored at pos, or None when it does not decode there."""
    try:
        name, root = nbt.parse(region.inflate(rec))
        level = root["Level"][1]
        return (name, root) if (level["xPos"][1], level["zPos"][1]) == pos else None
    except READ_ERRORS:
        return None


def drift(world_dir, area):
    """Claims of the chosen teams made since the area was built must not fall outside it."""
    claims = world.claims(world_dir)
    chosen = set().union(*(claims.get(t, set()) for t in area.teams))
    new = chosen - area.core - area.erased
    if new:
        raise Refusal(
            f"{jobs.count(len(new), 'chunk')} claimed since the area was made {'lies' if len(new) == 1 else 'lie'} outside it: "
            f"{jobs.listed_cells(new)}",
            "Make the area again (rwg2-migrator area ... --claims ...), or add these chunks with --add, then run preview again",
        )


def _plan_region(path, area):
    rx, rz = region.coords(path)
    with open(path, "rb") as f:
        recs = region.records(f.read())
    plan = RegionPlan()
    for i, (rec, ts) in sorted(recs.items()):
        pos = region.slot_pos(rx, rz, i)
        if pos not in area.kept:
            plan.dropped += 1
            continue
        got = decode(rec, pos)
        if got is None:
            plan.unreadable.append(pos)
            continue
        if not got[1]["Level"][1].get("TerrainPopulated", [nbt.BYTE, 0])[1]:
            plan.unpopulated.append(pos)
        plan.kept[i], plan.roots[pos] = (rec, ts), got
    return plan


def _refuse_bad(plans, area, opts):
    unreadable = [c for p in plans.values() for c in p.unreadable]
    if unreadable:
        raise Refusal(
            f"{jobs.count(len(unreadable), 'kept chunk')} {'does' if len(unreadable) == 1 else 'do'} not read in the working copy: "
            f"{jobs.listed_cells(unreadable)}",
            "Run preview again; if it fails while the world is stopped, restore those region files from a backup",
        )
    core = [c for p in plans.values() for c in p.unpopulated if c in area.core]
    if core and not opts.set_populated:
        raise Refusal(
            f"{jobs.count(len(core), 'chunk')} of the area to keep {'was' if len(core) == 1 else 'were'} never decorated by the game: "
            f"{jobs.listed_cells(core)}. "
            "RWG2 would decorate them as soon as their neighbours exist",
            "Run preview again with --set-populated to mark them decorated (a 1-byte change each, listed in the report), "
            "or leave them out of the area",
        )


def _set_populated(plan):
    for pos in plan.unpopulated:
        name, root = plan.roots[pos]
        root["Level"][1]["TerrainPopulated"] = [nbt.BYTE, 1]
        i = region.slot_index(*pos)
        plan.kept[i] = (region.deflate(nbt.write(name, root)), plan.kept[i][1])


def _top(root):
    ids = chunks.block_ids(root["Level"][1])[:, CENTRE, CENTRE]
    filled = np.flatnonzero(ids)
    return int(filled[-1]) + 1 if filled.size else EMPTY_COLUMN_TOP


class Mover:
    """Moves a player who stands outside kept in the overworld to the top of the nearest core chunk the copy holds."""

    def __init__(self, area, roots):
        self.area, self.roots = area, roots
        self.targets = sorted(c for c in area.core if c in roots)

    def move(self, player):
        """(from chunk, to position) after moving the player tag in place, or None when the player stays."""
        if player.get("Dimension", [nbt.INT, 0])[1] != 0 or not self.targets or "Pos" not in player:
            return None
        x, _, z = (struct.unpack(">d", b)[0] for b in player["Pos"][1][1])
        cell = (math.floor(x) // CHUNK, math.floor(z) // CHUNK)
        if cell in self.area.kept:
            return None
        tx, tz = min(self.targets, key=lambda c: ((c[0] - cell[0]) ** 2 + (c[1] - cell[1]) ** 2, c))
        to = (tx * CHUNK + CENTRE + STAND, float(_top(self.roots[(tx, tz)][1])), tz * CHUNK + CENTRE + STAND)
        player["Pos"] = [nbt.LIST, [nbt.DOUBLE, [struct.pack(">d", v) for v in to]]]
        return cell, to


def _gzip_nbt(name, root):
    return gzip.compress(nbt.write(name, root), mtime=0)


def _read_nbt(path):
    with open(path, "rb") as f:
        raw = nbt.unpack(f.read())
    return raw, *nbt.parse(raw)


def _without(root, keys, player):
    data = dict(root["Data"][1])
    for k in keys:
        data.pop(k, None)
    if player:
        data.pop("Player", None)
    return {**root, "Data": [nbt.COMPOUND, data]}


def _level_file(path, mover, moved):
    """New bytes of level.dat or level.dat_old: the 3 generator fields set, the single-player player moved; nothing else."""
    raw, name, root = _read_nbt(path)
    data = root["Data"][1]
    before = {}
    for k, (t, v) in FIELDS.items():
        if data.get(k, [None])[0] != t:
            raise Refusal(
                f"{os.path.basename(path)} has no Data.{k} of the expected type, so it is damaged",
                "Restore level.dat from level.dat_old or a backup, then run preview again",
            )
        before[k] = data[k][1]
        data[k] = [t, v]
    player = data.get("Player")
    hop = mover.move(player[1]) if player and mover else None
    if hop:
        moved.append({"name": f"single-player ({os.path.basename(path)})", "from": hop[0], "to": hop[1]})
    old = nbt.parse(raw)[1]
    if _without(old, FIELDS, hop) != _without(root, FIELDS, hop):
        raise AssertionError(f"{path}: the level.dat edit changed more than the generator fields")
    return _gzip_nbt(name, root), before


def _players(world_dir, mover, moved):
    out = {}
    for path in world.files(os.path.join(world_dir, "playerdata"), "*.dat"):
        try:
            _, name, root = _read_nbt(path)
        except (OSError, *READ_ERRORS):
            continue
        hop = mover.move(root)
        if hop:
            out[path] = _gzip_nbt(name, root)
            moved.append({"name": Path(path).stem, "from": hop[0], "to": hop[1]})
    return out


def structure_files(world_dir):
    """Stems of data/*.dat files that hold structure data (a data.Features compound)."""
    out = []
    for path in world.files(os.path.join(world_dir, "data"), "*.dat"):
        try:
            d = nbt.read_file(path).get("data")
        except (OSError, nbt.NBTError):
            continue
        if isinstance(d, dict) and "Features" in d:
            out.append(Path(path).stem)
    return out


def _data_changes(world_dir):
    drop = [f"data/{n}" for n in DROP_DATA if os.path.isfile(os.path.join(world_dir, "data", n))]
    noise = world.rwg_data(world_dir)
    noisy = noise["noiseImplementation"] not in (None, *world.PERLIN) and "error" not in noise
    if noisy:
        drop.append(RWG_DATA)
    reported = sorted(s for s in structure_files(world_dir) if s not in RWG2_STRUCTURES)
    return drop, reported, ("deleted" if noisy else "kept" if noise["noiseImplementation"] else "absent")


def _rel(world_dir, path):
    return os.path.relpath(path, world_dir).replace(os.sep, "/")


def _region_writes(world_dir, plans):
    writes = {}
    for path, plan in plans.items():
        rel = _rel(world_dir, path)
        if not plan.kept:
            writes[rel] = None
            continue
        data = region.build(plan.kept)
        problem = region.check(data, plan.kept)
        if problem:
            raise AssertionError(f"{rel}: {problem}")
        with open(path, "rb") as f:
            if f.read() != data:
                writes[rel] = data
    return writes


def _write_all(world_dir, writes):
    changed = {}
    for rel, data in sorted(writes.items()):
        jobs.checkpoint("transplant")
        path = os.path.join(world_dir, *rel.split("/"))
        if data is None:
            remove_file(path)
        else:
            write_atomic(path, data)
        changed[rel] = None if data is None else hashlib.sha256(data).hexdigest()
        jobs.checkpoint("transplant")
    return changed


def _validate(world_dir, changed):
    for rel, digest in changed.items():
        path = os.path.join(world_dir, *rel.split("/"))
        if digest is None and os.path.lexists(path):
            raise AssertionError(f"{rel} should have been deleted")
        if digest is not None and jobs.file_hash(path) != digest:
            raise AssertionError(f"{rel} does not read back as written")


def run(world_dir, a, opts):
    """Transplant the staging world in place; returns the record, whose 'changed' maps each file it wrote to its sha256 (None
    for a deleted file)."""
    area = Area.of(a)
    drift(world_dir, area)
    plans = {p: _plan_region(p, area) for p in region_files(world_dir)}
    _refuse_bad(plans, area, opts)
    for plan in plans.values():
        _set_populated(plan)
    roots = {pos: got for plan in plans.values() for pos, got in plan.roots.items()}
    mover, moved, before = (Mover(area, roots) if opts.move_players else None), [], {}
    writes = _region_writes(world_dir, plans)
    for name in LEVEL_FILES:
        path = os.path.join(world_dir, name)
        if os.path.isfile(path):
            writes[name], before[name] = _level_file(path, mover, moved)
    writes |= {_rel(world_dir, p): b for p, b in (_players(world_dir, mover, moved) if mover else {}).items()}
    drop, reported, noise = _data_changes(world_dir)
    writes |= dict.fromkeys(drop)
    changed = _write_all(world_dir, writes)
    _validate(world_dir, changed)
    flagged = [c for plan in plans.values() for c in plan.unpopulated]
    return {
        "generator": GENERATOR,
        "core": len(area.core),
        "kept": len(roots),
        "never_generated": sorted(area.kept - set(roots)),
        "chunks_dropped": sum(p.dropped for p in plans.values()),
        "regions_kept": sorted(os.path.basename(p) for p, plan in plans.items() if plan.kept),
        "regions_deleted": sorted(os.path.basename(p) for p, plan in plans.items() if not plan.kept),
        "populated_set": {"core": sorted(c for c in flagged if c in area.core), "margin": sorted(c for c in flagged if c not in area.core)},
        "players_moved": moved,
        "level": before,
        "data_deleted": drop,
        "data_reported": reported,
        "rwg_data": noise,
        "changed": changed,
    }
