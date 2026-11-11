"""An in-process stand-in for the generator's gen and sampler boots. A gen boot first makes and decorates the vanilla spawn area
around the gen-world's spawn, as the game does before any request. generate makes deterministic chunks whose ground is the fake
sampler's top of ground and whose biomes are those its render names, each with one pig; decoration puts a log into the +x
neighbour, as RWG2's trees reach into it; render is the fake sampler's. plant grows a small tree per spot (a trunk, leaves into
air, the grass under it turned to dirt and a snow layer on top, as real trees and weather change more than logs and leaves) until
the spot's group has its log budget. Install it with use(monkeypatch) or, in a subprocess, install()."""

import os
import struct
from types import SimpleNamespace

import fake_sampler
import numpy as np
from conftest import string, tag
from fake_sampler import FakeSampler

from rwg2migrator import chunks, cli_search, generator, genworld, jobs, nbt, provision, region, transplant, world

RELEASE = "2.9.0"
DELAY_ENV = "RWG2_FAKE_DELAY"
STONE, GRASS, DIRT, LOG, LEAVES, SNOW_LAYER = 1, 2, 3, 17, 18, 78
UNKNOWN = "UNKNOWN"
BIOME_IDS = {"Temperate Forest": 324, "Jade Cliffs": 72, "Hot Desert": 318, "Tainted Land": 193}
TRUNK = 4


def _columns(config, cx, cz):
    xs = np.arange(16)
    zz, xx = np.meshgrid(cz * 16 + xs, cx * 16 + xs, indexing="ij")
    return FakeSampler().columns(tuple(config), np.c_[xx.ravel(), zz.ravel()])


def top(config, cx, cz):
    return (np.floor(_columns(config, cx, cz)["h"].astype(np.float64)) + 1).astype(int).reshape(16, 16)


def biomes(config, cx, cz):
    names = [BIOME_IDS[n] for n in fake_sampler.NAMES]
    return np.array(names)[_columns(config, cx, cz)["palette"]].reshape(16, 16)


def _sections(ids):
    out = []
    for y in range(16):
        part = ids[y * 16 : (y + 1) * 16].reshape(-1)
        if part.any():
            z = np.zeros(chunks.VOLUME, np.uint8)
            out.append(chunks.encode_section(chunks.ENDLESSIDS, y, chunks.Voxels(part, np.zeros(chunks.VOLUME, np.uint16), z + 15, z)))
    return out


def pig(cx, cz, ground):
    pos = [struct.pack(">d", v) for v in (cx * 16 + 8.5, float(ground) + 1, cz * 16 + 8.5)]
    most, least = (cx * 1_000_003 + cz) & ((1 << 62) - 1), (cz * 1_000_033 + cx) & ((1 << 62) - 1)
    return {
        "id": string("Pig"),
        "Pos": tag(nbt.LIST, [nbt.DOUBLE, pos]),
        "UUIDMost": tag(nbt.LONG, most),
        "UUIDLeast": tag(nbt.LONG, least),
    }


def new_chunk(config, cx, cz):
    t = top(config, cx, cz)
    ids = np.zeros((chunks.HEIGHT, 16, 16), np.uint32)
    y = np.arange(chunks.HEIGHT)[:, None, None]
    ids[y < t[None] - 1] = STONE
    ids[y == t[None] - 1] = GRASS
    level = {
        "xPos": tag(nbt.INT, cx),
        "zPos": tag(nbt.INT, cz),
        "TerrainPopulated": tag(nbt.BYTE, 0),
        "InhabitedTime": tag(nbt.LONG, 0),
        "HeightMap": tag(nbt.INT_ARRAY, t.astype(">i4").tobytes()),
        "Sections": tag(nbt.LIST, [nbt.COMPOUND, _sections(ids)]),
        "Biomes16v2": tag(nbt.BYTE_ARRAY, chunks.encode_biomes(chunks.ENDLESSIDS, biomes(config, cx, cz))),
        "TileEntities": tag(nbt.LIST, [nbt.END, []]),
        "Entities": tag(nbt.LIST, [nbt.COMPOUND, [pig(cx, cz, t[8, 8])]]),
    }
    return {"Level": tag(nbt.COMPOUND, level)}


def _spill(root):
    """Decoration of the chunk to the west reaches into this one: a log on the ground of its first column."""
    level = root["Level"][1]
    ids = chunks.block_ids(level)
    g = int(np.flatnonzero(ids[:, 0, 0])[-1]) + 1 if ids[:, 0, 0].any() else 64
    secs = {s["Y"][1]: s for s in chunks.sections(level)}
    sec = secs.get(g >> 4)
    if sec is None:
        return
    v = chunks.decode_section(sec)
    v.ids[(g & 15) << 8] = LOG
    secs[g >> 4] = chunks.encode_section(chunks.section_format(sec), g >> 4, v, like=sec)
    level["Sections"][1][1] = [secs[k] for k in sorted(secs)]


def set_block(root, xyz, block, into_air=True):
    """Set one voxel of a chunk root (making its section if needed); with into_air, only where it is air. True if it changed."""
    x, y, z = xyz
    level = root["Level"][1]
    secs = {sec["Y"][1]: sec for sec in chunks.sections(level)}
    sec = secs.get(y >> 4)
    zeros = np.zeros(chunks.VOLUME, np.uint8)
    v = chunks.decode_section(sec) if sec else chunks.Voxels(np.zeros(chunks.VOLUME, np.uint32), zeros.astype(np.uint16), zeros + 15, zeros)
    i = ((y & 15) << 8) | ((z & 15) << 4) | (x & 15)
    if into_air and v.ids[i]:
        return False
    v.ids[i] = block
    secs[y >> 4] = chunks.encode_section(chunks.section_format(sec) if sec else chunks.ENDLESSIDS, y >> 4, v, like=sec or ())
    level["Sections"][1][1] = [secs[k] for k in sorted(secs)]
    return True


def _tree(roots, x, z):
    """A trunk of TRUNK logs on grass with 3x3 leaves round its top, the grass under it turned to dirt and a snow layer on the
    top leaf; the log voxels placed, 0 when the column has no grass on top."""
    root = roots[(x // 16, z // 16)]
    ids = chunks.block_ids(root["Level"][1])[:, z % 16, x % 16]
    g = int(np.flatnonzero(ids)[-1]) + 1 if ids.any() else 0
    if not g or ids[g - 1] != GRASS:
        return 0
    set_block(root, (x, g - 1, z), DIRT, into_air=False)
    logs = sum(set_block(root, (x, g + i, z), LOG) for i in range(TRUNK))
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            for y in (g + TRUNK - 2, g + TRUNK - 1):
                set_block(roots[((x + dx) // 16, (z + dz) // 16)], (x + dx, y, z + dz), LEAVES)
    set_block(root, (x, g + TRUNK, z), LEAVES)
    set_block(root, (x, g + TRUNK + 1, z), SNOW_LAYER)
    return logs


class FakeGen:
    def __init__(self, folder, kind, config):
        self.folder, self.kind, self.config = folder, kind, config
        self.record = {"pid": os.getpid(), "heap": "3G", "instance": "fake"}
        self.roots = {}
        for path in transplant.region_files(folder) if folder else []:
            rx, rz = region.coords(path)
            with open(path, "rb") as f:
                for i, (rec, _) in region.records(f.read()).items():
                    self.roots[region.slot_pos(rx, rz, i)] = nbt.parse(region.inflate(rec))[1]

    def call(self, op, **req):
        return getattr(self, "op_" + op)(**req)

    def boot(self):
        """The game loads the spawn area at boot, so its chunks are made and decorated before the first request."""
        data = nbt.read_file(os.path.join(self.folder, "level.dat"))["Data"]
        spawn = (data["SpawnX"] // 16, data["SpawnZ"] // 16)
        self._make(self.config, sorted(genworld.area.dilate({spawn}, genworld.SPAWN_CHUNKS)))
        self._populate()

    def _make(self, offsets, cells):
        made = 0
        for cx, cz in cells:
            if (cx, cz) not in self.roots:
                self.roots[(cx, cz)] = new_chunk(offsets, cx, cz)
                made += 1
        return made

    def _populate(self):
        populated = 0
        for (cx, cz), root in sorted(self.roots.items()):
            level = root["Level"][1]
            if level["TerrainPopulated"][1] or not all((cx + dx, cz + dz) in self.roots for dx in (0, 1) for dz in (0, 1)):
                continue
            level["TerrainPopulated"] = tag(nbt.BYTE, 1)
            _spill(self.roots[(cx + 1, cz)])
            populated += 1
        return populated

    def op_status(self):
        noise = world.rwg_data(self.folder)["noiseImplementation"] if self.folder else None
        return {"ok": True, "noise": noise or UNKNOWN, "usedHeapMB": 1}

    def op_generate(self, offsets, chunks):
        made = self._make(offsets, chunks)
        return {"ok": True, "existed": len(chunks) - made, "generated": made, "populatedByProbe": self._populate()}

    def op_plant(self, spots, budgets):
        logs, trees, attempts, missing = [0] * len(budgets), [0] * len(budgets), 0, 0
        for x, z, _, g in spots:
            if logs[g] >= budgets[g]:
                continue
            if not all((x // 16 + dx, z // 16 + dz) in self.roots for dx in (-1, 0, 1) for dz in (-1, 0, 1)):
                missing += 1
                continue
            attempts += 1
            n = _tree(self.roots, x, z)
            logs[g], trees[g] = logs[g] + n, trees[g] + (n > 0)
        reply = {"spots": len(spots), "attempts": attempts, "trees": sum(trees), "logs": sum(logs), "missingChunks": missing}
        return {"ok": True, "unknownBiome": 0, "groupTrees": trees, "groupLogs": logs} | reply

    def op_save(self):
        by_region = {}
        for (cx, cz), root in self.roots.items():
            by_region.setdefault((cx // 32, cz // 32), {})[region.slot_index(cx, cz)] = (region.deflate(nbt.write("", root)), 1700000000)
        for (rx, rz), slots in by_region.items():
            region.write_atomic(os.path.join(self.folder, "region", f"r.{rx}.{rz}.mca"), region.build(slots))
        return {"ok": True, "worlds": 1, "flushed": True}


class FakeGens:
    """cli_search.Generators with the fake boots; boots lists the kinds booted, across every instance."""

    boots = []
    exit_code = 0

    def __init__(self, cache, args, emit, overlay=None):
        self.cache, self.args, self.emit, self.overlay, self.gen, self.threads = cache, args, emit, overlay, None, 1
        self.config = jobs.tree_key(overlay)

    def context(self):
        return SimpleNamespace(release=RELEASE), {"release": RELEASE, "heap": "3G"}

    def start(self, boot):
        FakeGens.boots.append(boot.kind)
        folder = os.path.join(provision.pack_dir(self.cache, RELEASE), boot.world)
        self.gen = FakeGen(folder, boot.kind, boot.offsets)
        if boot.kind == "gen":
            self.gen.boot()
        return self.gen

    def sampler(self, seed, threads):
        def make():
            FakeGens.boots.append("sampler")
            self.gen = FakeGen(None, "sampler", None)
            return FakeSampler(float(os.environ.get(DELAY_ENV, 0)))

        return make

    def stop(self):
        code, self.gen = (FakeGens.exit_code if self.gen else None), None
        return code


def settings(cache):
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, "generator-settings.json"), "w") as f:
        f.write('{"release": "2.9.0", "java": "java", "heap": "3G"}')


def install():
    """The fake generator in a CLI or UI process the tests start; like conftest's no_host_jvms, the host's JVMs are not its."""
    cli_search.Generators = FakeGens
    generator.jvm_headroom = lambda: 0


def use(monkeypatch):
    FakeGens.boots = []
    FakeGens.exit_code = 0
    monkeypatch.setattr(cli_search, "Generators", FakeGens)
