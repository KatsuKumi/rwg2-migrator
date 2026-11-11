import gzip
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import psutil
import pytest

from rwg2migrator import chunks, nbt, procs, region, world

REGISTRY = {
    0: "minecraft:air",
    1: "minecraft:stone",
    2: "minecraft:grass",
    3: "minecraft:dirt",
    9: "minecraft:water",
    12: "minecraft:sand",
    17: "minecraft:log",
    18: "minecraft:leaves",
    31: "minecraft:tallgrass",
    78: "minecraft:snow_layer",
    90: "minecraft:portal",
    4000: "gregtech:gt.blockores2",
}
SURFACE = 64
OLD = 3600
LAZY_POS = (70, 45)
LAZY_LAYOUTS = (
    ("BlockLight", "Blocks", "Y", "Data2", "SkyLight", "Add", "Data", "Data1High", "BlocksB2Hi"),
    ("Blocks", "BlockLight", "Y", "SkyLight", "Add", "Data", "BlocksB2Hi"),
    ("SkyLight", "BlockLight", "Add", "Data", "Blocks", "Y"),
    ("SkyLight", "BlockLight", "Data", "Blocks", "Y"),
)


def golden(name):
    """A file derived from the author's test world, kept outside the repo in RWG2_FIXTURE/golden; the test skips without it."""
    fixture = os.environ.get("RWG2_FIXTURE")
    if not fixture:
        pytest.skip(f"set RWG2_FIXTURE: {name} is test-world data kept in RWG2_FIXTURE/golden, not in the repo", allow_module_level=True)
    path = Path(fixture, "golden", name)
    if not path.is_file():
        pytest.skip(f"{path} is missing: copy the test world's golden files into RWG2_FIXTURE/golden", allow_module_level=True)
    return path


@pytest.fixture(scope="session")
def green1_pick():
    """The test world's green #1 as a search candidate: its offsets and nearest Tainted Land, from RWG2_FIXTURE/golden."""
    pin = json.loads(golden("green1.json").read_text())
    return {"config": pin["config"], "taint": {k: pin["taint"][k] for k in ("blocks", "point", "direction")}}


def _lazy_array(key, rng):
    bits = chunks.EID_ARRAYS.get(key, (None, 0, chunks.NIBBLE_BITS))[2]
    return tag(nbt.BYTE_ARRAY, rng.bytes(chunks.VOLUME * bits // chunks.BYTE_BITS))


def lazy_chunk():
    """A chunk whose sections hold 5, 6, 7 and all 9 EndlessIDs arrays in the key orders GTNH 2.9 writes, as (raw NBT, parsed root)."""
    rng = np.random.default_rng(29)
    secs = [{k: tag(nbt.BYTE, y) if k == "Y" else _lazy_array(k, rng) for k in keys} for y, keys in enumerate(LAZY_LAYOUTS)]
    root = chunk_root(*LAZY_POS)
    root["Level"][1]["Sections"] = tag(nbt.LIST, [nbt.COMPOUND, secs])
    raw = nbt.write("", root)
    return raw, nbt.parse(raw)[1]


def tag(t, v):
    return [t, v]


def compound(**items):
    return tag(nbt.COMPOUND, dict(items))


def string(s):
    return tag(nbt.STRING, s)


def write_nbt(path, value, compress=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = nbt.write("", value)
    with open(path, "wb") as f:
        f.write(gzip.compress(raw, mtime=0) if compress else raw)


def level_dat(generator="RWG", dreamcraft="2.9.99", extra_mods=(), player=None):
    from rwg2migrator import world

    pack = world.releases()["releases"]["2.9.0"]["mods"]
    mods = {**pack, "dreamcraft": dreamcraft, **dict(extra_mods)}
    mod_list = [{"ModId": string(m), "ModVersion": string(v)} for m, v in mods.items()]
    item_data = [{"K": string("\x01" + n), "V": tag(nbt.INT, i)} for i, n in REGISTRY.items()]
    data = {
        "generatorName": string(generator),
        "generatorVersion": tag(nbt.INT, 0),
        "generatorOptions": string(""),
        "RandomSeed": tag(nbt.LONG, 1234567890123),
        "SpawnX": tag(nbt.INT, 8),
        "SpawnY": tag(nbt.INT, 64),
        "SpawnZ": tag(nbt.INT, 8),
        "LevelName": string("Test"),
        "LastPlayed": tag(nbt.LONG, 1700000000000),
    }
    if player:
        data["Player"] = tag(nbt.COMPOUND, player)
    fml = {"ModList": tag(nbt.LIST, [nbt.COMPOUND, mod_list]), "ItemData": tag(nbt.LIST, [nbt.COMPOUND, item_data])}
    return {"FML": tag(nbt.COMPOUND, fml), "Data": tag(nbt.COMPOUND, data)}


def player_dat(x, z, dim=0, bed=None):
    pos = [struct.pack(">d", v) for v in (x, SURFACE, z)]
    out = {"Pos": tag(nbt.LIST, [nbt.DOUBLE, pos]), "Dimension": tag(nbt.INT, dim)}
    if bed:
        out |= {"SpawnX": tag(nbt.INT, bed[0]), "SpawnY": tag(nbt.INT, SURFACE), "SpawnZ": tag(nbt.INT, bed[1])}
    return out


def voxels(seed):
    rng = np.random.default_rng(seed)
    return chunks.Voxels(
        rng.integers(0, 1 << 16, chunks.VOLUME).astype(np.uint16),
        rng.integers(0, 1 << 16, chunks.VOLUME).astype(np.uint16),
        rng.integers(0, 16, chunks.VOLUME).astype(np.uint8),
        rng.integers(0, 16, chunks.VOLUME).astype(np.uint8),
    )


def terrain(top=SURFACE, portal=0):
    """Sections 0..3 of stone up to top-1 with grass on top and tall grass above; `portal` portal blocks in section 4."""
    ids = np.zeros((chunks.HEIGHT, 16, 16), np.uint16)
    ids[: top - 1] = 1
    ids[top - 1] = 2
    ids[top] = 31
    ids[80].reshape(-1)[:portal] = 90
    return ids


def chunk_root(cx, cz, fmt=chunks.ENDLESSIDS, spec=None):
    """spec may set populated (1), inhabited (0), tes (tile entity ids, or (id, {extra tags})) and ids (a block id stack)."""
    spec = spec or {}
    ids = spec.get("ids", terrain())
    populated, inhabited, tes = spec.get("populated", 1), spec.get("inhabited", 0), spec.get("tes", ())
    secs = []
    for y in range(chunks.HEIGHT // chunks.SECTION):
        part = ids[y * 16 : (y + 1) * 16].reshape(-1)
        if part.any():
            zeros = np.zeros(chunks.VOLUME, np.uint8)
            secs.append(chunks.encode_section(fmt, y, chunks.Voxels(part, np.zeros(chunks.VOLUME, np.uint16), zeros + 15, zeros)))
    hm = np.full(256, SURFACE + 1, ">i4").tobytes()
    te_list = [
        {
            "id": string(t if isinstance(t, str) else t[0]),
            "x": tag(nbt.INT, cx * 16),
            "y": tag(nbt.INT, SURFACE),
            "z": tag(nbt.INT, cz * 16),
        }
        | ({} if isinstance(t, str) else t[1])
        for t in tes
    ]
    level = {
        "xPos": tag(nbt.INT, cx),
        "zPos": tag(nbt.INT, cz),
        "TerrainPopulated": tag(nbt.BYTE, populated),
        "InhabitedTime": tag(nbt.LONG, inhabited),
        "HeightMap": tag(nbt.INT_ARRAY, hm),
        "Sections": tag(nbt.LIST, [nbt.COMPOUND, secs]),
        chunks.BIOME_KEY[fmt]: tag(nbt.BYTE_ARRAY, chunks.encode_biomes(fmt, np.full(256, 4))),
        "TileEntities": tag(nbt.LIST, [nbt.COMPOUND if te_list else nbt.END, te_list]),
        "Entities": tag(nbt.LIST, [nbt.END, []]),
    }
    return {"Level": tag(nbt.COMPOUND, level)}


def write_region(path, roots, ts=1700000000):
    kept = {region.slot_index(cx, cz): (region.deflate(nbt.write("", root)), ts) for (cx, cz), root in roots.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(region.build(kept))


def make_world(root, claims=None, players=None, chunk_specs=None, **level):
    """A small GTNH server folder: server.properties, World/ with level.dat, data/, one region, claims, players."""
    from rwg2migrator import world

    w = os.path.join(root, "World")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "server.properties"), "w") as f:
        f.write("#Minecraft server properties\nlevel-name=World\n")
    write_nbt(os.path.join(w, "level.dat"), level_dat(**level))
    write_nbt(os.path.join(w, "data", "GregTech_OregenPattern.dat"), {"data": compound()})
    write_nbt(
        os.path.join(w, "data", "rwg_data.dat"), {"data": compound(noiseImplementation=string("UNKNOWN"), biomeListVersion=tag(nbt.INT, 1))}
    )
    specs = chunk_specs if chunk_specs is not None else {(x, z): {} for x in range(4) for z in range(4)}
    write_region(
        os.path.join(w, "region", "r.0.0.mca"), {c: chunk_root(*c, kw.get("fmt", chunks.ENDLESSIDS), kw) for c, kw in specs.items()}
    )
    for team, cells in (claims if claims is not None else {"alpha": [(1, 1), (2, 1)]}).items():
        entries = [{"x": tag(nbt.INT, x), "z": tag(nbt.INT, z)} for x, z in cells]
        write_nbt(
            os.path.join(w, world.CLAIMS, team + ".dat"), {"ClaimedChunks": compound(**{"0": tag(nbt.LIST, [nbt.COMPOUND, entries])})}
        )
    names = []
    for uuid, (name, x, z) in (players or {"u1": ("Alice", 24, 24)}).items():
        write_nbt(os.path.join(w, "playerdata", uuid + ".dat"), player_dat(x, z, bed=(x, z)))
        names.append({"name": name, "uuid": uuid})
    with open(os.path.join(root, "usercache.json"), "w") as f:
        json.dump(names, f)
    age(w)
    return w


def world_scene(sc, a, release):
    """The map scene of a world alone, with no match drawn: one window per part of the kept area."""
    from rwg2migrator import area, maps

    return maps.Scene(sc, a, release, None, [tuple(w) for w in area.windows(set(map(tuple, a["kept"])))])


def zero_body(path):
    """Keep a gzip file's 20-byte header and zero the rest, as a power cut can leave it."""
    p = Path(path)
    raw = p.read_bytes()
    p.write_bytes(raw[:20] + bytes(len(raw) - 20))


def age(world_dir, seconds=OLD):
    t = time.time() - seconds
    os.utime(os.path.join(world_dir, "level.dat"), (t, t))


@pytest.fixture(autouse=True)
def no_host_jvms(request, monkeypatch):
    """A JVM of the machine running the tests (a live GTNH server) is not part of a test, so its heap headroom must not decide
    whether a test's smoothing has the memory it needs; the fixture tests boot the real generator and keep the real check."""
    from rwg2migrator import generator

    if not request.node.get_closest_marker("fixture"):
        monkeypatch.setattr(generator, "jvm_headroom", lambda: 0)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "server"
    make_world(str(root))
    return str(root)


@pytest.fixture
def cache(tmp_path):
    return str(tmp_path / "cache")


class Unreadable:
    pid = 4242
    info = {"name": "javaw.exe"}

    def cwd(self):
        raise psutil.AccessDenied(self.pid)


SLEEP = "import time; time.sleep(60)"
STAND_IN_SECONDS = 10


@pytest.fixture
def java_child(tmp_path, server):
    """A real child process named java whose working directory is the server folder. On Windows the stand-in is a copy of this
    Python's launcher, which runs the interpreter as its own child."""
    exe = tmp_path / "bin" / ("java.exe" if os.name == "nt" else "java")
    exe.parent.mkdir()
    if os.name == "nt":
        shutil.copyfile(sys.executable, exe)
        venv = Path(sys.executable).parent.parent / "pyvenv.cfg"
        if venv.is_file():
            shutil.copyfile(venv, exe.parent / "pyvenv.cfg")
    else:
        exe.symlink_to(sys.executable)
    child = subprocess.Popen([str(exe), "-c", SLEEP, "-Dmarker=abc"], cwd=server)
    if not _named_java(child):
        kill_tree(child)
        pytest.fail(f"{exe} did not come up as a process named java")
    yield child
    kill_tree(child)


def _live(proc):
    """The helper as psutil sees it while it lives, else None. Its identity is taken before the handle is asked whether it still
    runs, so the process number cannot have passed to another program in between."""
    try:
        seen = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return None
    return seen if proc.poll() is None else None


def _named_java(child):
    """Whether the child shows up under a java name while it lives: it carries Python's name until its launcher has started."""
    deadline = time.monotonic() + STAND_IN_SECONDS
    while time.monotonic() < deadline and (seen := _live(child)):
        with suppress(psutil.NoSuchProcess):
            if seen.name().lower() in procs.JAVA_NAMES:
                return True
        time.sleep(0.05)
    return False


def kill_tree(proc):
    """End a helper this test started and still holds, with every process it started, and wait until all are gone. A helper
    that has already ended is left alone: its process number may belong to another program by now."""
    root = _live(proc)
    if root is None:
        return
    try:
        below = root.children(recursive=True)
    except psutil.NoSuchProcess:
        below = []
    proc.kill()
    for p in below:
        with suppress(psutil.NoSuchProcess):
            p.kill()
    proc.wait(STAND_IN_SECONDS)
    _, alive = psutil.wait_procs(below, timeout=STAND_IN_SECONDS)
    if alive:
        raise RuntimeError(f"helper processes did not end: {[p.pid for p in alive]}")


def stop_tree(proc, console_closed=False):
    """Stop a helper that leads its own session, with everything it started: its process group gets SIGTERM, or SIGHUP as when
    its console is closed. Windows has neither, so there every process of the tree is ended."""
    if os.name == "nt":
        kill_tree(proc)
    else:
        os.killpg(proc.pid, signal.SIGHUP if console_closed else signal.SIGTERM)


RWG2_CFG = """# Configuration file

general {
    B:"Some Other Key"=true
}

"expert / worldgen migration" {
    I:"Biome Placement Offset X"=0
    I:"Biome Placement Offset Z"=0
    I:"Landmass Placement Offset X"=0
    I:"Landmass Placement Offset Z"=0
}
"""
TAINT_CFG = "general {\n    I:biome_taint_spread=200\n    I:taint_biome_weight=2\n}\n"
OFFSETS = [-16000, 8000, 2048, 6144]
FAR_CHUNK = (100, 100)
STRUCTURES = ("Village", "Stronghold", "Mineshaft", "Temple")


def add_migration_parts(root):
    """A far overworld region with no kept chunk, a nether region, structure files and the two configs the migration edits."""
    w = os.path.join(root, "World")
    write_region(os.path.join(w, "region", "r.3.3.mca"), {FAR_CHUNK: chunk_root(*FAR_CHUNK)})
    write_region(os.path.join(w, "DIM-1", "region", "r.0.0.mca"), {(0, 0): chunk_root(0, 0)})
    for name in STRUCTURES:
        write_nbt(os.path.join(w, "data", name + ".dat"), {"data": compound(Features=compound())})
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    Path(root, "config", "rwg2.cfg").write_text(RWG2_CFG)
    Path(root, "config", "Thaumcraft.cfg").write_text(TAINT_CFG)
    Path(root, "visualprospecting").mkdir(exist_ok=True)
    Path(root, "visualprospecting", "cache.bin").write_bytes(b"veins")
    age(w)
    return w


def tree(folder, skip=()):
    """{relative path: sha256} of every file under folder."""
    from rwg2migrator import jobs, staging

    return {rel: jobs.file_hash(os.path.join(folder, rel)) for rel in staging.walk(folder)[1] if rel not in skip}


def search_job(data, area_file, config_dir, cand=None):
    """A finished search job of the area whose only candidate is cand (default: OFFSETS, no Tainted Land), sampled on config_dir."""
    from rwg2migrator import jobs

    a = json.loads(Path(area_file).read_text())
    job = jobs.create(data, "search", {}, a["world"])
    for rel, value in (
        ("candidates/candidates.json", {"reference": {}, "bans": [], "candidates": [cand or {"config": OFFSETS, "taint": None}]}),
        ("search/inputs.json", {"lmap": {"config": jobs.tree_key(config_dir)}}),
        ("area.json", a),
    ):
        path = Path(job.folder, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    job.update(state=jobs.DONE)
    return job


class Migration:
    """A server folder ready for preview: area.json of the claims (margin 1) and a search job with one candidate."""

    def __init__(self, root, base, **kw):
        """kw: cand (the candidate, default search_job's), world_dir (default root/World), cache (a shared scan cache; scans with
        every worker)."""
        from rwg2migrator import cli

        self.root, self.world = str(root), str(kw.get("world_dir") or os.path.join(str(root), "World"))
        self.data, self.cache, self.area = str(base / "data"), str(kw.get("cache") or base / "cache"), str(base / "area.json")
        self.workers = [] if kw.get("cache") else ["--workers", "1"]
        assert cli.main(["area", self.root, "--claims", "all", "--out", self.area, "--cache", self.cache, *self.workers]) == 0
        config = world.locate(self.world).config
        self.search = search_job(self.data, self.area, config, kw.get("cand"))

    def opts(self, *extra):
        return [*map(str, extra), "--data", self.data, "--cache", self.cache, *self.workers]

    def preview(self, *extra):
        from rwg2migrator import cli, jobs

        before = {j.id for j in jobs.all_jobs(self.data)}
        code = cli.main(["preview", self.root, "--area", self.area, "--search", self.search.id, "--no-generate", *self.opts(*extra)])
        new = [j for j in jobs.all_jobs(self.data) if j.id not in before]
        return code, (new[0] if new else None)

    def generate(self, *extra):
        """A preview that generates the new terrain (through whatever generator the test installed)."""
        from rwg2migrator import cli, jobs

        before = {j.id for j in jobs.all_jobs(self.data)}
        code = cli.main(["preview", self.root, "--area", self.area, "--search", self.search.id, *self.opts(*extra)])
        new = [j for j in jobs.all_jobs(self.data) if j.id not in before]
        return code, (new[0] if new else None)

    def apply(self, preview, *extra):
        from rwg2migrator import cli

        return cli.main(["apply", self.root, "--preview", preview.id, "--yes", "--auto-restart-off", *self.opts(*extra)])

    def rollback(self, *extra):
        from rwg2migrator import cli

        return cli.main(["rollback", self.root, "--yes", "--auto-restart-off", *self.opts(*extra)])


@pytest.fixture
def mig(server, tmp_path):
    add_migration_parts(server)
    return Migration(server, tmp_path, cand={"config": OFFSETS, "taint": {"blocks": 330.1, "point": [0, 0], "direction": "north"}})
