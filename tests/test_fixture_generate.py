"""M5 on a COPY of the real fixture server tree with the real generator (RWG2_FIXTURE, RWG2_CACHE holding a set-up generator):
generated surroundings, harvest, smoothing and post-check against the POC's live result, the refresh, power cuts in generate and
smooth, and the NEID and OPENSIMPLEX variants. Every test leaves the copy as it found it. Runs on a machine with the RAM for a
3 GB generator (the user's PC)."""

import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import pytest
from conftest import Migration, age, golden, search_job, tree

from rwg2migrator import (
    area,
    chunks,
    cli,
    cli_search,
    generator,
    genworld,
    jobs,
    nbt,
    provision,
    region,
    smooth,
    staging,
    transplant,
    world,
)
from rwg2migrator.ground import column_ground, ground_ids

FIXTURE, CACHE = os.environ.get("RWG2_FIXTURE"), os.environ.get("RWG2_CACHE")
pytestmark = [pytest.mark.fixture, pytest.mark.skipif(not (FIXTURE and CACHE), reason="set RWG2_FIXTURE (a COPY) and RWG2_CACHE")]
NOT_OURS = ("session.lock", "rwg2-migration.json")
TIMEOUT = 3600
NEID_CORE = os.environ.get("RWG2_NEID_CORE", "11,12:13,14")
SAME = {"ids": 0.0025, "meta": 0.0012, "biome_columns": 30, "ground": 0.005}
KILLED = (-9, 1, 15)


def chunks_of(world_dir):
    """{(cx, cz): (compression byte + payload, timestamp)} from the region headers, independent of region.py."""
    out = {}
    for p in sorted(Path(world_dir, "region").glob("r.*.*.mca")):
        data, (rx, rz) = p.read_bytes(), (int(v) for v in p.name.split(".")[1:3])
        for i in range(1024):
            loc = int.from_bytes(data[4 * i : 4 * i + 3], "big")
            if loc:
                start, length = loc * 4096, int.from_bytes(data[loc * 4096 : loc * 4096 + 4], "big")
                ts = int.from_bytes(data[4096 + 4 * i : 4100 + 4 * i], "big")
                out[(rx * 32 + i % 32, rz * 32 + i // 32)] = (data[start + 4 : start + 4 + length], ts)
    return out


def level(entry):
    return nbt.parse(region.inflate(len(entry[0]).to_bytes(4, "big") + entry[0]))[1]["Level"][1]


def peak_mb():
    info = psutil.Process().memory_info()
    return round(max(getattr(info, "peak_wset", 0), info.rss) / 2**20)


@pytest.fixture(scope="module")
def fx(tmp_path_factory, green1_pick):
    work = Path(FIXTURE, staging.WORK)
    existed = work.exists()
    m = Migration(Path(FIXTURE), tmp_path_factory.mktemp("m5"), cand=green1_pick, cache=CACHE)
    m.world0, m.config0 = tree(m.world), tree(os.path.join(m.root, "config"))
    m.base = Path(m.data).parent
    yield m
    assert tree(m.world) == m.world0 and tree(os.path.join(m.root, "config")) == m.config0
    if not existed:
        shutil.rmtree(work)


def report(job):
    return jobs.read_json(os.path.join(job.folder, staging.POSTCHECK, staging.REPORT))


def record(job, step):
    return jobs.read_json(os.path.join(job.folder, step, staging.RECORD))


def _keep(m, job, name):
    """Copy what later comparisons need before the next preview supersedes this one and deletes its working copy."""
    dest = m.base / name
    shutil.copytree(os.path.join(job.data["args"]["staging"], "region"), dest / "region")
    (dest / "final.json").write_text(json.dumps(staging.final(job)))
    for step in (genworld.HARVEST, "smooth"):
        (dest / f"{step}.json").write_text(json.dumps(record(job, step)))
    return dest


@pytest.fixture(scope="module")
def ref(fx):
    t0 = time.monotonic()
    code, job = fx.generate()
    seconds = time.monotonic() - t0
    assert code == 0
    print(f"preview with generation on the fixture: {seconds:.1f} s; tool peak memory {peak_mb()} MB")
    job.kept_copy = _keep(fx, job, "ref")
    job.seconds = seconds
    return job


def _facing_inner(have, bare, inner, k):
    """Log, leaf and plant voxels of the undecorated outer chunks, in the halves that face the decorated ring and elsewhere: only
    the population of a decorated neighbour can have put the first ones there."""
    half, n = chunks.SIDE // 2, Counter()
    for c in bare:
        mask = np.zeros((chunks.SIDE, chunks.SIDE), bool)
        mask[:, :half] |= (c[0] - 1, c[1]) in inner
        mask[:half, :] |= (c[0], c[1] - 1) in inner
        deco = np.isin(smooth.lookup(k.cat, chunks.block_ids(level(have[c]))), (smooth.LOG, smooth.LEAVES, smooth.PLANT))
        n.update(chunks_facing=int(mask.any()), facing=int(deco[:, mask].sum()), elsewhere=int(deco[:, ~mask].sum()))
    return dict(n)


def _harvested_whole(job, kept):
    """Every chunk the gen boot made outside kept is in staging (so the user's game never decorates a decorated one again) and so
    is every chunk a decorated one decorated into, byte-identical unless smoothed; the undecorated ones are only at the edge and
    hold what their decorated neighbours put into them. Returns (made, undecorated, their decoration voxels by half)."""
    h, st, gen_have = record(job, genworld.HARVEST), job.data["args"]["staging"], chunks_of(genworld.generated(job)["folder"])
    have, r = chunks_of(st), job.data["args"]["ring"]
    made, ring = set(gen_have) - kept, area.dilate(kept, r) - kept
    smoothed = set(map(tuple, record(job, "smooth")["chunks_changed"]))
    assert ring <= made <= set(have) and h["not_generated"] == 0 and all(have[c] == gen_have[c] for c in made - smoothed)
    deco = {c for c in made if level(gen_have[c])["TerrainPopulated"][1]}
    assert {(x + dx, z + dz) for x, z in deco for dx in (0, 1) for dz in (0, 1)} - kept <= set(have)
    bare = made - deco
    assert not bare & area.dilate(kept, r - 1) and len(bare) == h["not_decorated"] and h["beyond_ring"] == len(made - ring)
    spill = _facing_inner(have, bare, deco, smooth.classify(world.registry(world.read_level(st))))
    assert spill["facing"] > 0 or not bare
    return made, bare, spill


def test_harvest_registry_coverage_core_files_and_entities(fx, ref):
    a = json.loads(Path(fx.area).read_text())
    core, kept = {tuple(c) for c in a["core"]}, {tuple(c) for c in a["kept"]}
    h, g = record(ref, genworld.HARVEST), genworld.generated(ref)
    st = ref.data["args"]["staging"]
    have, src = chunks_of(st), chunks_of(fx.world)
    assert h["registry"]["entries"] == h["registry"]["shared"]
    near = area.dilate(kept, genworld.COVERAGE)
    assert near <= set(have) and all(level(have[c])["TerrainPopulated"][1] for c in near)
    assert sum(have[c] == src[c] for c in core) == len(core) == 908
    w, s = fx.world0, tree(st)
    outside = sorted(rel for rel in w if not rel.startswith("region/") and w[rel] != s.get(rel))
    assert outside == ["data/Stronghold.dat", "data/Village.dat", "level.dat", "level.dat_old"]
    assert not {rel for rel in s if not rel.startswith("region/")} - set(w)
    ids = lambda lv: {(e["UUIDMost"][1], e["UUIDLeast"][1]) for e in lv["Entities"][1][1] if "UUIDMost" in e}  # noqa: E731
    kept_ids = set().union(*(ids(level(have[c])) for c in kept if c in have))
    harvested = set(map(tuple, h["chunks"]))
    crossed = kept_ids & set().union(*(ids(level(have[c])) for c in harvested))
    assert crossed == set()
    made, bare, spill = _harvested_whole(ref, kept)
    print(
        f"harvest: registry {h['registry']}; {h['harvested']} chunks harvested: every chunk the gen boot made outside kept "
        f"({len(made)}; {h['ring_requested']} in the ring, {h['beyond_ring']} past it), byte-identical to the gen-world unless smoothed; "
        f"{len(bare)} undecorated, all at the edge; decoration voxels their decorated neighbours put into them, now kept: {spill}; "
        f"{len(kept_ids)} entity UUIDs in kept chunks, {h['entities']} in harvested, 0 shared"
    )
    print(
        f"gen boot: {g['seconds']}; peak working set {g['peak_working_set_mb']} MB at -Xmx{g['heap']}; heap used {g['heap_used_mb']} MB; "
        f"totals {g['totals']}; noise {g['noise']}; trees clipped at the band edge {h['trees_clipped']}"
    )


def test_game_vs_render_seams_and_smoothing_match_the_poc(ref):
    p, sm = report(ref), record(ref, "smooth")
    pin = json.loads(golden("smooth-report.json").read_text())
    nt = p["new_terrain"]
    assert nt["median"] == 0 and nt["within1_pct"] >= 98
    before, after = p["seams"]["before"]["kept_border"], p["seams"]["after"]["kept_border"]
    assert before["pairs"] == pin["before"]["old_kept_border"]["pairs"] == 3008
    assert abs(before["median"] - 1) <= 1 and abs(before["p90"] - 11) <= 1
    assert abs(before["over10_pct"] - 10.3) <= 0.3 and abs(before["over30_pct"] - 0.6) <= 0.3
    assert after["median"] == 0 and after["p90"] <= 1 and after["over10_pct"] == 0 and after["max"] <= 2
    assert abs(p["columns_reshaped"] - pin["columns_changed"]) <= 0.05 * pin["columns_changed"]
    assert p["core_columns_changed"] == 0 and sm["chunks_changed_in_core"] == 0 and sm["new_fluid_walls"] == 0
    print(f"post-check: {json.dumps(p)}")
    print(f"smoothing: {json.dumps({k: v for k, v in sm.items() if k not in ('chunks_changed', 'changed')})}")


def _without(world_dir, cells):
    for path in transplant.region_files(world_dir):
        rx, rz = region.coords(path)
        recs = region.records(Path(path).read_bytes())
        keep = {i: r for i, r in recs.items() if region.slot_pos(rx, rz, i) not in cells}
        if len(keep) < len(recs):
            Path(path).write_bytes(region.build(keep))


def _boot_on(job, world_dir, run, how=((), None)):
    """Boot the generator on a copy of world_dir's overworld without the chunks in drop (spawn at the centre of kept, as in the
    gen-world), call run(gen) and stop; how = (drop, offsets, None for the job's); (run's result, exit code, the copy's chunks
    afterwards)."""
    d = job.data["args"]
    drop, offsets = how
    gens = cli_search.Generators(d["cache"], type("Args", (), {"accept_eula": False})(), print, d["config_dir"])
    _, settings = gens.context()
    name = "oracle-" + job.id
    dest = os.path.join(provision.pack_dir(d["cache"], settings["release"]), name)
    region.rmtree(dest)
    for part in ("region", "data"):
        region.copy_tree(os.path.join(world_dir, part), os.path.join(dest, part))
    _without(dest, set(drop))
    kept = {tuple(c) for c in jobs.read_json(os.path.join(job.folder, "area.json"))["kept"]}
    genworld._level(os.path.join(world_dir, "level.dat"), os.path.join(dest, "level.dat"), genworld.centre(kept))
    with generator.lock(d["cache"]):
        gen = gens.start(generator.Boot(kind=generator.GEN, world=name, offsets=tuple(offsets or d["config"])))
        try:
            out = run(gen)
        finally:
            code = gens.stop()
    after = chunks_of(dest)
    region.rmtree(dest)
    return out, code, after


def _ids_differ(a, b, cells):
    return sum(int((chunks.block_ids(level(a[c])) != chunks.block_ids(level(b[c]))).sum()) for c in cells) / (65536 * len(cells))


def test_the_users_game_decorates_the_outer_layer_as_one_continuous_generation(fx, ref):
    """One more layer generated on the working copy (the user's game after Apply) gives the outer layer that one continuous
    generation gives (the gen-world generating further, as the POC's live pregen did). Left out of the harvest, the undecorated
    outer chunks come back with the halves that face the ring bare: nothing decorates them again."""
    kept, r = {tuple(c) for c in json.loads(Path(fx.area).read_text())["kept"]}, ref.data["args"]["ring"]
    inner, layers = area.dilate(kept, r - 1), area.dilate(kept, r + 1) - area.dilate(kept, r - 1)
    st = ref.data["args"]["staging"]
    have = chunks_of(st)
    bare = sorted(c for c in layers if c in have and not level(have[c])["TerrainPopulated"][1])
    k = smooth.classify(world.registry(world.read_level(st)))

    def grow(gen):
        totals = generator.generate(gen, ref.data["args"]["config"], layers)
        gen.call("save")
        return totals

    got = {}
    for name, (src, drop) in {"continuous": (genworld.generated(ref)["folder"], ()), "harvested": (st, ()), "left out": (st, bare)}.items():
        totals, code, after = _boot_on(ref, src, grow, (drop, None))
        assert code == 0 and all(level(after[c])["TerrainPopulated"][1] for c in bare), (name, totals)
        got[name] = (after, _facing_inner(after, bare, inner, k), totals)
    cont, harv, left = (got[n] for n in ("continuous", "harvested", "left out"))
    same, off = _ids_differ(cont[0], harv[0], bare), _ids_differ(cont[0], left[0], bare)
    print(
        f"outer layer ({len(bare)} chunks undecorated at harvest) after one more layer: continuous {cont[1]}, harvested {harv[1]}, "
        f"left out {left[1]}; block ids differing from the continuous run: harvested {same:.4%}, left out {off:.4%}"
    )
    assert same <= SAME["ids"] and left[1]["facing"] < cont[1]["facing"] / 2


def _decoded(world_dir, cells, ok):
    """{pos: (ids, meta, biomes, ground, populated)} of the cells in world_dir."""
    out = {}
    for pos, entry in chunks_of(world_dir).items():
        if pos in cells:
            lv = level(entry)
            ids = chunks.block_ids(lv)
            meta = np.zeros_like(ids)
            for sec in chunks.sections(lv):
                y = sec["Y"][1] * 16
                meta[y : y + 16] = chunks.decode_section(sec).meta.reshape(16, 16, 16)
            out[pos] = (ids, meta, chunks.biomes(lv), column_ground(ids, chunks.heightmap(lv), ok), lv["TerrainPopulated"][1])
    return out


def same_result(kept_copy, job, ok):
    """The rule: the same chunks present and decorated; in the generated and smoothed chunks at most 0.25% of block ids,
    0.12% of meta, 30 biome columns and 0.5% of ground columns differ; identical bytes everywhere else."""
    st = job.data["args"]["staging"]
    a, b = chunks_of(kept_copy), chunks_of(st)
    assert set(a) == set(b)
    fa, fb = json.loads((kept_copy / "final.json").read_text()), staging.final(job)
    assert {k: v for k, v in fa.items() if not k.startswith("region/")} == {k: v for k, v in fb.items() if not k.startswith("region/")}
    loose = set(map(tuple, record(job, genworld.HARVEST)["chunks"])) | set(map(tuple, record(job, "smooth")["chunks_changed"]))
    loose |= set(map(tuple, json.loads((kept_copy / "harvest.json").read_text())["chunks"]))
    loose |= set(map(tuple, json.loads((kept_copy / "smooth.json").read_text())["chunks_changed"]))
    assert all(a[c] == b[c] for c in set(a) - loose)
    da, db = _decoded(kept_copy, loose, ok), _decoded(st, loose, ok)
    n = Counter()
    for c in loose:
        (ia, ma, ba, ga, pa), (ib, mb, bb, gb, pb) = da[c], db[c]
        assert pa == pb
        both = np.isnan(ga) & np.isnan(gb)
        n.update(voxels=ia.size, ids=int((ia != ib).sum()), meta=int((ma != mb).sum()), biome_columns=int((ba != bb).sum()))
        n.update(columns=ga.size, ground=int((~((ga == gb) | both)).sum()))
    got = {
        "ids": n["ids"] / n["voxels"],
        "meta": n["meta"] / n["voxels"],
        "biome_columns": n["biome_columns"],
        "ground": n["ground"] / n["columns"],
    }
    print(f"compared {len(loose)} generated or smoothed chunks: {got}")
    assert all(got[k] <= SAME[k] for k in SAME), got


def _cli(m, *argv, kill=None):
    env = {k: v for k, v in os.environ.items() if k != jobs.KILL_ENV} | ({jobs.KILL_ENV: kill} if kill else {})
    args = [sys.executable, "-I", "-m", "rwg2migrator", *map(str, argv), *m.opts()]
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=TIMEOUT)


def _killed_then_resumed(fx, kill):
    r = _cli(fx, "preview", fx.root, "--area", fx.area, "--search", fx.search.id, kill=kill)
    assert r.returncode != 0, r.stderr[-2000:]
    job = jobs.all_jobs(fx.data)[-1]
    return job, r


def test_kill_inside_generate_then_resume_builds_the_gen_world_again(fx, ref):
    ok = ground_ids(world.registry(world.read_level(fx.world)))
    built = genworld.generated(ref)["built"]["regions"]
    job, r = _killed_then_resumed(fx, f"{genworld.GENERATE}:{built + 4}")
    assert genworld.generated(job) is None, "the kill came after the generate step finished"
    folder = genworld.folder(CACHE, genworld.generated(ref)["release"], job.id)
    half = sorted(os.listdir(os.path.join(folder, "region")))
    t0 = time.monotonic()
    r = _cli(fx, "jobs", "resume", job.id)
    assert r.returncode == 0, r.stderr[-3000:]
    print(f"killed in generate (exit {r.returncode}); the half-built gen-world held {half}; resume took {time.monotonic() - t0:.1f} s")
    assert genworld.generated(job)["exit_code"] == 0
    same_result(ref.kept_copy, job, ok)
    _keep(fx, job, "after-generate-kill")


def test_kill_inside_smooth_then_resume_resets_to_the_copy(fx, ref):
    ok = ground_ids(world.registry(world.read_level(fx.world)))
    sm = record(ref, "smooth")
    point = 1 + len(sm["chunks_changed"]) + max(1, len(sm["changed"]) // 2)
    job, _ = _killed_then_resumed(fx, f"smooth:{point}")
    assert genworld.generated(job) is not None and record(job, "smooth") is None
    st = job.data["args"]["staging"]
    Path(st, "planted-by-test.txt").write_text("x")
    r = _cli(fx, "jobs", "resume", job.id)
    assert r.returncode == 0, r.stderr[-3000:]
    assert not Path(st, "planted-by-test.txt").exists()
    same_result(ref.kept_copy, job, ok)


def _rewrite_chunk(world_dir, pos, change):
    path = Path(world_dir, "region", f"r.{pos[0] // 32}.{pos[1] // 32}.mca")
    raw = path.read_bytes()
    recs = region.records(raw)
    i = region.slot_index(*pos)
    name, root = nbt.parse(region.inflate(recs[i][0]))
    change(root["Level"][1])
    recs[i] = (region.deflate(nbt.write(name, root)), recs[i][1] + 60)
    path.write_bytes(region.build(recs))
    return path, raw


def _game_generates(job, m, offsets=None):
    """The user's game, after Apply, generates one more layer of chunks past everything the applied world holds, with the job's
    offsets or with others (the 4 lines wiped by a GTNH update); the new chunks are saved into the world. Returns how many."""
    have = chunks_of(m.world)
    layer = area.dilate(set(have), 1) - set(have)
    config = list(offsets or job.data["args"]["config"])

    def grow(gen):
        generator.generate(gen, config, layer)
        gen.call("save")

    _, code, after = _boot_on(job, m.world, grow, ((), offsets))
    assert code == 0
    new = {c: e for c, e in after.items() if c not in have}
    by_region = {}
    for (cx, cz), (payload, ts) in new.items():
        by_region.setdefault((cx // region.SIDE, cz // region.SIDE), {})[region.slot_index(cx, cz)] = (
            len(payload).to_bytes(4, "big") + payload,
            ts,
        )
    for (rx, rz), add in by_region.items():
        path = Path(m.world, "region", f"r.{rx}.{rz}.mca")
        path.write_bytes(region.build((region.records(path.read_bytes()) if path.exists() else {}) | add))
    return len(new)


def _verify(m, capsys):
    capsys.readouterr()
    code = cli.main(["verify", m.root, *m.opts()])
    return code, capsys.readouterr().out


def test_apply_refresh_after_play_reuses_the_gen_world(fx, capsys):
    code, job = fx.generate()
    assert code == 0
    before, gen0, render0 = report(job), genworld.generated(job), jobs.read_json(os.path.join(job.folder, "render", jobs.MANIFEST_FILE))
    a = json.loads(Path(fx.area).read_text())
    pos = tuple(a["core"][len(a["core"]) // 2])
    player = sorted(Path(fx.world, "playerdata").glob("*.dat"))[0]
    raw_player = player.read_bytes()
    region_path, raw_region = _rewrite_chunk(
        fx.world, pos, lambda lv: lv.__setitem__("InhabitedTime", [nbt.LONG, lv["InhabitedTime"][1] + 1200])
    )
    try:
        name, root = nbt.parse(nbt.unpack(raw_player))
        root["XpTotal"] = [nbt.INT, root.get("XpTotal", [nbt.INT, 0])[1] + 7]
        player.write_bytes(gzip.compress(nbt.write(name, root)))
        age(fx.world)
        capsys.readouterr()
        t0 = time.monotonic()
        assert fx.apply(job) == 0
        downtime = time.monotonic() - t0
        err = capsys.readouterr().err
        refresh = next(line for line in err.splitlines() if "apply: refresh" in line)
        assert genworld.generated(job) == gen0 and jobs.read_json(os.path.join(job.folder, "render", jobs.MANIFEST_FILE)) == render0
        after = report(job)
        assert {k: v for k, v in after.items() if k != "core_identical"} == {k: v for k, v in before.items() if k != "core_identical"}
        code, out = _verify(fx, capsys)
        assert code == 0 and "Chunks generated since Apply: 0" in out and "Seam at the kept border now: median 0, p90 1" in out, out
        with capsys.disabled():
            print(f"{refresh}; whole Apply (the downtime) {downtime:.1f} s")
        right = _game_generates(job, fx)
        code, out = _verify(fx, capsys)
        with capsys.disabled():
            print(f"verify after the game generated {right} chunks with the 4 lines: exit {code}; {out}")
        assert code == 0 and f"Chunks generated since Apply: {right:,}" in out and "No problem found." in out
        wrong = _game_generates(job, fx, (0, 0, 0, 0))
        code, out = _verify(fx, capsys)
        with capsys.disabled():
            print(f"verify after {wrong} more chunks generated without them: exit {code}; {out}")
        assert code == 2 and "do not follow the preview" in out
        assert fx.rollback() == 0
    finally:
        region_path.write_bytes(raw_region)
        player.write_bytes(raw_player)
        age(fx.world)


def _oracle(job, cells):
    """Dump each cell of the smoothed working copy through the probe's chunk op."""
    out, code, _ = _boot_on(job, job.data["args"]["staging"], lambda gen: {(x, z): gen.call("chunk", x=x, z=z)["data"] for x, z in cells})
    return out, code


def _ours(entry):
    lv = level(entry)
    ids = chunks.block_ids(lv)
    meta = np.zeros_like(ids)
    for sec in chunks.sections(lv):
        meta[sec["Y"][1] * 16 : sec["Y"][1] * 16 + 16] = chunks.decode_section(sec).meta.reshape(16, 16, 16)
    te = np.zeros(ids.shape, bool)
    for t in lv.get("TileEntities", [0, [0, []]])[1][1]:
        te[t["y"][1], t["z"][1] & 15, t["x"][1] & 15] = True
    return ids, meta, chunks.biomes(lv), te


def _game(raw):
    ids = np.frombuffer(raw[: 4 * 65536], ">i4").astype(np.int64).reshape(256, 16, 16)
    meta = np.frombuffer(raw[4 * 65536 : 6 * 65536], ">i2").astype(np.int64).reshape(256, 16, 16) & 0xFFFF
    return ids, meta, np.frombuffer(raw[6 * 65536 :], ">i2").reshape(16, 16)


def test_neid_variant_the_game_reads_every_smoothed_neid_chunk_as_we_decode_it(fx, tmp_path, green1_pick):
    """Every voxel of every smoothed NEID chunk, as the game loads it, equals our decode. Blocks that carry a tile entity are
    compared apart: GT rewrites its old ore blocks when a chunk loads (SESSION_STATE §4), in chunks we never touched too, so
    some NEID chunks the smoothing did not change are dumped as the control."""
    out = tmp_path / "neid-area.json"
    assert cli.main(["area", fx.root, "--add", NEID_CORE, "--margin", "1", "--out", str(out), "--cache", CACHE]) == 0
    m = Migration.__new__(Migration)
    m.__dict__ |= fx.__dict__ | {"area": str(out)}
    m.search = search_job(fx.data, str(out), world.locate(fx.world).config, green1_pick)
    code, job = m.generate()
    assert code == 0
    sm = record(job, "smooth")
    made, bare, spill = _harvested_whole(job, {tuple(c) for c in json.loads(out.read_text())["kept"]})
    h = record(job, genworld.HARVEST)
    print(
        f"NEID variant harvest: {len(made)} chunks made outside kept, all in staging ({h['beyond_ring']} past the ring, "
        f"{h['beyond_ring_decorated']} of those decorated); {len(bare)} undecorated at the edge, holding {spill}"
    )
    have = chunks_of(job.data["args"]["staging"])
    is_neid = lambda c: chunks.chunk_format(level(have[c])) == chunks.NEID  # noqa: E731
    smoothed = [tuple(c) for c in sm["chunks_changed"] if is_neid(tuple(c))]
    control = sorted(c for c in map(tuple, json.loads(out.read_text())["core"]) if is_neid(c))
    assert smoothed and control, (smoothed, control)
    dumps, code = _oracle(job, smoothed + control)
    assert code == 0
    reg = world.registry(world.read_level(job.data["args"]["staging"]))
    name = lambda i: reg.get(int(i), str(int(i)))  # noqa: E731
    seen = {}
    for c in smoothed + control:
        ids, meta, biomes, te = _ours(have[c])
        g_ids, g_meta, g_biomes = _game(dumps[c])
        bad = ~te & ((g_ids != ids) | (g_meta != meta))
        where = [tuple(int(v) for v in p) for p in np.argwhere(bad)[:5]]
        pairs = Counter(
            (name(a), int(ma), name(b), int(mb)) for a, ma, b, mb in zip(ids[bad], meta[bad], g_ids[bad], g_meta[bad], strict=True)
        )
        assert not bad.any(), (c, int(bad.sum()), where, pairs.most_common(6))
        assert np.array_equal(g_biomes, biomes), (c, int((g_biomes != biomes).sum()))
        moved = te & ((g_ids != ids) | (g_meta != meta))
        seen[c] = (int(te.sum()), Counter((name(a), name(b)) for a, b in zip(ids[moved], g_ids[moved], strict=True)))
    voxels = 65536 * len(smoothed)
    print(
        f"NEID variant: {len(smoothed)} smoothed NEID chunks ({voxels:,} voxels) and {len(control)} untouched ones dumped by the game "
        f"equal our decode on every voxel without a tile entity; tile-entity blocks the game rewrote on load, smoothed chunks: "
        f"{sum(sum(seen[c][1].values()) for c in smoothed)} of {sum(seen[c][0] for c in smoothed)}, untouched: "
        f"{sum(sum(seen[c][1].values()) for c in control)} of {sum(seen[c][0] for c in control)}; "
        f"as (ours, game's): {dict(sum((seen[c][1] for c in seen), Counter()).most_common(6))}"
    )
    print(f"NEID variant seams: {report(job)['seams']}")


def test_opensimplex_variant_deletes_rwg_data_and_the_gen_boot_uses_perlin(fx):
    path = Path(fx.world, "data", "rwg_data.dat")
    raw = path.read_bytes()
    name, root = nbt.parse(nbt.unpack(raw))
    root["data"][1]["noiseImplementation"] = [nbt.STRING, "OPENSIMPLEX"]
    try:
        path.write_bytes(gzip.compress(nbt.write(name, root)))
        age(fx.world)
        code, job = fx.generate()
        assert code == 0
        t, g, p = record(job, staging.TRANSPLANT), genworld.generated(job), report(job)
        assert t["rwg_data"] == "deleted" and not Path(job.data["args"]["staging"], "data", "rwg_data.dat").exists()
        assert g["noise"] in world.PERLIN
        assert p["new_terrain"]["median"] == 0 and p["new_terrain"]["within1_pct"] >= 98
        print(f"OPENSIMPLEX variant: gen boot noise {g['noise']}; game vs render {p['new_terrain']}")
    finally:
        path.write_bytes(raw)
        age(fx.world)
