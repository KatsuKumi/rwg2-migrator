"""M6 on a COPY of the real fixture server tree with the real generator (RWG2_FIXTURE, RWG2_CACHE as for M5): trees replanted on
the strips smoothing reshaped, the biome blend at the kept border, the refresh after play that replays the merge list, and the NEID
variant's byte biome arrays. Every test leaves the copy as it found it. Runs on the user's PC."""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import test_fixture_generate as m5
from conftest import Migration, age, search_job
from PIL import Image
from test_fixture_generate import CACHE, NEID_CORE, _ours, chunks_of, level, peak_mb, record, report

from rwg2migrator import area, blend, chunks, cli, genworld, jobs, nbt, plant, region, score, smooth, staging, world
from rwg2migrator.ground import column_ground, ground_ids

pytestmark = m5.pytestmark
fx = m5.fx
POC_LOGS_REMOVED = 2626
GREEN, BROWN, LEAF, STRIP, GREY = (90, 140, 60), (110, 80, 40), (40, 90, 30), (190, 210, 120), (128, 128, 128)
PAD = 3
GLASS = "minecraft:glass"
PLAYED_TICKS, TICKS_PER_SECOND = 1200, 20


@pytest.fixture(scope="module")
def m6(fx, tmp_path_factory):
    """The preview with --trees and --blend-biomes; the working copy's region files are copied just before the merge."""
    base = tmp_path_factory.mktemp("m6")
    real = plant.replay

    def spy(stage, a, merge, strips):
        shutil.copytree(os.path.join(stage, "region"), base / "premerge" / "region", dirs_exist_ok=True)
        return real(stage, a, merge, strips)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plant, "replay", spy)
        t0 = time.monotonic()
        code, job = fx.generate("--trees", "--blend-biomes")
        seconds = time.monotonic() - t0
    assert code == 0
    print(f"preview with --trees --blend-biomes on the fixture: {seconds:.1f} s; tool peak memory {peak_mb()} MB")
    job.premerge, job.base = base / "premerge", base
    return job


def _kept(job):
    a = jobs.read_json(os.path.join(job.folder, "area.json"))
    return {tuple(c) for c in a["core"]}, {tuple(c) for c in a["kept"]}


def test_trees_come_back_on_the_strips_as_logs_and_leaves_into_air_only(fx, m6):
    core, kept = _kept(m6)
    before, after = chunks_of(m6.premerge), chunks_of(m6.data["args"]["staging"])
    k = smooth.classify(world.registry(world.read_level(m6.data["args"]["staging"])))
    assert set(before) == set(after)
    added, margin_chunks, bad = {"log": 0, "leaves": 0}, 0, []
    for pos in sorted(after):
        if after[pos] == before[pos]:
            continue
        assert pos not in core, f"protected chunk {pos} changed"
        ids, meta, biomes, _ = _ours(after[pos])
        was_ids, was_meta, was_biomes, _ = _ours(before[pos])
        changed = (ids != was_ids) | (meta != was_meta)
        cat = smooth.lookup(k.cat, ids[changed])
        if (was_ids[changed] != 0).any() or not np.isin(cat, (smooth.LOG, smooth.LEAVES)).all():
            bad.append(pos)
        assert np.array_equal(biomes, was_biomes), pos
        margin_chunks += pos in kept
        added["log"] += int(np.sum(cat == smooth.LOG))
        added["leaves"] += int(np.sum(cat == smooth.LEAVES))
    assert not bad, f"{len(bad)} chunks gained something other than logs or leaves in air, the first {bad[0]}"
    t, b = report(m6)["trees"], record(m6, plant.PLANT)
    assert added == {"log": t["log"], "leaves": t["leaves"]} and t["chunks_changed_in_core"] == 0
    assert t["log_in_strips"] >= 0.5 * t["log_removed"], t
    assert t["log_removed"] == record(m6, smooth.SMOOTH)["tree_log_voxels_removed"]
    print(
        f"trees: {b['trees']:,} trees from {b['attempts']:,} attempts at {t['spots']:,} spots in {b['groups']} chunks; "
        f"{t['log']:,} log and {t['leaves']:,} leaf voxels added, {t['skipped_not_air']:,} skipped (not air in staging); "
        f"{t['log_in_strips']:,} log voxels in the reshaped strips = {100 * t['log_in_strips'] / t['log_removed']:.1f}% of the "
        f"{t['log_removed']:,} smoothing removed ({100 * t['log_in_strips'] / POC_LOGS_REMOVED:.1f}% of the POC's {POC_LOGS_REMOVED:,}); "
        f"{len([p for p in after if after[p] != before[p]])} chunks changed, {margin_chunks} of them margin chunks; "
        f"plant boot {b['seconds']}, peak working set {b['peak_working_set_mb']:,} MB; unknown biome {b['unknownBiome']}, "
        f"missing chunks {b['missingChunks']}"
    )
    _picture(m6, before, after, k)


def _picture(job, before, after, k):
    """A top view around the chunk that lost the most logs: the reshaped strips pale, logs brown, leaves dark, before and after
    the merge."""
    strips = plant.load_strips(os.path.join(job.folder, smooth.SMOOTH, plant.STRIPS))
    worst = max(strips, key=lambda c: strips[c][1])
    cells = sorted(area.dilate({worst}, PAD))
    x0, z0 = min(c[0] for c in cells), min(c[1] for c in cells)
    side = (2 * PAD + 1) * chunks.SIDE
    panels = []
    for state in (before, after):
        img = np.full((side, side, 3), GREY, np.uint8)
        for c in cells:
            if c not in state:
                continue
            ids = chunks.block_ids(level(state[c]))
            cat = smooth.lookup(k.cat, ids)
            strip = strips[c][0] if c in strips else np.zeros((chunks.SIDE, chunks.SIDE), bool)
            top = np.where((cat == smooth.LOG).any(0), 1, np.where((cat == smooth.LEAVES).any(0), 2, np.where(strip, 3, 0)))
            tile = np.array([GREEN, BROWN, LEAF, STRIP], np.uint8)[top]
            ox, oz = (c[0] - x0) * chunks.SIDE, (c[1] - z0) * chunks.SIDE
            img[oz : oz + chunks.SIDE, ox : ox + chunks.SIDE] = tile
        panels.append(img)
    path = job.base / f"trees_{worst[0]}_{worst[1]}.png"
    both = np.concatenate([panels[0], np.full((side, PAD, 3), 255, np.uint8), panels[1]], 1)
    Image.fromarray(both).resize((both.shape[1] * PAD, side * PAD), Image.NEAREST).save(path)
    print(f"tree picture (before | after the merge, centred on chunk {worst}): {path}")


def test_the_blend_breaks_the_grass_colour_line_at_the_kept_border(fx, m6):
    """Measured with the test's own pairs: the biomes without the blend are the world's in kept chunks and the generator's
    outside; with it, the working copy's. The ground is the working copy's in both."""
    core, kept = _kept(m6)
    have, made, src = chunks_of(m6.data["args"]["staging"]), chunks_of(genworld.generated(m6)["folder"]), chunks_of(fx.world)
    ok = ground_ids(world.registry(world.read_level(m6.data["args"]["staging"])))
    cells = area.dilate(kept, 1) - (core - area.boundary(core))
    grids = {}
    for c in cells:
        lv = level(have[c])
        no_blend = chunks.biomes(level(src[c])) if c in kept else chunks.biomes(level(made[c]))
        grids[c] = (chunks.biomes(lv), no_blend, column_ground(chunks.block_ids(lv), chunks.heightmap(lv), ok))
        if c in core:
            assert np.array_equal(grids[c][0], grids[c][1]), c
        if c not in kept:
            assert np.array_equal(grids[c][0], grids[c][1]), c
    inner, outer = score.edge_pairs(kept)
    colours = blend.Colours(genworld.generated(m6)["release"])

    def lab(xz, which):
        b = np.array([grids[(x // 16, z // 16)][which][z % 16, x % 16] for x, z in xz.tolist()])
        g = np.array([grids[(x // 16, z // 16)][2][z % 16, x % 16] for x, z in xz.tolist()])
        return colours.lab(b, g - 1)

    counts = {}
    for name, which in (("no blend", 1), ("blend", 0)):
        de = np.linalg.norm(lab(inner, which) - lab(outer, which), axis=1)
        good = np.isfinite(de)
        counts[name] = (int(good.sum()), int(np.sum(de[good] > blend.OVER_DE)))
    b = report(m6)["biome_blend"]
    print(f"biome blend: border pairs over dE 10, own count {counts}; tool {b}")
    assert counts["blend"][1] <= 0.5 * counts["no blend"][1] and counts["no blend"][1] > 0
    assert (counts["no blend"][1], counts["blend"][1]) == (b["border_over_before"], b["border_over_after"])


def _place(lv, idx, block):
    """Put block at the first voxel of idx that is air in lv and lies in a section lv holds; returns that voxel or None."""
    ids, have = chunks.block_ids(lv).reshape(-1), {s["Y"][1]: s for s in chunks.sections(lv)}
    i = next((int(i) for i in idx if ids[i] == 0 and i // chunks.VOLUME in have), None)
    if i is not None:
        sec = have[i // chunks.VOLUME]
        v = chunks.decode_section(sec)
        v.ids[i % chunks.VOLUME], v.meta[i % chunks.VOLUME] = block, 0
        new = chunks.encode_section(chunks.chunk_format(lv), i // chunks.VOLUME, v, like=dict(sec))
        sec.clear()
        sec.update(new)
    return i


def _play_at_the_margin(job, world_dir):
    """A minute of play along the kept border: every margin chunk the trees reach saved again (InhabitedTime, LastUpdate), and a
    glass block placed where the merge list grows a log or a leaf. Returns the raw region files, the played chunks and the glass."""
    a, stage = jobs.read_json(os.path.join(job.folder, "area.json")), job.data["args"]["staging"]
    core, kept = _kept(job)
    merge = plant.load_merge(os.path.join(job.folder, plant.PLANT, plant.MERGE_FILE))
    played = plant.plan(stage, a, plant.load_strips(os.path.join(job.folder, smooth.SMOOTH, plant.STRIPS))).reach(core) & kept
    glass_id = next(i for i, n in world.registry(world.read_level(world_dir)).items() if n == GLASS)
    raws, glass = {}, None
    for (rx, rz), mine in plant.by_region(played).items():
        path = Path(world_dir, "region", f"r.{rx}.{rz}.mca")
        raws[path] = path.read_bytes()
        recs = region.records(raws[path])
        for pos in mine:
            i = region.slot_index(*pos)
            name, root = nbt.parse(region.inflate(recs[i][0]))
            lv = root["Level"][1]
            for key in ("InhabitedTime", "LastUpdate"):
                lv[key] = [nbt.LONG, lv.get(key, [nbt.LONG, 0])[1] + PLAYED_TICKS]
            if glass is None and pos in merge:
                at = _place(lv, merge[pos][0], glass_id)
                glass = None if at is None else (pos, at, glass_id)
            recs[i] = (region.deflate(nbt.write(name, root)), recs[i][1] + PLAYED_TICKS // TICKS_PER_SECOND)
        path.write_bytes(region.build(recs))
    age(world_dir)
    assert glass is not None and played
    return raws, played, glass


def test_play_at_the_margin_then_a_refresh_replays_the_merge_list(fx, m6, capsys):
    """Apply after play in the margin chunks the trees reach: no plant boot, and every chunk of the applied world as the preview
    made it, except the played chunks, whose blocks and biomes are the preview's too, but for the glass the merge left alone."""
    want, plant0 = staging.final(m6), jobs.read_json(os.path.join(m6.folder, plant.PLANT, jobs.MANIFEST_FILE))
    trees0, before = report(m6)["trees"], chunks_of(m6.data["args"]["staging"])
    raws, played, (glass_pos, glass_at, glass_id) = _play_at_the_margin(m6, fx.world)
    try:
        capsys.readouterr()
        t0 = time.monotonic()
        assert fx.apply(m6) == 0
        downtime = time.monotonic() - t0
        err = capsys.readouterr().err
        assert "replanting trees" not in err and "and replayed the trees' merge list in" in err, err[-3000:]
        assert jobs.read_json(os.path.join(m6.folder, plant.PLANT, jobs.MANIFEST_FILE)) == plant0
        after, final = chunks_of(fx.world), staging.final(m6)
        assert set(after) == set(before) and {p for p in after if after[p] != before[p]} == played
        played_files = {f"region/r.{rx}.{rz}.mca" for rx, rz in plant.by_region(played)}
        assert {rel for rel in set(final) | set(want) if final.get(rel) != want.get(rel)} == played_files
        for pos in sorted(played):
            got, was = _ours(after[pos])[:3], _ours(before[pos])[:3]
            if pos == glass_pos:
                y, z, x = np.unravel_index(glass_at, got[0].shape)
                assert (got[0][y, z, x], got[1][y, z, x]) == (glass_id, 0)
                was[0][y, z, x], was[1][y, z, x] = glass_id, 0
            assert all(np.array_equal(g, w) for g, w in zip(got, was, strict=True)), pos
        t = report(m6)["trees"]
        assert t["skipped_not_air"] == 1 and t["log"] + t["leaves"] == trees0["log"] + trees0["leaves"] - 1
        same = ("spots", "log_removed", "chunks", "boot", "chunks_changed_in_core")
        assert {k: t[k] for k in same} == {k: trees0[k] for k in same}
        refresh = next(line for line in err.splitlines() if "apply: refresh" in line)
        with capsys.disabled():
            print(
                f"refresh after play ({len(played)} margin chunks the trees reach saved again, glass at {glass_pos} voxel {glass_at}): "
                f"whole Apply (the downtime) {downtime:.1f} s; {refresh}; trees {t}"
            )
    finally:
        assert fx.rollback() == 0
        for path, raw in raws.items():
            path.write_bytes(raw)
        age(fx.world)


def test_neid_variant_byte_biome_arrays_never_take_an_id_above_255(fx, tmp_path, capsys, green1_pick):
    out = tmp_path / "neid-area.json"
    assert cli.main(["area", fx.root, "--add", NEID_CORE, "--margin", "1", "--out", str(out), "--cache", CACHE]) == 0
    m = Migration.__new__(Migration)
    m.__dict__ |= fx.__dict__ | {"area": str(out)}
    m.search = search_job(fx.data, str(out), world.locate(fx.world).config, green1_pick)
    capsys.readouterr()
    code, job = m.generate("--blend-biomes")
    err = capsys.readouterr().err
    if code:
        assert "holds biome ids up to 255" in err and "Run preview again without --blend-biomes" in err, err[-3000:]
        print(f"NEID variant with --blend-biomes: refused: {next(line for line in err.splitlines() if 'Refused' in line)}")
        return
    have = chunks_of(job.data["args"]["staging"])
    neid = [c for c in have if chunks.chunk_format(level(have[c])) == chunks.NEID]
    assert neid and all(chunks.biomes(level(have[c])).max() <= chunks.BYTE_MAX for c in neid)
    print(f"NEID variant with --blend-biomes: applied, {len(neid)} NEID chunks all within 255; blend {report(job)['biome_blend']}")
