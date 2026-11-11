"""M6 biome blend on the synthetic world with the fake generator: the margin takes the new terrain's biomes behind a wavy line
inside the band, core and the new terrain keep theirs, and a byte biome array never takes an id above 255."""

import fake_gen
import numpy as np
import pytest
from conftest import Migration, add_migration_parts, age, make_world, terrain
from test_generate import area_of, chunks_of, level_of, report

from rwg2migrator import Refusal, blend, chunks, genworld, smooth

RING = 3


def setup(base, monkeypatch, fmt=chunks.ENDLESSIDS, ids=None):
    """The synthetic server with every chunk in fmt, holding ids (default: the test terrain)."""
    root = base / "server"
    stack = terrain() if ids is None else ids
    make_world(str(root), chunk_specs={(x, z): {"ids": stack, "fmt": fmt} for x in range(4) for z in range(4)})
    add_migration_parts(str(root))
    age(str(root / "World"))
    fake_gen.use(monkeypatch)
    m = Migration(root, base)
    fake_gen.settings(m.cache)
    return m


@pytest.fixture
def plains(tmp_path, monkeypatch):
    return setup(tmp_path, monkeypatch)


def voxels(world_dir):
    """{pos: (ids, meta, biomes)} of every overworld chunk."""
    out = {}
    for pos, rec in chunks_of(world_dir).items():
        level = level_of(rec)
        meta = np.zeros((chunks.HEIGHT, 16, 16), np.uint32)
        for sec in chunks.sections(level):
            y = sec["Y"][1] * chunks.SECTION
            meta[y : y + chunks.SECTION] = chunks.decode_section(sec).meta.reshape(chunks.SECTION, 16, 16)
        out[pos] = (chunks.block_ids(level), meta, chunks.biomes(level))
    return out


def test_the_biome_blend_takes_the_new_biomes_into_the_margin_and_breaks_the_colour_line(plains):
    code, job = plains.generate("--ring", RING, "--blend-biomes")
    assert code == 0
    b = report(job)["biome_blend"]
    assert b["border_over_before"] > 0 and b["border_over_after"] <= 0.5 * b["border_over_before"], b
    core, kept = area_of(plains)
    have, made = voxels(job.data["args"]["staging"]), voxels(genworld.generated(job)["folder"])
    old = voxels(plains.world)
    new_ids = set(fake_gen.BIOME_IDS.values())
    taken = 0
    for pos, (_, _, biomes) in have.items():
        if pos in core:
            assert np.array_equal(biomes, old[pos][2]), pos
        elif pos in kept:
            moved = biomes != old[pos][2]
            assert set(biomes[moved].tolist()) <= new_ids
            taken += int(moved.sum())
        else:
            assert np.array_equal(biomes, made[pos][2]), pos
    assert taken == b["columns_changed"] > 0 and report(job)["core_identical"] == len(core)


def test_a_biome_above_255_bound_for_a_byte_biome_array_is_refused(tmp_path, monkeypatch, capsys):
    with pytest.raises(Refusal, match="holds biome ids up to 255"):
        blend.refuse_bytes({(3, 4): (chunks.NEID, np.full((16, 16), 300))}, {300: "Temperate Forest"})
    blend.refuse_bytes({(3, 4): (chunks.ENDLESSIDS, np.full((16, 16), 300)), (5, 4): (chunks.NEID, np.full((16, 16), 255))}, {})
    m = setup(tmp_path, monkeypatch, fmt=chunks.NEID)
    for name in fake_gen.BIOME_IDS:
        monkeypatch.setitem(fake_gen.BIOME_IDS, name, 324)
    code, _ = m.generate("--ring", RING, "--blend-biomes")
    err = capsys.readouterr().err
    assert code == 1 and "would write biome 324 (Temperate Forest) into chunk" in err and "Run preview again without --blend-biomes" in err


def test_the_blend_needs_the_smoothed_border(plains, capsys):
    for extra in (("--no-smooth", "--blend-biomes"), ("--no-generate", "--blend-biomes")):
        code, job = plains.generate(*extra)
        assert code == 1 and job is None
        assert "the new terrain generated and smoothed" in capsys.readouterr().err


def test_the_blend_threshold_is_smooth_noise_and_its_weight_rises_across_the_band():
    x, z = np.meshgrid(np.arange(-40, 40), np.arange(-40, 40))
    n = blend.noise(x, z)
    assert n.min() >= 0 and n.max() <= 1 and np.array_equal(n, blend.noise(x, z))
    assert np.abs(np.diff(n, axis=1)).max() < 0.25 and n.std() > 0.05
    lay = smooth.Layout({(cx, 0) for cx in range(4)})
    kept, core = lay.mask({(0, 0), (1, 0), (2, 0)}), lay.mask({(0, 0), (1, 0)})
    w = blend.weight(lay, kept, core, blend.depth({(0, 0), (1, 0)}, {(0, 0), (1, 0), (2, 0)}))
    row = w[lay.ix((2, 0)), 8]
    assert np.all(np.diff(row) > 0) and row[0] < 0.05 and row[-1] > 0.95
    assert not w[lay.ix((1, 0))].any() and not w[lay.ix((3, 0))].any()
    wide = smooth.Layout({(cx, 0) for cx in range(6)})
    core3, kept3 = {(0, 0)}, {(cx, 0) for cx in range(4)}
    assert blend.depth(core3, kept3) == 3
    w3 = blend.weight(wide, wide.mask(kept3), wide.mask(core3), 3)
    row3 = np.concatenate([w3[wide.ix((cx, 0)), 8] for cx in (1, 2, 3)])
    assert np.all(np.diff(row3) > 0) and row3[0] < 0.01 and row3[-1] > 0.99
    names = np.full((4, 16, 16), 1)
    field = np.full((4, 16, 16), 4)
    field[lay.ix((3, 0))] = 72
    assert blend.rwg2_ids(names, field, lay.mask({(3, 0)})) == {1: 72}
