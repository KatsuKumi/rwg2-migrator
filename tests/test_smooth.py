"""The smoothing plan, the column rebuild, the tree flood and the light tables, on the POC selftest's synthetic cases."""

import struct

import numpy as np
import pytest

from rwg2migrator import Refusal, smooth
from rwg2migrator.smooth import NO_LEVEL, REACH, SEA_TOP, Fields, plan

N, KW = 64, 40
NAMES = {
    1: "minecraft:stone",
    2: "minecraft:grass",
    3: "minecraft:dirt",
    9: "minecraft:water",
    31: "minecraft:tallgrass",
    17: "minecraft:log",
    18: "minecraft:leaves",
    54: "minecraft:chest",
    500: "gregtech:gt.blockores2",
    106: "minecraft:vine",
}
STONE, GRASS, DIRT, WATER, TALL, CHEST, ORE = 1, 2, 3, 9, 31, 54, 500


LAYOUT = smooth.Layout({(x, z) for x in range(N // 16) for z in range(N // 16)})


def pack(dense, lay=LAYOUT):
    """A dense (..., z, x) field from block 0,0 on, as the layout's (..., n, 16, 16)."""
    return np.stack([dense[..., z * 16 : z * 16 + 16, x * 16 : x * 16 + 16] for x, z in lay.cells], axis=-3)


def unpack(packed, lay=LAYOUT):
    nx, nz = (max(c[i] for c in lay.cells) + 1 for i in (0, 1))
    out = np.zeros((*packed.shape[:-3], nz * 16, nx * 16), packed.dtype)
    for i, (x, z) in enumerate(lay.cells):
        out[..., z * 16 : z * 16 + 16, x * 16 : x * 16 + 16] = packed[..., i, :, :]
    return out


def _band():
    kept = np.zeros((N, N), bool)
    kept[:, :KW] = True
    zone = (np.abs(np.arange(N) - KW + 0.5) < REACH)[None].repeat(N, 0)
    return kept, zone


def _plan(cur, old, new, wtop=None):
    kept, zone = _band()
    wtop = np.full((N, N), NO_LEVEL) if wtop is None else wtop
    p = plan(Fields(*(pack(a) for a in (cur, old, new, kept, zone, wtop)), LAYOUT))
    return {k: unpack(v) if isinstance(v, np.ndarray) else v for k, v in p.items()}


def test_the_ramp_rises_monotonically_across_the_border_and_stays_inside_the_zone():
    kept, _ = _band()
    old, new = np.full((N, N), 70.0), np.full((N, N), 100.0)
    p = _plan(np.where(kept, old, new), old, new)
    row = p["target"][N // 2]
    assert row[KW - REACH - 1] == 70 and row[KW + REACH] == 100
    assert np.all(np.diff(row) >= 0) and np.abs(np.diff(row)).max() <= 2
    assert not p["change"][:, : KW - REACH].any() and not p["change"][:, KW + REACH :].any()


def test_ground_that_already_agrees_within_min_diff_is_left_alone():
    old = np.full((N, N), 70.0)
    assert not _plan(old.copy(), old, old + 1)["change"].any()


def test_a_lake_above_sea_level_is_never_drained_and_its_shore_is_raised():
    kept, _ = _band()
    old, new = np.full((N, N), 70.0), np.full((N, N), 100.0)
    cur = np.where(kept, old, new)
    lake = np.full((N, N), NO_LEVEL)
    lake[:, KW - 3] = 90
    p = _plan(cur, old, new, lake)
    assert not p["change"][:, KW - 3].any() and (p["target"][:, [KW - 4, KW - 2]] >= 91).all()
    lake[:, KW - 3] = 75
    p = _plan(cur, old, new, lake)
    assert p["change"][:, KW - 3].all() and (p["target"][:, KW - 3] > 75).all() and p["shore_cols"] == 0


def test_lowered_dry_ground_stays_dry_and_ground_under_the_sea_gets_sea_water():
    kept, _ = _band()
    old, new = np.full((N, N), 60.0), np.full((N, N), 40.0)
    cur = np.where(kept, old, new)
    pit = _plan(cur, old, new)
    assert pit["change"].any() and (pit["level"] == NO_LEVEL).all() and pit["shore_cols"] == 0
    sea = _plan(cur, old, new, np.full((N, N), SEA_TOP))
    assert (sea["level"][sea["change"]] == SEA_TOP).all() and sea["sea_filled_cols"] == 0


@pytest.fixture
def kinds():
    return smooth.classify(NAMES)


def test_block_kinds(kinds):
    cats = [int(kinds.cat[i]) for i in (STONE, TALL, 17, 18, CHEST, ORE, WATER, 106)]
    assert cats == [smooth.PLAIN, smooth.PLANT, smooth.LOG, smooth.LEAVES, smooth.OTHER, smooth.ORE, smooth.WATER, smooth.PLANT]
    assert kinds.vine[106] and kinds.liquid[WATER] and not kinds.liquid[STONE]
    assert smooth.lookup(kinds.cat, np.array([1 << 20]))[0] == smooth.OTHER


def test_air_stays_air_and_transparent_when_the_registry_names_it():
    k = smooth.classify({0: "minecraft:air"} | NAMES)
    assert k.cat[0] == smooth.AIR and k.opac[0] == 0 and not k.surface[0]
    col = np.zeros((256, 1, 1), np.uint32)
    col[:64] = STONE
    assert smooth.heightmap(col, k.opac)[0, 0] == 64 and smooth.skylight(col, k.opac)[100, 0, 0] == 15


def test_a_registry_without_the_basic_blocks_is_refused():
    with pytest.raises(Refusal, match="no minecraft:water"):
        smooth.classify({1: "minecraft:stone", 2: "minecraft:grass", 3: "minecraft:dirt"})


def _column(top=70):
    ids, meta = np.zeros(256, np.uint32), np.zeros(256, np.uint16)
    ids[1 : top - 4], ids[top - 4 : top - 1], ids[top - 1], ids[top] = STONE, DIRT, GRASS, TALL
    ids[top - 6], ids[top + 2] = ORE, CHEST
    meta[top - 2] = 2
    return ids, meta


def test_raising_a_column_fills_stone_then_the_old_surface_and_keeps_other_blocks(kinds):
    c = _column()
    smooth.rebuild(c, (70, 75, NO_LEVEL), kinds)
    assert list(c[0][66:76]) == [1, 1, 1, 1, 1, 3, 54, 3, 2, 0] and c[1][73] == 2 and c[0][64] == ORE


def test_lowering_below_the_sea_puts_dirt_under_water_and_keeps_the_chest(kinds):
    c = _column()
    smooth.rebuild(c, (70, 58, SEA_TOP), kinds)
    assert list(c[0][54:58]) == [3] * 4 and list(c[1][56:58]) == [2, 2]
    assert all(c[0][58:63] == WATER) and all(c[0][63:72] == 0) and c[0][72] == CHEST


def test_lowering_on_dry_land_moves_the_surface_down(kinds):
    c = _column()
    smooth.rebuild(c, (70, 62, NO_LEVEL), kinds)
    assert c[0][61] == GRASS and c[0][60] == DIRT and c[1][60] == 2 and all(c[0][62:72] == 0) and c[0][72] == CHEST
    c = _column()
    smooth.rebuild(c, (70, 67, NO_LEVEL), kinds)
    assert c[0][64] == ORE and c[0][66] == GRASS and all(c[0][67:72] == 0)


def test_a_cave_under_the_new_surface_gets_a_roof(kinds):
    c = _column()
    c[0][58:62] = 0
    smooth.rebuild(c, (70, 62, NO_LEVEL), kinds)
    assert list(c[0][57:62]) == [1, 3, 3, 3, 2] and c[0][72] == CHEST


def test_leaves_are_held_only_by_logs_within_reach_and_logs_flood_through_their_tree():
    lay = smooth.Layout({(0, 0), (1, 0)})
    shape = (6, 16, 32)
    logs, leaves = np.zeros(shape, bool), np.zeros(shape, bool)
    for x in (8, 24):
        logs[1:4, 8, x] = True
        leaves[2:5, 7:10, x - 1 : x + 2] = True
        leaves[2:5, 8, x] = False
    logs, leaves = pack(logs, lay), pack(leaves, lay)
    held = lay.flood(logs, logs | leaves, False, smooth.LEAF_REACH) & leaves
    assert held.sum() == leaves.sum()
    cut = logs.copy()
    cut[:, 0] = False
    assert (lay.flood(cut, cut | leaves, False, smooth.LEAF_REACH) & leaves)[:, 0].sum() == 0
    one = smooth.Layout({(0, 0)})
    assert one.flood(logs[:, :1] & (np.arange(6)[:, None, None, None] == 1), logs[:, :1], True).sum() == 3


def _dense_grow(m, full):
    out = m
    for ax in range(m.ndim):
        src = out if full else m
        out = out | smooth.shift(src, 1, ax) | smooth.shift(src, -1, ax)
    return out


def test_the_layout_gives_what_a_dense_field_over_the_bounding_box_gives():
    """Growth and box means cross into neighbour chunks, and a 26-neighbour step crosses the corner between (1,1) and (2,2),
    whose shared neighbours hold nothing: the layout holds every chunk next to the loaded ones."""
    rng = np.random.default_rng(7)
    loaded = {(1, 1), (2, 2), (3, 1)}
    lay = smooth.Layout(smooth.area.dilate(loaded, 1))
    where = pack(np.zeros((5, 4 * 16, 5 * 16), bool), lay)
    for c in loaded:
        where[:, lay.ix(c)] = True
    m = (rng.random(where.shape) < 0.03) & where
    m[:, lay.ix((1, 1)), 15, 15] = True
    for full in (True, False):
        assert np.array_equal(unpack(lay.grow(m, full), lay), _dense_grow(unpack(m, lay), full))
    a, w = rng.random(where.shape[1:]) * 100, where[0] & (rng.random(where.shape[1:]) < 0.7)
    da, dw = unpack(a, lay), unpack(w, lay)
    for r in (1, smooth.SPREAD):
        got = unpack(lay.box_mean(a, w, r), lay)
        for z, x in zip(*np.nonzero(unpack(where[0], lay)), strict=True):
            box = (slice(max(z - r, 0), z + r + 1), slice(max(x - r, 0), x + r + 1))
            want = da[box][dw[box]].mean() if dw[box].any() else np.nan
            assert got[z, x] == pytest.approx(want, nan_ok=True)


def test_skylight_and_heightmap_follow_opacity(kinds):
    col = np.zeros((256, 1, 1), np.uint32)
    col[:60], col[60:62], col[70], col[71] = STONE, WATER, 18, TALL
    sky = smooth.skylight(col, kinds.opac)[:, 0, 0]
    assert sky[72] == 15 and sky[71] == 15 and sky[70] == 14 and sky[69] == 13 and sky[62] == 6 and sky[61] == 3 and sky[60] == 0
    assert smooth.heightmap(col, kinds.opac)[0, 0] == 71


def test_a_reshaped_column_loses_the_entity_standing_on_it_and_the_count_is_kept():
    from conftest import REGISTRY, SURFACE, chunk_root, tag

    from rwg2migrator import nbt, region

    def cow(x, z):
        pos = [struct.pack(">d", v) for v in (5 * 16 + x, SURFACE + 1.0, 5 * 16 + z)]
        return {"id": tag(nbt.STRING, "Cow"), "Pos": tag(nbt.LIST, [nbt.DOUBLE, pos])}

    k = smooth.classify(REGISTRY | {59: "minecraft:wheat", 3001: "IC2:blockCrop"})
    assert k.cat[59] == k.cat[3001] == smooth.PLANT
    root = chunk_root(5, 5)
    root["Level"][1]["Entities"] = tag(nbt.LIST, [nbt.COMPOUND, [cow(8.5, 8.5), cow(2.5, 2.5)]])
    ch = smooth.Chunk((5, 5), region.deflate(nbt.write("", root)), 0)
    ch.snapshot()
    ground = int(np.flatnonzero(ch.ids[:, 8, 8])[-1]) + 1
    smooth.rebuild((ch.ids[:, 8, 8], ch.meta[:, 8, 8]), (ground, ground - 6, NO_LEVEL), k)
    change = np.zeros((16, 16), bool)
    change[8, 8] = True
    low = np.where(change, ground - 6 - smooth.PROFILE, smooth.HEIGHT).astype(np.int32)
    _, _, dropped = ch.encode(k, (change, low))
    assert dropped["Entities"] == 1 and len(ch.lvl["Entities"][1][1]) == 1
