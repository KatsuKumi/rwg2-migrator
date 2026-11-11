import re

import numpy as np

from .chunks import HEIGHT

GROUND = re.compile(
    r"stone|dirt|grass|sand|gravel|clay|mycelium|snow|ice|deepslate|ore|mud|bedrock|podzol|tuff|granite|diorite|andesite|"
    r"basalt|cleanDirt",
    re.I,
)
NOT_GROUND = re.compile(
    r"brick|slab|stair|wall|fence|glass|casing|machine|lamp|pane|tile|leaves|log|wood|plank|torch|_block_|bars|"
    r"tall|longGrass|garden|crop|sapling|snowbell|snow_layer|button|plate|wire",
    re.I,
)
MAX_SCAN = 48
ID_SPACE = 1 << 16
_Y = np.arange(HEIGHT)[:, None, None]
TOPS = ("none", "grass", "dirt", "rock", "sand", "mesa", "snow", "mud", "other")
NO_TOP, OTHER_TOP = 0, len(TOPS) - 1
TOP_RULES = (
    ("snow", r"snow|(?<![a-z])ice"),
    ("grass", r"minecraft:grass|BiomesOPlenty:longGrass"),
    ("dirt", r"minecraft:dirt|farmland"),
    ("mesa", r"hardened_clay"),
    ("sand", r"sand|hardDirt"),
    ("rock", r"stone|cobble|gravel"),
    ("mud", r"mud|quicksand"),
)
WATER = ("minecraft:water", "minecraft:flowing_water")
COVER = re.compile(r"snow_layer")
DEPTH_MAX = 255


def _table(registry):
    return np.zeros(max(ID_SPACE, max(registry, default=0) + 1), bool)


def ground_ids(registry):
    """Which block ids count as natural ground, from {id: 'modid:name'} (the rule of the POC's fixed rwg2_seams)."""
    ok = _table(registry)
    for i, name in registry.items():
        local = name.split(":", 1)[-1]
        ok[i] = bool(GROUND.search(local)) and not NOT_GROUND.search(local)
    return ok


def top_kind(name):
    return next((TOPS.index(kind) for kind, pattern in TOP_RULES if re.search(pattern, name)), OTHER_TOP)


def top_kinds(registry):
    """The look of each block id when it is the top ground block: an index into TOPS."""
    kinds = np.full(_table(registry).size, OTHER_TOP, np.uint8)
    for i, name in registry.items():
        kinds[i] = top_kind(name)
    return kinds


def cover_kinds(registry):
    """The look a block gives the column when it lies on the top ground block (a snow layer makes it snow), else NO_TOP."""
    cover = np.full(_table(registry).size, NO_TOP, np.uint8)
    cover[[i for i, n in registry.items() if COVER.search(n)]] = TOPS.index("snow")
    return cover


def water_ids(registry):
    wet = _table(registry)
    wet[[i for i, n in registry.items() if n in WATER]] = True
    return wet


def _lookup(flags, ids):
    return (ids < flags.size) & flags[np.minimum(ids, flags.size - 1)]


def _ground_mask(ids, hm, ok):
    top = np.minimum(hm, HEIGHT - 1)
    window = (top >= _Y) & (np.maximum(top - MAX_SCAN, 0) < _Y)
    return _lookup(ok, ids) & window, window


def column_ground(ids, hm, ok):
    """Ground height (top face of the highest ground block within MAX_SCAN of the HeightMap) per column, NaN where none."""
    m, _ = _ground_mask(ids, hm, ok)
    h = HEIGHT - np.argmax(m[::-1], 0)
    return np.where(m.any(0), h, np.nan).astype(np.float32)


def _at(table, ids, y):
    i = np.take_along_axis(ids, np.clip(y, 0, HEIGHT - 1)[None], 0)[0]
    return np.where(i < table.size, table[np.minimum(i, table.size - 1)], NO_TOP)


def column_top(ids, hm, rule):
    """(look of the top ground block as a TOPS index, depth of the water above it) per column; rule has ok, kinds, cover, wet."""
    m, window = _ground_mask(ids, hm, rule.ok)
    has = m.any(0)
    y = np.where(has, HEIGHT - 1 - np.argmax(m[::-1], 0), -1)
    cover = np.where(y + 1 < HEIGHT, _at(rule.cover, ids, y + 1), NO_TOP)
    kind = np.where(has, np.where(cover != NO_TOP, cover, _at(rule.kinds, ids, y)), NO_TOP).astype(np.uint8)
    depth = (_lookup(rule.wet, ids) & window & (y < _Y)).sum(0)
    return kind, np.minimum(depth, DEPTH_MAX).astype(np.uint8)
