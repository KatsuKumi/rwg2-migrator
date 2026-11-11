import hashlib
import json
import re
from collections import Counter
from functools import cache

import numpy as np

from . import Refusal, jobs, provision, verdicts
from .scan import GRAVE, UNREADABLE

CHUNK = 16
FAR = 512
MIN_MARGIN = 1
DEFAULT_MARGIN = 1
SUGGEST_TICKS = 72_000
SHOWN = 8
NEIGHBOURS = tuple((dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dz)
WORLDGEN_FILE = "worldgen-tes.json"
NATURAL_TES = frozenset({
    "GT_TileEntity_Ores",
    "MobSpawner",
    "etfuturum.glow_lichen",
    "etfuturum.cave_vines",
    "TileNode",
    "TileExtendedNode",
    "TileEldritchCap",
    "TileEldritchAltar",
    "TileEldritchObelisk",
    "TileManaPod",
    "LOOTGAMES_MASTER_TE",
    "tile.projectred.exploration.lily|0",
})
GRAVE_KEYS = ("name", "x", "y", "z", "stacks")
PLAYER_NAME = re.compile(r"\w{3,16}|[0-9a-f]{8}-?([0-9a-f]{4}-?){3}[0-9a-f]{12}", re.ASCII | re.IGNORECASE)
YOURS_WHY = "a player's name in their data, or a kind of block world generation never makes"
TE_LABELS = {
    "savedMultipart": "microblocks",
    "Chest": "chests",
    "Skull": "skulls",
    "FlowerPot": "flower pots",
    "doorTileEntity": "doors",
    "trapDoorTileEntity": "trapdoors",
    "fenceGateTileEntity": "fence gates",
    "forestry.Swarm": "beehives",
    "forestry.Wood": "Forestry wood blocks",
    "TConstruct.Servants": "smeltery blocks",
    "Trap": "dispensers",
    "Dropper": "droppers",
    "Hopper": "hoppers",
    "Furnace": "furnaces",
    "RecordPlayer": "jukeboxes",
    "Music": "note blocks",
    "Sign": "signs",
    "Cauldron": "cauldrons",
    "EnchantTable": "enchanting tables",
    "Comparator": "comparators",
    "TileBanner": "banners",
    "etfuturum.banner": "banners",
    "etfuturum.brewing_stand": "brewing stands",
    "BlockSkyChest": "sky stone chests",
    "TileCrucible": "crucibles",
    "BaseMetaTileEntity": "GregTech machines",
    "BaseMetaPipeEntity": "GregTech pipes and cables",
}
ID_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
ID_NOISE = {"tile", "entity", "te", "block"}


def dilate(cells, n):
    return {(x + dx, z + dz) for x, z in cells for dx in range(-n, n + 1) for dz in range(-n, n + 1)}


def claim_hash(cells):
    return hashlib.sha256(";".join(f"{x},{z}" for x, z in sorted(cells)).encode()).hexdigest()


def _pair(text):
    x, z = text.split(",")
    return int(x), int(z)


def rect(text):
    """Chunks of an inclusive rectangle written X1,Z1:X2,Z2 in chunk coordinates."""
    try:
        (x1, z1), (x2, z2) = (_pair(p) for p in text.split(":"))
    except ValueError:
        raise Refusal(
            f"{text!r} is not a chunk rectangle", "Write it as X1,Z1:X2,Z2 in chunk coordinates, for example 100,20:110,30"
        ) from None
    return {(x, z) for x in range(min(x1, x2), max(x1, x2) + 1) for z in range(min(z1, z2), max(z1, z2) + 1)}


def chunk_file(path):
    """Chunks listed in a file, one X,Z per line, or a JSON list of [X, Z] (an area.json's core also works)."""
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
        data = data["core"] if isinstance(data, dict) else data
        return {(int(x), int(z)) for x, z in data}
    except (ValueError, KeyError, TypeError):
        pass
    try:
        return {_pair(line) for line in text.split() if line.strip()}
    except ValueError:
        raise Refusal(f"{path} does not hold chunk coordinates", "Write one X,Z pair per line, or a JSON list of [X, Z] pairs") from None


def parts(cells):
    """Connected parts (8 neighbours), largest first."""
    left, out = set(cells), []
    while left:
        todo, part = [left.pop()], set()
        while todo:
            c = todo.pop()
            part.add(c)
            near = {(c[0] + dx, c[1] + dz) for dx, dz in NEIGHBOURS} & left
            left -= near
            todo += near
        out.append(part)
    return sorted(out, key=lambda p: (-len(p), min(p)))


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def windows(kept, ring=FAR):
    """One block window (x0, z0, x1, z1) per connected part of kept, grown by ring, chunk-aligned; overlapping windows merge."""
    boxes = [
        (
            min(x for x, _ in p) * CHUNK - ring,
            min(z for _, z in p) * CHUNK - ring,
            (max(x for x, _ in p) + 1) * CHUNK + ring,
            (max(z for _, z in p) + 1) * CHUNK + ring,
        )
        for p in parts(kept)
    ]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            j = next((j for j in range(i + 1, len(boxes)) if _overlap(boxes[i], boxes[j])), None)
            if j is not None:
                a, b = boxes[i], boxes.pop(j)
                boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                merged = True
                break
    return sorted(boxes)


def _gap_offsets(radius):
    """Chunk offsets whose box gap, in blocks, is at most radius."""
    n = radius // CHUNK + 1
    d = np.arange(-n, n + 1)
    dx, dz = np.meshgrid(d, d)
    gap = np.hypot(np.maximum(np.abs(dx) - 1, 0), np.maximum(np.abs(dz) - 1, 0)) * CHUNK
    return list(zip(dx[gap <= radius].tolist(), dz[gap <= radius].tolist(), strict=True))


def near(cells, radius):
    """Chunks whose box lies within radius blocks of a chunk in cells (the cells included)."""
    return {(x + dx, z + dz) for x, z in cells for dx, dz in _gap_offsets(radius)}


def boundary(cells):
    """Chunks of cells with a neighbour (8 directions) outside cells."""
    return {c for c in cells if any((c[0] + dx, c[1] + dz) not in cells for dx, dz in NEIGHBOURS)}


def distance(points, cells):
    """Euclidean distance in blocks from block columns (n, 2) to the nearest chunk of cells; 0 inside a chunk."""
    pts = np.asarray(points, np.int64).reshape(-1, 2)
    d = np.full(len(pts), np.inf, np.float32)
    for cx, cz in sorted(boundary(cells)):
        gx = np.maximum(np.maximum(cx * CHUNK - pts[:, 0], 0), pts[:, 0] - cx * CHUNK - (CHUNK - 1))
        gz = np.maximum(np.maximum(cz * CHUNK - pts[:, 1], 0), pts[:, 1] - cz * CHUNK - (CHUNK - 1))
        d = np.minimum(d, np.hypot(gx, gz).astype(np.float32))
    inside = np.array([tuple(c) in cells for c in (pts // CHUNK).tolist()], bool)
    d[inside] = 0
    return d


def nearest_point(point, cells):
    """The point of the chunk boxes of cells nearest to a block column."""
    p = np.asarray(point, float)
    boxes = np.array(sorted(boundary(cells)), float) * CHUNK
    q = np.clip(p, boxes, boxes + CHUNK - 1)
    return q[np.argmin(np.hypot(*(q - p).T))]


def chosen_claims(claims, teams):
    if not teams:
        return {}
    if teams == "all":
        return dict(claims)
    names = [t.strip() for t in teams.split(",") if t.strip()]
    unknown = [t for t in names if t not in claims]
    if unknown:
        raise Refusal(f"no claims for team {', '.join(unknown)}", f"Teams with claims here: {', '.join(sorted(claims)) or 'none'}")
    return {t: claims[t] for t in names}


def build(chosen, painted, erased, margin=DEFAULT_MARGIN):
    """core = chosen {team: claims} + painted - erased; kept = core grown by margin chunks."""
    if margin < MIN_MARGIN:
        raise Refusal(
            f"a margin of {margin} chunks leaves no safety band: new trees and lakes would spill into the kept area",
            "Use --margin 1 or more (1 is the default)",
        )
    claimed = set().union(*chosen.values()) if chosen else set()
    core = (claimed | set(painted)) - set(erased)
    if not core:
        raise Refusal("the area to keep is empty", "Import claims with --claims all, or add chunks with --add X1,Z1:X2,Z2 or --chunks FILE")
    kept = dilate(core, margin)
    return {
        "version": 1,
        "margin": margin,
        "teams": sorted(chosen),
        "claim_hash": claim_hash(claimed),
        "erased_claims": sorted(claimed & set(erased)),
        "core": sorted(core),
        "kept": sorted(kept),
        "windows": windows(kept),
    }


PARTS_WARNING = (
    "One RWG2 setting has to fit every part. Far-apart parts make a good match much less likely; keep only what you need, "
    "usually your base."
)


def suggest(table, min_ticks=SUGGEST_TICKS):
    """Chunks where players spent at least min_ticks (InhabitedTime), as a starting point to trim."""
    sel = table["inhabited"] >= min_ticks
    return set(zip(table["cx"][sel].tolist(), table["cz"][sel].tolist(), strict=True))


def gap(a, b):
    """The shortest distance in blocks between the chunk boxes of two sets of chunks."""
    pa, pb = (np.array(sorted(boundary(c)), np.int64) for c in (a, b))
    d = np.maximum(np.abs(pa[:, None, :] - pb[None, :, :]) - 1, 0)
    return float(np.hypot(d[..., 0], d[..., 1]).min() * CHUNK)


def _centre(cells):
    return np.mean(np.array(sorted(cells), float), axis=0) * CHUNK


def suggestion(table, claimed, min_ticks=SUGGEST_TICKS):
    """suggest() split into parts that stay apart once kept (MIN_MARGIN): the main part holds or touches the claimed chunks (without
    one, it is the part with the most InhabitedTime) and the others are optional extras, each with its size, its distance from the main
    part and the direction it lies in."""
    sel = suggest(table, min_ticks)
    groups = [p & sel for p in parts(dilate(sel, MIN_MARGIN))]
    if not groups:
        return {"main": [], "extra": []}
    ticks = dict(zip(cells(table), table["inhabited"].tolist(), strict=True))
    near = dilate(claimed, MIN_MARGIN)
    held = [i for i, p in enumerate(groups) if p & near] or [max(range(len(groups)), key=lambda i: sum(map(ticks.get, groups[i])))]
    main = set().union(*(groups[i] for i in held))
    extra = [groups[i] for i in range(len(groups)) if i not in held]
    return {"main": sorted(main), "extra": [_extra(p, main) for p in extra]}


def _extra(part, main):
    dx, dz = _centre(part) - _centre(main)
    return {"cells": sorted(part), "chunks": len(part), "blocks": round(gap(part, main)), "direction": verdicts.compass(dx, dz)}


def parts_warning(n):
    """The warning for a kept area of n separate parts, or None."""
    return f"The area to keep has {n} separate parts. {PARTS_WARNING}" if n > 1 else None


def _chunk(block_x, block_z):
    return block_x // CHUNK, block_z // CHUNK


def cells(table):
    return list(zip(table["cx"].tolist(), table["cz"].tolist(), strict=True))


def status(area, table):
    present_cells = cells(table)
    present = set(present_cells)
    populated = {c for c, p in zip(present_cells, table["populated"].tolist(), strict=True) if p}
    unreadable = {c for c, s in zip(present_cells, table["status"].tolist(), strict=True) if s == UNREADABLE}
    kept, core = set(map(tuple, area["kept"])), set(map(tuple, area["core"]))
    return {
        "core": len(core),
        "kept": len(kept),
        "margin_chunks": len(kept - core),
        "never_generated": sorted(kept - present),
        "unreadable_kept": sorted(kept & unreadable),
        "unpopulated_kept": sorted((kept & present) - populated - unreadable),
        "unpopulated_core": sorted((core & present) - populated - unreadable),
        "dropped": len(present - kept),
        "parts": len(parts(kept)),
        "windows": area["windows"],
    }


def unreadable_refusal(st):
    """The refusal of an area whose kept chunks do not all read in the world, or None: the preview's transplant could not keep
    them, so the search refuses it before its long run."""
    bad = st["unreadable_kept"]
    if not bad:
        return None
    shown = jobs.listed_cells(bad, SHOWN)
    return Refusal(
        f"{jobs.count(len(bad), 'chunk')} of the area to keep {'does' if len(bad) == 1 else 'do'} not read in the world ({shown}), "
        "so the preview could not keep them",
        "Close the game or stop the server, restore the region files that hold them from a backup (or leave those chunks out of "
        "the area), then make the area again",
    )


def record(area, table, scene):
    """(the area file's record, status, removed): the area with what it keeps and what leaving the rest removes."""
    st, rm = status(area, table), removed(area, table, scene)
    return area | {"status": {k: v for k, v in st.items() if k != "windows"}, "removed": rm}, st, rm


def scene(sc, claims, players, spawn):
    """What removed() and removed_cells() read from a world: its scan's tile entities, owner tags and graves, the claims, the
    players and the world spawn."""
    return {
        "tile_entities": sc.tile_entities(),
        "owners": sc.owners(),
        "graves": sc.graves(),
        "claims": claims,
        "players": players,
        "spawn": spawn,
    }


def grave_text(g):
    return f"{g['name']}'s grave at {g['x']}, {g['y']}, {g['z']} ({g['stacks']} stacks of items)"


@cache
def worldgen():
    """(tile-entity types, owner names) that world generation writes: sampled from chunks only the generator wrote, plus the
    structures a mod's code was seen to place (package data made by dev/sample_worldgen.py)."""
    d = json.loads(provision.data(WORLDGEN_FILE).read_text(encoding="utf-8"))
    return frozenset(d["types"]) | frozenset(d["structures"]), frozenset(d["owners"])


def player(name, worldgen_owners):
    """An owner name is a player's when it can be a Minecraft name or UUID (a mod's own name, such as Railcraft's "[Railcraft]" or
    BartWorks' "Ancient Cultures", cannot) and world generation never writes it."""
    return bool(PLAYER_NAME.fullmatch(name)) and name not in worldgen_owners


def player_signs(scene):
    """(cx, cz, type) of the tile entities that are a sign of a player: an owner tag naming a player, or a type world generation
    never makes, unless that tile entity names a mod as its owner (Railcraft's village workshop: "[Railcraft]")."""
    types, owners = worldgen()
    known = types | NATURAL_TES
    mods = {(cx, cz, t) for cx, cz, t, name in scene["owners"] if not player(name, owners)}
    made = {(cx, cz, t) for cx, cz, t, _ in scene["tile_entities"] if t not in known and (cx, cz, t) not in mods}
    return made | {(cx, cz, t) for cx, cz, t, name in scene["owners"] if player(name, owners)}


def player_chunks(scene):
    """Chunks with a sign of a player."""
    return {(cx, cz) for cx, cz, _ in player_signs(scene)}


def label(te):
    """Plain words for a tile-entity id: a known kind, else the words of its last part ("RCRollingMachineTile": "RC rolling
    machine")."""
    if te in TE_LABELS:
        return TE_LABELS[te]
    words = [w for w in ID_WORD.findall(re.split(r"[.:]", te)[-1]) if w.lower() not in ID_NOISE]
    return " ".join(w if w.isupper() else w.lower() for w in words) or te


def _group(counts):
    named = Counter()
    for te, n in counts.items():
        named[label(te)] += n
    return {"count": sum(counts.values()), "types": dict(counts.most_common()), "named": dict(named.most_common())}


def removed_cells(area, table, scene):
    """The chunks outside kept behind each line of removed(), so the UI can show them and add them to the area."""
    kept, chosen = set(map(tuple, area["kept"])), set(area["teams"])
    mine, never = player_chunks(scene), NATURAL_TES | {GRAVE}
    claimed = set().union(*(c for t, c in scene["claims"].items() if t not in chosen))
    layers = {
        "yours": {(cx, cz) for cx, cz, name, _ in scene["tile_entities"] if (cx, cz) in mine and name not in never},
        "graves": {tuple(g["chunk"]) for g in scene["graves"]},
        "portals": {c for c, n in zip(cells(table), table["portals"].tolist(), strict=True) if n},
        "claims": claimed,
        "players": {tuple(p["chunk"]) for p in scene["players"] if p["dim"] == 0},
        "beds": {tuple(p["bed_chunk"]) for p in scene["players"] if p["bed_chunk"]},
        "spawn": {_chunk(*scene["spawn"])},
    }
    return {k: sorted(v - kept) for k, v in layers.items()}


def _split(scene, kept):
    """(probably yours, beside them, from world generation): tile entities outside kept by type. Probably yours: those that are a
    player's sign themselves. Beside them: the others of a chunk with such a sign, of kinds world generation makes too (a
    village's chests above a mine's lights, or a player's chest by their machines), so they are not called yours. The natural
    ones (ores before conversion, spawners, plants, nodes) are on no line: new terrain makes them again. Graves have their own."""
    signs = player_signs(scene)
    mine, groups = {(cx, cz) for cx, cz, _ in signs}, (Counter(), Counter(), Counter())
    for cx, cz, name, n in scene["tile_entities"]:
        if (cx, cz) in kept or name in NATURAL_TES or name == GRAVE:
            continue
        groups[0 if (cx, cz, name) in signs else 1 if (cx, cz) in mine else 2][name] += n
    return (_group(c) for c in groups)


def removed(area, table, scene):
    """What leaves the overworld outside kept: tile entities probably yours, beside them and from world generation, graves,
    portal blocks, claims, players, beds, spawn."""
    kept = set(map(tuple, area["kept"]))
    yours, beside, made = _split(scene, kept)
    portals = sum(n for c, n in zip(cells(table), table["portals"].tolist(), strict=True) if c not in kept)
    others = {t: len(cells - kept) for t, cells in scene["claims"].items() if t not in area["teams"] and cells - kept}
    dim0 = [p for p in scene["players"] if p["dim"] == 0]
    return {
        "yours": yours,
        "beside": beside,
        "worldgen": made,
        "graves": [{k: g[k] for k in GRAVE_KEYS} for g in scene["graves"] if tuple(g["chunk"]) not in kept],
        "portal_blocks": portals,
        "other_teams_claims": others,
        "erased_claims": len(area["erased_claims"]),
        "players": [p["name"] for p in dim0 if tuple(p["chunk"]) not in kept],
        "beds": [p["name"] for p in scene["players"] if p["bed_chunk"] and tuple(p["bed_chunk"]) not in kept],
        "spawn_in_kept": _chunk(*scene["spawn"]) in kept,
    }
