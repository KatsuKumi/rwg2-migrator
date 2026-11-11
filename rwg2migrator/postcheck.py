"""The post-check of a preview, and of an applied world later: game vs generator render in the new chunks, the seams at the core
and kept borders before and after smoothing, protected chunks, files outside the overworld and region structure [POC
rwg2_postcheck, rwg2_smooth.report]."""

import os

import numpy as np

from . import area, chunks, region, score, transplant, world
from .ground import column_ground, ground_ids

PCT = 100
P90 = 90
DECIMALS = 2
SIDE = chunks.SIDE
STEPS = (10, 30)
OFF_BLOCKS = 1
CLUSTER = 4


class Columns:
    """Ground per column from {(cx, cz): 16x16 grid}; NaN where no chunk is known."""

    def __init__(self, grids):
        self.grids = grids

    def at(self, xz):
        xz = np.asarray(xz, np.int64).reshape(-1, 2)
        out = np.full(len(xz), np.nan)
        cells = xz // SIDE
        for i, (c, (x, z)) in enumerate(zip(map(tuple, cells.tolist()), xz.tolist(), strict=True)):
            g = self.grids.get(c)
            if g is not None:
                out[i] = g[z - c[1] * SIDE, x - c[0] * SIDE]
        return out


def registry_ok(world_dir):
    return ground_ids(world.registry(world.read_level(world_dir)))


def chunk_ground(world_dir, cells, ok=None):
    """Columns of the cells present in world_dir's overworld region files."""
    ok = registry_ok(world_dir) if ok is None else ok
    out, wanted = {}, {(x // region.SIDE, z // region.SIDE) for x, z in cells}
    for path in transplant.region_files(world_dir):
        rx, rz = region.coords(path)
        if (rx, rz) not in wanted:
            continue
        with open(path, "rb") as f:
            recs = region.records(f.read())
        for i, (rec, _) in recs.items():
            pos = region.slot_pos(rx, rz, i)
            got = transplant.decode(rec, pos) if pos in cells and rec is not None else None
            if got is not None:
                level = got[1]["Level"][1]
                out[pos] = column_ground(chunks.block_ids(level), chunks.heightmap(level), ok)
    return Columns(out)


def seam(columns, cells):
    """Steps across the border of cells where both columns have ground: pairs, median, p90, % over 10 and 30, max."""
    inner, outer = score.edge_pairs(cells)
    a, b = columns.at(inner), columns.at(outer)
    good = np.isfinite(a) & np.isfinite(b)
    if not good.any():
        return {"pairs": 0}
    r = score.seam(a[good], b[good])
    out = {"pairs": r["pairs"], "median": r["median"], "p90": r["p90"], "max": r["max"]}
    return out | {f"over{t}_pct": round(PCT * r[f"over{t}"], DECIMALS) for t in STEPS}


def new_terrain(columns, render, cells):
    """Game ground vs RWG2's top of ground from the render, over every column of cells."""
    xz = np.array([(cx * SIDE + x, cz * SIDE + z) for cx, cz in sorted(cells) for z in range(SIDE) for x in range(SIDE)], np.int64)
    if not len(xz):
        return {"columns": 0}
    game, want = columns.at(xz), render.at(xz)
    good = np.isfinite(game) & np.isfinite(want)
    d = np.abs(game[good] - want[good])
    if not d.size:
        return {"columns": 0, "outside_render": int(np.sum(np.isfinite(game) & ~np.isfinite(want)))}
    return {
        "columns": int(d.size),
        "median": float(np.median(d)),
        "exact_pct": round(PCT * float(np.mean(d == 0)), DECIMALS),
        "within1_pct": round(PCT * float(np.mean(d <= 1)), DECIMALS),
        "outside_render": int(np.sum(np.isfinite(game) & ~np.isfinite(want))),
    }


def reshaped(before, after, cells):
    """Columns of cells whose ground differs between two states."""
    xz = np.array([(cx * SIDE + x, cz * SIDE + z) for cx, cz in sorted(cells) for z in range(SIDE) for x in range(SIDE)], np.int64)
    if not len(xz):
        return 0
    a, b = before.at(xz), after.at(xz)
    return int(np.sum(np.isfinite(a) & np.isfinite(b) & (a != b)))


def region_problems(world_dir):
    """Region files that are not whole sectors, have overlapping or out-of-file entries, or a record that does not read."""
    out = []
    for path in transplant.region_files(world_dir):
        with open(path, "rb") as f:
            data = f.read()
        recs = region.records(data)
        bad = [i for i, (rec, _) in recs.items() if rec is None]
        problem = f"slot {bad[0]} has a bad header entry" if bad else region.check(data, recs)
        if problem:
            out.append(f"{os.path.basename(path)}: {problem}")
    return out


def border_cells(a):
    """The chunks whose ground the border seams read: core's own border chunks, the margin band and the first ring."""
    core, kept = ({tuple(c) for c in a[k]} for k in ("core", "kept"))
    return (area.dilate(kept, 1) - core) | area.boundary(core)


def seams(columns, a):
    core, kept = ({tuple(c) for c in a[k]} for k in ("core", "kept"))
    return {"core_border": seam(columns, core), "kept_border": seam(columns, kept)}


def saved_since(world_dir, applied_at, skip=()):
    """Chunks saved at or after applied_at, outside the region files named in skip."""
    out = set()
    for path in transplant.region_files(world_dir):
        if os.path.basename(path) in skip:
            continue
        rx, rz = region.coords(path)
        with open(path, "rb") as f:
            out |= {region.slot_pos(rx, rz, i) for i, *_, ts in region.slots(f.read()) if ts >= applied_at}
    return out


def off_chunks(columns, render, cells):
    """(chunks of cells whose game ground is off the render by a median of OFF_BLOCKS or more, chunks compared at all)."""
    off, compared = set(), 0
    for cx, cz in sorted(cells):
        game = columns.grids.get((cx, cz))
        if game is None:
            continue
        d = np.abs(game - render.grid((cx * SIDE, cz * SIDE, (cx + 1) * SIDE, (cz + 1) * SIDE)))
        d = d[np.isfinite(d)]
        compared += bool(d.size)
        if d.size and np.median(d) >= OFF_BLOCKS:
            off.add((cx, cz))
    return off, compared


def mismatch(since):
    """Whether the chunks generated since Apply hold a patch of CLUSTER or more touching chunks that are each off the preview: a
    single chunk can be off (lakes and features the render does not hold), a patch generated without the offsets is."""
    return bool(since) and since.get("off_cluster", 0) >= CLUSTER


def since(world_dir, a, preview, applied):
    """In an applied world: game vs the preview's render in the chunks generated since Apply, and the seam at the borders now.
    preview = (the harvested chunks, the render); applied = (Apply's time, region files to leave out: leftovers of an upload
    merged into the old folder hold old terrain, which verify reports on its own)."""
    harvested, render = preview
    kept = {tuple(c) for c in a["kept"]}
    new = saved_since(world_dir, *applied) - kept - {tuple(c) for c in harvested}
    ok = registry_ok(world_dir)
    cols = chunk_ground(world_dir, new, ok)
    off, compared = off_chunks(cols, render, new)
    patch = area.parts(off)[0] if off else set()
    first = min(patch, default=None)
    return {
        "new_chunks": len(new),
        "new_terrain": new_terrain(cols, render, new),
        "compared_chunks": compared,
        "off_chunks": len(off),
        "off_cluster": len(patch),
        "off_at": [first[0] * SIDE + SIDE // 2, first[1] * SIDE + SIDE // 2] if first else None,
        "seams": seams(chunk_ground(world_dir, border_cells(a), ok), a),
    }
