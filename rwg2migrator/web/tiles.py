"""Map tiles for the UI: 256-pixel PNGs cut from larger blocks, cached on disk per layer, zoom and position."""

import io
import os
import threading
from collections import defaultdict

import numpy as np
from PIL import Image

from .. import appearance, maps, score, search
from ..region import write_atomic

VERSION = 2
TILE = 256
ZOOMS = (-4, -3, -2, -1, 0)
WORLD_BLOCK = 1024
RENDER_SPAN = 2048
LOOK_STRIDE = 4
BIOME_IDS = 1 << 16
UNKNOWN_LOOK = appearance.LOOKS.index("dull")
UNSEEN = -2
WATER = appearance.LOOKS.index("water")
FIELDS = ("ground", "biome", "top", "water")


def look_rules():
    """What a tile's colours follow beyond its layer's own inputs, read at each call: a change to any of them gives every layer
    a new cache folder."""
    a = appearance
    looks = [a.LOOKS, a.NAME_RULES, a.BAD_LOOKS, a.BAD_SHARE, a.GOOD, a.BAD_ZERO, a.DELTA_E_ZERO, a.SURFACES, a.BAD, a.PARTIAL]
    return [*looks, a.MARSH_DEPTH, maps.COLOURS, maps.TAINT]


def bpp_of(zoom):
    if zoom not in ZOOMS:
        raise ValueError(f"zoom {zoom} is not one of {ZOOMS}")
    return 1 << -zoom


def render_block(bpp):
    """Pixels a side of one candidate render: the generator's cost follows the chunks a render covers, so coarse zooms render
    smaller blocks."""
    return max(TILE, min(WORLD_BLOCK, RENDER_SPAN // bpp))


def biome_lut(sc, ref, release):
    """Look index per biome id over the whole scan (every LOOK_STRIDE-th column), so tiles never change colour at their edges;
    UNSEEN for ids the scan does not hold."""
    parts = {k: [] for k in FIELDS}
    for npz in sc.regions.values():
        with np.load(npz) as z:
            for k, v in parts.items():
                v.append(z[k][::LOOK_STRIDE, ::LOOK_STRIDE].ravel())
    lut = np.full(BIOME_IDS, UNSEEN, np.int8)
    if parts["ground"]:
        for bid, _, look in maps.biome_looks({k: np.concatenate(v) for k, v in parts.items()}, ref, release):
            lut[bid] = look
    return lut


def merge(first, then):
    """A look table that judges each biome as first does, and the ones first never saw as then does: a working copy's map then
    colours the kept chunks exactly as the world's map does (a biome is judged at its median height, and the kept part alone of
    a biome can sit far higher than the whole world's)."""
    return np.where(first != UNSEEN, first, then).astype(np.int8)


def real_looks(sc, lut, box, bpp):
    """(look index, ground) per pixel of the real world over box; -1 where nothing was explored."""
    g, b, w = (sc.grid(k, box, bpp) for k in ("ground", "biome", "water"))
    look = lut[b]
    look[look == UNSEEN] = UNKNOWN_LOOK
    look[w > 0] = WATER
    look[np.isnan(g)] = -1
    return look, g


def world_draw(sc, lut):
    """Draws the scanned world over a box; one pixel more on each side keeps the hill shade even across block edges."""

    def draw(box, bpp):
        x0, z0, x1, z1 = box
        look, g = real_looks(sc, lut, (x0 - bpp, z0 - bpp, x1 + bpp, z1 + bpp), bpp)
        return maps.shade(maps.palette_rgb(look), g, bpp)[1:-1, 1:-1]

    return draw


def candidate_draw(scene, lut, factory):
    """Draws a candidate: the kept area as it is, RWG2 around it (Tainted Land marked), from cached generator renders."""
    kept = set(map(tuple, scene.area["kept"]))
    ref = maps.reference_of(scene)

    def draw(box, bpp):
        c_look, top, taint = maps.candidate_looks(maps.Renderer(scene.pick, factory), scene, ref, (box, bpp))
        look, g = real_looks(scene.scan, lut, box, bpp)
        inside = maps.cell_mask(kept, box, bpp)
        rgb = maps.palette_rgb(np.where(inside, look, c_look))
        rgb[taint & ~inside] = maps.TAINT
        return maps.shade(rgb, np.where(inside, g, top), bpp)

    return draw


def planned_draw(now, render, judge, kept):
    """Draws what a preview planned: the kept area as the world holds it now (now = (scan, look table)), and around it the new
    terrain of the preview's render (Tainted Land marked), unexplored past the render."""
    sc, lut = now
    names = render.names or ["?"]
    looks = judge.lut(names)[0]
    taint = search.taint_ids(names)

    def draw(box, bpp):
        top, idx = render.grid(box, bpp), render.biome_grid(box, bpp)
        r_look = np.where(idx >= 0, looks[np.maximum(idx, 0)], -1)
        r_look = np.where(np.isfinite(top) & (top < score.SEA), WATER, r_look)
        look, g = real_looks(sc, lut, box, bpp)
        inside = maps.cell_mask(kept, box, bpp)
        rgb = maps.palette_rgb(np.where(inside, look, r_look))
        rgb[np.isin(idx, taint) & ~inside] = maps.TAINT
        return maps.shade(rgb, np.where(inside, g, top), bpp)

    return draw


def png(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", compress_level=1)
    return buf.getvalue()


class Layer:
    """Tiles of one picture of the world. A missing tile draws its whole block once, and every tile of it is kept on disk, so a
    second visit draws nothing. drawn counts the blocks drawn, redrawn those drawn again in this process; each listener gets
    (folder, zoom, (first tile x, first tile y, tiles a side)) once a block's tiles are on disk."""

    drawn = redrawn = 0
    _count, _seen = threading.Lock(), set()
    listeners = []

    def __init__(self, folder, draw, block=lambda bpp: WORLD_BLOCK):
        self.folder, self.draw, self.block = folder, draw, block
        self._locks, self._guard = defaultdict(threading.Lock), threading.Lock()

    def path(self, zoom, tx, ty):
        return os.path.join(self.folder, str(zoom), f"{tx}_{ty}.png")

    def tile(self, zoom, tx, ty):
        """The tile's file, drawn first (with its whole block) when missing."""
        path = self.path(zoom, tx, ty)
        if os.path.isfile(path):
            return path
        bpp = bpp_of(zoom)
        n = self.block(bpp) // TILE
        bx, by = tx // n, ty // n
        with self._guard:
            lock = self._locks[(zoom, bx, by)]
        with lock:
            if os.path.isfile(path):
                return path
            self._draw_block(zoom, (bx, by), n)
        return path

    def _draw_block(self, zoom, block, n):
        bpp, (bx, by) = bpp_of(zoom), block
        span = n * TILE * bpp
        rgb = self.draw((bx * span, by * span, (bx + 1) * span, (by + 1) * span), bpp)
        os.makedirs(os.path.join(self.folder, str(zoom)), exist_ok=True)
        for j in range(n):
            for i in range(n):
                write_atomic(self.path(zoom, bx * n + i, by * n + j), png(rgb[j * TILE : (j + 1) * TILE, i * TILE : (i + 1) * TILE]))
        with Layer._count:
            Layer.drawn += 1
            Layer.redrawn += (self.folder, zoom, block) in Layer._seen
            Layer._seen.add((self.folder, zoom, block))
        for listen in Layer.listeners:
            listen(self.folder, zoom, (bx * n, by * n, n))


def hexa(rgb):
    return "#" + "".join(f"{int(c):02x}" for c in rgb)


def heat(points, steps):
    """[[x, z, step, colour], ...] for the seam-step dots along a border."""
    return [[int(x), int(z), float(s), hexa(maps.heat_colour(s))] for (x, z), s in zip(points.tolist(), steps.tolist(), strict=True)]


def kept_border(sc, kept, outside_top):
    """Steps across the kept border: the real ground inside, outside_top(points) outside; (outer points, steps)."""
    inner, outer = score.edge_pairs(set(map(tuple, kept)))
    real = sc.at("ground", inner)
    ok = np.isfinite(real)
    top = np.asarray(outside_top(outer[ok]), np.float64)
    known = np.isfinite(top)
    return outer[ok][known], np.abs(real[ok][known] - top[known])


def legend():
    looks = [{"label": appearance.LOOK_LABELS[k], "colour": hexa(maps.COLOURS[k])} for k in appearance.LOOKS]
    return {
        "looks": looks + [{"label": maps.UNEXPLORED.capitalize(), "colour": hexa(maps.UNGENERATED)}],
        "taint": hexa(maps.TAINT),
        "heat": [{"step": t, "colour": hexa(maps.heat_colour(t))} for t in maps.HEAT_TICKS],
    }
