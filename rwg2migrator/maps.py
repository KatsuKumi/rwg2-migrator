"""POC-style PNG maps: hill-shaded look colours, the kept outline and margin, seam heat, Tainted Land markers, legend, scale bar."""

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import Refusal, appearance, score, search
from .generator import RECORD

CHUNK = 16
BPPS = (1, 2, 4, 8, 16)
MAX_PANEL = 1024
COLOURS = {
    "green": (0x4F, 0x9A, 0x3A),
    "dull": (0xA6, 0xA2, 0x48),
    "sand": (0xEE, 0xD9, 0x9A),
    "snow": (0xF4, 0xF7, 0xFB),
    "wet": (0x6E, 0x55, 0x38),
    "water": (0x28, 0x7E, 0xB3),
    "other": (0xB0, 0x3C, 0xC8),
}
UNGENERATED = (40, 40, 44)
UNEXPLORED = "not explored"
MAPS, RENDER_FILE = "maps", "render.npz"
HEAT = (
    (0, (0xC7, 0xE9, 0xA0)),
    (2, (0xE6, 0xD5, 0x6A)),
    (5, (0xF2, 0x9E, 0x3E)),
    (10, (0xDF, 0x5A, 0x34)),
    (20, (0xB0, 0x22, 0x2B)),
    (30, (0x5E, 0x0A, 0x18)),
)
HEAT_TICKS = (0, 2, 5, 10, 20, 30)
CORE_LINE = (0, 0, 0)
CORE_HALO = (255, 255, 255)
KEPT_LINE = (255, 255, 255)
TAINT = (0xFF, 0x2E, 0xD1)
TEXT = (255, 255, 255)
BACK = (0, 0, 0)
SHADE_SLOPE, SHADE_LO, SHADE_HI = 0.06, 0.55, 1.35
GAP = 16
PAD = 12
TITLE_SIZE, TEXT_SIZE = 20, 15
LEGEND_W = 250
ROW = 22
SWATCH = 16
HEAT_W = 200
HEAT_H = 12
BAR_BLOCKS = (250, 500, 1000, 2000)
BAR_MIN_PX = 80
HEAT_DOT = 2
MARK = 9
ARROW = 26


def font(size):
    return ImageFont.load_default(size=size)


def heat_colour(step):
    """Green to orange to red with falling lightness, so the order also reads without colour."""
    xs = [s for s, _ in HEAT]
    cs = np.array([c for _, c in HEAT], float)
    return tuple(int(round(np.interp(min(step, xs[-1]), xs, cs[:, i]))) for i in range(3))


def choose_bpp(window, bpp=None):
    x0, z0, x1, z1 = window
    if bpp is not None:
        if bpp not in BPPS:
            raise Refusal(f"--bpp {bpp} is not one of {', '.join(map(str, BPPS))}", "Pick one of these")
        return bpp
    return next((b for b in BPPS if max(x1 - x0, z1 - z0) / b <= MAX_PANEL), BPPS[-1])


def window_arg(text):
    try:
        x0, z0, size = (int(v) for v in text.split(","))
    except ValueError:
        raise Refusal(f"--window {text!r} is not X0,Z0,SIZE", "Write it in blocks, for example --window=1000,-500,2048") from None
    return x0, z0, x0 + size, z0 + size


def shade(rgb, height, bpp):
    h = np.nan_to_num(height, nan=np.nanmean(height) if np.isfinite(height).any() else score.SEA)
    gz, gx = np.gradient(h)
    light = np.clip(1.0 + (-gx - gz) / bpp * SHADE_SLOPE, SHADE_LO, SHADE_HI)
    out = rgb.astype(np.float32) * light[..., None]
    out[np.isnan(height)] = UNGENERATED
    return np.clip(out, 0, 255).astype(np.uint8)


def palette_rgb(look_idx):
    lut = np.array([COLOURS[k] for k in appearance.LOOKS] + [UNGENERATED], np.uint8)
    return lut[np.where(look_idx < 0, len(appearance.LOOKS), look_idx)]


@dataclass
class Scene:
    scan: object
    area: dict
    release: str
    pick: dict
    windows: list
    bpp: int | None = None


def pick(job, rank):
    """The candidate of a search job at rank (1 = best), with the job's reference look and bans; the alternatives are numbered on
    after the candidates."""
    path = os.path.join(job.folder, "candidates", "candidates.json")
    if not os.path.isfile(path):
        raise search.no_candidates(job)
    result = search.load(path)
    pool = result["candidates"] + result.get("alternatives", [])
    if not 1 <= rank <= len(pool):
        raise Refusal(
            f"job {job.id} has {len(result['candidates'])} candidates and {len(pool) - len(result['candidates'])} alternatives, "
            f"not #{rank}",
            f"Pick one of them with --pick N (1 to {len(pool)}; rwg2-migrator candidates {job.id} --all lists them)",
        )
    return {"job": job, "candidate": pool[rank - 1], "reference": result["reference"], "bans": result["bans"], "rank": rank}


def reference_of(scene):
    if scene.pick:
        return appearance.Reference.from_dict(scene.pick["reference"])
    return appearance.reference(scene.scan, scene.area["core"], scene.release)


def real_looks(scene, ref, box, bpp):
    """(look index per pixel, ground per pixel) of the real world over box, -1 where nothing was explored."""
    sl = (slice(None, None, bpp), slice(None, None, bpp))
    grids = {k: scene.scan.grid(k, box)[sl] for k in ("ground", "biome", "top", "water")}
    look = np.full(grids["ground"].shape, -1, np.int8)
    for _, sel, value in biome_looks(grids, ref, scene.release):
        look[sel] = value
    look[(grids["water"] > 0) & (look >= 0)] = appearance.LOOKS.index("water")
    return look, grids["ground"]


def biome_looks(cols, ref, release):
    """(biome id, its explored columns, look index) per biome in cols (ground, biome, top and water arrays of one shape), judged
    over those columns: their top-block shares and their colour at their median ground height."""
    names, tab = appearance.biome_names(release), appearance.table(release)
    for bid in np.unique(cols["biome"]):
        sel = (cols["biome"] == bid) & np.isfinite(cols["ground"])
        if not sel.any():
            continue
        e = tab.get(appearance.fold(names.get(int(bid), "")))
        if e is None:
            yield int(bid), sel, appearance.LOOKS.index("dull")
            continue
        sh = appearance.shares(cols["top"][sel], cols["water"][sel])
        y = int(np.median(cols["ground"][sel])) - 1
        value = appearance.score(sh, appearance.colour_at(e, y, int(bid)), ref)
        yield int(bid), sel, appearance.LOOKS.index(appearance.look_of(e["name"], sh, e["ocean"], value))


class Renderer:
    """RWG2 renders of a picked candidate, cached in its job folder as steps with output manifests, so a render torn by a kill or
    a power cut is drawn again instead of trusted; the generator starts only for a missing one."""

    def __init__(self, chosen, factory):
        self.pick = chosen
        self.sampler = search.Lazy(factory)

    def step(self, window, bpp):
        """(step name, input keys) of one render; the keys hold the [seed, config key] the generator boots on (pick's
        "generator"), so a render a generator for another world or config drew is never taken for this one."""
        config = self.pick["candidate"]["config"]
        name = "render-" + "_".join(map(str, config)) + f"-{window[0]}-{window[1]}-{window[2]}-{bpp}"
        keys = {"config": list(config), "window": list(window), "bpp": bpp, "generator": self.pick.get("generator")}
        return os.path.join(MAPS, name), keys

    def cached(self, window, bpp):
        return self.pick["job"].step_valid(*self.step(window, bpp))

    def render(self, window, bpp):
        job, (step, keys) = self.pick["job"], self.step(window, bpp)
        path = os.path.join(job.step_dir(step), RENDER_FILE)
        if self.cached(window, bpp):
            with np.load(path) as z:
                return z["rec"].view(RECORD).reshape(z["shape"]), [str(n) for n in z["names"]]
        rec, names = self.sampler.render(self.pick["candidate"]["config"], window, bpp)
        np.savez_compressed(path, rec=np.ascontiguousarray(rec).view(np.uint8).ravel(), shape=np.array(rec.shape), names=np.array(names))
        job.finish_step(step, [RENDER_FILE], keys)
        return rec, names


def cached(chosen, windows, bpp):
    r = Renderer(chosen, None)
    return all(r.cached(render_window(snap(w), choose_bpp(snap(w), bpp)), choose_bpp(snap(w), bpp)) for w in windows)


def snap(box):
    """The chunk-aligned box that covers box: its sides are whole pixels at every bpp, so the world's and the render's looks of a
    window line up."""
    x0, z0, x1, z1 = box
    return x0 // CHUNK * CHUNK, z0 // CHUNK * CHUNK, -(-x1 // CHUNK) * CHUNK, -(-z1 // CHUNK) * CHUNK


def render_window(box, bpp):
    """A square chunk-aligned render window (x0, z0, size) that covers box at bpp."""
    x0, z0, x1, z1 = box
    unit = CHUNK * bpp
    size = -(-max(x1 - x0, z1 - z0) // unit) * unit
    return (x0 // CHUNK) * CHUNK, (z0 // CHUNK) * CHUNK, size


def candidate_looks(renderer, scene, ref, view):
    """(look index, top of ground, taint mask) per pixel of the candidate over the view (box, bpp)."""
    box, bpp = view
    win = render_window(box, bpp)
    rec, names = renderer.render(win, bpp)
    oz, ox = (box[1] - win[1]) // bpp, (box[0] - win[0]) // bpp
    h, w = (box[3] - box[1]) // bpp, (box[2] - box[0]) // bpp
    rec = rec[oz : oz + h, ox : ox + w]
    judge = score.Judge(ref, frozenset(scene.pick["bans"]), appearance.table(scene.release))
    look, _, _ = judge.columns(rec, names)
    top = score.surface(rec["h"])
    look = np.where(top < score.SEA, appearance.LOOKS.index("water"), look)
    return look.astype(np.int8), top, np.isin(rec["palette"], search.taint_ids(names))


def cell_mask(cells, box, bpp):
    x0, z0, x1, z1 = box
    m = np.zeros(((z1 - z0) // bpp, (x1 - x0) // bpp), bool)
    for cx, cz in cells:
        px0, pz0 = max(cx * CHUNK - x0, 0) // bpp, max(cz * CHUNK - z0, 0) // bpp
        px1, pz1 = min(cx * CHUNK + CHUNK - x0, x1 - x0) // bpp, min(cz * CHUNK + CHUNK - z0, z1 - z0) // bpp
        if px0 < px1 and pz0 < pz1:
            m[pz0:pz1, px0:px1] = True
    return m


def outline(draw, cells, view, style):
    """Lines along every chunk edge of cells that borders a chunk outside cells; view is (box, bpp), style (colour, width)."""
    (x0, z0, _, _), bpp = view
    colour, width = style
    s = CHUNK / bpp
    for cx, cz in cells:
        px, pz = (cx * CHUNK - x0) / bpp, (cz * CHUNK - z0) / bpp
        sides = (((0, -1), (px, pz, px + s, pz)), ((0, 1), (px, pz + s, px + s, pz + s)))
        sides += (((-1, 0), (px, pz, px, pz + s)), ((1, 0), (px + s, pz, px + s, pz + s)))
        for (dx, dz), seg in sides:
            if (cx + dx, cz + dz) not in cells:
                draw.line(seg, fill=colour, width=width)


def kept_steps(scene, top, box, bpp):
    """Seam steps where the candidate meets the kept area: (pixel x, pixel z, step) per kept-border pair inside the window."""
    kept = set(map(tuple, scene.area["kept"]))
    inner, outer = score.edge_pairs(kept)
    real = scene.scan.at("ground", inner)
    px, pz = (outer[:, 0] - box[0]) // bpp, (outer[:, 1] - box[1]) // bpp
    ok = np.isfinite(real) & (px >= 0) & (pz >= 0) & (px < top.shape[1]) & (pz < top.shape[0])
    return px[ok], pz[ok], np.abs(real[ok] - top[pz[ok], px[ok]])


def heat(draw, steps):
    px, pz, step = steps
    for x, z, s in sorted(zip(px.tolist(), pz.tolist(), step.tolist(), strict=True), key=lambda t: t[2]):
        draw.rectangle([x - HEAT_DOT, z - HEAT_DOT, x + HEAT_DOT, z + HEAT_DOT], fill=heat_colour(s))


def taint_marker(img, box, bpp, taint):
    """A marker at the nearest Tainted Land, or an arrow at the window edge pointing to it."""
    if not taint:
        return
    d = ImageDraw.Draw(img)
    x, z = (taint["point"][0] - box[0]) / bpp, (taint["point"][1] - box[1]) / bpp
    label = f"Tainted Land {round(taint['blocks'])} blocks {taint['direction']}"
    inside = 0 <= x < img.width and 0 <= z < img.height
    if not inside:
        cx, cz = img.width / 2, img.height / 2
        t = min((img.width / 2 - ARROW) / max(abs(x - cx), 1), (img.height / 2 - ARROW) / max(abs(z - cz), 1))
        x, z = cx + (x - cx) * t, cz + (z - cz) * t
        label += " (off the map)"
    d.ellipse([x - MARK, z - MARK, x + MARK, z + MARK], outline=TAINT, width=3)
    d.line([x - MARK, z - MARK, x + MARK, z + MARK], fill=TAINT, width=3)
    d.line([x - MARK, z + MARK, x + MARK, z - MARK], fill=TAINT, width=3)
    tx = min(max(x + MARK + 4, PAD), img.width - d.textlength(label, font=font(TEXT_SIZE)) - PAD)
    d.text((tx, z - MARK), label, font=font(TEXT_SIZE), fill=TEXT, stroke_width=2, stroke_fill=BACK)


def scale_bar(img, bpp):
    d = ImageDraw.Draw(img)
    blocks = next((b for b in BAR_BLOCKS if b / bpp >= BAR_MIN_PX), BAR_BLOCKS[-1])
    w = blocks / bpp
    x, y = PAD, img.height - PAD - HEAT_H
    d.rectangle([x, y, x + w, y + HEAT_H // 2], fill=TEXT, outline=BACK, width=2)
    d.text((x, y - TEXT_SIZE - 6), f"{blocks} blocks", font=font(TEXT_SIZE), fill=TEXT, stroke_width=2, stroke_fill=BACK)


def title(img, text):
    ImageDraw.Draw(img).text((PAD, PAD), text, font=font(TITLE_SIZE), fill=TEXT, stroke_width=3, stroke_fill=BACK)


def legend(height, with_candidate):
    img = Image.new("RGB", (LEGEND_W, height), (24, 24, 28))
    d = ImageDraw.Draw(img)
    y = PAD
    for look in appearance.LOOKS:
        d.rectangle([PAD, y, PAD + SWATCH, y + SWATCH], fill=COLOURS[look], outline=BACK)
        d.text((PAD + SWATCH + 8, y), appearance.LOOK_LABELS[look], font=font(TEXT_SIZE), fill=TEXT)
        y += ROW
    d.rectangle([PAD, y, PAD + SWATCH, y + SWATCH], fill=UNGENERATED, outline=(90, 90, 90))
    d.text((PAD + SWATCH + 8, y), UNEXPLORED.capitalize(), font=font(TEXT_SIZE), fill=TEXT)
    y += ROW * 2
    d.line([PAD, y + 8, PAD + SWATCH, y + 8], fill=CORE_HALO, width=4)
    d.line([PAD, y + 8, PAD + SWATCH, y + 8], fill=CORE_LINE, width=2)
    d.text((PAD + SWATCH + 8, y), "Protected area (kept as is)", font=font(TEXT_SIZE), fill=TEXT)
    y += ROW
    d.line([PAD, y + 8, PAD + SWATCH, y + 8], fill=KEPT_LINE, width=2)
    d.text((PAD + SWATCH + 8, y), "Kept area (with safety margin)", font=font(TEXT_SIZE), fill=TEXT)
    if with_candidate:
        y += ROW
        d.ellipse([PAD + 2, y + 2, PAD + SWATCH - 2, y + SWATCH - 2], outline=TAINT, width=3)
        d.text((PAD + SWATCH + 8, y), "Tainted Land", font=font(TEXT_SIZE), fill=TEXT)
        y += ROW * 2
        d.text((PAD, y), "Step at the kept edge (blocks)", font=font(TEXT_SIZE), fill=TEXT)
        y += ROW
        for i in range(HEAT_W):
            d.line([PAD + i, y, PAD + i, y + HEAT_H], fill=heat_colour(HEAT_TICKS[-1] * i / (HEAT_W - 1)))
        for t in HEAT_TICKS:
            x = PAD + (HEAT_W - 1) * t / HEAT_TICKS[-1]
            d.line([x, y + HEAT_H, x, y + HEAT_H + 4], fill=TEXT)
            d.text((x - 4, y + HEAT_H + 5), f"{t}+" if t == HEAT_TICKS[-1] else str(t), font=font(TEXT_SIZE - 2), fill=TEXT)
    return img


def panel(rgb, height, bpp):
    return Image.fromarray(shade(rgb, height, bpp))


def window_panels(scene, renderer, box, ref):
    box = snap(box)
    bpp = choose_bpp(box, scene.bpp)
    core, kept = set(map(tuple, scene.area["core"])), set(map(tuple, scene.area["kept"]))
    look, ground = real_looks(scene, ref, box, bpp)
    now = panel(palette_rgb(look), ground, bpp)
    panels, facts = [now], {"window": list(box), "bpp": bpp}
    if scene.pick:
        c_look, top, taint = candidate_looks(renderer, scene, ref, (box, bpp))
        inside = cell_mask(kept, box, bpp)
        mixed = np.where(inside, look, c_look)
        rgb = palette_rgb(mixed)
        rgb[taint & ~inside] = TAINT
        panels.append(panel(rgb, np.where(inside, ground, top), bpp))
    for p in panels:
        d = ImageDraw.Draw(p)
        outline(d, kept, (box, bpp), (KEPT_LINE, 2))
        outline(d, core, (box, bpp), (CORE_HALO, 4))
        outline(d, core, (box, bpp), (CORE_LINE, 2))
        scale_bar(p, bpp)
    if scene.pick:
        steps = kept_steps(scene, top, box, bpp)
        heat(ImageDraw.Draw(panels[1]), steps)
        taint_marker(panels[1], box, bpp, scene.pick["candidate"]["taint"])
        facts |= {"seam_pairs": int(len(steps[2])), "seam_median": float(np.median(steps[2])) if len(steps[2]) else None}
    title(now, "NOW: your world")
    if scene.pick:
        bx, bz, lx, lz = scene.pick["candidate"]["config"]
        title(panels[1], f"CANDIDATE #{scene.pick['rank']}: your kept area inside RWG2 Smooth")
        facts["config"] = [bx, bz, lx, lz]
    return panels, facts


def draw(scene, renderer, out):
    """One row per window (NOW, plus CANDIDATE when a search pick is given), and a legend; returns what was drawn."""
    ref = reference_of(scene)
    rows, facts = [], []
    for box in scene.windows:
        panels, f = window_panels(scene, renderer, box, ref)
        rows.append(panels)
        facts.append(f)
    width = max(sum(p.width for p in r) + GAP * (len(r) - 1) for r in rows) + GAP + LEGEND_W
    height = sum(max(p.height for p in r) for r in rows) + GAP * (len(rows) - 1)
    sheet = Image.new("RGB", (width, max(height, ROW * (len(appearance.LOOKS) + 14))), BACK)
    y = 0
    for r in rows:
        x = 0
        for p in r:
            sheet.paste(p, (x, y))
            x += p.width + GAP
        y += max(p.height for p in r) + GAP
    sheet.paste(legend(sheet.height, bool(scene.pick)), (width - LEGEND_W, 0))
    sheet.save(out)
    return {"out": os.path.abspath(out), "windows": facts, "size": [sheet.width, sheet.height]}
