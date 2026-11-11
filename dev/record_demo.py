"""Records the README demo from a finished fixture search and its preview, played back by `ui --replay` (no generator needed):
Playwright drives the page on what it shows and films it, a real Apply runs on a throwaway copy, and imageio-ffmpeg cuts the
waits short into docs/media/demo.mp4 and a palette docs/media/demo.gif 960 px wide, beside three stills. The world's folder,
seed, offsets, mods beyond the pack, graves' places and players, team and player names and the pointer's coordinates
are masked.

  record_demo.py --bundle B --fixture SERVER --work DIR [--out docs/media]
      copies the fixture server and the bundle (tests/e2e_fixture.py bundle) into DIR, finishes the preview if it was stopped,
      starts `ui --replay` on it and records
  record_demo.py --warm --url URL --world PATH --pick N
      walks the same views up to Apply on a running `ui --replay` (the PC's, through ssh -L), so its generator draws every map
      the recording shows into the search job; no video"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "tests")]
from e2e_fixture import BUNDLE, relocate  # noqa: E402

from rwg2migrator import world  # noqa: E402

VIEW = {"width": 1280, "height": 720}
CAPTION_H = 44
FPS, GIF_FPS, CRF = 25, 12, 22
SPEED = 180.0
FAST, FASTER = 4, 20
SHORT, LONG = 60_000, 1800_000
POLL_MS, PRESS_MS, TYPE_MS = 100, 120, 30
REDRAW_TRIES, SETTLED_S = 50, 1.0
GLIDE_STEPS, STEP_MS = 12, 15
WIDE_SWIPE, NARROW_SWIPE, SWIPE_PX_PER_STEP = (0.2, 0.8, 0.5), (0.3, 0.7, 0.5), 12
SETTLE_S, SETTLE_MAX_S = 4.0, 240.0
OK = 200
PAGE = {"x": 0, "y": CAPTION_H, "width": VIEW["width"], "height": VIEW["height"] - CAPTION_H}
GIF_SIZE = (960, 540)
GIF_STILL, GIF_BLOCK, GIF_BLOCK_MEAN = 10, 20, 4
GIF_LIMIT = 10 * 1024 * 1024
SERVER = "GTNH-server"
SHOWN_PATH = os.path.join("…", SERVER)
PARTS = ("World", "config", "serverutilities", "server.properties")
PLACE = r"-?\d+, -?\d+, -?\d+"
PATTERNS = ((r'(Offset [XZ]"=)-?\d+', "$1(hidden)"), (rf"\w+'s at {PLACE}", "a player's at (place hidden)"), (PLACE, "(place hidden)"))
MASK_JS = """([masks, patterns]) => {
  const res = patterns.map(([re, to]) => [new RegExp(re, 'g'), to]);
  const scrub = (node) => {
    const walk = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    for (let n = walk.currentNode; n; n = walk.nextNode()) {
      if (n.nodeType !== 3) continue;
      let t = n.nodeValue;
      for (const [from, to] of masks) t = t.split(from).join(to);
      for (const [re, to] of res) t = t.replace(re, to);
      if (t !== n.nodeValue) n.nodeValue = t;
    }
  };
  const watch = { subtree: true, childList: true, characterData: true };
  new MutationObserver((ms) => ms.forEach((m) => scrub(m.target))).observe(document, watch);
  document.addEventListener('DOMContentLoaded', () => scrub(document.body));
}"""
STAGE_JS = """(capH) => {
  const css = `body { margin-top: ${capH}px !important; height: calc(100% - ${capH}px) !important; }
#demo-cap { position: fixed; inset: 0 0 auto 0; height: ${capH}px; z-index: 99998; display: flex; align-items: center; gap: 12px;
  padding: 0 16px; background: #15191d; border-bottom: 1px solid #2b3137; color: #eef1f3; font: 600 17px system-ui, sans-serif; }
#demo-cap b { background: #6cc24a; color: #10200a; border-radius: 6px; padding: 1px 10px; }
#demo-cap b:empty { display: none; }
#demo-cur { position: fixed; z-index: 99999; left: -60px; top: -60px; margin: -2px 0 0 -3px; pointer-events: none; }
.demo-still #demo-cur { visibility: hidden; }
.demo-ring { position: fixed; z-index: 99999; width: 36px; height: 36px; margin: -18px 0 0 -18px; border-radius: 50%;
  border: 3px solid #6cc24a; pointer-events: none; animation: demo-ring .5s ease-out forwards; }
@keyframes demo-ring { from { transform: scale(.3); opacity: 1; } to { transform: scale(1.3); opacity: 0; } }
.readout { display: none !important; }`;
  const arrow = '<svg id="demo-cur" width="24" height="24" viewBox="0 0 24 24"><path d="M3 2v17l4.6-4.3 3.1 6.6 3-1.4-3.1-6.4h6.3z"'
    + ' fill="#fff" stroke="#111" stroke-width="1.5" stroke-linejoin="round"/></svg>';
  window.__cap = (step, text) => {
    const cap = document.getElementById('demo-cap');
    cap.firstChild.textContent = step;
    cap.lastChild.textContent = text;
  };
  const opener = (e) => e.target.id === 'world-open' && window.__real;
  addEventListener('click', (e) => { if (opener(e)) document.getElementById('world-path').value = window.__real; }, true);
  addEventListener('click', (e) => { if (opener(e)) document.getElementById('world-path').value = window.__shown; });
  document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = css;
    document.head.append(style);
    document.body.insertAdjacentHTML('beforeend', `<div id="demo-cap"><b></b><span></span></div>${arrow}`);
    const cur = document.getElementById('demo-cur');
    addEventListener('pointermove', (e) => { cur.style.left = `${e.clientX}px`; cur.style.top = `${e.clientY}px`; }, true);
    addEventListener('pointerdown', (e) => {
      const ring = document.createElement('div');
      ring.className = 'demo-ring';
      ring.style.left = `${e.clientX}px`;
      ring.style.top = `${e.clientY}px`;
      document.body.append(ring);
      setTimeout(() => ring.remove(), 600);
    }, true);
  });
}"""
TILES_JS = """(sel) => {
  const tiles = [...document.querySelectorAll(`${sel} img.leaflet-tile`)];
  return tiles.length > 0 && tiles.every((i) => i.complete);
}"""


class Film:
    """The recording's timeline: (seconds into the video, speed) marks, a speed of 0 cutting what follows. In warm mode nothing
    is filmed: a hold waits instead until no candidate tile has been refused for SETTLE_S seconds, so the generator draws every
    map on screen."""

    def __init__(self, page, out, warm):
        self.page, self.out, self.warm = page, out, warm
        self.t0, self.marks, self.pos = time.monotonic(), [(0.0, 0)], (VIEW["width"] / 2, VIEW["height"] / 2)
        self.refused = time.monotonic()
        page.on("response", self._tile)

    def _tile(self, r):
        if "/tiles/cand." in r.url and r.status != OK:
            self.refused = time.monotonic()

    def pace(self, speed):
        if self.marks[-1][1] != speed:
            self.marks.append((time.monotonic() - self.t0, speed))

    def settle(self):
        end = time.monotonic() + SETTLE_MAX_S
        self.page.wait_for_timeout(SETTLE_S * 1000)
        while time.monotonic() - self.refused < SETTLE_S and time.monotonic() < end:
            self.page.wait_for_timeout(SETTLE_S * 250)

    def say(self, step, text):
        self.page.evaluate("([step, text]) => window.__cap(step, text)", [step, text])

    def hold(self, seconds):
        if self.warm:
            self.settle()
        else:
            self.page.wait_for_timeout(seconds * 1000)

    def until(self, locator, timeout=SHORT, speed=None):
        """Waits until locator shows, played speed times faster; the error panel showing instead stops the recording."""
        end = time.monotonic() + timeout / 1000
        error = self.page.get_by_role("alertdialog")
        if speed:
            self.pace(speed)
        while not locator.first.is_visible():
            if error.is_visible():
                raise SystemExit(f"the error panel says: {error.inner_text()}")
            if time.monotonic() > end:
                raise SystemExit(f"timed out waiting for {locator}")
            self.page.wait_for_timeout(POLL_MS)
        self.pace(1)
        return locator.first

    def tiles(self, sel, speed=FAST):
        """Waits, played speed times faster, until sel shows map tiles and every one has loaded or failed."""
        self.pace(speed)
        self.page.wait_for_function(TILES_JS, arg=sel, timeout=SHORT)
        self.pace(1)

    def still(self, name, sel=None):
        if self.warm:
            return
        b = self.page.locator(sel).first.bounding_box() if sel else PAGE
        self.page.evaluate("() => document.documentElement.classList.add('demo-still')")
        shot = Image.open(io.BytesIO(self.page.screenshot()))
        self.page.evaluate("() => document.documentElement.classList.remove('demo-still')")
        box = [round(v) for v in (b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"])]
        shot.crop(box).save(self.out / name, optimize=True)


def cuts(marks):
    """(start, end, speed) of each kept stretch of the video, from marks ending in a speed of 0."""
    return [(a, b, s) for (a, s), (b, _) in pairwise(marks) if s]


def ease(t):
    return t * t * (3 - 2 * t)


def glide(f, x, y, steps=GLIDE_STEPS):
    """The pointer moves to x, y in steps, easing in and out, so the film shows where it goes."""
    x0, y0 = f.pos
    for i in range(1, steps + 1):
        e = ease(i / steps)
        f.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
        f.page.wait_for_timeout(STEP_MS)
    f.pos = (x, y)


def where(f, el):
    """The centre of el, scrolled into view; tried again while the page redraws it."""
    for _ in range(REDRAW_TRIES):
        try:
            el.scroll_into_view_if_needed()
            b = el.bounding_box()
        except PlaywrightError:
            f.page.wait_for_timeout(POLL_MS)
            continue
        return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    raise SystemExit(f"{el} kept being redrawn")


def press(f, locator, act="click"):
    el = locator.first
    glide(f, *where(f, el))
    f.page.wait_for_timeout(PRESS_MS)
    getattr(el, act)()


def button(page, name):
    return page.get_by_role("button", name=name, exact=True)


def drag_swipe(f, sel, stops=WIDE_SWIPE):
    """Drags the swipe handle of the comparison in sel through stops (fractions of the map's width)."""
    box = f.page.locator(f"{sel} .leaflet-container").first.bounding_box()
    handle = f.page.locator(f"{sel} .swipe").first.bounding_box()
    y = box["y"] + box["height"] / 2
    glide(f, handle["x"] + handle["width"] / 2, y)
    f.page.mouse.down()
    for frac in stops:
        x = box["x"] + box["width"] * frac
        glide(f, x, y, max(GLIDE_STEPS, round(abs(x - f.pos[0]) / SWIPE_PX_PER_STEP)))
    f.page.mouse.up()


def show_left(f, sel, label):
    """Puts label on the left map of the comparison in sel, and again until it has stayed SETTLED_S: the page may draw the map
    once more with its own choice."""
    left, end, since = f.page.locator(f"{sel} .side-label").first, time.monotonic() + SHORT / 1000, None
    while time.monotonic() < end:
        if left.inner_text() != label:
            f.page.locator(sel).get_by_label("Left map").select_option(label=label)
            since = None
        elif since is None:
            since = time.monotonic()
        elif time.monotonic() - since >= SETTLED_S:
            return
        f.page.wait_for_timeout(POLL_MS)
    raise SystemExit(f"the left map of {sel} did not keep {label}")


def opening(f, path):
    p = f.page
    f.say("", "rwg2-migrator: move a GTNH world from legacy RWG to RWG2 Smooth and keep your base")
    f.until(p.get_by_text("Move your GTNH world to RWG2 Smooth"))
    f.pace(1)
    f.hold(1)
    press(f, button(p, "Start with a world"))
    f.say("1", "Open your world: the tool checks it and reads its map (the author's server, copied)")
    p.evaluate("([real, shown]) => { window.__real = real; window.__shown = shown; }", [path, SHOWN_PATH])
    field = p.get_by_label("World folder")
    press(f, field)
    field.press_sequentially(SHOWN_PATH, delay=TYPE_MS)
    press(f, button(p, "Open"))
    f.until(p.get_by_text("in the overworld"), LONG, FAST)
    press(f, p.get_by_label(re.compile("I accept the")), "check")
    f.hold(0.6)
    press(f, button(p, "Continue: choose the area to keep"))


def area(f):
    p = f.page
    f.say("2", "Choose what to keep: import your claims; a safety margin goes around them")
    claims = f.until(button(p, "Import claims"))
    f.tiles("#screen-area")
    press(f, claims)
    f.until(p.locator("#area-removed li", has_text="become new terrain"))
    f.tiles("#screen-area")
    f.hold(1)
    f.still("area.png")
    p.locator("#area-removed li", has_text="Probably yours").evaluate("(e) => e.scrollIntoView({ behavior: 'smooth', block: 'center' })")
    f.hold(1.6)
    press(f, button(p, "Continue: find the best matches"))


def matches(f, pick, speed):
    p = f.page
    f.say("3", f"Best matches fill in live as the search runs (shown {speed:.0f}x faster: a real search takes 15 to 30 minutes)")
    start = f.until(button(p, "Find the best matches"))
    while not start.is_enabled():
        p.wait_for_timeout(POLL_MS)
    press(f, start)
    f.until(button(p, "Pick this one"), LONG)
    f.hold(0.5)
    card = p.locator("#m-cards .cand", has=p.locator(".rank", has_text=re.compile(rf"^#{pick}$")))
    press(f, card)
    f.say("4", "NOW | CANDIDATE: drag the slider to compare today's terrain with the match")
    f.until(p.locator("#m-compare .swipe"))
    f.tiles("#m-compare")
    f.hold(0.5)
    drag_swipe(f, "#m-compare")
    f.still("best-matches.png")
    f.hold(0.5)
    press(f, card.get_by_role("button", name="Pick this one"))


def preview(f):
    p = f.page
    f.say("5", "Preview: the migrated world is built in a copy: new terrain, smoothed border, checks (sped up)")
    press(f, f.until(button(p, "Build the preview")))
    f.until(button(p, "Continue: Apply"), LONG, FAST)
    f.until(p.locator("#p-compare .swipe"), LONG, FAST)
    f.pace(FAST)
    show_left(f, "#p-compare", "NOW")
    f.tiles("#p-compare")
    f.say("5", "Preview: your world NOW | SMOOTHED, the migrated world built in a copy, its border smoothed")
    drag_swipe(f, "#p-compare", NARROW_SWIPE)
    f.hold(0.3)


def applying(f):
    p = f.page
    press(f, button(p, "Continue: Apply"))
    f.say("6", "Apply (world stopped): the preview replaces your world, which is kept as the backup (sped up)")
    f.until(p.locator("#a-list li").first)
    press(f, p.get_by_label(re.compile("I turned off anything that restarts")).filter(visible=True), "check")
    f.hold(1)
    press(f, button(p, "Apply now"))
    press(f, f.until(p.get_by_role("dialog").get_by_role("button", name="Apply")))
    f.until(p.get_by_role("heading", name="Your world now uses RWG2 Smooth"), LONG, FASTER)
    f.say("", "Done: your world now uses RWG2 Smooth around your base; Verify and Roll back stay available")
    f.hold(1.8)


def after(f):
    p = f.page
    press(f, button(p, "Go to Verify"))
    f.say("7", "Verify: your world BEFORE APPLY and NOW, side by side: the base as it was, RWG2 around it (sped up)")
    f.until(p.locator("#v-compare .side-label", has_text="NOW"), LONG, FASTER)
    f.pace(FASTER)
    show_left(f, "#v-compare", "BEFORE APPLY")
    f.tiles("#v-compare", FASTER)
    press(f, p.locator("#v-compare .cmp-bar button", has_text="Side by side"))
    f.tiles("#v-compare", FASTER)
    f.hold(0.8)
    f.still("before-after.png", "#v-compare .cmp-maps")
    f.hold(2)


def masks(root):
    """(text, replacement) pairs: the folder, the seed, the mods beyond the pack, team names and player names or ids."""
    w = world.locate(root)
    level, report = world.check(w.world)
    found, _ = world.players(w.world, w.root, level)
    out = [(w.root, SHOWN_PATH), (str(level["Data"]["RandomSeed"]), "(seed hidden)")]
    beyond = ", ".join(report["mod_diff"]["beyond_pack"])
    out += [(beyond, "(hidden)")] if beyond else []
    out += [(team, f"#{i}") for i, team in enumerate(sorted(world.claims(w.world)), 1)]
    out += [(p["name"], f"player {i}") for i, p in enumerate(found, 1)]
    return out


def prepare(opts):
    """A throwaway copy of the fixture server and of the bundle's jobs, relocated here, with the preview finished."""
    work = Path(opts.work)
    root = work / SERVER
    for part in PARTS:
        src = Path(opts.fixture, part)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, root / part)
    bundle = Path(opts.bundle)
    pc = json.loads((bundle / BUNDLE).read_text())
    for part in ("data", "cache"):
        shutil.copytree(bundle / part, work / part)
    shutil.copytree(bundle / "work", root / "rwg2-work")
    relocate(work / "data", {pc["root"]: str(root), pc["data"]: str(work / "data"), pc["cache"]: str(work / "cache")})
    subprocess.run(cli("jobs", "resume", pc["preview"], "--data", work / "data", "--cache", work / "cache"), check=False)
    return root, pc


def cli(*argv):
    return [sys.executable, "-m", "rwg2migrator", *map(str, argv)]


def start_ui(work, search, speed):
    proc = subprocess.Popen(cli("ui", "--no-browser", "--replay", search, "--speed", speed, "--data", work / "data", "--cache",
                                work / "cache"), stdout=subprocess.PIPE, text=True)  # fmt: skip
    return proc, proc.stdout.readline().strip()


def steady(frames, still=GIF_STILL, block=GIF_BLOCK):
    """The frames with each pixel kept until it moves more than still from what is shown, or its block of block x block pixels
    moves more than GIF_BLOCK_MEAN on average (a faint panel that went away), so the video's codec noise does not redraw the
    whole GIF every frame."""
    shown = None
    for frame in frames:
        if shown is None:
            shown = frame.copy()
            yield shown
            continue
        moved = np.abs(frame.astype(np.int16) - shown).max(axis=2)
        rows, cols = moved.shape[0] // block, moved.shape[1] // block
        faint = moved.reshape(rows, block, cols, block).mean(axis=(1, 3)) > GIF_BLOCK_MEAN
        changed = (moved > still) | faint.repeat(block, 0).repeat(block, 1)
        shown[changed] = frame[changed]
        yield shown


def raw_frames(proc, size):
    width, height = size
    while len(raw := proc.stdout.read(width * height * 3)) == width * height * 3:
        yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def gif(mp4, out):
    """A palette GIF of mp4 at GIF_FPS and GIF_SIZE, its frames held steady."""
    ffmpeg, size = imageio_ffmpeg.get_ffmpeg_exe(), "{}x{}".format(*GIF_SIZE)
    palette = "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    dec = subprocess.Popen([ffmpeg, "-loglevel", "error", "-i", str(mp4), "-vf", f"fps={GIF_FPS},scale={size}:flags=lanczos", "-f",
                            "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)  # fmt: skip
    enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", str(GIF_FPS),
                            "-i", "-", "-vf", palette, str(out)], stdin=subprocess.PIPE)  # fmt: skip
    for frame in steady(raw_frames(dec, GIF_SIZE)):
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    if enc.wait() or dec.wait():
        raise SystemExit(f"ffmpeg could not make {out}")


def encode(video, marks, out):
    """demo.mp4 (the waits cut short) and the palette demo.gif made from it; sizes in bytes and the length in seconds."""
    ffmpeg, parts = imageio_ffmpeg.get_ffmpeg_exe(), cuts(marks)
    split = f"[0:v]split={len(parts)}" + "".join(f"[s{i}]" for i in range(len(parts)))
    trims = [f"[s{i}]trim=start={a:.3f}:end={b:.3f},setpts=(PTS-STARTPTS)/{s}[p{i}]" for i, (a, b, s) in enumerate(parts)]
    joined = "".join(f"[p{i}]" for i in range(len(parts))) + f"concat=n={len(parts)},fps={FPS}[v]"
    mp4, gif_path = out / "demo.mp4", out / "demo.gif"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", ";".join([split, *trims, joined]),
                    "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags",
                    "+faststart", str(mp4)], check=True)  # fmt: skip
    gif(mp4, gif_path)
    seconds = sum((b - a) / s for a, b, s in parts)
    return {"seconds": round(seconds, 1), "mp4": mp4.stat().st_size, "gif": gif_path.stat().st_size}


def walk(f, opts):
    opening(f, opts.world)
    area(f)
    matches(f, opts.pick, opts.speed)
    preview(f)
    if not f.warm:
        applying(f)
        after(f)
    f.pace(0)


def record(opts, mask):
    """Films the walk on opts.url; the video's path (None when warm) and its marks."""
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    video = {} if opts.warm else {"record_video_dir": str(Path(opts.work, "video")), "record_video_size": VIEW}
    with sync_playwright() as pw:
        ctx = pw.chromium.launch().new_context(viewport=VIEW, **video)
        page = ctx.new_page()
        page.add_init_script(f"({MASK_JS})({json.dumps([mask, PATTERNS])})")
        page.add_init_script(f"({STAGE_JS})({CAPTION_H})")
        f = Film(page, out, opts.warm)
        page.goto(opts.url)
        try:
            walk(f, opts)
        finally:
            ctx.close()
        return (None if opts.warm else page.video.path()), f.marks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle")
    ap.add_argument("--fixture")
    ap.add_argument("--work")
    ap.add_argument("--out", default=str(HERE.parent / "docs" / "media"))
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--url")
    ap.add_argument("--world")
    ap.add_argument("--pick", type=int)
    ap.add_argument("--speed", type=float, default=SPEED)
    opts = ap.parse_args()
    if opts.warm:
        record(opts, [])
        return
    root, pc = prepare(opts)
    opts.world = str(root)
    opts.pick = opts.pick or json.loads(Path(opts.work, "data", "jobs", pc["preview"], "job.json").read_text())["args"]["pick"]
    proc, opts.url = start_ui(Path(opts.work), pc["search"], opts.speed)
    try:
        video, marks = record(opts, masks(str(root)))
    finally:
        proc.terminate()
    sizes = encode(video, marks, Path(opts.out))
    print(json.dumps({"marks": marks} | sizes))
    if sizes["gif"] > GIF_LIMIT:
        raise SystemExit(f"demo.gif is {sizes['gif']:,} bytes, over the README's 10 MB")


if __name__ == "__main__":
    main()
