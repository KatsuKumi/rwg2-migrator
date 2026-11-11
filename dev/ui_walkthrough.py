"""Drives the UI with Playwright from World to Apply using only what the page shows (roles, labels, button names), and measures
the live maps: candidate tile latency, best-so-far card updates, re-attach after a reload, zoom levels, re-renders on a revisit,
page width at 1920x1080 and 1366x768, and frame times while panning and zooming. Every wait that times out is recorded as a stuck
point with a screenshot.

  ui_walkthrough.py --out DIR --fake                     a synthetic world, the fake generator, the UI started here
  ui_walkthrough.py --out DIR --url URL --world PATH     a running UI (for the PC: through ssh -L PORT:127.0.0.1:PORT)
  options: --pick N (the candidate to preview), --apply (also Apply, then Verify), --trees, --blend,
           --stop-after S (a reduced budget: click "Stop and use best so far" S seconds after the first matches show)"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "tests")
VIEWPORTS = {"1920x1080": (1920, 1080), "1366x768": (1366, 768)}
SHORT, WORK, LONG = 30_000, 300_000, 4 * 3600_000
CARD_POLL = 1.0
RELOAD_AFTER_CARDS = 2
SLOW_FRAME_MS = 50
PAN_STEPS, PAN_PX = 30, 600
WHEEL_PX = 120
TILE_LIMIT_S = 2.0
OK = 200
SETTLE_S, SETTLE_MAX_S = 3.0, 90.0
PAN_FROM = 0.3
ZOOM_WAIT_MS = 700
READOUT_AT, READOUT_TRIES = (80, 140), 30
LEGEND_JS = """(field) => { const C = S[field]; const b = cellBounds([...C.kept].map(unkey));
  return C.maps.map((m) => { const lg = m.getContainer().querySelector('.legend'); if (!lg) return null;
    const r = lg.getBoundingClientRect(); const c = m.getContainer().getBoundingClientRect();
    const p0 = m.latLngToContainerPoint(LL(b[0], b[1])); const p1 = m.latLngToContainerPoint(LL(b[2], b[3]));
    const k = { l: c.left + Math.min(p0.x, p1.x), r: c.left + Math.max(p0.x, p1.x),
      t: c.top + Math.min(p0.y, p1.y), b: c.top + Math.max(p0.y, p1.y) };
    return { open: lg.open, clear: !(r.right > k.l && r.left < k.r && r.bottom > k.t && r.top < k.b) }; }).filter(Boolean); }"""
REBUILD_JS = """([field, mode]) => { const C = S[field];
  C.maps.forEach((m) => m.remove()); C.maps = []; C.view = null; C.mode = mode; C.build(); }"""
FRAMES_JS = """
window.__frames = []; window.__rec = false;
(function tick(t) { if (window.__rec) window.__frames.push(t); requestAnimationFrame(tick); })(performance.now());
"""


class Stuck(Exception):
    pass


class Walk:
    def __init__(self, page, out):
        self.page, self.out, self.report, self.shot_n = page, out, {"stuck": [], "width": {}, "screens": []}, 0
        self.tiles, self.windows, self.last_wait = None, [], time.time()
        page.on("response", self._response)

    def _response(self, r):
        if "/tiles/cand." not in r.url:
            return
        if r.status != OK:
            self.last_wait = time.time()
        if self.tiles is None:
            return
        url = re.sub(r"&r=\d+$", "", r.url)
        t = self.tiles.setdefault(url, {"statuses": []})
        t["statuses"].append(r.status)
        if r.status == OK and "age" not in t:
            t["age"] = float(r.headers.get("x-tile-age", "nan"))
            t["waited"] = len(t["statuses"]) > 1

    def still(self, on):
        """Candidate tiles are timed only while the view does not move: a tile panned away from is not waited for."""
        if on:
            self.tiles = {}
        elif self.tiles is not None:
            self.windows.append(self.tiles)
            self.tiles = None

    def settle(self):
        """Wait until no candidate tile has been waiting for SETTLE_S seconds."""
        end = time.time() + SETTLE_MAX_S
        while time.time() - self.last_wait < SETTLE_S and time.time() < end:
            self.page.wait_for_timeout(500)

    def shot(self, name):
        self.shot_n += 1
        path = os.path.join(self.out, f"{self.shot_n:02d}-{name}.png")
        self.page.screenshot(path=path)
        return path

    def stuck(self, step, expected):
        self.report["stuck"].append({"step": step, "expected": expected, "screenshot": self.shot(f"stuck-{step}")})
        raise Stuck(f"{step}: {expected}")

    def see(self, step, locator, timeout=SHORT):
        """The first match of locator once visible; the error panel showing instead is a stuck point too."""
        error = self.page.get_by_role("alertdialog")
        try:
            locator.first.or_(error).first.wait_for(state="visible", timeout=timeout)
        except PlaywrightTimeout:
            self.stuck(step, str(locator))
        if error.is_visible():
            self.stuck(step, f"{locator}, but the error panel says: {error.inner_text()}")
        return locator.first

    def button(self, name):
        return self.page.get_by_role("button", name=name, exact=True)

    def text(self, value):
        return self.page.get_by_text(value)

    def widths(self, screen):
        """Whether the page scrolls sideways, at both sizes; the layout ends at the size it started at."""
        out = {}
        for label, (w, h) in reversed(list(VIEWPORTS.items())):
            self.page.set_viewport_size({"width": w, "height": h})
            self.page.wait_for_timeout(300)
            out[label] = self.page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        self.report["width"][screen] = out
        self.report["screens"].append(screen)
        self.shot(screen)


def open_world(w, path):
    w.see("home", w.text("Move your GTNH world to RWG2 Smooth"))
    w.widths("home")
    w.button("Start with a world").click()
    w.see("world", w.page.get_by_role("heading", name="Which world?"))
    w.page.get_by_label("World folder").fill(path)
    w.button("Open").click()
    w.see("world-card", w.text("in the overworld"), LONG)
    w.report["world_card"] = w.page.locator("#world-card").inner_text()
    w.page.get_by_label(re.compile("I accept the")).check()
    w.widths("world")
    w.button("Continue: choose the area to keep").click()


def count(w, label):
    return int(w.page.locator(".counts div", has_text=label).locator("b").inner_text().replace(",", ""))


def counts(w):
    w.page.wait_for_timeout(1500)
    return {k: count(w, label) for k, label in (("core", "protected"), ("kept", "kept with the margin"), ("never", "never explored"))}


def area(w, fake):
    w.see("area", w.page.get_by_role("heading", name="Area to keep"))
    if fake:
        suggest_then_clear(w)
        import_ticked(w)
    w.see("area-claims", w.button("Import claims")).click()
    w.see("area-counts", w.page.locator("#area-removed li", has_text="become new terrain"), SHORT)
    w.report["area"] = {"claims": counts(w), "removed": w.page.locator("#area-removed").inner_text()}
    if fake:
        area_edits(w)
    map_checks(w, "#area-map")
    w.widths("area")
    w.see("area-continue", w.button("Continue: find the best matches")).click()


def suggest_then_clear(w):
    w.see("area-suggest", w.button("Suggest from where you've played")).click()
    w.see("area-suggested", w.page.locator("#c-core", has_text=re.compile("^[1-9]")), SHORT)
    suggested = counts(w)
    w.button("Clear").click()
    w.see("area-clear", w.page.get_by_role("dialog").get_by_role("button", name="Clear")).click()
    w.see("area-cleared", w.page.locator("#c-core", has_text=re.compile("^0$")), SHORT)
    w.report["area_suggest"] = {"suggested": suggested, "cleared": counts(w)}


def team_ticks(w):
    return w.page.locator("#area-team-list input").evaluate_all("els => els.map((e) => e.checked)")


def import_ticked(w):
    """Tick one team, then Import claims: only that team's claims are kept. Unticking it again leaves nothing chosen."""
    w.page.get_by_label(re.compile("Team alpha")).check()
    w.button("Import claims").click()
    w.report["import_ticked"] = {"ticks": team_ticks(w), "core": counts(w)["core"]}
    w.page.get_by_label(re.compile("Team alpha")).uncheck()
    w.report["import_ticked"]["unticked_core"] = counts(w)["core"]


def area_edits(w):
    """Painting, the margin and 'Add these chunks', each read back from the counts."""
    w.page.get_by_label(re.compile("Team beta")).uncheck()
    base = counts(w)
    added = w.page.locator("#area-removed li", has_text="Other teams' claims").get_by_role("button", name=re.compile("Add these chunks"))
    n = int(re.search(r"\((\d+)\)", added.inner_text()).group(1))
    added.click()
    after = counts(w)
    w.report["area"]["add_these"] = {"listed": n, "core_before": base["core"], "core_after": after["core"]}
    w.button("Paint").click()
    box = w.page.locator("#area-map").bounding_box()
    w.page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2)
    painted = counts(w)
    w.button("Erase").click()
    w.page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2)
    erased = counts(w)
    w.button("Move").click()
    w.page.get_by_label(re.compile("Safety margin")).fill("2")
    margin2 = counts(w)
    w.page.get_by_label(re.compile("Safety margin")).fill("1")
    w.report["area"]["edits"] = {"painted": painted, "erased": erased, "margin2": margin2, "margin1": counts(w)}


def drawn(w):
    return w.page.evaluate("fetch('/api/state', {headers: {'X-Token': document.querySelector('meta[name=token]').content}})"
                           ".then(r => r.json()).then(s => [s.drawn, s.redrawn])")  # fmt: skip


def frames(w):
    t = w.page.evaluate("window.__rec = false; window.__frames")
    gaps = [b - a for a, b in zip(t, t[1:], strict=False)]
    if not gaps:
        return {}
    gaps.sort()
    return {"frames": len(t), "median_ms": round(statistics.median(gaps), 1), "p95_ms": round(gaps[int(0.95 * (len(gaps) - 1))], 1),
            "max_ms": round(gaps[-1], 1), "over_50ms": sum(g > SLOW_FRAME_MS for g in gaps)}  # fmt: skip


def pan(w, sel, dx):
    box = w.page.locator(sel).bounding_box()
    cx, cy = box["x"] + box["width"] * (PAN_FROM if dx > 0 else 1 - PAN_FROM), box["y"] + box["height"] / 2
    w.page.mouse.move(cx, cy)
    w.page.mouse.down()
    for i in range(1, PAN_STEPS + 1):
        w.page.mouse.move(cx + dx * i / PAN_STEPS, cy)
    w.page.mouse.up()
    w.page.wait_for_timeout(1500)


def zoom_all_the_way(w, button):
    """Clicks until the button turns disabled; the number of clicks."""
    clicks = 0
    while True:
        w.page.wait_for_timeout(ZOOM_WAIT_MS)
        if "leaflet-disabled" in (button.get_attribute("class") or ""):
            return clicks
        button.click()
        clicks += 1


def zoom_levels(w, sel):
    zoom_all_the_way(w, w.page.locator(f"{sel} .leaflet-control-zoom-out"))
    return zoom_all_the_way(w, w.page.locator(f"{sel} .leaflet-control-zoom-in")) + 1


def map_checks(w, sel):
    """Zoom levels, frame times while panning and zooming, and tiles drawn again when the view comes back."""
    out = {"zoom_levels": zoom_levels(w, sel)}
    w.page.locator(f"{sel} .leaflet-control-zoom-out").click()
    w.page.wait_for_timeout(2500)
    out["overlay_tiles_zoomed_out"] = w.page.locator(f"{sel} .leaflet-overlay-pane canvas").count()
    w.page.evaluate("window.__frames = []; window.__rec = true")
    pan(w, sel, PAN_PX)
    box = w.page.locator(sel).bounding_box()
    w.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    for d in (WHEEL_PX, WHEEL_PX, -WHEEL_PX, -WHEEL_PX):
        w.page.mouse.wheel(0, d)
        w.page.wait_for_timeout(600)
    out["pan_zoom"] = frames(w)
    w.settle()
    before = drawn(w)
    pan(w, sel, -PAN_PX)
    pan(w, sel, PAN_PX)
    w.settle()
    after = drawn(w)
    out["revisit"] = {"drawn": after[0] - before[0], "redrawn_total": after[1]}
    w.report.setdefault("maps", {})[sel] = out


def card_texts(w):
    """Every card's text, Details included: two best-so-far matches can read the same in plain words."""
    return tuple(w.page.locator("#m-cards .cand").evaluate_all("els => els.map((e) => e.textContent)"))


def search(w, stop_after=None):
    w.see("matches", w.page.get_by_role("heading", name="Best matches"))
    w.report["base"] = w.page.get_by_label("Your base looks like").input_value()
    w.see("search-start", w.button("Find the best matches")).click()
    w.see("search-running", w.button("Stop and use best so far"), LONG)
    w.still(True)
    t0, seen, reloaded, titles, stopped = time.time(), [], None, set(), None
    while not w.button("Pick this one").count():
        titles.add(re.sub(r"\d+:\d\d", "m:ss", w.page.title()))
        cards = card_texts(w)
        if cards and (not seen or seen[-1][1] != cards):
            seen.append((round(time.time() - t0, 1), cards))
        if stop_after is not None and stopped is None and seen and time.time() - t0 - seen[0][0] > stop_after:
            stopped = round(time.time() - t0, 1)
            w.button("Stop and use best so far").click()
        if reloaded is None and len(seen) >= RELOAD_AFTER_CARDS:
            w.still(False)
            reloaded = reload_check(w)
            w.still(True)
        if time.time() - t0 > LONG / 1000:
            w.stuck("search", "the search to finish ('Pick this one')")
        time.sleep(CARD_POLL)
    w.still(False)
    try:
        w.page.wait_for_function("document.title.startsWith('Done')", timeout=WORK)
    except PlaywrightTimeout:
        w.stuck("search-done", "the search to finish its last stage (the title 'Done – rwg2-migrator')")
    titles.add(w.page.title())
    w.report["search_job"] = w.page.evaluate(
        "fetch('/api/job?id=' + location.hash.split('/')[1], {headers: {'X-Token': document.querySelector('meta[name=token]').content}})"
        ".then((r) => r.json()).then((j) => ({id: j.job.id, prefs: j.args.prefs, candidates: j.candidates}))"
    )
    w.report["search"] = {"seconds": round(time.time() - t0, 1), "stopped_at": stopped, "card_updates": [s for s, _ in seen],
                          "reload": reloaded, "titles": sorted(titles), "badges": badges(w),
                          "final_cards": card_texts(w), "stage": w.page.locator("#m-stage").inner_text()}  # fmt: skip
    w.shot("matches-done")


def badges(w):
    """The badges of each card, and whether the page explains them."""
    cards = w.page.locator("#m-cards .cand").evaluate_all(
        "els => els.map((e) => [...e.querySelectorAll('.badge')].map((b) => b.textContent))"
    )
    return {"cards": cards, "explained": w.page.locator("#m-badges").is_visible()}


def metrics_clear(w):
    """At each size, side by side: what is drawn at the centre of each map (the kept base) is the map, not a panel over it."""
    out = {}
    for label, (width, height) in VIEWPORTS.items():
        w.page.set_viewport_size({"width": width, "height": height})
        w.page.wait_for_timeout(800)
        out[label] = w.page.evaluate("""() => [...document.querySelectorAll('#m-compare .cmp-maps > div')].map((m) => {
            const r = m.getBoundingClientRect();
            const el = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            return !!el && !!el.closest('.leaflet-container') && !el.closest('.leaflet-control') && !el.closest('#m-metrics'); })""")
    out["text"] = w.page.locator("#m-metrics").inner_text()
    w.page.set_viewport_size(dict(zip(("width", "height"), VIEWPORTS["1920x1080"], strict=True)))
    return out


def legend_clear(w, field):
    """For each size and mode, the comparison drawn afresh as a visit at that size draws it: whether each map's legend is open
    and whether it stays off the box of the kept area (the base the player compares)."""
    out = {}
    for mode in ("side", "swipe"):
        for label, (width, height) in VIEWPORTS.items():
            w.page.set_viewport_size({"width": width, "height": height})
            w.page.evaluate(REBUILD_JS, [field, mode])
            w.page.wait_for_timeout(1000)
            out[f"{mode} {label}"] = w.page.evaluate(LEGEND_JS, field)
    w.page.set_viewport_size(dict(zip(("width", "height"), VIEWPORTS["1920x1080"], strict=True)))
    w.page.evaluate(REBUILD_JS, [field, "swipe"])
    return out


def reload_check(w):
    w.page.reload()
    ok = True
    try:
        w.button("Stop and use best so far").first.wait_for(state="visible", timeout=SHORT)
        w.page.locator("#m-cards .cand").first.wait_for(state="visible", timeout=SHORT)
    except PlaywrightTimeout:
        ok = w.button("Pick this one").count() > 0
    shot = w.shot("matches-after-reload")
    return {"reattached": ok, "stage": w.page.locator("#m-stage").inner_text(), "screenshot": shot}


def pick(w, rank):
    card = w.page.locator("#m-cards .cand", has=w.page.locator(".rank", has_text=re.compile(rf"^#{rank}$")))
    w.see("pick", card)
    w.still(True)
    card.click()
    w.see("compare", w.page.locator("#m-compare .side-label", has_text="CANDIDATE"), WORK)
    w.settle()
    w.still(False)
    w.button("Side by side").click()
    w.page.wait_for_timeout(3000)
    w.report["metrics"] = metrics_clear(w)
    w.widths("matches-side-by-side")
    w.report["legend"] = legend_clear(w, "compare")
    w.button("Side by side").click()
    w.page.wait_for_timeout(1500)
    w.report["readout"] = readouts(w)
    w.button("Swipe").click()
    w.page.wait_for_timeout(3000)
    map_checks(w, "#m-compare")
    w.widths("matches-swipe")
    card.get_by_role("button", name="Details").click()
    w.report["picked"] = card.inner_text()
    card.get_by_role("button", name="Pick this one").click()


def readouts(w):
    """The hover readout of NOW and of the candidate at the same spot of each pane, outside the kept area."""
    panes, out = w.page.locator("#m-compare .leaflet-container"), {}
    for i, side in enumerate(("now", "candidate")):
        box = panes.nth(i).bounding_box()
        for dx in range(READOUT_TRIES):
            w.page.mouse.move(box["x"] + READOUT_AT[0] + dx, box["y"] + READOUT_AT[1])
            w.page.wait_for_timeout(ZOOM_WAIT_MS)
            out[side] = w.page.locator("#m-compare .readout").nth(i).inner_text()
            if not re.search("Not drawn here yet|Point at the map", out[side]):
                break
    return out


def preview(w, opts):
    w.see("preview", w.button("Build the preview"))
    if opts.trees:
        w.page.get_by_label(re.compile("Replant trees")).check()
    if opts.blend:
        w.page.get_by_label(re.compile("Blend the biomes")).check()
    w.button("Build the preview").click()
    t0 = time.time()
    w.see("preview-done", w.button("Continue: Apply"), LONG)
    w.page.wait_for_timeout(8000)
    w.report["preview"] = {"seconds": round(time.time() - t0, 1), "steps": w.page.locator("#p-steps").inner_text(),
                           "result": w.page.locator("#p-result").inner_text()}  # fmt: skip
    w.widths("preview")
    w.button("Continue: Apply").click()


def apply_screen(w, go):
    w.see("apply", w.page.get_by_role("heading", name="Apply", exact=True))
    w.see("apply-list", w.page.locator("#a-list li").first)
    w.page.get_by_label(re.compile("I turned off anything that restarts")).filter(visible=True).check()
    w.report["apply"] = {"checklist": w.page.locator("#a-list").inner_text(), "enabled": w.button("Apply now").is_enabled()}
    w.widths("apply")
    if not go:
        return
    w.button("Apply now").click()
    w.see("apply-confirm", w.page.get_by_role("dialog").get_by_role("button", name="Apply")).click()
    w.see("after", w.page.get_by_role("heading", name="Your world now uses RWG2 Smooth"), LONG)
    w.page.wait_for_timeout(2000)
    w.report["after"] = w.page.locator("#screen-after").inner_text()
    w.widths("after")
    w.button("Go to Verify").click()
    w.see("verify-map", w.page.locator("#v-compare .side-label", has_text="NOW"), LONG)
    w.report["verify_map"] = verify_map(w)
    w.button("Check my world").click()
    w.see("verify", w.page.locator("#v-result"), LONG)
    w.report["verify"] = w.page.locator("#v-result").inner_text()
    w.widths("verify")


def verify_map(w):
    """The map after Apply: its labels and menus, tiles shown, the seam heat, and the legend off the base."""
    w.page.wait_for_timeout(1000)
    w.settle()
    w.page.wait_for_function("document.querySelectorAll('#v-compare img.leaflet-tile-loaded').length > 0", timeout=WORK)
    out = {
        "labels": w.page.locator("#v-compare .side-label").all_inner_texts(),
        "choices": w.page.locator("#v-compare").get_by_label("Left map").locator("option").all_inner_texts(),
        "tiles": w.page.locator("#v-compare img.leaflet-tile-loaded").count(),
        "heat": w.page.evaluate("Object.values(S.vcompare.heat).map((d) => d.length)"),
        "shot": w.shot("verify-map"),
    }
    out["legend"] = legend_clear(w, "vcompare")
    return out


def tile_summary(w):
    w.still(False)
    seen = [t for win in w.windows for t in win.values()]
    waited = sorted(t["age"] for t in seen if t.get("waited"))
    return {"tiles": len(seen), "drawn_while_waiting": len(waited), "max_age_s": waited[-1] if waited else None,
            "median_age_s": statistics.median(waited) if waited else None, "over_2s": sum(a > TILE_LIMIT_S for a in waited),
            "never_200": sum(OK not in t["statuses"] for t in seen)}  # fmt: skip


def fake_ui(out):
    """A synthetic world and the UI with the fake generator; (url, world path, process)."""
    sys.path[:0] = [TESTS]
    from conftest import add_migration_parts
    from test_web import fake_world

    root = os.path.join(out, "fake")
    world = fake_world(os.path.join(root, "server"))
    add_migration_parts(os.path.join(root, "server"))
    env = os.environ | {"PYTHONPATH": os.path.join(HERE, ".."), "RWG2_FAKE_DELAY": os.environ.get("RWG2_FAKE_DELAY", "0.3")}
    cmd = [sys.executable, os.path.join(TESTS, "run_fake_ui.py"), os.path.join(root, "data"), os.path.join(root, "cache")]
    proc = subprocess.Popen(cmd, cwd=TESTS, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.console = [proc.stdout.readline().rstrip("\n") for _ in range(2)]
    return proc.console[0], world, proc


def walk(page, opts, fake):
    w = Walk(page, opts.out)
    page.add_init_script(FRAMES_JS)
    page.goto(opts.url)
    try:
        open_world(w, opts.world)
        area(w, fake)
        search(w, opts.stop_after)
        pick(w, opts.pick)
        preview(w, opts)
        apply_screen(w, opts.apply)
    except (Stuck, PlaywrightError) as e:
        w.report["stopped"] = str(e)
        w.shot("stopped")
    w.report["candidate_tiles"] = tile_summary(w)
    return w.report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url")
    ap.add_argument("--world")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--pick", type=int, default=1)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--trees", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--stop-after", type=float, metavar="S")
    opts = ap.parse_args()
    os.makedirs(opts.out, exist_ok=True)
    proc = None
    if opts.fake:
        opts.url, opts.world, proc = fake_ui(opts.out)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=dict(zip(("width", "height"), VIEWPORTS["1920x1080"], strict=True)))
        report = walk(page, opts, opts.fake)
        browser.close()
    if proc:
        proc.terminate()
        out, err = proc.communicate()
        report["console"] = {"stdout": proc.console + out.splitlines(), "stderr": err.splitlines()}
    with open(os.path.join(opts.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report.get(k) for k in ("stopped", "stuck", "width", "maps", "candidate_tiles", "search")}, indent=1))


if __name__ == "__main__":
    main()
