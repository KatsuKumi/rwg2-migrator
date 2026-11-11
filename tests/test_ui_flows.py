"""The page after a restart, driven with Playwright on on-screen text: Continue on a search attaches the Best matches screen live,
many tabs share the server's connections, Continue on an Apply cut off between its two renames finishes it, and a migrated world
reopened later reaches Verify, the rules and a Roll back that reads as done."""

import os
import re
import subprocess
import sys
import time

import psutil
from conftest import stop_tree
from test_kill import KILLED, cli, fresh
from test_ui_walkthrough import chromium
from test_web import HERE, fake_world, free_port, http_json, until

from rwg2migrator import apply, jobs
from rwg2migrator.web import server

DELAY = "0.3"
TABS = 8
ANSWER_S = 10.0
HELD_POLL_MS = 50
SETTLE_MS = 500
SHORT = 30_000
LONG = 180_000
TICKS = 3
PAUSE_S = 7200
RAN_S = 300
RESUMED_S = 60
RESUMED_LAST_S = 1
HELD_MS = 1500
REPLAYED_AGO_S = 3 * 24 * 3600


def launch(data, cache, port, delay=None):
    env = os.environ | {"PYTHONPATH": str(HERE.parent)} | ({"RWG2_FAKE_DELAY": delay} if delay else {})
    proc = subprocess.Popen([sys.executable, str(HERE / "run_fake_ui.py"), str(data), str(cache), str(port)], cwd=HERE, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)  # fmt: skip
    url = proc.stdout.readline().strip()
    return proc, url, url.split("t=")[1]


def close(proc, console_closed=False):
    stop_tree(proc, console_closed)
    proc.wait(LONG / 1000)


def browser(p):
    return p.chromium.launch().new_context(viewport={"width": 1366, "height": 768})


def places_tried(page):
    m = re.search(r"([\d,]+) places tried", page.locator("#m-time").inner_text())
    return int(m.group(1).replace(",", "")) if m else 0


def state_answers(page):
    """Seconds /api/state takes from this tab, None when it does not answer within ANSWER_S."""
    return page.evaluate(
        """async (limit) => { const c = new AbortController(); const t0 = performance.now(); setTimeout(() => c.abort(), limit * 1000);
        try { await fetch('/api/state', {signal: c.signal, headers: {'X-Token': document.querySelector('meta[name=token]').content}});
        return (performance.now() - t0) / 1000; } catch (e) { return null; } }""",
        ANSWER_S,
    )


def test_continue_after_the_console_closed_follows_the_search_live_in_many_tabs(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    w = fake_world(tmp_path / "server")
    data, cache, port = tmp_path / "data", tmp_path / "cache", free_port()
    ui, _, token = launch(data, cache, port, DELAY)
    for path, body in (("world", {"path": w}), ("eula", {"accept": True})):
        http_json(port, token, path, body)
    until(lambda: http_json(port, token, "state")["world"]["scan"].get("chunks"), "the scan")
    http_json(port, token, "area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
    http_json(port, token, "search", {"base": None, "banned": None, "tolerance": 1})
    job = jobs.Job(str(data), until(lambda: http_json(port, token, "state")["runner"]["job"], "the search job"))
    until(lambda: any(e.get("event") == "batch" for e in job.events()), "a search batch")
    step = psutil.Process(job.data["pid"])
    close(ui, console_closed=True)
    step.wait(LONG / 1000)
    ui, url, token = launch(data, cache, port, DELAY)
    loads = []
    try:
        with sync_playwright() as p:
            ctx = browser(p)
            page = ctx.new_page()
            page.goto(url)
            page.on("load", lambda pg: loads.append(pg.url))
            page.get_by_role("button", name="Continue", exact=True).click()
            page.get_by_role("button", name="Stop and use best so far").wait_for(state="visible", timeout=SHORT)
            tried = [places_tried(page)]
            while len(set(tried)) < TICKS and not page.get_by_role("button", name="Pick this one").count():
                page.wait_for_timeout(500)
                tried.append(places_tried(page))
            assert len(set(tried)) >= TICKS and page.locator("#m-stage").inner_text() not in ("", "Starting…"), tried
            tabs = [ctx.new_page() for _ in range(TABS - 1)]
            for tab in tabs:
                tab.goto(f"{url}#matches/{job.id}", wait_until="domcontentloaded")
            tabs[-1].wait_for_timeout(2000)
            answers = [state_answers(t) for t in (page, *tabs)]
            assert all(a is not None and a < ANSWER_S for a in answers), answers
            for tab in tabs:
                tab.close()
            page.get_by_role("button", name="Pick this one").first.wait_for(state="visible", timeout=LONG)
            assert job.state() == jobs.DONE and not page.get_by_role("alertdialog").is_visible()
            assert loads == [], loads
    finally:
        close(ui)


def test_continue_an_apply_cut_off_between_its_renames_then_verify_and_roll_back_after_a_restart(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    m = fresh(tmp_path / "m")
    _, preview = m.preview()
    r = cli(m, "apply", m.root, "--preview", preview.id, "--yes", "--auto-restart-off", kill=f"{apply.SWAP}:4")
    assert r.returncode in KILLED and not os.path.exists(m.world)
    port = free_port()
    ui, url, _ = launch(m.data, m.cache, port)
    try:
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(url)
            page.get_by_role("button", name="Continue", exact=True).click()
            page.get_by_role("heading", name="Your world now uses RWG2 Smooth").wait_for(timeout=LONG)
            assert os.path.isdir(m.world) and apply.record_of(m.world)["state"] == apply.APPLIED
            assert "worldgen migration" in page.locator("#screen-after").inner_text()
        close(ui)
        ui, url, _ = launch(m.data, m.cache, port)
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(url)
            page.locator("#home-worlds").get_by_role("button", name="Open").click()
            page.get_by_text("This tool migrated this world to RWG2 Smooth").wait_for(timeout=SHORT)
            page.get_by_role("button", name="Show the rules").click()
            page.get_by_role("heading", name="Your world now uses RWG2 Smooth").wait_for(timeout=SHORT)
            page.get_by_role("button", name="Go to Verify").click()
            page.get_by_role("button", name="Check my world").click()
            page.get_by_text("The 4 offset lines match the migration record.").wait_for(timeout=LONG)
            page.get_by_label("I turned off anything that restarts my server by itself").check()
            page.get_by_role("button", name="Roll back").click()
            page.get_by_role("dialog").get_by_role("button", name="Roll back").click()
            page.get_by_text("Your world from before Apply is back.").wait_for(timeout=LONG)
            assert not page.get_by_role("alertdialog").is_visible() and apply.record_of(m.world) is None
            until(lambda: page.title() == "Done – rwg2-migrator", "the tab title")
        close(ui)
        ui, url, token = launch(m.data, m.cache, port)
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(url)
            row = page.locator("#home-finished li", has_text="Apply")
            row.get_by_text("Rolled back: the world from before Apply is back").wait_for(timeout=SHORT)
            assert not row.get_by_role("button").count()
            http_json(port, token, "world", {"path": m.world})
            page = browser(p).new_page()
            page.goto(f"{url}#after")
            page.get_by_text("was rolled back, so it uses the old terrain generator again").wait_for(timeout=SHORT)
            assert not page.get_by_role("heading", name="Your world now uses RWG2 Smooth").count()
            assert "Open it on the World screen" in page.locator("#after-none").inner_text()
            page = browser(p).new_page()
            page.goto(url)
            page.locator("#home-finished li", has_text="Preview").get_by_role("button", name="See the preview").click()
            page.get_by_role("button", name="Build a new preview of this match").wait_for(timeout=LONG)
            assert not page.get_by_role("button", name="Continue: Apply").is_visible()
            page.get_by_role("navigation").get_by_role("button", name="5 Apply").click()
            line = page.locator("#a-list li.bad").first
            line.wait_for(timeout=SHORT)
            text = line.inner_text()
            assert text.startswith("This preview was applied, and that Apply was rolled back"), text
            assert "Build a new preview of this match" in text and not re.search(r"\d{8}-\d{6}|rwg2-migrator ", text), text
    finally:
        close(ui)


def search_events(now):
    """A search cut off RAN_S seconds into its run PAUSE_S ago and resumed RESUMED_S ago, as the page gets its events: the
    resumed run replays its finished stages (their own config counts, their lower shares) before it samples again."""
    first, again = now - PAUSE_S, now - RESUMED_S
    raw = [
        {"t": first, "step": "job", "event": "start"},
        {"t": first + RAN_S / 3, "step": "search", "event": "batch", "stage": "coarse", "cache": 8000},
        {"t": first + RAN_S / 3, "step": "search", "event": "stage", "stage": "coarse", "configs": 8000},
        {"t": first + RAN_S, "step": "search", "event": "batch", "stage": "coarse-land", "cache": 8001},
        {"t": again, "step": "job", "event": "start"},
        *({"t": again + RESUMED_LAST_S, "step": "search", "event": "stage", "stage": st, "configs": c} for st, c in
          (("baseline", 1), ("coarse", 8000), ("coarse-full", 600))),
    ]  # fmt: skip
    return [server.labelled(e) for e in raw]


def test_a_resumed_search_counts_on_and_times_what_is_left_on_its_events_clock_without_the_pause(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    ui, url, _ = launch(tmp_path / "data", tmp_path / "cache", free_port())
    shown = """(ev) => { S.search = {started: ev[0].t}; ev.forEach(searchEvent);
        return [$('#m-time').textContent, $('#m-stage').textContent, $('#m-bar').style.width]; }"""
    try:
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(url)
            events = search_events(time.time())
            share = server.stage_label("coarse-land")[1]
            left_min = round((RAN_S + RESUMED_LAST_S) * (1 - share) / share / 60)
            assert page.evaluate(shown, events) == [
                f"8,001 places tried · about {left_min} min left",
                "Checking that the best places stay on land",
                f"{100 * share:g}%",
            ]
            assert page.evaluate(shown, search_events(time.time() - REPLAYED_AGO_S)) == page.evaluate(shown, events)
            reset = [
                {"step": "search", "event": "reset", "changed": ["config"]},
                {"step": "search", "event": "batch", "stage": "baseline", "cache": 10},
            ]
            got = page.evaluate(shown, events + [server.labelled(e | {"t": time.time()}) for e in reset])
            label, first_share = server.stage_label("baseline")
            assert got[0].startswith("10 places tried") and got[1:] == [label, f"{100 * first_share:g}%"]
    finally:
        close(ui)


def test_a_tab_left_on_a_job_that_is_gone_says_what_to_do_in_on_screen_terms(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    try:
        http_json(port, token, "world", {"path": fake_world(tmp_path / "server")})
        until(lambda: http_json(port, token, "state")["world"]["scan"].get("chunks"), "the scan")
        http_json(port, token, "area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#matches/20260101-000000-gone00")
            panel = page.get_by_role("alertdialog")
            panel.wait_for(timeout=SHORT)
            assert "Home screen" in panel.inner_text() and "rwg2-migrator jobs" not in panel.inner_text(), panel.inner_text()
    finally:
        close(ui)


def test_a_tab_on_one_world_cannot_save_its_area_for_a_world_another_tab_opened(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    a, b = fake_world(tmp_path / "A"), fake_world(tmp_path / "B")
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    try:
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(url)
            page.get_by_role("button", name="Start with a world").click()
            page.get_by_label("World folder").fill(a)
            page.get_by_role("button", name="Open", exact=True).click()
            page.get_by_text("in the overworld").first.wait_for(timeout=SHORT)
            page.get_by_role("button", name="Continue: choose the area to keep").click()
            page.get_by_role("button", name="Import claims").click()
            page.wait_for_function("!document.querySelector('#area-continue').disabled", timeout=SHORT)
            http_json(port, token, "world", {"path": b})
            until(lambda: http_json(port, token, "state")["world"]["scan"].get("chunks"), "the scan of B")
            page.get_by_role("button", name="Continue: find the best matches").click()
            panel = page.get_by_role("alertdialog")
            panel.wait_for(timeout=SHORT)
            assert "still shows" in panel.inner_text() and "Reload this page" in panel.inner_text(), panel.inner_text()
            assert http_json(port, token, "state")["area"] is None
            page.reload()
            page.wait_for_function("(w) => S.world === w", arg=b, timeout=SHORT)
    finally:
        close(ui)


def caught(page, held):
    """Whether a route handler has put a request into held; waiting on the page lets Playwright run that handler."""
    for _ in range(SHORT // HELD_POLL_MS):
        if held:
            break
        page.wait_for_timeout(HELD_POLL_MS)
    return bool(held)


def open_area(page, world_dir):
    """Open world_dir on the World screen and go on to its Area screen once its card shows it scanned."""
    page.get_by_label("World folder").fill(world_dir)
    page.get_by_role("button", name="Open", exact=True).click()
    page.locator("#world-card", has_text=world_dir).filter(has_text="in the overworld").wait_for(timeout=SHORT)
    page.get_by_role("button", name="Continue: choose the area to keep").click()
    page.get_by_role("heading", name="Area to keep").wait_for(timeout=SHORT)


def suggest_three(page):
    page.get_by_role("button", name="Suggest from where you've played").click()
    page.wait_for_function("document.querySelector('#c-core').textContent === '3'", timeout=SHORT)
    assert not page.get_by_role("alertdialog").is_visible()


def test_the_area_screen_takes_no_click_before_its_world_data_has_arrived(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    w = fake_world(tmp_path / "server")
    ui, url, _ = launch(tmp_path / "data", tmp_path / "cache", free_port())
    held = []
    try:
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.route("**/api/world", lambda route: held.append(route) if route.request.method == "GET" else route.continue_())
            page.goto(url)
            page.get_by_role("button", name="Start with a world").click()
            open_area(page, w)
            locked = "document.querySelector('#screen-area').inert"
            assert caught(page, held) and page.evaluate(locked)
            page.get_by_role("button", name="Home", exact=True).click()
            page.get_by_role("button", name="2 Area to keep").click()
            page.wait_for_timeout(HELD_POLL_MS)
            assert len(held) == 2, "entering the screen again asks for the world's data again"
            held[0].continue_()
            page.wait_for_timeout(SETTLE_MS)
            assert page.evaluate(locked), "the answer to the first, abandoned entry must not unlock the screen"
            held[1].continue_()
            page.unroute("**/api/world")
            suggest_three(page)
            assert not page.evaluate(locked)
            page.evaluate("document.querySelector('#screen-area').inert = true")
            page.get_by_role("button", name="Home", exact=True).click()
            page.get_by_role("button", name="2 Area to keep").click()
            page.wait_for_function(f"!{locked}", timeout=SHORT)
    finally:
        close(ui)


def test_the_area_screen_works_for_a_second_world_opened_in_the_same_page(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    worlds = [fake_world(tmp_path / name) for name in ("A", "B")]
    ui, url, _ = launch(tmp_path / "data", tmp_path / "cache", free_port())
    try:
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(url)
            page.get_by_role("button", name="Start with a world").click()
            for world_dir in worlds:
                open_area(page, world_dir)
                suggest_three(page)
                page.get_by_role("button", name="1 World").click()
    finally:
        close(ui)


def finished_search(port, token, w):
    for path, body in (("world", {"path": w}), ("eula", {"accept": True})):
        http_json(port, token, path, body)
    until(lambda: http_json(port, token, "state")["world"]["scan"].get("chunks"), "the scan")
    http_json(port, token, "area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
    http_json(port, token, "search", {"base": None, "banned": None, "tolerance": 1})
    job = until(lambda: http_json(port, token, "state")["runner"]["job"], "the search job")
    until(lambda: not http_json(port, token, "state")["runner"]["running"], "the search")
    return job


def test_a_match_whose_maps_are_refused_says_why_in_the_large_view(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    try:
        job = finished_search(port, token, fake_world(tmp_path / "server"))
        os.remove(tmp_path / "data" / "jobs" / job / "site" / "reference.json")
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#matches/{job}")
            panel = page.get_by_role("alertdialog")
            panel.wait_for(timeout=SHORT)
            assert "has not measured your base yet" in panel.inner_text(), panel.inner_text()
            assert "cannot be drawn" in page.locator("#m-compare .empty").inner_text()
    finally:
        close(ui)


def test_a_config_changed_since_the_search_says_so_on_the_maps_and_search_again_draws_them(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    w = fake_world(tmp_path / "server")
    cfg = tmp_path / "server" / "config" / "rwg2.cfg"
    cfg.parent.mkdir()
    cfg.write_text('"expert / worldgen migration" {\n    I:"Biome Placement Offset X"=0\n}\n')
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    try:
        job = finished_search(port, token, w)
        cfg.write_text('"client" {\n    B:"some setting"=true\n}\n')
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#matches/{job}")
            panel = page.get_by_role("alertdialog")
            panel.wait_for(timeout=SHORT)
            assert "changed since this search ran" in panel.inner_text() and "Search again" in panel.inner_text(), panel.inner_text()
            note = page.locator("#m-compare .mapnote")
            note.wait_for(timeout=SHORT)
            assert note.inner_text().startswith("The new terrain cannot be drawn now") and "Search again" in note.inner_text()
            page.locator("#error-close").click()
            page.get_by_role("button", name="Search again").click()
            page.get_by_role("button", name="Find the best matches").click()
            page.get_by_role("button", name="Pick this one").first.wait_for(timeout=LONG)
            drawn = "[...document.querySelectorAll('#m-compare img')].some((i) => i.src.includes('/tiles/cand.') && i.naturalWidth)"
            page.wait_for_function(drawn, timeout=LONG)
            assert not note.count() and not panel.is_visible(), panel.inner_text()
    finally:
        close(ui)


def hover_until(page, at, readout, pattern):
    """Move the pointer a pixel at a time from at until the readout matches (the tile under it may still be drawing)."""
    for dx in range(60):
        page.mouse.move(at[0] + dx, at[1])
        page.wait_for_timeout(500)
        if pattern.search(readout.inner_text()):
            break
    return readout.inner_text()


def test_the_candidate_map_reads_out_the_new_terrain_and_the_now_map_todays_world(tmp_path):
    from fake_sampler import NAMES
    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    new_terrain = re.compile(rf"^({'|'.join(NAMES)}), ground at y \d+$", re.M)
    try:
        job = finished_search(port, token, fake_world(tmp_path / "server"))
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#matches/{job}")
            page.get_by_role("button", name="Side by side", exact=True).click()
            panes = page.locator("#m-compare .leaflet-container")
            panes.nth(1).wait_for(timeout=SHORT)
            box = panes.nth(1).bounding_box()
            readouts = page.locator("#m-compare .readout")
            text = hover_until(page, (box["x"] + 40, box["y"] + 120), readouts.nth(1), new_terrain)
            assert new_terrain.search(text), text
            left = panes.nth(0).bounding_box()
            page.mouse.move(left["x"] + left["width"] / 2, left["y"] + left["height"] / 2)
            page.wait_for_timeout(1000)
            now = readouts.nth(0).inner_text()
            assert "ground at y" in now and not new_terrain.search(now), now
    finally:
        close(ui)


def test_a_seam_heat_refused_after_its_view_is_gone_raises_no_error(tmp_path):
    import json

    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    refused = {"now": False}
    gone = json.dumps({"error": {"reason": "the world already uses RWG2", "action": "Check it on the Verify screen"}})

    def heat(route):
        body, status = (gone, 400) if refused["now"] else ('{"pending": "measuring"}', 200)
        route.fulfill(status=status, content_type="application/json", body=body)

    try:
        job = finished_search(port, token, fake_world(tmp_path / "server"))
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.route("**/api/heat?*", heat)
            page.goto(f"{url}#matches/{job}")
            page.locator("#m-compare .leaflet-container").first.wait_for(timeout=SHORT)
            refused["now"] = True
            page.get_by_role("alertdialog").wait_for(timeout=SHORT)
            page.locator("#error-close").click()
            refused["now"] = False
            page.locator("#m-cards .cand").nth(1).click()
            page.wait_for_timeout(3000)
            page.get_by_role("button", name="1 World").click()
            refused["now"] = True
            page.wait_for_timeout(5000)
            assert not page.get_by_role("alertdialog").is_visible()
    finally:
        close(ui)


def test_the_stop_button_stops_a_search_without_an_error_and_continue_finishes_it(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port, DELAY)
    try:
        for path, body in (("world", {"path": fake_world(tmp_path / "server")}), ("eula", {"accept": True})):
            http_json(port, token, path, body)
        until(lambda: http_json(port, token, "state")["world"]["scan"].get("chunks"), "the scan")
        http_json(port, token, "area", {"teams": ["alpha"], "painted": [], "erased": [], "margin": 1, "save": True})
        http_json(port, token, "search", {"base": None, "banned": None, "tolerance": 1})
        job = until(lambda: http_json(port, token, "state")["runner"]["job"], "the search job")
        with sync_playwright() as p:
            page = browser(p).new_page()
            dropped = []
            page.route("**/api/stop", lambda route: route.continue_() if dropped else dropped.append(route.abort()))
            page.goto(f"{url}#matches/{job}")
            page.get_by_role("button", name="Stop and use best so far").wait_for(timeout=SHORT)
            page.get_by_role("banner").get_by_role("button", name="Stop").click()
            page.get_by_role("dialog").get_by_role("button", name="Stop").click()
            page.get_by_role("button", name="Continue the search").wait_for(timeout=SHORT)
            assert dropped, "the first stop request was not dropped, so its retry is untested"
            assert not page.get_by_role("alertdialog").is_visible() and jobs.Job(str(tmp_path / "data"), job).state() == jobs.INTERRUPTED
            assert "stopped before it finished" in page.locator("#m-stopped").inner_text()
            page.get_by_role("button", name="Continue the search").click()
            page.get_by_role("button", name="Pick this one").first.wait_for(timeout=LONG)
            assert not page.get_by_role("alertdialog").is_visible() and page.locator("#m-stopped").is_hidden()
    finally:
        close(ui)


def test_after_the_area_changes_best_matches_asks_for_a_new_search_and_offers_no_pick(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    try:
        finished_search(port, token, fake_world(tmp_path / "server"))
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#area")
            page.wait_for_function("document.querySelector('#c-kept').textContent === '15'", timeout=SHORT)
            page.get_by_label(re.compile("Safety margin")).fill("2")
            page.wait_for_function("document.querySelector('#c-kept').textContent === '35'", timeout=SHORT)
            page.get_by_role("button", name="Continue: find the best matches").click()
            page.get_by_text("You changed the area to keep since the last search").wait_for(timeout=SHORT)
            assert page.get_by_role("button", name="Find the best matches").is_visible()
            assert not page.get_by_role("button", name="Pick this one").count()
            page.get_by_role("navigation").get_by_role("button", name="Home").click()
            page.locator("#home-finished li", has_text="Best-match search").get_by_role("button", name="See the matches").click()
            page.get_by_text("These matches were found for another area to keep").wait_for(timeout=SHORT)
            page.locator("#m-cards .cand").first.wait_for(timeout=SHORT)
            assert not page.get_by_role("button", name="Pick this one").count()
            assert page.get_by_role("button", name="Search again for this area").is_visible()
    finally:
        close(ui)


def test_the_downloaded_generator_is_removed_from_the_world_screen(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    try:
        (tmp_path / "cache" / "packs" / "2.9.0").mkdir(parents=True)
        (tmp_path / "cache" / "packs" / "2.9.0" / "server.jar").write_bytes(b"jar")
        assert http_json(port, token, "state")["generator"]
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#world")
            page.get_by_role("button", name="Remove the downloaded generator").click()
            page.locator("#confirm").get_by_role("button", name="Remove").click()
            page.get_by_text("is downloaded and set up when the search starts").wait_for(timeout=SHORT)
        assert not http_json(port, token, "state")["generator"] and not (tmp_path / "cache" / "packs").exists()
        assert (tmp_path / "cache").is_dir() and (tmp_path / "data").is_dir()
    finally:
        close(ui)


def test_build_the_preview_shows_only_once_the_page_knows_the_match_and_then_posts_it(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    held = []
    try:
        job = finished_search(port, token, fake_world(tmp_path / "server"))

        def the_match(u):
            return f"/api/job?id={job}" in u

        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#matches/{job}")
            pick = page.get_by_role("button", name="Pick this one").first
            pick.wait_for(timeout=SHORT)
            page.route(the_match, lambda route: held.append(route))
            pick.click()
            page.get_by_role("heading", name="Preview").wait_for(timeout=SHORT)
            page.wait_for_timeout(HELD_MS)
            build = page.get_by_role("button", name="Build the preview", exact=True)
            assert held and build.is_hidden()
            for route in held:
                route.continue_()
            page.unroute(the_match)
            with page.expect_request(lambda r: r.url.endswith("/api/preview")) as sent:
                build.click(timeout=SHORT)
            assert sent.value.post_data_json["search"] == job and sent.value.post_data_json["pick"] == 1
            assert sent.value.response().status == 200 and not page.get_by_role("alertdialog").is_visible()
    finally:
        close(ui)


def test_the_config_folder_picked_on_the_world_screen_is_what_the_search_runs_on(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    port = free_port()
    ui, url, token = launch(tmp_path / "data", tmp_path / "cache", port)
    try:
        w = fake_world(tmp_path / "download")
        os.remove(tmp_path / "download" / "server.properties")
        mine = tmp_path / "mine"
        mine.mkdir()
        (mine / "rwg2.cfg").write_text("")
        http_json(port, token, "world", {"path": w})
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#world")
            page.get_by_text("No config folder of its own was found next to this world").wait_for(timeout=SHORT)
            page.get_by_label("Config folder").fill(str(mine))
            page.get_by_role("button", name="Use", exact=True).click()
            page.get_by_text(f"In use: {mine} (chosen by you)").wait_for(timeout=SHORT)
        job = finished_search(port, token, w)
        assert jobs.Job(str(tmp_path / "data"), job).data["args"]["config_dir"] == str(mine)
    finally:
        close(ui)


def test_verify_without_a_config_folder_shows_the_lines_to_set_by_hand(tmp_path):
    from playwright.sync_api import sync_playwright

    chromium()
    m = fresh(tmp_path / "m")
    _, preview = m.preview()
    assert m.apply(preview, "--taint-spread-off") == 0
    os.rename(os.path.join(m.root, "config"), os.path.join(m.root, "config-away"))
    port = free_port()
    ui, url, token = launch(m.data, m.cache, port)
    try:
        http_json(port, token, "world", {"path": m.world})
        with sync_playwright() as p:
            page = browser(p).new_page()
            page.goto(f"{url}#verify")
            page.get_by_role("button", name="Check my world").click()
            page.get_by_text("Pick its config folder on the World screen to check them, or:").wait_for(timeout=LONG)
            text = page.locator("#v-result").inner_text()
            assert apply.TAINT_LINE in text and '"Biome Placement Offset X"' in text and "No other problem found." in text
    finally:
        close(ui)
