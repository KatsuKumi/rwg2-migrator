"""The CLI commands of the migration itself: preview, apply, rollback and verify."""

import json
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict

from . import Refusal, apply, cli_search, generator, genworld, jobs, maps, postcheck, search, smooth, staging, transplant, world
from .region import write_atomic

EXIT_PROBLEMS = 2
NOTHING_CHANGED = "Every check passed; nothing was changed."
SAVED_SINCE = (
    "The world was played after the swap: the {files} your game saved since then are not compared with the preview, and the "
    "protected chunks among them are not counted as identical."
)


def _err(line):
    print(line, file=sys.stderr, flush=True)


def _cells(cells):
    return jobs.listed_cells(cells) or "none"


def preview_cmd(args, scan_world):
    data = cli_search._data(args)
    found = jobs.find(data, args.search)
    place = cli_search.searched_place(args, found)
    a = cli_search.read_area(args.area, place)
    level, _ = world.check(place.world)
    if args.ring < genworld.MIN_RING:
        raise Refusal(
            f"a ring of {args.ring} chunks leaves the chunks next to the kept area undecorated",
            f"Use --ring {genworld.MIN_RING} or more ({genworld.RING} is the default)",
        )
    if (args.trees or args.blend_biomes) and (args.no_generate or args.no_smooth):
        raise Refusal(
            "--trees and --blend-biomes work on the smoothed border, so they need the new terrain generated and smoothed",
            "Leave out --no-generate and --no-smooth, or leave out --trees and --blend-biomes",
        )
    apply.refuse_unfinished(data, place.world)
    chosen = maps.pick(found, args.pick)
    searched = jobs.read_json(os.path.join(found.folder, cli_search.AREA_FILE)) or {}
    if [sorted(map(tuple, searched.get(k, []))) for k in ("core", "kept")] != [sorted(a[k]) for k in ("core", "kept")]:
        raise Refusal(
            f"search job {found.id} was made for another area than {args.area}",
            f"Search this area first: {apply.search_again(place, args.area)}",
        )
    changed = jobs.config_change(place.config, search.sampled_config(found), f"search job {found.id}")
    if changed:
        raise Refusal(
            f"{changed}, so its candidates may no longer be what RWG2 makes",
            f"Run the search again: {apply.search_again(place, args.area)}",
        )
    cli_search.defaults_note(place.config)
    sc = scan_world(place, args)
    c = chosen["candidate"]
    job_args = {
        "search": found.id,
        "pick": args.pick,
        "config": c["config"],
        "taint": c.get("taint"),
        "options": asdict(transplant.Options(set_populated=args.set_populated)),
        "no_generate": args.no_generate,
        "smooth": not args.no_smooth,
        "trees": args.trees,
        "blend_biomes": args.blend_biomes,
        "ring": args.ring,
        "seed": str(level["Data"]["RandomSeed"]),
        "cache": cli_search._cache(args),
        "threads": cli_search._threads(args),
        "place": asdict(place),
        "root": place.root,
        "config_dir": place.config,
    }
    job = jobs.create(data, "preview", job_args, place.world)
    job.update(args=job_args | {"staging": staging.staging_dir(place.root, job.id)})
    write_atomic(os.path.join(job.folder, "area.json"), (json.dumps(a) + "\n").encode())
    _err(f"Job {job.id} ({job.folder})")
    return _run_preview(args, job, sc)


def describe(e):
    """One console line per preview event worth showing."""
    kind, step = e.get("event"), e.get("step")
    if step == "boot" and kind == "ready":
        return f"generator: {e['kind']} boot ready in {e['seconds']} s"
    if step == "generate" and kind == "progress":
        return f"generate: {e['done']:,}/{e['total']:,} chunks"
    if kind in ("done", "failed", "interrupted") and step in ("generate", "render", "job"):
        return f"{step}: {kind}" + (f" ({e['error']})" if "error" in e else "")
    return None


def _show(e):
    line = describe(e)
    if line:
        _err(line)


def generators(job, args, emit=_show):
    """The generator of a preview that generates its new terrain, else None."""
    d = job.data["args"]
    if d.get("no_generate", True):
        return None
    gens = cli_search.Generators(d["cache"], args, emit, d.get("config_dir"))
    gens.threads = d["threads"]
    return gens


def _run_preview(args, job, sc):
    opts = transplant.Options(**job.data["args"]["options"])
    gens = generators(job, args)
    job.listeners.append(_show)

    def work(j):
        apply.supersede(j.root, j.data["world"], j.id)
        return staging.build(j, staging.Work(sc, opts, gens, _err))

    try:
        with generator.lock(gens.cache) if gens else nullcontext():
            record = jobs.run(job, work)
    finally:
        if gens:
            gens.stop()
    print(json.dumps(record | {"job": job.id}, indent=1) if args.json else "\n".join(preview_lines(job, record)))
    return 0


def resume_preview(args, job, scan_world):
    place = world.Place(**job.data["args"]["place"])
    return _run_preview(args, job, scan_world(place, args))


def _seam(s):
    if not s.get("pairs"):
        return "no pairs"
    return f"median {s['median']:g}, p90 {s['p90']:g}, {s['over10_pct']:g}% over 10, {s['over30_pct']:g}% over 30, max {s['max']:g}"


def generated_lines(job):
    """What the generated terrain, the harvest, the smoothing and the post-check found."""
    g = genworld.generated(job)
    h = jobs.read_json(os.path.join(job.folder, genworld.HARVEST, staging.RECORD))
    sm = jobs.read_json(os.path.join(job.folder, smooth.SMOOTH, staging.RECORD))
    p = jobs.read_json(os.path.join(job.folder, staging.POSTCHECK, staging.REPORT))
    t, nt, kb = g["totals"], p["new_terrain"], p["seams"]
    lines = [
        f"  New terrain: {g['requested']:,} chunks around the kept area (ring {job.data['args']['ring']}), {t.get('generated', 0):,} of "
        f"them new. {h['harvested']:,} chunks copied into the working copy ({h.get('beyond_ring', 0):,} of them past the ring, made by "
        f"the generator's start or its trees); {h['not_decorated']:,} of them, at the edge, will be decorated by your game once it "
        "generates past them",
        f"  Generator: boot {g['seconds']['boot']:g} s, generate {g['seconds']['generate']:g} s, save {g['seconds']['save']:g} s; "
        f"peak memory {g['peak_working_set_mb']:,} MB (heap {g['heap']}); noise {g['noise']}",
        f"  Game vs the candidate's render in the new chunks: median {nt.get('median', 'n/a')}, {nt.get('exact_pct', 0):g}% exact, "
        f"{nt.get('within1_pct', 0):g}% within 1 block ({nt['columns']:,} columns)",
        f"  Seam at the kept border before smoothing: {_seam(kb['before']['kept_border'])} "
        f"({kb['before']['kept_border']['pairs']:,} pairs)",
    ]
    if sm:
        lines += [
            f"  Seam at the kept border after smoothing: {_seam(kb['after']['kept_border'])}",
            f"  Smoothing: {p['columns_reshaped']:,} columns reshaped, {p['core_columns_changed']} in protected chunks; "
            f"{sm['new_fluid_walls']} new fluid walls; {sm['tree_log_voxels_removed']:,} log voxels removed",
            *_smoothed_away_lines(p.get("smoothing_removed")),
        ]
    lines += _extras_lines(p)
    c = h["trees_clipped"]
    lines.append(
        f"  Trees cut at the kept border (kept chunks are never copied): {c['logs']:,} log and {c['leaves']:,} leaf voxels "
        f"in {c['chunks']} chunks"
    )
    out = p["outside_overworld"]
    lines.append(
        f"  Files outside the overworld: {out['identical']:,} of {out['files']:,} identical; changed by the transplant: "
        f"{', '.join(out['changed_by_transplant']) or 'none'}"
    )
    return lines


def _smoothed_away_lines(removed):
    """What smoothing took off the columns it reshaped; a preview built before this was counted has no such line."""
    if not removed:
        return []
    return [
        f"  Removed on the reshaped columns: {removed['entities']:,} animals or other entities, "
        f"{removed['tile_entities']:,} crops, flower pots or similar blocks"
    ]


def _extras_lines(p):
    """The replanting and the biome blend, when the preview made them."""
    lines, t, b = [], p.get("trees"), p.get("biome_blend")
    if t:
        share = 100 * t["log_in_strips"] / t["log_removed"] if t["log_removed"] else 0
        lines.append(
            f"  Trees replanted: {t['boot']['trees']:,} trees; {t['log']:,} log and {t['leaves']:,} leaf voxels added, only into air "
            f"({t['log_in_strips']:,} log voxels in the reshaped strips: {share:.0f}% of the {t['log_removed']:,} smoothing removed)"
        )
    if b:
        lines.append(
            f"  Biome blend: {b['columns_changed']:,} margin columns took the new terrain's biome; grass colour steps over dE 10 "
            f"across the kept border: {b['border_over_pct_before']:g}% without the blend, {b['border_over_pct_after']:g}% with it "
            f"({b['border_pairs']:,} pairs)"
        )
    return lines


def preview_lines(job, r):
    d = job.data
    regions = [rel for rel in staging.source(job)["files"] if staging.dim0_region(rel)]
    names = world.player_names(d["args"]["root"])
    moved = ", ".join(f"{names.get(m['name'], m['name'])} (from chunk {m['from'][0]},{m['from'][1]})" for m in r["players_moved"]) or "none"
    flags = r["populated_set"]
    lines = [
        f"Preview {job.id} is ready to apply. Working copy: {d['args']['staging']}",
        f"  Kept: {r['kept']:,} chunks; {r['core_identical']:,} protected chunks byte-identical to the world",
        f"  Never explored kept chunks: {len(r['never_generated'])} (they become new terrain)",
        f"  Overworld region files: {len(r['regions_kept'])} kept ({', '.join(r['regions_kept'])}); "
        f"{len(regions) - len(r['regions_kept'])} removed",
        f"  Marked decorated: {len(flags['margin'])} margin chunks ({_cells(flags['margin'])}), "
        f"{len(flags['core'])} protected chunks ({_cells(flags['core'])})",
        f"  Players moved into the kept area: {moved}",
        f"  level.dat and level.dat_old: generator {', '.join(sorted({v['generatorName'] for v in r['level'].values()}))} "
        f"-> {r['generator']}",
        f"  Deleted: {', '.join(r['data_deleted']) or 'nothing'}; other structure files kept: {', '.join(r['data_reported']) or 'none'}",
    ]
    if d["args"].get("no_generate", True):
        lines.append("  New terrain: RWG2 generates it on your server or game around the kept area (--no-generate)")
    else:
        lines += generated_lines(job)
    place = world.Place(**d["args"]["place"])
    if place.kind == "server":
        return [*lines, f"Next (keep the server stopped, and turn off its auto-restart): {apply_command(d['world'], job.id, place)}"]
    return [*lines, f"Next (keep the game closed): {apply_command(d['world'], job.id, place)}"]


def _restart_flag(place, remote=False):
    return " --auto-restart-off" if place.kind == "server" and not remote else ""


def apply_command(world_dir, preview_id, place, remote=False):
    """The apply command that runs as it stands: a server's Apply also needs --auto-restart-off."""
    return f"rwg2-migrator apply {world_dir} --preview {preview_id} --yes" + _restart_flag(place, remote)


def _options(args):
    return apply.Options(
        yes=args.yes,
        remote=args.remote,
        zip=args.zip,
        backup_copy=args.backup_copy and world.normalise(args.backup_copy),
        taint_off=args.taint_spread_off,
        leave_players=args.leave_players,
        allow_shared_instance=args.allow_shared_instance,
        confirm_stopped=args.confirm_stopped,
        auto_restart_off=args.auto_restart_off,
        reset_journeymap=args.reset_journeymap,
    )


def _yours(rm):
    """The probably-yours count of an area's removed record; an area made before the split counted every non-natural one."""
    return rm["yours"]["count"] if "yours" in rm else rm.get("tile_entities_other", 0)


def _configs_written(preview, opts, config):
    if config:
        taint = f" and biome_taint_spread=0 in {os.path.join(config, apply.TAINT_CFG)}" if apply.taint_off(preview, opts) else ""
        return f"the 4 offset lines in {os.path.join(config, apply.RWG2_CFG)}" + taint
    return "none, as no config folder was found: Apply shows the 4 offset lines to add by hand and saves them next to the world"


def checklist_lines(preview, opts, place, config):
    a = jobs.read_json(os.path.join(preview.folder, "area.json"))
    rm = a.get("removed", {})
    others = ", ".join(f"{t} {n}" for t, n in rm.get("other_teams_claims", {}).items())
    cfg = _configs_written(preview, opts, config)
    return [
        f"Apply preview {preview.id} to {place.world}:",
        "  - the world must be stopped: the tool checks for a running game or server and for a level.dat saved in the last 2 minutes",
        f"  - removed for good: {a.get('status', {}).get('dropped', 0):,} overworld chunks become new terrain, "
        f"with {jobs.count(_yours(rm), 'block')} probably yours"
        + (f" and {jobs.count(len(rm['graves']), 'grave')}" if rm.get("graves") else ""),
        f"  - other teams' claims outside the kept area: {others or 'none'}",
        "  - players outside the kept area are moved to its nearest protected chunk"
        + (" (not: --leave-players)" if opts.leave_players else ""),
        f"  - configs written: {cfg}",
        f"  - your current world is kept, renamed, as the backup in {staging.work_dir(place.root)}",
        "  - rolling back later means losing all play since Apply, in every dimension",
    ]


def apply_cmd(args, scan_world):
    data = cli_search._data(args)
    preview = jobs.find(data, args.preview)
    named = world.locate(args.world, exists=False)
    if os.path.normcase(os.path.realpath(named.world)) != os.path.normcase(os.path.realpath(preview.data.get("world", ""))):
        raise Refusal(f"job {preview.id} is not a preview of {named.world}", "Pass a preview of this world: rwg2-migrator jobs list")
    place = world.Place(**preview.data["args"]["place"])
    opts = _options(args)
    config = world.with_config(place, args.config_dir).config
    job = apply.apply_job(data, preview, asdict(opts) | {"config": config})
    _ready(job, preview, place)
    if not opts.yes and not apply.is_open(job):
        print("\n".join(checklist_lines(preview, opts, place, config)))
        try:
            apply.check(apply.Ctx(place, preview, job, opts, None, config, _err))
        except apply.GoAhead:
            again = apply_command(named.world, preview.id, place, opts.remote) + (f" --config-dir {config}" if args.config_dir else "")
            print(f"{NOTHING_CHANGED}\nNext: {again}")
            return 0
    return _run_apply(args, job, scan_world)


def _run_apply(args, job, scan_world):
    data = job.data["args"]
    preview = jobs.find(job.root, data["preview"])
    place = world.Place(**preview.data["args"]["place"])
    options = apply.Options(**{k: v for k, v in data["options"].items() if k != "config"})
    gens = generators(preview, args)
    ctx = apply.Ctx(place, preview, job, options, lambda: scan_world(place, args), data["options"]["config"], _err, gens)
    try:
        with generator.lock(gens.cache) if gens else nullcontext():
            rec = jobs.run(job, lambda j: apply.run(ctx))
    finally:
        if gens:
            gens.stop()
    print(json.dumps(rec, indent=1) if args.json else "\n".join(after_lines(rec, place)))
    return 0


def _ready(job, preview, place):
    """The preview must be ready until its apply job has written configs or moved folders; from then on its journal decides."""
    if not apply.is_open(job):
        apply.ready(preview, place)


def resume_apply(args, job, scan_world):
    preview = jobs.find(job.root, job.data["args"]["preview"])
    _ready(job, preview, world.Place(**preview.data["args"]["place"]))
    return _run_apply(args, job, scan_world)


def after_lines(rec, place):
    backup = apply.resolve(rec["backup"], place.root)
    lines = [
        f"Applied: {place.world} now uses RWG2 Smooth; {rec['core_identical']:,} of {rec['core_present']:,} protected chunks "
        "are identical.",
        f"Your world from before Apply is kept in {backup}" + (f" (also copied to {rec['backup_copy']})" if rec.get("backup_copy") else ""),
        *([SAVED_SINCE.format(files=jobs.count(rec["saved_since_swap"], "file"))] if rec.get("saved_since_swap") else []),
        "",
        *_offset_rule_lines(rec, place),
        *(f"Rule: {r}." for r in rec["rules"][1:]),
        "",
    ]
    if rec.get("journeymap"):
        lines.append(f"The JourneyMap overworld map was renamed to {apply.resolve(rec['journeymap'], place.root)} (waypoints are kept).")
    else:
        lines.append(apply.MAP_TIP)
    if rec.get("upload"):
        steps, zip_path = rec["upload"]
        lines += ["", *steps] + ([f"Upload zip: {zip_path}"] if zip_path else [])
    return lines


def _offset_rule_lines(rec, place):
    """The first rule with the 4 lines; when Apply had no config folder, first the warning, where the lines go and their file."""
    if not rec.get("by_hand"):
        written = ", ".join(apply.resolve(path, place.root) for path in sorted(rec["configs"]))
        return [
            *([f"Config files written: {written}."] if written else []),
            f"First rule: {rec['rules'][0]}. The 4 lines:",
            *(f"    {line}" for line in rec["offset_lines"]),
        ]
    return [
        f"Warning: {apply.NO_CONFIG}.",
        *apply.by_hand_lines(rec["offsets"], rec["by_hand"]["taint"]),
        f"These lines are saved in {apply.resolve(rec['by_hand']['file'], place.root)}.",
        "",
        f"First rule: {rec['rules'][0]}.",
    ]


def _plain_options(args):
    return apply.Options(yes=args.yes, remote=args.remote, confirm_stopped=args.confirm_stopped, auto_restart_off=args.auto_restart_off)


def rollback_lines(place):
    rec = apply.applied_record(place)
    return [
        f"Roll back {place.world}:",
        "  - the world must be stopped: the tool checks for a running game or server",
        f"  - the world from before Apply comes back from {apply.resolve(rec['backup'], place.root)}",
        f"  - the migrated world is moved aside into {staging.work_dir(place.root)}, not deleted",
        f"  - config keys Apply changed are put back in {', '.join(sorted(rec['configs']))}"
        if rec["configs"]
        else "  - no config to put back: Apply wrote none",
        "  - all play since Apply is lost, in every dimension",
    ]


def _unfinished_rollback(place, remote):
    """A Roll back cut after its first rename left the world without its record, or without its folder: only --yes finishes it."""
    started = apply.Journal(apply.rollback_journal(place)).first("start")
    if started and not apply.record_of(place.world):
        raise Refusal(
            f"a Roll back of {place.world} stopped before it finished",
            f"Finish it: rwg2-migrator rollback {place.world} --yes{_restart_flag(place, remote)}",
        )


def rollback_cmd(args, scan_world):
    place = world.locate(args.world, exists=False)
    data = cli_search._data(args)
    if not args.yes:
        _unfinished_rollback(place, args.remote)
        print("\n".join(rollback_lines(place)))
    try:
        with jobs.lock(data, place.world, "rollback"), jobs.keep_awake():
            out = apply.rollback(place, _plain_options(args), data)
    except apply.GoAhead:
        print(f"{NOTHING_CHANGED}\nNext: rwg2-migrator rollback {place.world} --yes{_restart_flag(place, args.remote)}")
        return 0
    print(json.dumps(out, indent=1) if args.json else rolled_back_text(place, out))
    return 0


def rolled_back_text(place, out):
    configs = f"config keys restored in {', '.join(out['configs'])}" if out["configs"] else "no config restored"
    if not out["configs"] and not out["configs_by_hand"]:
        configs = "no config to restore (Apply wrote none)"
    gone = "".join(
        f"\nNot restored, its folder is gone: {path}. If a server or game still uses that file, set these lines back there:\n"
        + "\n".join(f"    {line}" for line in lines)
        for path, lines in out["configs_by_hand"].items()
    )
    kept = f"the migrated world is kept in {out['rolled_back']}"
    return f"Rolled back: {place.world} is the world from before Apply again; {kept}; {configs}{gone}"


def _since(place, r, data):
    """The post-check of the chunks generated since Apply, from the preview job's render; None when that job is gone."""
    rec = r["record"]
    preview = jobs.Job(data, rec.get("preview_job", ""))
    harvest = jobs.read_json(os.path.join(preview.folder, genworld.HARVEST, staging.RECORD))
    if not harvest or not os.path.isdir(os.path.join(preview.folder, genworld.RENDER)):
        return None
    a = jobs.read_json(os.path.join(preview.folder, "area.json"))
    leftovers = {f["file"] for f in r["leftovers"]}
    return postcheck.since(place.world, a, (harvest["chunks"], genworld.Render(preview)), (rec["applied_at"], leftovers))


def verify_cmd(args, scan_world):
    place = world.locate(args.world)
    r = apply.verify(place, args.restore_offsets, args.config_dir and world.config_dir(args.config_dir))
    r["since"] = _since(place, r, cli_search._data(args))
    if args.json:
        print(json.dumps({k: v for k, v in r.items() if k != "record"}, indent=1))
    else:
        print("\n".join(verify_lines(r)))
    left = set(r["wrong"]) - set(r["restored"])
    return EXIT_PROBLEMS if left or r["leftovers"] or postcheck.mismatch(r["since"]) else 0


def _too_few(since):
    return bool(since) and since["new_chunks"] > 0 and since["compared_chunks"] < postcheck.CLUSTER


def since_lines(since):
    if since is None:
        return ["New chunks: not checked (the preview job is no longer in the data folder, or it did not generate the new terrain)"]
    nt, kb = since["new_terrain"], since["seams"]["kept_border"]
    lines = [f"Chunks generated since Apply: {since['new_chunks']:,}"]
    if nt.get("columns"):
        lines.append(
            f"  Game vs the preview there: median {nt['median']:g}, {nt['within1_pct']:g}% within 1 block ({nt['columns']:,} columns; "
            f"{nt['outside_render']:,} more columns lie beyond the preview's map and are not compared); "
            f"{since['off_chunks']:,} of {since['compared_chunks']:,} chunks are off by a block or more"
        )
    if _too_few(since):
        lines.append(
            f"  Too few to judge yet: {since['compared_chunks']:,} of them lie on the preview's map (it reaches 4 chunks past the "
            f"generated terrain) and {postcheck.CLUSTER} are needed; run verify again once players have explored there"
        )
    if postcheck.mismatch(since):
        x, z = since["off_at"]
        lines.append(
            f"  {since['off_cluster']:,} touching chunks around x {x}, z {z} do not follow the preview. They were generated while the 4 "
            "offset lines were missing or different (a GTNH update replaces config/), or by another RWG2 version. They keep that "
            "terrain; look at them in game. The lines decide only the chunks generated from now on, so keep them as the record says"
        )
    lines.append(f"Seam at the kept border now: {_seam(kb)} ({kb.get('pairs', 0):,} pairs)")
    return lines


def verify_lines(r):
    if r["config"] is None:
        lines = [
            "Config: none of its own found next to the world, so the 4 offset lines were not checked (pass --config-dir to check them).",
            *(f"  {line}" for line in r["by_hand"]),
        ]
    else:
        lines = [f"Config {r['config']}:"]
    if r["config"] and not r["wrong"]:
        lines.append("  the 4 offset lines match the migration record")
    for key, (want, got) in r["wrong"].items():
        done = " (restored)" if key in r["restored"] else ""
        lines.append(f"  {key} is {got if got is not None else 'missing'}, the migration wrote {want}{done}")
    if r["wrong"] and not r["restored"]:
        lines.append("  Fix: run verify again with --restore-offsets, or set these lines in its 'expert / worldgen migration' section:")
        lines += [f"    {line}" for line in r["lines"]]
    for f in r["leftovers"]:
        lines.append(
            f"Leftover: region/{f['file']} holds {jobs.count(f['old_chunks'], 'chunk')} saved before Apply; the old terrain came back "
            "with an upload that merged into the old folder. Delete the world folder on the host and upload the migrated one again."
        )
    lines += since_lines(r.get("since"))
    if not r["wrong"] and not r["leftovers"] and not postcheck.mismatch(r.get("since")):
        found = "No problem found" if r["config"] else "No other problem found"
        lines.append(f"{found} so far (the new chunks are too few to judge)." if _too_few(r.get("since")) else f"{found}.")
    return lines


RESUME = {"preview": resume_preview, "apply": resume_apply}


def _stop_flags(p):
    p.add_argument("--remote", action="store_true", help="this world was copied from another machine; you confirm its server is stopped")
    p.add_argument("--confirm-stopped", action="store_true", help="I checked that the game or server using this world is stopped")
    p.add_argument("--auto-restart-off", action="store_true", help="I turned off anything that restarts my server by itself")
    p.add_argument("--yes", action="store_true", help="go ahead (without it, the list of what happens is printed)")
    p.add_argument("--data", help="jobs folder (default: the per-user data folder)")


def add_parsers(sub, common):
    p = sub.add_parser("preview", parents=[common], help="build the migrated world in a working copy next to the world")
    p.add_argument("world")
    p.add_argument("--area", required=True, metavar="AREA.json")
    p.add_argument("--search", required=True, metavar="JOB", help="a search job id, or latest")
    p.add_argument("--pick", type=int, default=1, help="candidate rank (default 1)")
    p.add_argument("--no-generate", action="store_true", help="transplant only; your server generates the new terrain")
    p.add_argument("--no-smooth", action="store_true", help="keep the step at the kept border as the generator leaves it")
    p.add_argument(
        "--trees",
        action="store_true",
        help="replant trees on the reshaped strips (each biome grows its own; needs a third start of the generator)",
    )
    p.add_argument("--blend-biomes", action="store_true", help="blend the margin's biomes into the new terrain's (no straight colour line)")
    p.add_argument(
        "--ring", type=int, default=genworld.RING, help=f"chunks of new terrain to generate around the kept area (default {genworld.RING})"
    )
    p.add_argument("--set-populated", action="store_true", help="mark never-decorated protected chunks decorated (1 byte each)")
    p.add_argument("--config-dir", help=cli_search.CONFIG_HELP)
    p.add_argument("--threads", type=int, help="generator threads for the render (default: cores - 2)")
    p.add_argument(
        "--accept-eula", action="store_true", help="I accept the Minecraft EULA (https://aka.ms/MinecraftEULA); sets up the generator"
    )
    p.add_argument("--data")
    a = sub.add_parser("apply", parents=[common], help="swap the ready preview in; your world is kept as the backup")
    a.add_argument("world")
    a.add_argument("--preview", required=True, metavar="JOB")
    a.add_argument("--config-dir", help=cli_search.CONFIG_HELP)
    a.add_argument("--zip", action="store_true", help="with --remote: also write an upload zip")
    a.add_argument("--backup-copy", metavar="DIR", help="also copy the world from before Apply into DIR (another drive)")
    taint = a.add_mutually_exclusive_group()
    taint.add_argument("--taint-spread-off", dest="taint_spread_off", action="store_const", const=True, help="set biome_taint_spread=0")
    taint.add_argument("--keep-taint-spread", dest="taint_spread_off", action="store_const", const=False)
    a.add_argument("--leave-players", action="store_true", help="do not move players standing outside the kept area")
    a.add_argument("--allow-shared-instance", action="store_true", help="other RWG2 worlds on the same config folder may shift")
    a.add_argument("--reset-journeymap", action="store_true", help="single player: rename the JourneyMap overworld map (game closed)")
    _stop_flags(a)
    r = sub.add_parser("rollback", parents=[common], help="put the world from before Apply back")
    r.add_argument("world")
    _stop_flags(r)
    v = sub.add_parser("verify", parents=[common], help="check the offset lines and flag leftover region files (read-only by default)")
    v.add_argument("world")
    v.add_argument("--config-dir", help=cli_search.CONFIG_HELP)
    v.add_argument("--restore-offsets", action="store_true", help="write the 4 offset lines back into config/rwg2.cfg")
    v.add_argument("--data")
    return {"preview": preview_cmd, "apply": apply_cmd, "rollback": rollback_cmd, "verify": verify_cmd}
