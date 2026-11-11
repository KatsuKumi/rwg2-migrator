"""Apply a ready preview to its world, roll it back, and verify it later."""

import json
import os
import shutil
import struct
import time
import zipfile
from dataclasses import dataclass, replace

from . import Refusal, __version__, forgecfg, generator, genworld, jobs, nbt, plant, procs, staging, transplant, world
from .jobs import read_json
from .region import copy_tree, fsync_dir, remove_file, rmtree, slots, write_atomic

TOOL = "rwg2-migrator"
MARKER = "rwg2-migration.json"
SESSION_LOCK = "session.lock"
JOURNAL = "journal.jsonl"
RWG2_CFG = world.RWG2_CFG
TAINT_CFG, TAINT_KEY, TAINT_OFF = "Thaumcraft.cfg", ("general", "biome_taint_spread"), ("I", "0")
TAINT_NEAR = 2000
WINDOWS = os.name == "nt"
WINDOWS_PATH_MAX = 259
SHOWN = 3
MB = 1 << 20
HEADROOM = 512 * MB
RENAME_TRIES = 6
RENAME_WAIT = 0.25
LOCKED = (5, 32)
STAMP = "%Y%m%d-%H%M%S"
MS = 1000
STAGED, APPLIED = "staged", "applied"
OURS, OTHER = "ours", "other"
NOT_SWAPPED, BETWEEN, SWAPPED, RECREATED, UNKNOWN = "not swapped", "between the renames", "swapped", "recreated", "unknown"
SWAP, ROLLBACK = "swap", "rollback"
TRIAL = ".rwg2-trial"
VP = "visualprospecting"
JM_DIM0, JM_LEGACY = "DIM0", "DIM0-legacy-rwg"
CONFIGS_OUT, ENDED = ("write_configs", "rename1", "rename2"), ("verified", "reversed", "rolled_back")
SIDE_FILES = {"source": "-source.json", "journal": "-journal.jsonl", "visualprospecting": "-visualprospecting"}
RULES = (
    "After every GTNH update, run 'rwg2-migrator verify' on this world, or set the 4 offset lines again in the "
    f"'{generator.RWG2_CATEGORY}' section of config/rwg2.cfg: "
    "without them every new chunk beyond the migrated area uses no offset and cliffs appear about 8 chunks out",
    "Never change those 4 lines while this world exists",
    "Never run /rwg_noise or /rwg_biomelists",
    "Keep RWG-alpha (legacy RWG) and config/RWG.cfg installed: the kept chunks store its biome ids",
)
NO_CONFIG = "no config folder of its own was found, so Apply wrote no config file"
BY_HAND = "-rwg2-offsets.txt"
TAINT_LINE = "I:biome_taint_spread=0"
MAP_TIP = (
    "Your JourneyMap still shows the old terrain outside your base: rename the overworld map folder (DIM0) of this world so it "
    "redraws; waypoints are kept. VisualProspecting veins stay valid."
)


@dataclass
class Options:
    yes: bool = False
    remote: bool = False
    zip: bool = False
    backup_copy: str | None = None
    taint_off: bool | None = None
    leave_players: bool = False
    allow_shared_instance: bool = False
    confirm_stopped: bool = False
    auto_restart_off: bool = False
    reset_journeymap: bool = False


@dataclass
class Ctx:
    place: world.Place
    preview: jobs.Job
    job: jobs.Job
    opts: Options
    scan: object
    config: str | None
    say: object = print
    gens: object = None


class GoAhead(Refusal):
    """Every check passed and only --yes is missing; nothing was written."""


class Journal:
    """Append-only phases; each one is fsynced before the action it names."""

    def __init__(self, path):
        self.path = path

    def entries(self):
        return jobs.read_lines(self.path)

    def note(self, phase, **data):
        jobs.append_line(self.path, {"phase": phase, "t": round(time.time(), 3)} | data)

    def first(self, phase):
        return next((e for e in self.entries() if e["phase"] == phase), None)

    def latest(self, phase):
        return next((e for e in reversed(self.entries()) if e["phase"] == phase), None)

    def last(self, *phases):
        return next((e["phase"] for e in reversed(self.entries()) if e["phase"] in phases), None)


def stamp():
    return time.strftime(STAMP)


def free_bytes(path):
    return shutil.disk_usage(path).free


def rel_to(path, root):
    """path relative to root when it lies inside root, else absolute: the record survives a moved server folder."""
    path, root = os.path.abspath(path), os.path.abspath(root)
    try:
        inside = os.path.commonpath([path, root]) == root
    except ValueError:
        inside = False
    return os.path.relpath(path, root) if inside else path


def resolve(path, root):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))


def free(dst):
    if os.path.lexists(dst):
        raise Refusal(f"{dst} already exists", "Move it out of the way (the tool never replaces a folder), then run this again")


def rename(src, dst):
    """os.rename, never onto an existing path; on Windows a file held open by Defender or Explorer is retried with backoff.
    A Refusal means nothing was renamed."""
    free(dst)
    wait = RENAME_WAIT
    for attempt in range(RENAME_TRIES):
        try:
            os.rename(src, dst)
            break
        except OSError as e:
            if getattr(e, "winerror", None) not in LOCKED or attempt == RENAME_TRIES - 1:
                raise Refusal(
                    f"{src} cannot be renamed to {dst} ({e})",
                    f"Close any window showing files in {src} (Explorer, an editor, OneDrive), then run this again",
                ) from e
            time.sleep(wait)
            wait *= 2
    fsync_dir(os.path.dirname(os.path.abspath(dst)))


def record_of(world_dir):
    rec = read_json(os.path.join(world_dir, MARKER)) if os.path.isdir(world_dir) else None
    return rec if isinstance(rec, dict) and rec.get("tool") == TOOL else None


def swap_state(world_dir, backup, stage, apply_id):
    """The state of an Apply, from what exists; anything else is UNKNOWN and is never guessed at. backup is None while
    the journal says Apply has not moved the world there, so a folder found at that path never counts as the backup."""
    moved = backup is not None and os.path.isdir(backup)
    key = (os.path.isdir(world_dir), _mark(world_dir, apply_id), moved, os.path.isdir(stage))
    table = {
        (True, None, False, True): NOT_SWAPPED,
        (False, None, True, True): BETWEEN,
        (True, OURS, True, False): SWAPPED,
        (True, None, True, True): RECREATED,
    }
    return table.get(key, UNKNOWN)


def _mark(world_dir, apply_id):
    """None for a world without a record of this tool, OURS for this apply job's record, OTHER for another one."""
    rec = record_of(world_dir)
    return None if rec is None else OURS if rec.get("apply_job") == apply_id else OTHER


def moved(j, backup):
    """backup when the journal says Apply moved the world there and has not undone it, else None."""
    return backup if j.last("rename1", "unmoved", "reversed") == "rename1" else None


def is_open(job):
    """An apply job that wrote configs or renamed folders and has neither finished, reversed nor been rolled back: only its own
    journal knows the old config values and where the world went."""
    d = job.data
    if d.get("kind") != "apply" or d.get("state") == jobs.DONE:
        return False
    return Journal(os.path.join(job.folder, JOURNAL)).last(*CONFIGS_OUT, *ENDED) not in (None, *ENDED)


def unfinished(data_root, preview_id):
    return next((j for j in jobs.all_jobs(data_root) if j.data.get("args", {}).get("preview") == preview_id and is_open(j)), None)


def needed_by(data_root, job):
    """The open apply job that deleting job would strand: job itself, or the apply job of this preview."""
    if is_open(job):
        return job
    return unfinished(data_root, job.id) if job.data.get("kind") == "preview" else None


def _previews(data_root, world_dir, keep=None):
    same = os.path.normcase(os.path.realpath(world_dir))
    for j in jobs.all_jobs(data_root):
        d = j.data
        if j.id != keep and d.get("kind") == "preview" and os.path.normcase(os.path.realpath(d.get("world", ""))) == same:
            yield j


def refuse_unfinished(data_root, world_dir):
    """A new preview must wait for an Apply of this world that wrote configs or renamed folders."""
    for j in _previews(data_root, world_dir):
        busy = unfinished(data_root, j.id)
        if busy:
            raise Refusal(
                f"apply job {busy.id} of preview {j.id} is not finished",
                f"Resume it first: rwg2-migrator jobs resume {busy.id}. It either finishes Apply or puts your config back, "
                "and says what to do next",
            )


def supersede(data_root, world_dir, keep):
    """Mark the world's older previews superseded and delete their working copies."""
    refuse_unfinished(data_root, world_dir)
    for j in _previews(data_root, world_dir, keep):
        if j.data.get("state") != jobs.SUPERSEDED and not j.data.get("applied"):
            drop_working_copy(j)
            j.update(state=jobs.SUPERSEDED)


def drop_working_copy(job):
    """Delete a preview's working copy and its generated world; an applied preview has none left: its copy became the world."""
    stage = job.data.get("args", {}).get("staging")
    if job.data.get("kind") == "preview" and stage and not job.data.get("applied"):
        rmtree(os.path.dirname(stage))
        genworld.discard(job)


def ready(preview, place):
    d = preview.data
    if d.get("kind") != "preview":
        raise Refusal(f"job {preview.id} is a {d.get('kind')} job, not a preview", "Pass the id of a preview job: rwg2-migrator jobs list")
    if os.path.normcase(os.path.realpath(d["world"])) != os.path.normcase(os.path.realpath(place.world)):
        raise Refusal(f"preview {preview.id} was made for {d['world']}, not {place.world}", "Apply a preview of this world")
    if d.get("state") == jobs.SUPERSEDED:
        raise Refusal(
            f"preview {preview.id} was replaced by a newer preview of this world", "Apply the newest preview (rwg2-migrator jobs list)"
        )
    if d.get("applied") and jobs.Job(preview.root, d["applied"]).data.get("rolled_back"):
        a = d["args"]
        raise Refusal(
            f"preview {preview.id} was applied, and that Apply was rolled back: its working copy is the migrated world set aside then",
            f"Build a new preview of the same match: rwg2-migrator preview {place.world} --area AREA.json --search {a['search']} "
            f"--pick {a['pick']}",
        )
    if d.get("applied"):
        raise Refusal(f"preview {preview.id} is already applied", "There is nothing left to do; run rwg2-migrator verify on the world")
    if d.get("state") != jobs.DONE or not os.path.isdir(staging.paths(preview)[1]):
        raise Refusal(
            f"preview {preview.id} is not ready to apply ({d.get('state')})", f"Finish it first: rwg2-migrator jobs resume {preview.id}"
        )


def apply_job(data_root, preview, options):
    """The unfinished apply job of this preview, or a new one. A job that wrote configs or moved folders keeps its config folder."""
    for j in jobs.all_jobs(data_root):
        d = j.data
        if d.get("kind") == "apply" and d.get("args", {}).get("preview") == preview.id and d.get("state") != jobs.DONE:
            if is_open(j):
                _keeps_config(j, options["config"])
            j.update(args=d["args"] | {"options": options})
            return j
    return jobs.create(data_root, "apply", {"preview": preview.id, "options": options}, preview.data["world"])


def _keeps_config(job, config):
    then = job.data["args"]["options"].get("config")
    if then == config or (then and config and world.same(then, config)):
        return
    raise Refusal(
        f"apply job {job.id} is part way through and started {_with(then)}; it must finish that way, not {_with(config)}",
        f"Finish it as it started: rwg2-migrator jobs resume {job.id}. Afterwards, to set the 4 offset lines in the config folder you "
        "want, run rwg2-migrator verify on the world with it",
    )


def _with(config):
    return f"with the config folder {config}" if config else "without a config folder"


def other_rwg2_saves(place, config):
    """Names of the other worlds that already use RWG2 and run on the config folder Apply writes into: its rwg2.cfg is shared by
    all of them (the saves of a game instance, the world of a server)."""
    out = []
    for other in world.worlds_on(config) if config else []:
        if world.same(other, place.world):
            continue
        try:
            name = nbt.read_file(os.path.join(other, world.LEVEL))["Data"].get("generatorName")
        except (OSError, nbt.NBTError, KeyError):
            continue
        if name in world.MIGRATED:
            out.append(os.path.basename(other))
    return sorted(out)


def _shared_refusal(place, config, shared):
    action = (
        "Copy the instance first (Prism: right-click > Copy) and migrate the save there"
        if place.kind == "singleplayer"
        else "Move this world into a folder of its own and search and preview it there: Apply then writes no config file and shows "
        "the 4 offset lines to add by hand"
    )
    return Refusal(
        f"other worlds that run on the config folder {config} already use RWG2 ({', '.join(shared)}); its rwg2.cfg is shared, so the "
        "offsets would shift their unexplored terrain",
        f"{action}, or run again with --allow-shared-instance if only their unexplored terrain may shift",
    )


def _paths(ctx, plan):
    stage = staging.paths(ctx.preview)[1]
    return ctx.place.world, stage, stage + TRIAL, plan["backup"]


def _plan(ctx, j):
    """The noted plan, or one named after this run: the first run that passes the checks notes it (_note_plan), so the backup
    carries the time of the Apply that made it, not of an earlier refused one."""
    p = j.first("plan")
    if p is None:
        name = os.path.basename(ctx.place.world)
        return {"backup": os.path.join(staging.work_dir(ctx.place.root), f"{name}-pre-rwg2-{stamp()}")}
    return p


def _note_plan(j, plan):
    if j.first("plan") is None:
        j.note("plan", backup=plan["backup"])


def _usage(ctx):
    p = ctx.place
    usage = procs.remote() if ctx.opts.remote else procs.in_use(p.root, p.world)
    procs.require_stopped(usage, ctx.opts.confirm_stopped)


def taint_off(preview, opts):
    """Turn off taint spreading when asked, and by default when the candidate's Tainted Land is less than 2 km away."""
    if opts.taint_off is not None:
        return opts.taint_off
    taint = preview.data["args"].get("taint")
    return bool(taint) and taint["blocks"] < TAINT_NEAR


def _destinations(ctx, j, plan):
    ours = {"visualprospecting"} if j.first("vp_copy") else set()
    out = [plan["backup"], *(plan["backup"] + s for k, s in SIDE_FILES.items() if k not in ours)]
    if ctx.opts.backup_copy and not j.first("backup_copy"):
        out.append(os.path.join(ctx.opts.backup_copy, os.path.basename(plan["backup"])))
    jm = _journeymap(ctx)
    if jm:
        out.append(os.path.join(os.path.dirname(jm), JM_LEGACY))
    return out


def _backup_copy_target(ctx, j, plan):
    if not ctx.opts.backup_copy or j.first("backup_copy"):
        return
    dest, work = os.path.join(ctx.opts.backup_copy, os.path.basename(plan["backup"])), staging.work_dir(ctx.place.root)
    if procs.inside(dest, ctx.place.world) or procs.inside(dest, work):
        raise Refusal(
            f"the folder for the extra backup copy, {ctx.opts.backup_copy}, is inside the world or inside {work}, where Apply "
            "itself keeps the world from before Apply",
            "Pass a folder on another drive to --backup-copy",
        )


def _journeymap(ctx):
    if not ctx.opts.reset_journeymap or ctx.place.kind != "singleplayer":
        return None
    return os.path.join(ctx.place.root, "journeymap", "data", "sp", os.path.basename(ctx.place.world), JM_DIM0)


def _bytes(folder):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(folder) for f in fs)


def _disk(ctx, j):
    m = staging.source(ctx.preview)
    need = sum(size for _, size in staging.pending(m, ctx.place.world)) + _bytes(os.path.join(ctx.place.root, VP)) + HEADROOM
    free = free_bytes(ctx.place.root)
    if free < need:
        raise Refusal(
            f"Apply needs {need / MB:,.0f} MB free on the drive of {ctx.place.root} to copy what changed since the preview; "
            f"{free / MB:,.0f} MB is free",
            "Free some space on that drive, then run apply again",
        )
    dest = ctx.opts.backup_copy
    if dest and not j.first("backup_copied"):
        if not os.path.isdir(dest):
            raise Refusal(f"the backup folder {dest} does not exist", "Create it, or pass another folder to --backup-copy")
        copy = sum(e["size"] for e in m["files"].values())
        need = copy + HEADROOM + (need if staging.device(dest) == staging.device(ctx.place.root) else 0)
        if free_bytes(dest) < need:
            raise Refusal(f"the backup copy needs {need / MB:,.0f} MB free in {dest}", "Free some space there, or pick another drive")


def _long_paths(ctx, plan):
    if not WINDOWS:
        return
    files = staging.source(ctx.preview)["files"]
    w, _, trial, backup = _paths(ctx, plan)
    longest = max((staging.at(f, rel) for f in (w, trial, backup) for rel in files), key=_windows_len, default="")
    if _windows_len(longest) > WINDOWS_PATH_MAX:
        raise Refusal(
            f"a path would be {_windows_len(longest)} characters long, more than Windows allows ({longest})",
            "Move the server or instance folder closer to the drive root (for example D:\\server), then run preview and apply again",
        )


def _windows_len(path):
    """The length Windows counts: UTF-16 units, so a character outside the basic plane counts twice."""
    return len(path.encode("utf-16-le")) // 2


def _same_drive(ctx):
    stage = staging.paths(ctx.preview)[1]
    if staging.device(stage) != staging.device(ctx.place.world):
        raise Refusal(
            f"the working copy {stage} is on another drive than {ctx.place.world}", "Run preview again; it keeps its copy next to the world"
        )


def preflight(ctx, j, plan):
    """Every check that refuses Apply before anything is written; each refusal names the next action."""
    p, o = ctx.place, ctx.opts
    _usage(ctx)
    if p.kind == "server" and not o.remote and not o.auto_restart_off:
        raise Refusal(
            "a server that restarts by itself (a panel's auto-restart, a watchdog script, systemd) could open the world while Apply runs",
            "Turn that off until Apply finishes, then run apply again with --auto-restart-off",
        )
    shared = other_rwg2_saves(p, ctx.config)
    if shared and not o.allow_shared_instance:
        raise _shared_refusal(p, ctx.config, shared)
    if ctx.config and not os.path.isdir(ctx.config):
        raise Refusal(
            f"the config folder {ctx.config} does not exist, and Apply writes the 4 offset lines into its rwg2.cfg",
            "Copy the server's (or instance's) config folder there, or run apply with --config-dir pointing at it",
        )
    drift(ctx)
    _same_drive(ctx)
    for dest in _destinations(ctx, j, plan):
        free(dest)
    _backup_copy_target(ctx, j, plan)
    _disk(ctx, j)
    _long_paths(ctx, plan)
    if not o.yes:
        raise GoAhead("Apply needs your go-ahead", "Read the list above, then run apply again with --yes")
    return shared


def check(ctx):
    """Apply's checks, writing nothing (apply without --yes); GoAhead when only --yes is missing."""
    j = Journal(os.path.join(ctx.job.folder, JOURNAL))
    preflight(ctx, j, _plan(ctx, j))


def _differences(old, new, versions=False):
    """The first few differences between two {name: value} maps, in words."""
    out = [f"{k} added" for k in sorted(set(new) - set(old))] + [f"{k} removed" for k in sorted(set(old) - set(new))]
    out += [f"{k} {old[k]} -> {new[k]}" if versions else f"{k} changed" for k in sorted(set(old) & set(new)) if old[k] != new[k]]
    return jobs.listed(out, SHOWN)


def _generator_mods(level):
    """The GTNH release and the RWG2 version a world was last opened with: they decide the terrain the search matched."""
    mods = world.mods(level)
    return {"GTNH": world.release_for(mods, world.releases())[0], "rwg2": mods.get("rwg2")}


def search_again(place, area_file):
    """The search command of place's world on its config folder."""
    flag = f" --config-dir {place.config}" if place.config and world.holds_rwg2(place.config) else ""
    return f"rwg2-migrator search {place.world} --area {area_file}{flag}"


def drift(ctx):
    """Refuse when the world, its mods or its config are no longer what the search and the preview were made for: Apply would
    make offsets permanent that were matched for other generator inputs."""
    w, p = ctx.place.world, ctx.preview
    level, then = world.read_level(w), world.read_level(staging.paths(p)[1])
    again = f"search and preview again: {search_again(replace(ctx.place, config=ctx.config), os.path.join(p.folder, 'area.json'))}"
    seed, want = str(level["Data"]["RandomSeed"]), p.data["args"]["seed"]
    if seed != want:
        raise Refusal(
            f"{w} has the seed {seed}, but preview {p.id} was made for a world with the seed {want}",
            f"Put the previewed world back, or {again}",
        )
    changed = _differences(_generator_mods(then), _generator_mods(level), versions=True)
    if changed:
        raise Refusal(
            f"the world was opened with another GTNH or RWG2 version since preview {p.id} ({changed})",
            f"Its matches were found for the old one: {again}",
        )
    keys = read_json(os.path.join(p.folder, staging.SNAPSHOT, staging.SNAPSHOT_FILES[2]))
    changed = _differences(dict(keys["mods"]), dict(map(tuple, staging.mods_listing(ctx.place.root))))
    if changed:
        raise Refusal(
            f"the mods folder changed since preview {p.id} ({changed})",
            "Put the mods back as they were, or open the world once with these mods, close it, then run the preview again",
        )
    changed = jobs.config_change(ctx.config, keys["config"], f"preview {p.id}")
    if changed:
        files = _differences(keys.get("config_files") or {}, jobs.tree_hashes(ctx.config) or {}) or "its settings"
        raise Refusal(
            f"{changed} ({files}), so its offsets may no longer match what RWG2 makes", f"Put those settings back as they were, or {again}"
        )


def _config_plan(ctx):
    if not ctx.config:
        return {}
    edits = {RWG2_CFG: generator.offset_edits(ctx.preview.data["args"]["config"])}
    if taint_off(ctx.preview, ctx.opts):
        edits[TAINT_CFG] = {TAINT_KEY: TAINT_OFF}
    files = {}
    for name, keys in edits.items():
        path = os.path.join(ctx.config, name)
        exists = os.path.exists(path)
        rows = []
        for key, (kind, value) in keys.items():
            old = forgecfg.get(path, key) if exists else None
            if exists and old is None:
                raise Refusal(
                    f"{path} has no {key[-1]} setting in its {key[0]} section",
                    "Start the server (or the game) once with this GTNH version so it writes the full file, stop it, then run apply again",
                )
            rows.append({"key": list(key), "type": kind, "old": old, "new": value})
        files[path] = {"created": not exists, "keys": rows}
    return files


def write_configs(files, which):
    """Write the "old" or the "new" value of every journaled key, file by file, and read them back."""
    for path, f in files.items():
        if which == "old" and f["created"]:
            remove_file(path)
            continue
        forgecfg.edit(path, {tuple(row["key"]): (row["type"], row[which]) for row in f["keys"]})
        wrong = [row["key"][-1] for row in f["keys"] if forgecfg.get(path, row["key"]) != row[which]]
        if wrong:
            raise Refusal(
                f"{path} does not read back the values just written ({', '.join(wrong)})",
                "Check that nothing else edits it, then run again",
            )


def _configs(ctx, j):
    """Each attempt plans its own edits on the config folder it checked; an earlier attempt's edits were undone before this one."""
    j.note("configs", files=_config_plan(ctx))
    j.note("write_configs")
    write_configs(j.latest("configs")["files"], "new")


def _backup_copy(ctx, j, plan):
    dest_dir = ctx.opts.backup_copy
    if not dest_dir:
        return
    dest = os.path.join(dest_dir, os.path.basename(plan["backup"]))
    j.note("backup_copy", dest=dest)
    rmtree(dest)
    copy_tree(ctx.place.world, dest)
    files = {rel: e["sha256"] for rel, e in staging.source(ctx.preview)["files"].items()}
    bad = staging.mismatches(dest, files)
    if bad:
        raise Refusal(
            f"the backup copy in {dest} differs from your world ({bad[0][0]}: {bad[0][1]})", "Check that drive, then run apply again"
        )
    j.note("backup_copied", dest=dest)


def _before_swap(ctx, j, plan):
    shared = preflight(ctx, j, plan)
    _note_plan(j, plan)
    stage = staging.paths(ctx.preview)[1]
    if not j.first("refresh"):
        bad = staging.mismatches(stage, staging.final(ctx.preview))
        if bad:
            raise Refusal(
                f"the working copy of preview {ctx.preview.id} changed since it was made: {jobs.count(len(bad), 'file')}, "
                f"the first {bad[0][0]} ({bad[0][1]})",
                "Run preview again and apply the new preview",
            )
    j.note("refresh", shared=shared)
    args = ctx.preview.data["args"]["options"]
    opts = transplant.Options(set_populated=args["set_populated"], move_players=not ctx.opts.leave_players)
    t0 = time.monotonic()
    record = staging.build(ctx.preview, staging.Work(ctx.scan(), opts, ctx.gens, ctx.say), refresh=True)
    seconds = round(time.monotonic() - t0, 1)
    bad = staging.mismatches(stage, staging.final(ctx.preview))
    if bad:
        raise Refusal(
            f"after Apply caught up with your world, the working copy is not what it should be ({bad[0][0]}: {bad[0][1]})",
            "Run apply again",
        )
    j.note("refreshed", files=len(record["refreshed"]), seconds=seconds, post=record.get("post"))
    ctx.say(
        f"apply: copied {jobs.count(len(record['refreshed']), 'changed file')} again and rebuilt {_redone(ctx, record)} in {seconds:g} s"
    )
    _backup_copy(ctx, j, plan)
    _configs(ctx, j)
    j.note("configs_written")


def _redone(ctx, record):
    post = record.get("post")
    if not post:
        return "the kept area"
    steps = "the kept area, the new terrain and the smoothing"
    if not post.get("trees"):
        return steps
    if _preview_file(ctx, plant.TREES, staging.RECORD)["replayed"]:
        return f"{steps}, and replayed the trees' merge list"
    return f"{steps}, and planted the trees again (one more start of the generator: the ground where they grow changed)"


def _marker(ctx, plan, state):
    a = ctx.preview.data["args"]
    return {
        "tool": TOOL,
        "version": __version__,
        "state": state,
        "preview_job": ctx.preview.id,
        "apply_job": ctx.job.id,
        "backup": rel_to(plan["backup"], ctx.place.root),
        "mode": "Smooth",
        "generator": transplant.GENERATOR,
        "offsets": dict(zip(generator.OFFSET_KEYS, a["config"], strict=True)),
    }


def _save_vp(ctx, j, plan):
    """VisualProspecting's cache as it was before Apply, copied next to the backup before the world moves: after the swap a game
    may already have written the migrated world's veins into it."""
    vp, saved = os.path.join(ctx.place.root, VP), plan["backup"] + SIDE_FILES["visualprospecting"]
    if not os.path.isdir(vp):
        return
    j.note("vp_copy")
    for path in (saved, saved + ".partial"):
        rmtree(path)
    copy_tree(vp, saved + ".partial")
    rename(saved + ".partial", saved)


def _swap(ctx, j, plan):
    w, stage, trial, backup = _paths(ctx, plan)
    _save_vp(ctx, j, plan)
    _usage(ctx)
    late = staging.pending(staging.source(ctx.preview), w)
    if late:
        raise Refusal(
            f"the world changed while Apply was running ({jobs.count(len(late), 'file')}, the first {late[0][0]})",
            "Make sure nothing runs the world, then run apply again",
        )
    write_atomic(os.path.join(stage, SESSION_LOCK), struct.pack(">q", int(time.time() * MS)))
    write_atomic(os.path.join(stage, MARKER), json.dumps(_marker(ctx, plan, STAGED), indent=1).encode())
    jobs.checkpoint(SWAP)
    j.note("trial")
    rename(stage, trial)
    jobs.checkpoint(SWAP)
    rename(trial, stage)
    jobs.checkpoint(SWAP)
    _move_world(j, w, backup)
    jobs.checkpoint(SWAP)


def _move_world(j, w, backup):
    """rename1, journaled only once its destination is known to be free; a refused rename journals 'unmoved', so a folder found at
    the backup path is never taken for the world."""
    try:
        free(backup)
        j.note("rename1", backup=backup)
        rename(w, backup)
    except Refusal:
        j.note("unmoved")
        raise


def _undo_configs(j):
    """Before the world has moved, put back every config key an earlier attempt or this one wrote, and close that attempt."""
    if j.last(*CONFIGS_OUT, *ENDED) in CONFIGS_OUT:
        _restore(j)
        j.note("reversed")


def _up_to_rename1(ctx, j, plan):
    """Every step before the world moves; whatever stops it (a refusal or another error now, a power cut found by the next run)
    leaves the config keys as they were, so a world the user keeps playing never carries the new offsets."""
    _undo_configs(j)
    try:
        _before_swap(ctx, j, plan)
        _swap(ctx, j, plan)
    except Exception:
        _undo_configs(j)
        raise


def _second(ctx, j, plan):
    w, stage, _, _ = _paths(ctx, plan)
    if os.path.lexists(w):
        _reverse(ctx, j, plan)
    j.note("rename2")
    rename(stage, w)
    jobs.checkpoint(SWAP)


def _restore(j):
    e = j.latest("configs")
    if e:
        write_configs(e["files"], "old")


def _reverse(ctx, j, plan):
    """Put the original world back after something recreated <world> between the renames, and undo the config writes. The
    decision is journaled first, so a later run keeps reversing instead of swapping the migrated world in."""
    w, _, _, backup = _paths(ctx, plan)
    if j.last("rename1", "reverse") != "reverse":
        j.note("reverse")
    if moved(j, backup) and os.path.isdir(backup):
        if os.path.lexists(w):
            usage = procs.in_use(ctx.place.root, None)
            if usage.state != procs.STOPPED:
                _restore(j)
                raise Refusal(
                    f"something created a new {w} after Apply moved your world aside, and it is running ({usage.detail}). "
                    f"Your world is safe in {backup}; the config changes were undone",
                    "Stop that server and turn off its auto-restart, then run apply again: it puts your world back first",
                )
            aside = os.path.join(staging.work_dir(ctx.place.root), f"{os.path.basename(w)}-recreated-{stamp()}")
            j.note("aside", path=aside)
            rename(w, aside)
        rename(backup, w)
    _restore(j)
    j.note("reversed")
    raise Refusal(
        f"something created a new {w} between the two renames of Apply (a server that restarted itself?). Apply put your world "
        f"back as it was and undid the config changes; the folder it found is kept in {staging.work_dir(ctx.place.root)}",
        "Turn off the auto-restart, then run apply again",
    )


def _copy_side(ctx, j, plan):
    backup = plan["backup"]
    source = os.path.join(ctx.preview.folder, staging.COPY, staging.SOURCE)
    for path, suffix in ((source, SIDE_FILES["source"]), (j.path, SIDE_FILES["journal"])):
        with open(path, "rb") as f:
            write_atomic(backup + suffix, f.read())
    saved = backup + SIDE_FILES["visualprospecting"]
    return saved if os.path.isdir(saved) else None


def _reset_journeymap(ctx):
    jm = _journeymap(ctx)
    if not jm:
        return None
    legacy = os.path.join(os.path.dirname(jm), JM_LEGACY)
    if os.path.isdir(jm) and not os.path.lexists(legacy):
        rename(jm, legacy)
    return legacy if os.path.isdir(legacy) else None


def offset_lines(offsets):
    return [f'I:"{k}"={v}' for k, v in offsets.items()]


def by_hand_lines(offsets, taint=False):
    """Where the offset lines go when the tool has no config folder to write them into, and the lines themselves."""
    lines = [
        "Set these 4 lines in config/rwg2.cfg of the server or game instance that will run this world, before it starts,",
        f"in its '{generator.RWG2_CATEGORY}' section, in place of the lines with the same names:",
        *(f"    {line}" for line in offset_lines(offsets)),
    ]
    if taint:
        lines += ["And in config/Thaumcraft.cfg, in its 'general' section, to stop Tainted Land spreading:", f"    {TAINT_LINE}"]
    return lines


def _by_hand(ctx, offsets):
    """The offset lines in a text file next to the world, for an Apply that had no config folder to write them into."""
    if ctx.config:
        return None
    taint = taint_off(ctx.preview, ctx.opts)
    path = os.path.join(ctx.place.root, os.path.basename(ctx.place.world) + BY_HAND)
    write_atomic(path, ("\n".join([f"Warning: {NO_CONFIG}.", *by_hand_lines(offsets, taint)]) + "\n").encode())
    return {"file": rel_to(path, ctx.place.root), "taint": taint}


def _preview_file(ctx, *parts):
    return read_json(os.path.join(ctx.preview.folder, *parts))


def checklist(ctx, rec, zip_path):
    name = os.path.basename(ctx.place.world)
    cfgs = " and ".join(f"config/{os.path.basename(p)}" for p in rec["configs"])
    by_hand = rec["by_hand"] and resolve(rec["by_hand"]["file"], ctx.place.root)
    before = [rel for rel in staging.source(ctx.preview)["files"] if staging.dim0_region(rel)]
    deleted = len(before) - len(rec["regions"])
    return [
        "Upload checklist (the server on the host must stay stopped until step 5):",
        "1. Check that the server on the host is stopped.",
        f"2. Delete the old {name} folder on the host first: the migration removed {jobs.count(deleted, 'overworld region file')}, uploads "
        "merge by default, and leftover files would bring the old terrain back next to the new.",
        f"3. Upload the folder {ctx.place.world} as {name}/"
        + (f", or upload {zip_path} and extract it in the server folder." if zip_path else "."),
        f"4. Upload {cfgs}." if cfgs else f"4. On the host, set the lines saved in {by_hand} in its config folder, as that file says.",
        f"5. Start the server. Later, download {name} and config again and run: rwg2-migrator verify <the downloaded world>",
    ]


def _upload(ctx, rec):
    work, name = staging.work_dir(ctx.place.root), os.path.basename(ctx.place.world)
    zip_path = os.path.join(work, f"{name}-upload.zip") if ctx.opts.zip else None
    lines = checklist(ctx, rec, zip_path)
    write_atomic(os.path.join(work, f"{name}-upload.txt"), ("\n".join(lines) + "\n").encode())
    return lines, zip_path


def _zip(ctx, rec, zip_path):
    """Built from the world only once its record is complete: verify on the host needs the applied record, not the staged marker."""
    name, tmp = os.path.basename(ctx.place.world), zip_path + ".partial"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for rel in staging.walk(ctx.place.world)[1]:
            z.write(staging.at(ctx.place.world, rel), f"{name}/{rel}")
        for path in rec["configs"]:
            z.write(resolve(path, ctx.place.root), f"config/{os.path.basename(path)}")
    os.replace(tmp, zip_path)


def _record(ctx, j, plan):
    w = ctx.place.world
    root = ctx.place.root
    configs = j.latest("configs")["files"]
    a = _preview_file(ctx, "area.json")
    offsets = _marker(ctx, plan, APPLIED)["offsets"]
    rec = _marker(ctx, plan, APPLIED) | {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "applied_at": int(j.first("rename1")["t"]),
        "config_dir": ctx.config and rel_to(ctx.config, root),
        "by_hand": _by_hand(ctx, offsets),
        "offset_lines": offset_lines(offsets),
        "taint_spread_off": TAINT_CFG in {os.path.basename(p) for p in configs},
        "configs": {rel_to(p, root): f for p, f in configs.items()},
        "backup_copy": (j.first("backup_copied") or {}).get("dest"),
        "core": len(a["core"]),
        "kept": len(a["kept"]),
        "regions": sorted(os.path.basename(p) for p in transplant.region_files(w)),
        "affected_saves": (j.first("refresh") or {}).get("shared", []),
        "remote": ctx.opts.remote,
        "level_sha256": _level_sha(ctx),
        "rules": list(RULES),
    }
    saved = _copy_side(ctx, j, plan)
    rec["visualprospecting"] = rel_to(saved, root) if saved else None
    legacy = _reset_journeymap(ctx)
    rec["journeymap"] = rel_to(legacy, root) if legacy else None
    rec["upload"] = _upload(ctx, rec) if ctx.opts.remote else None
    write_atomic(os.path.join(w, MARKER), json.dumps(rec, indent=1).encode())
    jobs.checkpoint(SWAP)
    if rec["upload"] and rec["upload"][1]:
        _zip(ctx, rec, rec["upload"][1])
    j.note("recorded")
    work = os.path.dirname(staging.paths(ctx.preview)[1])
    if os.path.isdir(work) and not os.listdir(work):
        os.rmdir(work)
    return rec


def _level_sha(ctx):
    return staging.final(ctx.preview).get(world.LEVEL)


def _played(ctx, j, bad):
    """The differing files a game saved after the swap: none while level.dat is still the one Apply swapped in (a game that ran
    saved it), else those present and modified after the second rename."""
    level = os.path.join(ctx.place.world, world.LEVEL)
    if not os.path.isfile(level) or jobs.file_hash(level) == _level_sha(ctx):
        return []
    since = next(e["t"] for e in reversed(j.entries()) if e["phase"] == "rename2")
    return [b for b in bad if b[1] != "missing" and os.path.getmtime(staging.at(ctx.place.world, b[0])) >= since]


def _verify_after(ctx, j, rec):
    w = ctx.place.world
    bad = staging.mismatches(w, staging.final(ctx.preview), ignore=(SESSION_LOCK, MARKER))
    played = _played(ctx, j, bad)
    wrong = [b for b in bad if b not in played]
    if wrong:
        loses = " (this loses the play since Apply)" if played else ""
        raise Refusal(
            f"after the swap {jobs.count(len(wrong), 'file')} of {w} {'differs' if len(wrong) == 1 else 'differ'} from the preview, "
            f"the first {wrong[0][0]} ({wrong[0][1]})",
            f"Roll back{loses}: rwg2-migrator rollback {w}",
        )
    snap = _preview_file(ctx, staging.SNAPSHOT, staging.SNAPSHOT_FILES[0])
    flagged = {tuple(c) for c in _preview_file(ctx, staging.TRANSPLANT, staging.RECORD)["populated_set"]["core"]}
    names = {f"{x},{z}" for x, z in _preview_file(ctx, "area.json")["core"] if (x, z) not in flagged} & set(snap)
    got = staging.chunk_hashes(w, {tuple(map(int, n.split(","))) for n in names})
    same = sum(got.get(n) == snap[n] for n in names)
    ctx.preview.update(applied=ctx.job.id)
    genworld.discard(ctx.preview)
    j.note("verified", core_identical=same, saved_since=len(played))
    return rec | {"core_identical": same, "core_present": len(names), "saved_since_swap": len(played)}


def run(ctx):
    """Apply, or resume an Apply: the state of the folders decides where it continues."""
    j = Journal(os.path.join(ctx.job.folder, JOURNAL))
    plan = _plan(ctx, j)
    w, stage, trial, backup = _paths(ctx, plan)
    if os.path.isdir(trial) and not os.path.lexists(stage):
        rename(trial, stage)
    state = swap_state(w, moved(j, backup), stage, ctx.job.id)
    if state == UNKNOWN:
        raise Refusal(
            f"the folders do not match any state Apply can be in "
            f"(world {_exists(w)}, backup {_exists(backup)}, working copy {_exists(stage)})",
            f"Nothing was changed now. Look in {staging.work_dir(ctx.place.root)} for your world's folders, "
            f"or ask for help with these lines and the file {j.path}",
        )
    if state == RECREATED or (state == BETWEEN and j.last("rename1", "reverse") == "reverse"):
        _reverse(ctx, j, plan)
    if state == NOT_SWAPPED and j.last("rename1", "reverse", "reversed") == "reverse":
        _reverse(ctx, j, plan)
    if state == NOT_SWAPPED:
        _up_to_rename1(ctx, j, plan)
    if state in (NOT_SWAPPED, BETWEEN):
        _second(ctx, j, plan)
    _swapped_stopped(ctx)
    return _verify_after(ctx, j, _record(ctx, j, plan))


def _swapped_stopped(ctx):
    """A resume after the swap may find the migrated world started (a boot autostart, a player); the record is written only into
    a stopped world."""
    try:
        require_stopped(ctx.place, ctx.opts, _level_sha(ctx))
    except Refusal as e:
        raise Refusal(f"Apply swapped the migrated world in, but before it wrote its record {e.reason}", e.action) from e


def _exists(path):
    return "present" if os.path.lexists(path) else "missing"


def rollback_state(world_dir, backup, rolled, apply_id):
    key = (os.path.isdir(world_dir), _mark(world_dir, apply_id), os.path.isdir(backup), os.path.isdir(rolled))
    table = {(True, OURS, True, False): NOT_SWAPPED, (False, None, True, True): BETWEEN, (True, None, False, True): SWAPPED}
    return table.get(key, UNKNOWN)


def require_stopped(place, opts, level_sha):
    """The level.dat Apply swapped in has a fresh mtime; only a level.dat a game saved since counts as a sign of a running world."""
    level = os.path.join(place.world, world.LEVEL)
    untouched = os.path.isfile(level) and jobs.file_hash(level) == level_sha
    usage = procs.remote() if opts.remote else procs.in_use(place.root, None if untouched else place.world)
    procs.require_stopped(usage, opts.confirm_stopped)


def _rollback_checks(place, opts, rec):
    backup = resolve(rec["backup"], place.root)
    require_stopped(place, opts, rec.get("level_sha256"))
    if place.kind == "server" and not opts.remote and not opts.auto_restart_off:
        raise Refusal(
            "a server that restarts by itself could open the world while Roll back runs",
            "Turn that off until it finishes, then run rollback again with --auto-restart-off",
        )
    m = read_json(backup + SIDE_FILES["source"])
    if not os.path.isdir(backup) or not m:
        raise Refusal(
            f"the backup {backup}, or the list of its files kept next to it, is missing",
            "Restore it from your own backup copy; nothing was changed",
        )
    bad = staging.mismatches(backup, {rel: e["sha256"] for rel, e in m["files"].items()})
    if bad:
        raise Refusal(
            f"the backup {backup} changed since Apply ({bad[0][0]}: {bad[0][1]}); nothing was rolled back",
            "Restore it from your own backup copy",
        )
    if not opts.yes:
        raise GoAhead("Rollback loses all play since Apply, in every dimension", "Run rollback again with --yes if that is what you want")


def _restore_side(place, rec, rolled):
    saved = rec.get("visualprospecting") and resolve(rec["visualprospecting"], place.root)
    vp = os.path.join(place.root, VP)
    if saved and os.path.isdir(saved):
        if os.path.isdir(vp):
            rename(vp, rolled + "-" + VP)
        rename(saved, vp)
    pregen = os.path.join(place.root, "saves", os.path.basename(place.world), "pregenerationFiles")
    if os.path.isdir(pregen) and not procs.inside(pregen, place.world):
        rename(pregen, rolled + "-pregenerationFiles")
    by_hand = rec.get("by_hand") and resolve(rec["by_hand"]["file"], place.root)
    if by_hand and os.path.isfile(by_hand):
        rename(by_hand, rolled + BY_HAND)
    legacy = rec.get("journeymap") and resolve(rec["journeymap"], place.root)
    if legacy and os.path.isdir(legacy) and not os.path.lexists(os.path.join(os.path.dirname(legacy), JM_DIM0)):
        rename(legacy, os.path.join(os.path.dirname(legacy), JM_DIM0))


def rollback_journal(place):
    return os.path.join(staging.work_dir(place.root), f"{os.path.basename(place.world)}-rollback.jsonl")


def rollback(place, opts, data_root):
    """Put the world from before Apply back, restore only the config keys Apply changed, and keep the migrated world aside."""
    name = os.path.basename(place.world)
    j = Journal(rollback_journal(place))
    started = j.first("start")
    start = started or {
        "record": applied_record(place),
        "rolled": os.path.join(staging.work_dir(place.root), f"{name}-rwg2-rolled-back-{stamp()}"),
    }
    rec, rolled = start["record"], start["rolled"]
    backup = resolve(rec["backup"], place.root)
    state = rollback_state(place.world, backup, rolled, rec["apply_job"])
    if state == NOT_SWAPPED or not started:
        _rollback_checks(place, opts, rec)
    if not started:
        j.note("start", **start)
    if state == NOT_SWAPPED:
        j.note("rename1")
        rename(place.world, rolled)
        jobs.checkpoint(ROLLBACK)
        state = BETWEEN
    if state == BETWEEN:
        j.note("rename2")
        rename(backup, place.world)
        jobs.checkpoint(ROLLBACK)
        state = SWAPPED
    if state != SWAPPED:
        raise Refusal(
            f"the folders do not match any state Rollback can be in (world {_exists(place.world)}, backup {_exists(backup)}, "
            f"rolled-back copy {_exists(rolled)})",
            f"Nothing was changed now; see {j.path}",
        )
    reachable, by_hand = _reachable({p: (resolve(p, place.root), f) for p, f in rec["configs"].items()})
    write_configs({path: f for path, f in reachable.values()}, "old")
    _restore_side(place, rec, rolled)
    _close(data_root, rec["apply_job"])
    j.note("done")
    os.replace(j.path, j.path[: -len(".jsonl")] + f"-{stamp()}.done.jsonl")
    return {"world": place.world, "rolled_back": rolled, "configs": sorted(reachable), "configs_by_hand": by_hand}


def _reachable(configs):
    """The journaled config files whose folder is still there, and the old lines of the others, to put back by hand."""
    here = {p: v for p, v in configs.items() if os.path.isdir(os.path.dirname(v[0]))}
    gone = {v[0]: [_old_line(row) for row in v[1]["keys"]] for p, v in configs.items() if p not in here and not v[1]["created"]}
    return here, gone


def _old_line(row):
    line = f'{row["type"]}:"{row["key"][-1]}"'
    return f"remove the line {line}" if row["old"] is None else f"{line}={row['old']}"


def applied_record(place):
    rec = record_of(place.world)
    if not rec or rec.get("state") != APPLIED:
        raise Refusal(f"{place.world} carries no migration record of {TOOL}", "Only a world this tool applied can be rolled back")
    return rec


def _close(data_root, apply_id):
    """A rolled-back Apply is over, whatever phase its journal had reached, so it no longer blocks a new preview."""
    job = jobs.Job(data_root, apply_id)
    if job.data:
        Journal(os.path.join(job.folder, JOURNAL)).note("rolled_back")
        job.update(state=jobs.DONE, rolled_back=True)


def leftover_regions(world_dir, rec):
    """Overworld region files the applied world did not have that hold chunks saved before Apply: the leftovers of a merged upload."""
    out = []
    for path in transplant.region_files(world_dir):
        if os.path.basename(path) in set(rec["regions"]):
            continue
        with open(path, "rb") as f:
            old = sum(ts < rec["applied_at"] for *_, ts in slots(f.read()))
        if old:
            out.append({"file": os.path.basename(path), "old_chunks": old})
    return out


def config_in_use(place, rec, given=None):
    """The config folder verify reads: the one given, else the one Apply wrote into while it is still this world's (recorded
    absolute, or the folder found next to the world now), else the one found next to it; None when none holds rwg2.cfg."""
    if given:
        return given
    recorded = (rec or {}).get("config_dir")
    path = recorded and resolve(recorded, place.root)
    mine = path and (os.path.isabs(recorded) or (place.config and world.same(path, place.config)))
    return next((c for c in (path if mine else None, place.config) if c and world.holds_rwg2(c)), None)


def _offsets_check(cfg, rec, restore):
    """(wrong, restored) of the 4 offset lines in cfg against the record; nothing is checked without a config folder."""
    if cfg is None:
        return {}, []
    keys = {(generator.RWG2_CATEGORY, k): str(v) for k, v in rec["offsets"].items()}
    wrong = {k[-1]: [v, forgecfg.get(cfg, k)] for k, v in keys.items() if forgecfg.get(cfg, k) != v}
    restored = []
    if restore and wrong:
        forgecfg.edit(cfg, {k: ("I", v) for k, v in keys.items()})
        restored = [k for k in wrong if forgecfg.get(cfg, (generator.RWG2_CATEGORY, k)) == str(rec["offsets"][k])]
    return wrong, restored


def _taint_turned_off(rec):
    """Whether Apply turned taint spreading off, in the config it wrote or in the lines it left to set by hand."""
    return bool(rec.get("taint_spread_off") or (rec.get("by_hand") or {}).get("taint"))


def verify(place, restore, config=None):
    """Compare config/rwg2.cfg with the world's record (optionally restoring the 4 offset lines) and flag leftover region files.
    With no config folder at all the lines are not checked: "config" is None and the caller says where they go."""
    rec = record_of(place.world)
    if not rec or rec.get("state") != APPLIED:
        raise Refusal(f"{place.world} carries no migration record of {TOOL}", "Verify works on a world this tool applied")
    folder = config_in_use(place, rec, config)
    cfg = folder and os.path.join(folder, RWG2_CFG)
    wrong, restored = _offsets_check(cfg, rec, restore)
    return {
        "config": cfg,
        "wrong": wrong,
        "restored": restored,
        "lines": offset_lines(rec["offsets"]),
        "by_hand": None if cfg else by_hand_lines(rec["offsets"], _taint_turned_off(rec)),
        "leftovers": leftover_regions(place.world, rec),
        "record": rec,
    }
