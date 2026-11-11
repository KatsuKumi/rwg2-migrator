"""The per-job working copy of a world: the copy and its source manifest, the snapshot, reset and refresh, the final manifest."""

import errno
import hashlib
import io
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import Refusal, blend, generator, genworld, jobs, plant, postcheck, procs, region, smooth, transplant, world
from . import area as areas
from .jobs import append_line, file_hash, key_of, read_json, read_lines
from .region import remove_file, rmtree, write_atomic

COPY_BLOCK = 1 << 20
WORK = "rwg2-work"
STAGING = "staging"
COPY_TRIES = 5
RETRY_SECONDS = 0.5
SOURCE, PROGRESS, FINAL, RECORD = "source.json", "copied.jsonl", "final.json", "record.json"
SNAPSHOT_FILES = ("kept.json", "ground.npz", "keys.json")
SNAPSHOT_RING = 2
COPY, SNAPSHOT, TRANSPLANT, FINAL_STEP, POSTCHECK = "copy", "snapshot", "transplant", "final", "postcheck"
BEFORE, REPORT = "before.npz", "report.json"
MB = 1 << 20
HEADROOM = 512 * MB


def work_dir(root):
    return os.path.join(root, WORK)


def staging_dir(root, job_id):
    return os.path.join(root, WORK, job_id, STAGING)


def device(path):
    return os.stat(path).st_dev


def _unreadable(err):
    if not isinstance(err, FileNotFoundError):
        raise Refusal(
            f"the tool cannot read the folder {err.filename} ({err.strerror})",
            "Give your user account read access to that folder, then run this again",
        )


def _is_link(path):
    return os.path.islink(path) or os.path.isjunction(path)


def walk(folder):
    """(relative folders, relative files) under folder, with / separators, sorted. A link or a folder that cannot be read is
    refused, never skipped: it would be missing from the copy and from every check made against the same listing."""
    dirs, files = [], []
    for d, ds, fs in os.walk(folder, onerror=_unreadable):
        ds.sort()
        links = sorted(n for n in ds + fs if _is_link(os.path.join(d, n)))
        if links:
            raise Refusal(
                f"{os.path.join(d, links[0])} is a link to another place (a symlink or junction), which the tool cannot copy "
                "into the migrated world",
                "Move the real folder or file back into the world in place of the link, then run this again",
            )
        rel = os.path.relpath(d, folder).replace(os.sep, "/")
        dirs.append(rel)
        files += [f if rel == "." else f"{rel}/{f}" for f in sorted(fs)]
    return dirs, files


def at(folder, rel):
    """The path of a /-separated relative path under folder, with the OS's own separators."""
    return os.path.join(folder, *rel.split("/"))


def _stat(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def dim0_region(rel):
    """(rx, rz) of an overworld region file given by its path relative to the world, else None."""
    head, _, name = rel.partition("/")
    return region.coords(name) if head == "region" and region.NAME.fullmatch(name) else None


def kept_regions(cells):
    return {(x // region.SIDE, z // region.SIDE) for x, z in cells}


@dataclass(frozen=True)
class Kept:
    cells: frozenset
    regions: frozenset

    @classmethod
    def of(cls, cells):
        cells = frozenset(map(tuple, cells))
        return cls(cells, frozenset(kept_regions(cells)))


def torn_chunk(path, rxz, kept):
    """The first kept chunk of a region file that does not decode at its slot, or None."""
    with open(path, "rb") as f:
        recs = region.records(f.read())
    for i, (rec, _) in sorted(recs.items()):
        pos = region.slot_pos(*rxz, i)
        if pos in kept and transplant.decode(rec, pos) is None:
            return pos
    return None


def _read_through(src, dst):
    h, n = hashlib.sha256(), 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as f, open(dst, "wb") as out:
        while block := f.read(COPY_BLOCK):
            out.write(block)
            h.update(block)
            n += len(block)
            jobs.checkpoint(COPY)
        out.flush()
        os.fsync(out.fileno())
    return h.hexdigest(), n


def copy_file(src, dst, check, rel):
    """Copy one file of a world that may be running. It is copied again while it changes under the copy or while check(dst)
    names a kept chunk that does not decode; the entry keeps the stat taken before the copy, so a refresh sees later changes."""
    for _ in range(COPY_TRIES):
        before = _stat(src)
        digest, n = _read_through(src, dst)
        torn = check(dst)
        if torn is None and before == _stat(src):
            break
        time.sleep(RETRY_SECONDS)
    if torn is not None:
        raise Refusal(
            f"the kept chunk {torn[0]},{torn[1]} in {rel} did not read in {COPY_TRIES} copies: "
            "your world was saving, or that chunk is damaged",
            "Try again; if it fails while the world is stopped, restore that region file from a backup",
        )
    return {"size": before[0], "mtime_ns": before[1], "sha256": digest, "bytes": n, "copied": True}


def _take(src, dst, rel, kept):
    """The manifest entry of one world file: copied, or only hashed for an overworld region file that holds no kept chunk;
    None when the file vanished (a running game renames level.dat_new into place)."""
    s, d = at(src, rel), at(dst, rel)
    rxz = dim0_region(rel)
    try:
        if rxz is not None and rxz not in kept.regions:
            size, mtime = _stat(s)
            return {"size": size, "mtime_ns": mtime, "sha256": file_hash(s), "copied": False}
        check = (lambda p: torn_chunk(p, rxz, kept.cells)) if rxz is not None else (lambda p: None)
        return copy_file(s, d, check, rel)
    except FileNotFoundError:
        if os.path.exists(s):
            raise
        return None


def _staged(dst, rel, e):
    return not e["copied"] or (os.path.isfile(at(dst, rel)) and os.path.getsize(at(dst, rel)) == e["bytes"])


def require_free(folder, need, action):
    """Refuse unless the drive of folder has need bytes free plus HEADROOM, which a world that may be running keeps for its saves."""
    free = shutil.disk_usage(folder).free
    if free < need + HEADROOM:
        raise Refusal(
            f"this needs {need / MB:,.0f} MB on the drive of {folder}, plus {HEADROOM / MB:,.0f} MB left free for your world's own "
            f"saves; {free / MB:,.0f} MB is free",
            action,
        )


def _size(path):
    try:
        return os.path.getsize(path)
    except FileNotFoundError:
        return 0


def to_copy(src, left, kept):
    """Bytes copy() writes for the files left: overworld region files without a kept chunk are only hashed."""
    return sum(_size(at(src, rel)) for rel in left if dim0_region(rel) is None or dim0_region(rel) in kept.regions)


def copy(job, src, dst, kept):
    """Copy the world into staging, leaving out overworld region files that hold no kept chunk; resumes file by file."""
    progress = os.path.join(job.step_dir(COPY), PROGRESS)
    done = {e["rel"]: e for e in read_lines(progress)}
    dirs, files = walk(src)
    left = [rel for rel in files if not (rel in done and _staged(dst, rel, done[rel]))]
    require_free(dst, to_copy(src, left, kept), "Free some space on that drive, then run the preview again")
    for d in dirs:
        os.makedirs(at(dst, d), exist_ok=True)
    for rel in left:
        jobs.checkpoint(COPY)
        entry = _take(src, dst, rel, kept)
        if entry:
            append_line(progress, {"rel": rel} | entry)
    entries = {e["rel"]: {k: v for k, v in e.items() if k != "rel"} for e in read_lines(progress)}
    manifest = {"world": src, "dirs": dirs, "files": entries}
    write_atomic(os.path.join(job.step_dir(COPY), SOURCE), json.dumps(manifest).encode())
    return manifest


def source(job):
    return read_json(os.path.join(job.folder, COPY, SOURCE))


def _save_source(job, m):
    write_atomic(os.path.join(job.step_dir(COPY), SOURCE), json.dumps(m).encode())


def pending(m, src):
    """World files that differ from the source manifest by size or mtime, or are new or gone: (rel, size now) pairs."""
    _, now = walk(src)
    present = set(now)
    gone = [(rel, 0) for rel in m["files"] if rel not in present]
    out = []
    for rel in now:
        e, st = m["files"].get(rel), _stat(at(src, rel))
        if not e or (e["size"], e["mtime_ns"]) != st:
            out.append((rel, st[0]))
    return out + gone


def _catch_up(m, src, dst, kept):
    files = m["files"]
    dirs, _ = walk(src)
    for d in dirs:
        os.makedirs(at(dst, d), exist_ok=True)
    changed = pending(m, src)
    for rel, _ in changed:
        entry = _take(src, dst, rel, kept)
        if entry:
            files[rel] = entry
        else:
            files.pop(rel, None)
    m["dirs"] = dirs
    return [rel for rel, _ in changed]


def paths(job):
    """(world, staging) of a preview job."""
    return job.data["world"], job.data["args"]["staging"]


def sync(job, kept, refresh=False):
    """Reset staging to the copy: re-copy every staged file whose hash differs from the source manifest and delete files the copy
    did not make. With refresh, first catch up with every file the world changed since (the world is stopped by then)."""
    src, dst = paths(job)
    m = source(job)
    changed = _catch_up(m, src, dst, kept) if refresh else []
    files = m["files"]
    _, have = walk(dst)
    for rel in have:
        if not files.get(rel, {}).get("copied"):
            remove_file(at(dst, rel))
            jobs.checkpoint(COPY)
    for rel, e in sorted(files.items()):
        path = at(dst, rel)
        if e["copied"] and (not os.path.isfile(path) or file_hash(path) != e["sha256"]):
            jobs.checkpoint(COPY)
            entry = _take(src, dst, rel, kept)
            if entry:
                files[rel] = entry
    _save_source(job, m)
    return changed


def chunk_hashes(world_dir, cells):
    """{"x,z": [sha256 of compression byte + payload, timestamp]} of every chunk of cells present in world_dir."""
    out, wanted = {}, kept_regions(cells)
    for path in transplant.region_files(world_dir):
        rx, rz = region.coords(path)
        if (rx, rz) not in wanted:
            continue
        with open(path, "rb") as f:
            recs = region.records(f.read())
        for i, (rec, ts) in recs.items():
            pos = region.slot_pos(rx, rz, i)
            if pos in cells and rec is not None:
                out[f"{pos[0]},{pos[1]}"] = [hashlib.sha256(rec[4:]).hexdigest(), ts]
    return out


def mods_listing(root):
    folder = os.path.join(root, "mods")
    return sorted((os.path.basename(p), os.path.getsize(p)) for p in world.files(folder, "*") if os.path.isfile(p))


def snapshot(job, dst, a, sc):
    """What later steps use instead of the original world: kept-chunk hashes, the old ground around kept, claim and config keys."""
    kept, teams = {tuple(c) for c in a["kept"]}, a.get("teams", [])
    folder = job.step_dir(SNAPSHOT)
    write_atomic(os.path.join(folder, SNAPSHOT_FILES[0]), json.dumps(chunk_hashes(dst, kept)).encode())
    boxes = areas.windows(areas.dilate(kept, SNAPSHOT_RING), ring=0)
    buf = io.BytesIO()
    grids = {f"ground{i}": sc.grid("ground", b) for i, b in enumerate(boxes)}
    np.savez_compressed(buf, boxes=np.array(boxes, np.int64).reshape(-1, 4), **grids)
    write_atomic(os.path.join(folder, SNAPSHOT_FILES[1]), buf.getvalue())
    claims = world.claims(dst)
    keys = {
        "claims": areas.claim_hash(set().union(*(claims.get(t, set()) for t in teams))),
        "config": jobs.tree_key(job.data["args"].get("config_dir")),
        "config_files": jobs.tree_hashes(job.data["args"].get("config_dir")),
        "mods": mods_listing(job.data["args"]["root"]),
    }
    write_atomic(os.path.join(folder, SNAPSHOT_FILES[2]), json.dumps(keys).encode())


def final_files(m, changed):
    """{rel: sha256} of every file staging must hold after the transplant."""
    files = {rel: e["sha256"] for rel, e in m["files"].items() if e["copied"]}
    for rel, digest in changed.items():
        if digest is None:
            files.pop(rel, None)
        else:
            files[rel] = digest
    return files


def final(job):
    return read_json(os.path.join(job.folder, FINAL_STEP, FINAL))["files"]


def mismatches(folder, files, ignore=()):
    """Files of folder that are missing, differ from files {rel: sha256}, or should not be there, as (rel, why) pairs."""
    _, have = walk(folder)
    have = [rel for rel in have if rel not in ignore]
    files = {rel: digest for rel, digest in files.items() if rel not in ignore}
    out = [(rel, "missing") for rel in sorted(set(files) - set(have))]
    out += [(rel, "not in the preview") for rel in sorted(set(have) - set(files))]
    return out + [(rel, "changed") for rel in sorted(set(have) & set(files)) if file_hash(at(folder, rel)) != files[rel]]


def _check_core(job, dst, a, record):
    want = read_json(os.path.join(job.folder, SNAPSHOT, SNAPSHOT_FILES[0]))
    core = {tuple(c) for c in a["core"]} - {tuple(c) for c in record["populated_set"]["core"]}
    names = {f"{x},{z}" for x, z in core}
    got = chunk_hashes(dst, core)
    bad = sorted(n for n in names & set(want) if got.get(n) != want[n])
    if bad:
        raise AssertionError(f"{len(bad)} protected chunks changed in the working copy, the first {bad[0]}")
    return len(names & set(want))


class OldGround:
    """The snapshot's ground grids (the world's ground around kept, as copied), read like Scan.grid."""

    def __init__(self, job):
        with np.load(os.path.join(job.folder, SNAPSHOT, SNAPSHOT_FILES[1])) as z:
            self.parts = [(tuple(int(v) for v in b), z[f"ground{i}"]) for i, b in enumerate(z["boxes"])]

    def grid(self, field, box):
        x0, z0, x1, z1 = box
        out = np.full((z1 - z0, x1 - x0), np.nan, np.float32)
        for (bx0, bz0, bx1, bz1), g in self.parts:
            ax0, ax1, az0, az1 = max(x0, bx0), min(x1, bx1), max(z0, bz0), min(z1, bz1)
            if ax0 < ax1 and az0 < az1:
                out[az0 - z0 : az1 - z0, ax0 - x0 : ax1 - x0] = g[az0 - bz0 : az1 - bz0, ax0 - bx0 : ax1 - bx0]
        return out


@dataclass
class Work:
    """What a build needs besides the job: the world scan, the transplant options, the generator (None for a transplant-only
    preview) and a line printer."""

    scan: object
    opts: transplant.Options
    gens: object = None
    say: object = print


def _hash(job, step, name):
    path = os.path.join(job.folder, step, name)
    return file_hash(path) if os.path.isfile(path) else None


def _record(job, step):
    return read_json(os.path.join(job.folder, step, RECORD))


def _save(job, step, record, keys):
    write_atomic(os.path.join(job.step_dir(step), RECORD), json.dumps(record).encode())
    job.finish_step(step, [RECORD], keys)


class Steps:
    """The input keys of every step from the transplant on, and which of them still hold."""

    def __init__(self, job, work, a):
        self.job, self.work, self.a = job, work, a
        args = job.data["args"]
        self.generating = not args.get("no_generate", True)
        self.smoothing = self.generating and args.get("smooth", False)
        self.trees = self.smoothing and args.get("trees", False)
        self.blend = self.smoothing and args.get("blend_biomes", False)

    @property
    def settings(self):
        return generator.settings(self.work.gens.cache) if self.generating else None

    def keys(self, step):
        j, h = self.job, _hash
        rwg_data = (_record(j, TRANSPLANT) or {}).get("rwg_data")
        return {
            TRANSPLANT: lambda: {"source": h(j, COPY, SOURCE), "options": asdict(self.work.opts)},
            genworld.GENERATE: lambda: genworld.generate_key(j, self.a, self.settings) | {"rwg_data": rwg_data},
            genworld.RENDER: lambda: genworld.render_key(j, self.a, self.settings),
            genworld.HARVEST: lambda: {"transplant": h(j, TRANSPLANT, RECORD), "generate": h(j, genworld.GENERATE, genworld.GENERATED)},
            smooth.SMOOTH: lambda: {
                "harvest": h(j, genworld.HARVEST, RECORD),
                "render": h(j, genworld.RENDER, jobs.MANIFEST_FILE),
                "snapshot": h(j, SNAPSHOT, jobs.MANIFEST_FILE),
                "blend": self.blend,
            },  # fmt: skip
            plant.TREES: lambda: {"smooth": h(j, smooth.SMOOTH, RECORD), "plant": h(j, plant.PLANT, jobs.MANIFEST_FILE)},
        }[step]()

    def valid(self, step):
        if step == genworld.GENERATE:
            return genworld.gen_valid(self.job, self.keys(step))
        return self.job.step_valid(step, self.keys(step))

    def chain(self):
        out = [TRANSPLANT]
        out += [genworld.GENERATE, genworld.RENDER, genworld.HARVEST] if self.generating else []
        return out + ([smooth.SMOOTH] if self.smoothing else []) + ([plant.TREES] if self.trees else [])


def _transplant(job, work, a, refresh):
    """Reset staging to the copy (catching up with the world first on a refresh), then snapshot and transplant."""
    kept = Kept.of(a["kept"])
    dst = paths(job)[1]
    for step in (TRANSPLANT, genworld.HARVEST, smooth.SMOOTH, plant.TREES, FINAL_STEP, POSTCHECK):
        job.drop_step(step)
    changed = sync(job, kept, refresh)
    job.finish_step(COPY, [SOURCE], {"world": paths(job)[0], "kept": key_of(sorted(kept.cells))})
    source_key = file_hash(os.path.join(job.folder, COPY, SOURCE))
    if not job.step_valid(SNAPSHOT, source_key):
        snapshot(job, dst, a, work.scan)
        job.finish_step(SNAPSHOT, list(SNAPSHOT_FILES), source_key)
    record = transplant.run(dst, a, work.opts) | {"refreshed": changed}
    record["core_identical"] = _check_core(job, dst, a, record)
    _save(job, TRANSPLANT, record, {"source": source_key, "options": asdict(work.opts)})
    return record


def _generated(job, st, work, record):
    """The gen-world and the render, reused while their keys hold; then the harvest and the smoothing of staging."""
    a = st.a
    if not st.valid(genworld.GENERATE):
        work.say("preview: generating the new terrain around the kept area (one start of the generator, about 2 minutes)")
        genworld.run_generate(job, work.gens, a, lambda: st.keys(genworld.GENERATE))
    if not st.valid(genworld.RENDER):
        work.say("preview: rendering the chosen candidate (a second start of the generator, about 1 to 2.5 minutes)")
        genworld.run_render(job, work.gens, lambda: st.keys(genworld.RENDER))
    if not st.valid(genworld.HARVEST):
        h = genworld.harvest(job, a, genworld.generated(job), record["never_generated"])
        h["core_identical"] = _check_core(job, paths(job)[1], a, record)
        _save(job, genworld.HARVEST, h, st.keys(genworld.HARVEST))
    if st.smoothing and not st.valid(smooth.SMOOTH):
        colours = blend.Colours(st.settings["release"]) if st.blend else None
        try:
            rec, before, strips = smooth.run(paths(job)[1], a, (OldGround(job), genworld.Render(job)), colours)
        except smooth.LowMemory as e:
            raise Refusal(e.reason, f"{e.action}: rwg2-migrator jobs resume {job.id}") from e
        _save_ground(job, before)
        plant.save_strips(os.path.join(job.step_dir(smooth.SMOOTH), plant.STRIPS), strips)
        rec["core_identical"] = _check_core(job, paths(job)[1], a, record)
        write_atomic(os.path.join(job.step_dir(smooth.SMOOTH), RECORD), json.dumps(rec).encode())
        job.finish_step(smooth.SMOOTH, [RECORD, BEFORE, plant.STRIPS], st.keys(smooth.SMOOTH))
    if st.trees and not st.valid(plant.TREES):
        step = plant.Step(job, work.gens, a, st.settings)
        rec = plant.run(step, os.path.join(job.folder, smooth.SMOOTH, plant.STRIPS), work.say)
        rec["core_identical"] = _check_core(job, paths(job)[1], a, record)
        _save(job, plant.TREES, rec, st.keys(plant.TREES))


def _save_ground(job, grids):
    buf = io.BytesIO()
    pos = np.array(sorted(grids), np.int64).reshape(-1, 2)
    np.savez_compressed(buf, pos=pos, ground=np.array([grids[tuple(p)] for p in pos.tolist()], np.float32).reshape(-1, 16, 16))
    write_atomic(os.path.join(job.step_dir(smooth.SMOOTH), BEFORE), buf.getvalue())


def before_smoothing(job):
    with np.load(os.path.join(job.folder, smooth.SMOOTH, BEFORE)) as z:
        return {tuple(p): g for p, g in zip(z["pos"].tolist(), z["ground"], strict=True)}


def _changes(job, st):
    records = [_record(job, s) for s in st.chain() if s in (TRANSPLANT, genworld.HARVEST, smooth.SMOOTH, plant.TREES)]
    changed = {}
    for r in records:
        changed |= r["changed"]
    return changed, key_of([r["changed"] for r in records])


def _final(job, st):
    changed, key = _changes(job, st)
    if not job.step_valid(FINAL_STEP, key):
        files = final_files(source(job), changed)
        write_atomic(os.path.join(job.step_dir(FINAL_STEP), FINAL), json.dumps({"files": files}).encode())
        job.finish_step(FINAL_STEP, [FINAL], key)


def build(job, work, refresh=False):
    """build_steps, with a full drive refused in plain words; a preview's partial working copy is deleted then, so a world that may
    be running has room to save again (Apply's refresh keeps it: the world is stopped and the copy is the ready preview)."""
    try:
        return build_steps(job, work, refresh)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        dst = paths(job)[1]
        drop = not refresh and procs.inside(e.filename or dst, dst)
        if drop:
            rmtree(dst)
            job.drop_step(COPY)
        kept = "deleted" if drop else "kept"
        raise Refusal(
            f"the drive filled up while writing {e.filename}; the working copy of preview {job.id} was {kept}",
            "Free some space on that drive, then run this again",
        ) from e


def build_steps(job, work, refresh=False):
    """The preview pipeline on the job's world: copy, reset, snapshot, transplant, then (unless transplant-only) the generated
    terrain, the harvest and the smoothing, the final manifest and the post-check. Each step is skipped while its manifest still
    matches its inputs; any step from the transplant on that does not resets staging to the copy and redoes them all, while the
    gen-world and the render, which live outside staging, are reused while their keys hold. With refresh (Apply) the copy
    catches up with the world first."""
    a = read_json(os.path.join(job.folder, "area.json"))
    (src, dst), kept = paths(job), Kept.of(a["kept"])
    keys = {"world": src, "kept": key_of(sorted(kept.cells))}
    if not job.step_valid(COPY, keys):
        os.makedirs(dst, exist_ok=True)
        if device(dst) != device(src):
            raise Refusal(
                f"the working folder {dst} is on another drive than {src}", "Apply only renames folders; keep rwg2-work next to the world"
            )
        copy(job, src, dst, kept)
        job.finish_step(COPY, [SOURCE], keys)
    st = Steps(job, work, a)
    if refresh or not all(st.valid(s) for s in st.chain()):
        _transplant(job, work, a, refresh)
    record = _record(job, TRANSPLANT)
    if st.generating:
        _generated(job, st, work, record)
    _final(job, st)
    return record | ({"post": _postcheck(job, st)} if st.generating else {})


def _outside(job, record):
    """Files outside the overworld region folder: identical to the world's, except those the transplant names."""
    src, files = source(job)["files"], final(job)
    rels = [rel for rel in files if not rel.startswith("region/")]
    by_transplant = sorted(rel for rel in record["changed"] if not rel.startswith("region/"))
    differ = sorted(rel for rel in rels if rel not in by_transplant and files[rel] != src.get(rel, {}).get("sha256"))
    if differ:
        raise AssertionError(f"{jobs.count(len(differ), 'file')} outside the overworld changed after the transplant, the first {differ[0]}")
    return {"files": len(rels), "identical": len(rels) - len(set(by_transplant) & set(rels)), "changed_by_transplant": by_transplant}


TREE_FIELDS = ("spots", "log_removed", "log", "log_in_strips", "leaves", "skipped_not_air", "chunks")


def _trees(job):
    """What the replanting added, against the log voxels smoothing removed."""
    r = _record(job, plant.TREES)
    in_core = len(set(map(tuple, r["chunks_changed"])) & _core(job))
    return {k: r.get(k, 0) for k in TREE_FIELDS} | {"boot": r["boot"], "chunks_changed_in_core": in_core}


def _core(job):
    return {tuple(c) for c in read_json(os.path.join(job.folder, "area.json"))["core"]}


def _blended(job):
    """The grass-colour step across the kept border with and without the blend, and how many band columns took RWG2's biome."""
    r = _record(job, smooth.SMOOTH)
    out = {k.removeprefix("blend_"): v for k, v in r.items() if k.startswith("blend_")}
    pairs = out.get("border_pairs", 0)
    shares = {f"border_over_pct_{w}": blend.share(pairs, out.get(f"border_over_{w}", 0)) for w in ("before", "after")}
    return out | shares


def _smoothed_away(job):
    """What smoothing took off the columns it reshaped: the entities standing there (animals, item frames, dropped items) and the
    blocks that carry data of their own (crops, flower pots)."""
    removed = _record(job, smooth.SMOOTH).get("removed", {})
    return {"entities": removed.get("Entities", 0), "tile_entities": removed.get("TileEntities", 0)}


def _postcheck(job, st):
    """Game vs render in the new chunks, the seams before and after smoothing, and the integrity of the final staging."""
    key = {"final": _hash(job, FINAL_STEP, FINAL)}
    path = os.path.join(job.folder, POSTCHECK, REPORT)
    if job.step_valid(POSTCHECK, key):
        return read_json(path)
    a, stage = st.a, paths(job)[1]
    bad = mismatches(stage, final(job))
    if bad:
        raise AssertionError(f"the working copy does not match its final manifest ({bad[0][0]}: {bad[0][1]})")
    problems = postcheck.region_problems(stage)
    if problems:
        raise AssertionError(f"a region file of the working copy is damaged: {problems[0]}")
    record, core = _record(job, TRANSPLANT), {tuple(c) for c in a["core"]}
    after = postcheck.chunk_ground(stage, postcheck.border_cells(a))
    before = postcheck.Columns(before_smoothing(job)) if st.smoothing else after
    harvested = genworld.harvest_ground(job)
    report = {
        "new_terrain": postcheck.new_terrain(postcheck.Columns(harvested), genworld.Render(job), set(harvested)),
        "seams": {"before": postcheck.seams(before, a), "after": postcheck.seams(after, a)},
        "columns_reshaped": postcheck.reshaped(before, after, postcheck.border_cells(a) - core),
        "core_columns_changed": postcheck.reshaped(before, after, areas.boundary(core)),
        "smoothing_removed": _smoothed_away(job) if st.smoothing else None,
        "core_identical": _check_core(job, stage, a, record),
        "outside_overworld": _outside(job, record),
        "trees": _trees(job) if st.trees else None,
        "biome_blend": _blended(job) if st.blend else None,
        "region_files": len(transplant.region_files(stage)),
        "final_files": len(final(job)),
    }
    write_atomic(os.path.join(job.step_dir(POSTCHECK), REPORT), json.dumps(report, indent=1).encode())
    job.finish_step(POSTCHECK, [REPORT], key)
    return report
