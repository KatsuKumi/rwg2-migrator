import errno
import json
import os
import shutil
from pathlib import Path

import pytest
from conftest import tree

from rwg2migrator import cli, jobs, nbt, region, staging, transplant

TORN = (1, 1)
LATER = (2, 1)


def _record(path, pos):
    return region.records(Path(path).read_bytes())[region.slot_index(*pos)]


def _rewritten(path, pos, inhabited):
    """The record of pos with another InhabitedTime, and the file offset of its sectors."""
    data = Path(path).read_bytes()
    i = region.slot_index(*pos)
    loc, count = int.from_bytes(data[4 * i : 4 * i + 3], "big"), data[4 * i + 3]
    name, root = transplant.decode(region.record(data, loc, count), pos)
    root["Level"][1]["InhabitedTime"] = [nbt.LONG, inhabited]
    rec = region.deflate(nbt.write(name, root))
    assert len(rec) <= count * region.SECTOR
    return rec, loc * region.SECTOR


def _write_at(path, offset, data):
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def test_a_kept_chunk_torn_by_a_writer_during_the_copy_is_copied_again(mig, monkeypatch):
    src = os.path.join(mig.world, "region", "r.0.0.mca")
    rec, offset = _rewritten(src, TORN, 123_456)
    real, calls = staging._read_through, []

    def writer_during_copy(a, b):
        calls.append(a)
        if a == src and calls.count(src) == 1:
            _write_at(src, offset, rec[: len(rec) // 2])
            out = real(a, b)
            _write_at(src, offset + len(rec) // 2, rec[len(rec) // 2 :])
            return out
        return real(a, b)

    found, check = [], staging.torn_chunk
    monkeypatch.setattr(staging, "torn_chunk", lambda *a: found.append(check(*a)) or found[-1])
    monkeypatch.setattr(staging, "_read_through", writer_during_copy)
    monkeypatch.setattr(staging, "RETRY_SECONDS", 0)
    code, job = mig.preview()
    assert code == 0 and calls.count(src) == 2 and found[:2] == [TORN, None]
    m = staging.source(job)
    events = [e for e in jobs.read_lines(os.path.join(job.folder, "copy", staging.PROGRESS)) if e["rel"] == "region/r.0.0.mca"]
    assert len(events) == 1 and m["files"]["region/r.0.0.mca"]["sha256"] == jobs.file_hash(src)
    st = job.data["args"]["staging"]
    assert _record(os.path.join(st, "region", "r.0.0.mca"), TORN) == _record(src, TORN)
    monkeypatch.setattr(staging, "_read_through", real)
    rec, offset = _rewritten(src, LATER, 777)
    _write_at(src, offset, rec)
    Path(mig.world, "playerdata", "new.dat").write_bytes(Path(mig.world, "playerdata", "u1.dat").read_bytes())
    os.utime(os.path.join(mig.world, "level.dat"), (0, 0))
    assert mig.apply(job) == 0
    assert _record(os.path.join(mig.world, "region", "r.0.0.mca"), LATER)[0] == rec
    applied = tree(mig.world, skip=("session.lock", "rwg2-migration.json"))
    assert applied == staging.final(job) and "playerdata/new.dat" in applied


def test_reset_puts_staging_back_to_the_copy(mig):
    code, job = mig.preview()
    st = job.data["args"]["staging"]
    lock = Path(st, "session.lock")
    lock.write_bytes(b"flipped!")
    os.remove(os.path.join(st, "data", "Mineshaft.dat"))
    Path(st, "stray.rwg2tmp").write_bytes(b"x")
    kept = staging.Kept.of(json.loads(Path(job.folder, "area.json").read_text())["kept"])
    staging.sync(job, kept)
    copied = {rel: e["sha256"] for rel, e in staging.source(job)["files"].items() if e["copied"]}
    assert code == 0 and tree(st) == copied


def test_refresh_catches_up_with_changed_new_and_deleted_files(mig):
    code, job = mig.preview()
    w = mig.world
    Path(w, "data", "Temple.dat").unlink()
    Path(w, "data", "extra.dat").write_bytes(b"new")
    Path(w, "region", "r.3.3.mca").write_bytes(Path(w, "region", "r.3.3.mca").read_bytes() + bytes(region.SECTOR))
    m = staging.source(job)
    assert sorted(rel for rel, _ in staging.pending(m, w)) == ["data/Temple.dat", "data/extra.dat", "region/r.3.3.mca"]
    kept = staging.Kept.of(json.loads(Path(job.folder, "area.json").read_text())["kept"])
    changed = staging.sync(job, kept, refresh=True)
    m = staging.source(job)
    assert code == 0 and sorted(changed) == ["data/Temple.dat", "data/extra.dat", "region/r.3.3.mca"]
    assert "data/Temple.dat" not in m["files"] and m["files"]["data/extra.dat"]["copied"]
    assert m["files"]["region/r.3.3.mca"]["sha256"] == jobs.file_hash(os.path.join(w, "region", "r.3.3.mca"))
    assert not os.path.exists(os.path.join(job.data["args"]["staging"], "region", "r.3.3.mca"))
    assert staging.pending(m, w) == []


def _link_folder(real, link):
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(real), str(link))
    else:
        os.symlink(real, link)


def test_a_linked_dimension_is_refused_instead_of_left_out(mig, tmp_path, capsys):
    real = tmp_path / "nether-disk" / "DIM-1"
    real.parent.mkdir()
    os.rename(os.path.join(mig.world, "DIM-1"), real)
    _link_folder(real, Path(mig.world, "DIM-1"))
    capsys.readouterr()
    code, job = mig.preview()
    err = capsys.readouterr().err
    assert code == 1 and f"{os.path.join(mig.world, 'DIM-1')} is a link to another place" in err
    assert job.state() == jobs.FAILED and staging.source(job) is None


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a POSIX folder mode that this user cannot list")
def test_a_folder_the_tool_cannot_list_is_refused_instead_of_left_out(mig, capsys):
    hidden = Path(mig.world, "DIM7")
    (hidden / "region").mkdir(parents=True)
    os.chmod(hidden, 0o300)
    try:
        capsys.readouterr()
        code, _ = mig.preview()
    finally:
        os.chmod(hidden, 0o755)
    assert code == 1 and f"the tool cannot read the folder {hidden} (Permission denied)" in capsys.readouterr().err


def test_a_missing_folder_walks_empty():
    assert staging.walk(os.path.join("no", "such", "folder")) == ([], [])


def test_a_preview_without_room_on_the_world_drive_refuses_before_copying(mig, monkeypatch, capsys):
    free = {"bytes": staging.HEADROOM}
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda p: real(p)._replace(free=free["bytes"]))
    code, job = mig.preview()
    err = capsys.readouterr().err
    assert code == 1 and "left free for your world's own saves" in err and "Next: Free some space on that drive" in err, err
    assert "Traceback" not in err and tree(staging.paths(job)[1]) == {}
    free["bytes"] = 1 << 50
    assert mig.preview()[0] == 0


def test_a_drive_that_fills_up_during_the_copy_deletes_the_partial_copy(mig, monkeypatch, capsys):
    real, copied = staging._read_through, []

    def fills_up(src, dst):
        if copied:
            raise OSError(errno.ENOSPC, "No space left on device", dst)
        copied.append(dst)
        return real(src, dst)

    monkeypatch.setattr(staging, "_read_through", fills_up)
    code, job = mig.preview()
    err = capsys.readouterr().err
    assert code == 1 and "the drive filled up while writing" in err and "was deleted" in err and "Traceback" not in err, err
    assert os.path.isfile(copied[0]) is False and not os.path.exists(staging.paths(job)[1])
    monkeypatch.undo()
    assert cli.main(["jobs", "resume", job.id, *mig.opts()]) == 0 and job.state() == jobs.DONE
