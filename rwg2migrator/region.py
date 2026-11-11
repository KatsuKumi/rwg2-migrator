import gzip
import os
import re
import shutil
import stat
import zlib
from contextlib import suppress

SECTOR = 4096
SLOTS = 1024
SIDE = 32
HEADER = 2 * SECTOR
MAX_SECTORS = 255
GZIP, ZLIB = 1, 2
INFLATE = {GZIP: gzip.decompress, ZLIB: zlib.decompress}
NAME = re.compile(r"r\.(-?\d+)\.(-?\d+)\.mca")
TMP_SUFFIX = ".rwg2tmp"


class RegionError(ValueError):
    pass


def coords(path):
    m = NAME.fullmatch(os.path.basename(path))
    if not m:
        raise RegionError(f"{path} is not a region file name")
    return int(m.group(1)), int(m.group(2))


def slot_index(cx, cz):
    return (cx & (SIDE - 1)) + (cz & (SIDE - 1)) * SIDE


def slot_pos(rx, rz, i):
    return rx * SIDE + i % SIDE, rz * SIDE + i // SIDE


def slots(data):
    """Yield (slot, sector, sector count, timestamp) for every used header entry."""
    if len(data) < HEADER:
        return
    for i in range(SLOTS):
        loc = int.from_bytes(data[4 * i : 4 * i + 3], "big")
        if loc:
            yield i, loc, data[4 * i + 3], int.from_bytes(data[SECTOR + 4 * i : SECTOR + 4 * i + 4], "big")


def record(data, loc, count):
    """The chunk record (4-byte length, compression byte, payload) or None if the header entry is bad."""
    start = loc * SECTOR
    length = int.from_bytes(data[start : start + 4], "big", signed=True)
    if loc < HEADER // SECTOR or count < 1 or length < 1 or length + 4 > count * SECTOR or start + 4 + length > len(data):
        return None
    return data[start : start + 4 + length]


def inflate(rec):
    try:
        return INFLATE[rec[4]](rec[5:])
    except KeyError:
        raise RegionError(f"unknown chunk compression {rec[4]}") from None
    except (zlib.error, EOFError, gzip.BadGzipFile) as e:
        raise RegionError(f"chunk payload does not decompress: {e}") from e


def deflate(raw):
    body = zlib.compress(raw)
    return (len(body) + 1).to_bytes(4, "big") + bytes([ZLIB]) + body


def records(data):
    """{slot: (record, timestamp)}; a bad header entry maps to (None, timestamp)."""
    return {i: (record(data, loc, count), ts) for i, loc, count, ts in slots(data)}


def build(kept):
    """Region bytes holding kept = {slot: (record, timestamp)}, records packed in slot order."""
    head, body, sector = bytearray(HEADER), bytearray(), HEADER // SECTOR
    for i in sorted(kept):
        rec, ts = kept[i]
        n = -(-len(rec) // SECTOR)
        if n > MAX_SECTORS:
            raise RegionError(f"slot {i} needs {n} sectors, more than a region header can address")
        head[4 * i : 4 * i + 4] = (sector << 8 | n).to_bytes(4, "big")
        head[SECTOR + 4 * i : SECTOR + 4 * i + 4] = ts.to_bytes(4, "big")
        body += rec + bytes(n * SECTOR - len(rec))
        sector += n
    return bytes(head + body)


def check(data, kept):
    """None if data is a valid region holding exactly kept, else the first problem found."""
    if len(data) % SECTOR or len(data) < HEADER:
        return f"file size {len(data)} is not a whole number of sectors past the header"
    used, got = set(), {}
    for i, loc, count, ts in slots(data):
        span = set(range(loc, loc + count))
        rec = record(data, loc, count)
        if rec is None or loc + count > len(data) // SECTOR or used & span:
            return f"slot {i} has a bad or overlapping header entry"
        used |= span
        got[i] = (rec, ts)
    if got.keys() != kept.keys():
        return f"slots differ from the kept set: extra {sorted(got.keys() - kept.keys())}, missing {sorted(kept.keys() - got.keys())}"
    bad = [i for i in kept if got[i] != kept[i]]
    return f"payload or timestamp differs in slots {bad}" if bad else None


def write_atomic(path, data):
    """Replace path in one rename, keeping its mode: a read-only file stays read-only on every OS."""
    tmp = path + TMP_SUFFIX
    remove_file(tmp)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(path):
            shutil.copymode(path, tmp)
        _replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            remove_file(tmp)
        raise
    fsync_dir(os.path.dirname(os.path.abspath(path)))


def _replace(tmp, path):
    locked = os.name == "nt" and os.path.exists(path) and not os.stat(path).st_mode & stat.S_IWRITE
    if not locked:
        os.replace(tmp, path)
        return
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)  # Windows refuses to rename over a read-only file
    try:
        os.replace(tmp, path)
    finally:
        os.chmod(path, stat.S_IREAD)


def _add_write(path):
    os.chmod(path, os.stat(path).st_mode | stat.S_IRWXU)


def remove_file(path):
    """Delete path if it exists, read-only or not."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except PermissionError:
        _add_write(path)
        os.remove(path)


def rmtree(path):
    """Delete a folder if it exists; read-only files and folders inside it do not stop it, other failures raise."""

    def retry_writable(func, failed, exc):
        if not isinstance(exc, PermissionError):
            raise exc
        if failed != path:
            _add_write(os.path.dirname(failed))
        _add_write(failed)
        func(failed)

    if os.path.lexists(path):
        shutil.rmtree(path, onexc=retry_writable)


def _raise(e):
    raise e


def copy_tree(src, dst):
    """Copy files and folders but never their modes, so a read-only source still gives a copy the tool can change and delete."""
    for root, _, files in os.walk(src, onerror=_raise, followlinks=True):
        out = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(out, exist_ok=True)
        for name in files:
            shutil.copyfile(os.path.join(root, name), os.path.join(out, name))


def fsync_dir(path):
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
