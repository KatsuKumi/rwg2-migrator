"""Named Binary Tag reader and writer that round-trips Minecraft files byte for byte.

A tag is held as ``[type, value]``. A compound value is a ``dict`` of name -> tag, kept in file order. A list value
is ``[element_type, [values]]``. Floats and doubles stay raw bytes, byte and int arrays stay ``bytes``.
"""

import gzip
import os
import struct
import zlib

END, BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, BYTE_ARRAY, STRING, LIST, COMPOUND, INT_ARRAY = range(12)
TEXT = "surrogateescape"
GZIP_MAGIC = b"\x1f\x8b"
_NUMBER = {BYTE: "b", SHORT: "h", INT: "i", LONG: "q"}
_NUMBER_STRUCT = {t: struct.Struct(">" + c) for t, c in _NUMBER.items()}
_RAW = {FLOAT: 4, DOUBLE: 8}
_RAW_FORMAT = {FLOAT: ">f", DOUBLE: ">d"}
_ARRAY_WIDTH = {BYTE_ARRAY: 1, INT_ARRAY: 4}
_LENGTH = struct.Struct(">i")
_NAME = struct.Struct(">H")


class NBTError(ValueError):
    pass


def _read_number(t):
    s = _NUMBER_STRUCT[t]

    def read(b, p):
        return s.unpack_from(b, p)[0], p + s.size

    return read


def _read_raw(t):
    n = _RAW[t]

    def read(b, p):
        return bytes(b[p : p + n]), p + n

    return read


def _read_array(t):
    width = _ARRAY_WIDTH[t]

    def read(b, p):
        n = _LENGTH.unpack_from(b, p)[0] * width
        if n < 0 or p + 4 + n > len(b):
            raise NBTError(f"array length {n} out of range at {p}")
        return bytes(b[p + 4 : p + 4 + n]), p + 4 + n

    return read


def _read_string(b, p):
    n = _NAME.unpack_from(b, p)[0]
    return bytes(b[p + 2 : p + 2 + n]).decode("utf-8", TEXT), p + 2 + n


def _read_list(b, p):
    et, n = b[p], _LENGTH.unpack_from(b, p + 1)[0]
    p += 5
    if n <= 0:
        return [et, []], p
    if et in _NUMBER:
        s = struct.Struct(f">{n}{_NUMBER[et]}")
        return [et, list(s.unpack_from(b, p))], p + s.size
    read = _reader(et)
    out = []
    for _ in range(n):
        v, p = read(b, p)
        out.append(v)
    return [et, out], p


def _read_compound(b, p):
    out = {}
    while b[p]:
        t, n = b[p], _NAME.unpack_from(b, p + 1)[0]
        name = bytes(b[p + 3 : p + 3 + n]).decode("utf-8", TEXT)
        v, p = _reader(t)(b, p + 3 + n)
        out[name] = [t, v]
    return out, p + 1


_READ = {
    **{t: _read_number(t) for t in _NUMBER},
    **{t: _read_raw(t) for t in _RAW},
    **{t: _read_array(t) for t in _ARRAY_WIDTH},
    STRING: _read_string,
    LIST: _read_list,
    COMPOUND: _read_compound,
}


def _reader(t):
    try:
        return _READ[t]
    except KeyError:
        raise NBTError(f"bad NBT tag type {t}") from None


def _write_number(t):
    s = _NUMBER_STRUCT[t]

    def write(v, out):
        out += s.pack(v)

    return write


def _write_raw(v, out):
    out += v


def _write_array(t):
    width = _ARRAY_WIDTH[t]

    def write(v, out):
        out += _LENGTH.pack(len(v) // width) + v

    return write


def _write_string(v, out):
    e = v.encode("utf-8", TEXT)
    out += _NAME.pack(len(e)) + e


def _write_list(v, out):
    et, items = v
    out += bytes([et]) + _LENGTH.pack(len(items))
    if not items:
        return
    if et in _NUMBER:
        out += struct.pack(f">{len(items)}{_NUMBER[et]}", *items)
        return
    write = _writer(et)
    for item in items:
        write(item, out)


def _write_compound(v, out):
    for name, (t, value) in v.items():
        e = name.encode("utf-8", TEXT)
        out += bytes([t]) + _NAME.pack(len(e)) + e
        _writer(t)(value, out)
    out += b"\0"


_WRITE = {
    **{t: _write_number(t) for t in _NUMBER},
    **dict.fromkeys(_RAW, _write_raw),
    **{t: _write_array(t) for t in _ARRAY_WIDTH},
    STRING: _write_string,
    LIST: _write_list,
    COMPOUND: _write_compound,
}


def _writer(t):
    try:
        return _WRITE[t]
    except KeyError:
        raise NBTError(f"bad NBT tag type {t}") from None


def parse(raw):
    """Return (root name, root compound) of uncompressed NBT bytes."""
    try:
        if raw[0] != COMPOUND:
            raise NBTError("NBT root is not a compound")
        name, p = _read_string(raw, 1)
        value, end = _read_compound(raw, p)
    except (IndexError, struct.error) as e:
        raise NBTError(f"truncated NBT: {e}") from e
    if end != len(raw):
        raise NBTError(f"NBT has {len(raw) - end} trailing bytes")
    return name, value


def write(name, value):
    out = bytearray([COMPOUND])
    _write_string(name, out)
    _write_compound(value, out)
    return bytes(out)


def unpack(raw):
    return gzip.decompress(raw) if raw[:2] == GZIP_MAGIC else raw


def read_file(path):
    """Plain Python view of an NBT file (gzip or not); a file that does not decode raises NBTError naming it."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return plain(COMPOUND, parse(unpack(raw))[1])
    except (NBTError, zlib.error, EOFError, gzip.BadGzipFile, struct.error) as e:
        raise NBTError(f"{os.path.basename(path)} is not a valid NBT file: {e}") from e


def plain(t, v):
    if t == COMPOUND:
        return {k: plain(tt, vv) for k, (tt, vv) in v.items()}
    if t == LIST:
        return [plain(v[0], x) for x in v[1]]
    if t in _RAW:
        return struct.unpack(_RAW_FORMAT[t], v)[0]
    if t == INT_ARRAY:
        return list(struct.unpack(f">{len(v) // 4}i", v))
    return v
