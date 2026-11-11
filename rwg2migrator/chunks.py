from dataclasses import dataclass

import numpy as np

from . import nbt

ENDLESSIDS, NEID = "EndlessIDs", "NEID"
FORMATS = (ENDLESSIDS, NEID)
SECTION_KEYS = {
    ENDLESSIDS: ("BlockLight", "Blocks", "Y", "Data2", "SkyLight", "Add", "Data", "Data1High", "BlocksB2Hi", "BlocksB3"),
    NEID: ("BlockLight", "Data16", "Y", "SkyLight", "Blocks16"),
}
NIBBLE_BITS, BYTE_BITS = 4, 8
EID_ARRAYS = {
    "Blocks": ("ids", 0, BYTE_BITS),
    "Add": ("ids", 8, NIBBLE_BITS),
    "BlocksB2Hi": ("ids", 12, NIBBLE_BITS),
    "BlocksB3": ("ids", 16, BYTE_BITS),
    "Data": ("meta", 0, NIBBLE_BITS),
    "Data1High": ("meta", 4, NIBBLE_BITS),
    "Data2": ("meta", 8, BYTE_BITS),
}
EID_REQUIRED = frozenset(("Blocks", "Data", "Y", "SkyLight", "BlockLight"))
BIOME_KEY = {ENDLESSIDS: "Biomes16v2", NEID: "Biomes"}
BIOME_DTYPE = {ENDLESSIDS: "<u2", NEID: "u1"}
HEIGHT = 256
SECTION = 16
SIDE = 16
VOLUME = SECTION * SIDE * SIDE
NIBBLE = 15
BYTE_MAX = 255
U16_MAX = 65535


@dataclass
class Voxels:
    ids: np.ndarray
    meta: np.ndarray
    sky: np.ndarray
    light: np.ndarray


def nibbles(b):
    a = np.frombuffer(b, np.uint8)
    out = np.empty(a.size * 2, np.uint8)
    out[0::2], out[1::2] = a & NIBBLE, a >> 4
    return out


def pack_nibbles(a):
    a = a.astype(np.uint8)
    return (a[0::2] | (a[1::2] << 4)).tobytes()


def _raw(sec, key):
    return sec[key][1]


def _unpack(raw, bits):
    return (nibbles(raw) if bits == NIBBLE_BITS else np.frombuffer(raw, np.uint8)).astype(np.uint32)


def _pack(part, bits):
    part = part & ((1 << bits) - 1)
    return pack_nibbles(part) if bits == NIBBLE_BITS else part.astype(np.uint8).tobytes()


def section_format(sec):
    """NEID's fixed layout, or EndlessIDs: the vanilla arrays plus any of the higher arrays it adds only when a value needs one."""
    keys = frozenset(sec)
    if keys == frozenset(SECTION_KEYS[NEID]):
        return NEID
    return ENDLESSIDS if EID_REQUIRED <= keys <= frozenset(SECTION_KEYS[ENDLESSIDS]) else None


def sections(level):
    return level["Sections"][1][1] if "Sections" in level else []


def chunk_format(level):
    """ENDLESSIDS or NEID when every section and the biome array agree, else None."""
    found = {section_format(s) for s in sections(level)}
    found |= {f for f, key in BIOME_KEY.items() if key in level}
    return found.pop() if len(found) == 1 and None not in found else None


def _eid_field(sec, field):
    """Like EndlessIDs, read the arrays of one field from the lowest bits up and stop at the first one that is missing."""
    out = np.zeros(VOLUME, np.uint32)
    for key, shift, bits in [(k, s, b) for k, (f, s, b) in EID_ARRAYS.items() if f == field]:
        if key not in sec:
            break
        out |= _unpack(_raw(sec, key), bits) << shift
    return out


def section_ids(sec):
    if "Blocks16" in sec:
        return np.frombuffer(_raw(sec, "Blocks16"), ">u2").astype(np.uint32)
    return _eid_field(sec, "ids")


def decode_section(sec):
    meta = np.frombuffer(_raw(sec, "Data16"), ">u2").astype(np.uint32) if "Blocks16" in sec else _eid_field(sec, "meta")
    return Voxels(section_ids(sec), meta, nibbles(_raw(sec, "SkyLight")), nibbles(_raw(sec, "BlockLight")))


def _arrays(fmt, v, keep):
    if fmt == NEID:
        if v.ids.max(initial=0) > U16_MAX:
            raise ValueError(f"block id {v.ids.max()} does not fit the 16-bit block array of an old-format chunk")
        return {"Blocks16": v.ids.astype(">u2").tobytes(), "Data16": v.meta.astype(">u2").tobytes()}
    out = {}
    for key, (field, shift, bits) in EID_ARRAYS.items():
        part = getattr(v, field).astype(np.uint32) >> shift
        if shift == 0 or key in keep or part.any():
            out[key] = _pack(part, bits)
    return out


def encode_section(fmt, y, v, like=()):
    """A section compound. `like`, the section it replaces, keeps its arrays and key order, as the game keeps an array once made."""
    tags = {k: [nbt.BYTE_ARRAY, b] for k, b in _arrays(fmt, v, like).items()}
    tags |= {"Y": [nbt.BYTE, y], "SkyLight": [nbt.BYTE_ARRAY, pack_nibbles(v.sky)], "BlockLight": [nbt.BYTE_ARRAY, pack_nibbles(v.light)]}
    return {k: tags[k] for k in dict.fromkeys([*like, *SECTION_KEYS[fmt]]) if k in tags}


def block_ids(level):
    """Block ids as a (y, z, x) = (256, 16, 16) array."""
    out = np.zeros((HEIGHT, SIDE, SIDE), np.uint32)
    for s in sections(level):
        y = s["Y"][1] * SECTION
        out[y : y + SECTION] = section_ids(s).reshape(SECTION, SIDE, SIDE)
    return out


def biomes(level):
    for fmt, key in BIOME_KEY.items():
        if key in level:
            return np.frombuffer(level[key][1], BIOME_DTYPE[fmt]).astype(np.uint16).reshape(SIDE, SIDE)
    return None


def encode_biomes(fmt, grid):
    a = np.asarray(grid).reshape(-1)
    if fmt == NEID and a.max(initial=0) > BYTE_MAX:
        raise ValueError(f"biome id {a.max()} does not fit the byte biome array of an old-format chunk")
    return a.astype(BIOME_DTYPE[fmt]).tobytes()


def heightmap(level):
    return np.frombuffer(level["HeightMap"][1], ">i4").astype(np.int32).reshape(SIDE, SIDE)
