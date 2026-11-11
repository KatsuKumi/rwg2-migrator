import gzip
import os
import stat

import numpy as np
import pytest
from conftest import chunk_root, golden, lazy_chunk, tag, voxels, write_region

from rwg2migrator import chunks, nbt, region

BLOB = {
    "a": [nbt.BYTE, -3],
    "s": [nbt.SHORT, 300],
    "i": [nbt.INT, -70000],
    "l": [nbt.LONG, 1 << 40],
    "f": [nbt.FLOAT, b"\x7f\xa0\x00\x01"],
    "d": [nbt.DOUBLE, bytes(8)],
    "ba": [nbt.BYTE_ARRAY, b"\x00\xff\x10"],
    "st": [nbt.STRING, "x\udcc0\udc80y"],
    "li": [nbt.LIST, [nbt.COMPOUND, [{"q": [nbt.INT, 1]}, {}]]],
    "nums": [nbt.LIST, [nbt.INT, [1, -2, 3]]],
    "dbl": [nbt.LIST, [nbt.DOUBLE, [b"\x3f\xf0" + bytes(6)]]],
    "e": [nbt.LIST, [nbt.END, []]],
    "ia": [nbt.INT_ARRAY, (1).to_bytes(4, "big") + (-2).to_bytes(4, "big", signed=True)],
    "c": [nbt.COMPOUND, {"n": [nbt.STRING, "ok"]}],
}


def test_nbt_round_trip_every_tag_type():
    raw = nbt.write("root", {"Level": [nbt.COMPOUND, BLOB]})
    assert nbt.parse(raw) == ("root", {"Level": [nbt.COMPOUND, BLOB]})
    assert nbt.write(*nbt.parse(raw)) == raw


def test_nbt_plain_view(tmp_path):
    p = tmp_path / "x.dat"
    p.write_bytes(gzip.compress(nbt.write("", BLOB)))
    v = nbt.read_file(p)
    assert v["dbl"] == [1.0] and v["ia"] == [1, -2] and v["nums"] == [1, -2, 3] and v["li"] == [{"q": 1}, {}] and v["c"] == {"n": "ok"}


@pytest.mark.parametrize("raw", [b"\x01\x00\x00", nbt.write("", BLOB)[:-3], nbt.write("", BLOB) + b"\x00", b"\x0a\x00\x00\x0d\x00\x00\x00"])
def test_nbt_rejects_bad_input(raw):
    with pytest.raises(nbt.NBTError):
        nbt.parse(raw)


@pytest.mark.parametrize("fmt", chunks.FORMATS)
def test_section_round_trip_random_voxels(fmt):
    v = voxels(7)
    sec = chunks.encode_section(fmt, 5, v)
    assert tuple(sec) == tuple(k for k in chunks.SECTION_KEYS[fmt] if k != "BlocksB3") and chunks.section_format(sec) == fmt
    back = chunks.decode_section(sec)
    for a, b in zip((v.ids, v.meta, v.sky, v.light), (back.ids, back.meta, back.sky, back.light), strict=True):
        assert np.array_equal(a, b)
    assert chunks.encode_section(fmt, 5, back) == sec


LAZY = [
    (255, 15, ()),
    (256, 15, ("Add",)),
    (4096, 15, ("Add", "BlocksB2Hi")),
    (65536, 15, ("Add", "BlocksB2Hi", "BlocksB3")),
    (255, 16, ("Data1High",)),
    (4096, 256, ("Add", "BlocksB2Hi", "Data1High", "Data2")),
]


@pytest.mark.parametrize("case", LAZY)
def test_endlessids_writes_a_higher_array_only_when_a_value_needs_it(case):
    top_id, top_meta, extra = case
    v = voxels(3)
    v.ids, v.meta = v.ids.astype(np.uint32) % (top_id + 1), v.meta % (top_meta + 1)
    v.ids[0], v.meta[0] = top_id, top_meta
    sec = chunks.encode_section(chunks.ENDLESSIDS, 2, v)
    assert set(sec) == chunks.EID_REQUIRED | set(extra) and chunks.section_format(sec) == chunks.ENDLESSIDS
    back = chunks.decode_section(sec)
    assert np.array_equal(back.ids, v.ids) and np.array_equal(back.meta, v.meta)
    game_order = dict(reversed(sec.items()))
    assert list(chunks.encode_section(chunks.ENDLESSIDS, 2, back, like=game_order).items()) == list(game_order.items())


def test_kept_arrays_stay_and_reads_stop_at_the_first_missing_array():
    v = voxels(4)
    v.ids, v.meta = v.ids % 256, v.meta % 16
    like = chunks.encode_section(chunks.ENDLESSIDS, 0, voxels(4))
    assert tuple(chunks.encode_section(chunks.ENDLESSIDS, 0, v, like=like)) == tuple(like)
    v.ids[0] = 4096
    sec = chunks.encode_section(chunks.ENDLESSIDS, 0, v)
    del sec["Add"]
    assert chunks.section_format(sec) == chunks.ENDLESSIDS and chunks.decode_section(sec).ids[0] == 0
    with pytest.raises(ValueError, match="does not fit"):
        chunks.encode_section(chunks.NEID, 0, chunks.Voxels(v.ids.astype(np.uint32) + 65536, v.meta, v.sky, v.light))


@pytest.mark.parametrize("keys", [{"Blocks16", "Add"}, {"Blocks", "Data", "Y", "SkyLight", "BlockLight", "Data16"}, {"Blocks", "Y"}])
def test_unknown_section_layouts(keys):
    assert chunks.section_format(dict.fromkeys(keys)) is None


def assert_lazy_sections_round_trip(raw, root):
    level = root["Level"][1]
    layouts = {len(s) for s in chunks.sections(level)}
    assert layouts == {5, 6, 7, 9} and chunks.chunk_format(level) == chunks.ENDLESSIDS
    for s in chunks.sections(level):
        v = chunks.decode_section(s)
        assert set(chunks.encode_section(chunks.ENDLESSIDS, s["Y"][1], v)) == set(s)
        s |= chunks.encode_section(chunks.ENDLESSIDS, s["Y"][1], v, like=s)
    assert nbt.write("", root) == raw


def test_chunk_with_lazy_sections_round_trips():
    raw, root = lazy_chunk()
    assert_lazy_sections_round_trip(raw, root)
    assert chunks.block_ids(root["Level"][1]).max() >= 1 << 12


@pytest.mark.fixture
def test_real_gtnh_29_chunk_with_lazy_sections_round_trips():
    raw = gzip.decompress(golden("eid-lazy-chunk-74-40.nbt.gz").read_bytes())
    root = nbt.parse(raw)[1]
    assert_lazy_sections_round_trip(raw, root)
    assert chunks.block_ids(root["Level"][1]).max() == 11208


def test_chunk_format_and_biomes():
    root = chunk_root(0, 0, chunks.NEID)["Level"][1]
    assert chunks.chunk_format(root) == chunks.NEID and chunks.biomes(root)[3, 3] == 4
    mixed = dict(root) | {"Biomes16v2": tag(nbt.BYTE_ARRAY, bytes(512))}
    assert chunks.chunk_format(mixed) is None
    assert chunks.chunk_format(chunk_root(0, 0)["Level"][1]) == chunks.ENDLESSIDS
    assert np.frombuffer(chunks.encode_biomes(chunks.ENDLESSIDS, np.full(256, 300)), "<u2")[0] == 300
    with pytest.raises(ValueError, match="does not fit"):
        chunks.encode_biomes(chunks.NEID, np.full(256, 300))


def test_block_ids_stacks_sections():
    ids = chunks.block_ids(chunk_root(0, 0)["Level"][1])
    assert ids.shape == (256, 16, 16) and ids[62, 3, 3] == 1 and ids[63, 3, 3] == 2 and ids[64, 3, 3] == 31 and ids[65].sum() == 0


def _records():
    roots = {(-32, 0): chunk_root(-32, 0), (-1, 5): chunk_root(-1, 5, chunks.NEID), (-20, 31): chunk_root(-20, 31)}
    return {region.slot_index(*c): (region.deflate(nbt.write("", r)), 1700000000 + i) for i, (c, r) in enumerate(roots.items())}


def test_region_build_check_and_tamper():
    kept = _records()
    data = region.build(kept)
    assert region.check(data, kept) is None and len(data) % region.SECTOR == 0
    assert region.records(data) == kept and region.build(region.records(data)) == data
    assert "slots differ" in region.check(data, {i: kept[i] for i in list(kept)[:1]})
    assert "whole number" in region.check(data[:-1], kept)
    flipped = bytearray(data)
    flipped[-region.SECTOR + 10] ^= 1
    assert region.check(bytes(flipped), kept)
    pos = {region.slot_pos(-1, 0, i) for i in kept}
    assert pos == {(-32, 0), (-1, 5), (-20, 31)}


def test_region_bad_header_entry_reads_as_none():
    kept = _records()
    data = bytearray(region.build(kept))
    i = min(kept)
    data[4 * i : 4 * i + 4] = (1 << 8 | 1).to_bytes(4, "big")
    assert region.records(bytes(data))[i][0] is None


def test_inflate_errors_and_coords(tmp_path):
    with pytest.raises(region.RegionError):
        region.inflate(b"\x00\x00\x00\x05\x07abcd")
    with pytest.raises(region.RegionError):
        region.inflate(b"\x00\x00\x00\x05\x02abcd")
    assert region.coords("/x/r.-3.12.mca") == (-3, 12)
    with pytest.raises(region.RegionError):
        region.coords("/x/level.dat")


def test_write_atomic_over_a_read_only_file_keeps_it_read_only(tmp_path):
    p = tmp_path / "rwg2.cfg"
    stale = tmp_path / ("rwg2.cfg" + region.TMP_SUFFIX)
    p.write_bytes(b"old")
    stale.write_bytes(b"left by a crash")
    for f in (p, stale):
        os.chmod(f, stat.S_IREAD)
    region.write_atomic(str(p), b"new")
    assert p.read_bytes() == b"new" and not os.access(p, os.W_OK) and os.listdir(tmp_path) == ["rwg2.cfg"]


def test_write_atomic_leaves_no_tmp_when_the_rename_fails(tmp_path, monkeypatch):
    p = tmp_path / "a.cfg"
    p.write_bytes(b"old")
    os.chmod(p, stat.S_IREAD)

    def refuse(*_):
        raise PermissionError("held by another program")

    monkeypatch.setattr(region.os, "replace", refuse)
    with pytest.raises(PermissionError):
        region.write_atomic(str(p), b"new")
    assert p.read_bytes() == b"old" and not os.access(p, os.W_OK) and os.listdir(tmp_path) == ["a.cfg"]


def test_copy_tree_drops_read_only_and_rmtree_removes_it(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "locked.cfg").write_text("x")
    os.chmod(src / "sub" / "locked.cfg", stat.S_IREAD)
    os.chmod(src / "sub", stat.S_IREAD | stat.S_IEXEC)
    region.copy_tree(str(src), str(dst))
    assert (dst / "sub" / "locked.cfg").read_text() == "x"
    assert os.access(dst / "sub" / "locked.cfg", os.W_OK) and os.access(dst / "sub", os.W_OK)
    region.rmtree(str(src))
    region.rmtree(str(src))
    assert os.listdir(tmp_path) == ["dst"]
    with pytest.raises(FileNotFoundError):
        region.copy_tree(str(src), str(dst))


@pytest.mark.skipif(os.name == "nt", reason="Windows keeps only the read-only flag, which the read-only test covers")
def test_write_atomic_replaces_and_keeps_mode(tmp_path):
    p = str(tmp_path / "r.0.0.mca")
    write_region(p, {(0, 0): chunk_root(0, 0)})
    os.chmod(p, 0o640)
    region.write_atomic(p, b"new")
    assert (tmp_path / "r.0.0.mca").read_bytes() == b"new" and os.stat(p).st_mode & 0o777 == 0o640
    assert not os.path.exists(p + region.TMP_SUFFIX)
