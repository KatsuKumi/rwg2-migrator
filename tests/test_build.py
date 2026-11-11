import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

import build  # noqa: E402

LINUX = [
    "_internal/libssl.so.3", "_internal/libcrypto.so.3", "_internal/libz.so.1", "_internal/libbz2.so.1.0", "_internal/liblzma.so.5",
    "_internal/libffi.so.8", "_internal/libexpat.so.1", "_internal/libstdc++.so.6", "_internal/libgcc_s.so.1",
    "_internal/libtinfo.so.6", "_internal/libpython3.12.so.1.0", "_internal/libavif-c6062d5f.so.16.4.2",
    "_internal/python3.12/lib-dynload/_decimal.cpython-312-x86_64-linux-gnu.so",
    "_internal/python3.12/lib-dynload/_json.cpython-312-x86_64-linux-gnu.so",
    "_internal/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so",
    "_internal/psutil/_psutil_linux.abi3.so", "rwg2-migrator", "README.md",
]  # fmt: skip
WINDOWS = [
    "_internal/libssl-3-x64.dll", "_internal/libffi-8.dll", "_internal/pyexpat.pyd", "_internal/_bz2.pyd", "_internal/_lzma.pyd",
    "_internal/_decimal.pyd", "_internal/select.pyd", "_internal/python312.dll", "_internal/VCRUNTIME140.dll",
    "_internal/api-ms-win-crt-heap-l1-1-0.dll", "_internal/numpy.libs/libscipy_openblas64_-43e11ff0749b8cbe0a615c9cf6737e0e.dll",
    "_internal/PIL/_imaging.cp312-win_amd64.pyd",
]  # fmt: skip


def test_every_bundled_native_file_has_its_notice_and_readline_is_refused():
    linux = build.components(LINUX, {})
    assert {k: len(v) for k, v in linux.items()} == {
        "openssl": 2, "zlib": 1, "bzip2": 1, "xz": 1, "libffi": 1, "expat": 1, "gcc-runtime": 2, "ncurses": 1, "python": 2,
        "numpy or pillow": 1, "mpdecimal": 1, "numpy": 1, "psutil": 1,
    }  # fmt: skip
    windows = build.components(WINDOWS, {})
    assert set(windows) == {"openssl", "libffi", "expat", "bzip2", "xz", "mpdecimal", "python", "numpy", "pillow"}
    assert all((build.TEXTS / f"{name}.txt").is_file() for name in {*linux, *windows} - {"python", "numpy or pillow", *build.BUNDLED})
    with pytest.raises(SystemExit, match="GNU readline"):
        build.components(["_internal/python3.12/lib-dynload/readline.cpython-312-x86_64-linux-gnu.so"], {})
    with pytest.raises(SystemExit, match="no license notice covers _internal/libfoo.so.1"):
        build.components([*LINUX, "_internal/libfoo.so.1"], {})


MACOS_PILLOW = [
    "libbrotlicommon.1.2.0.dylib", "libbrotlidec.1.2.0.dylib", "liblcms2.2.dylib", "libpng16.16.dylib", "libxcb.1.1.0.dylib",
    "libjpeg.62.4.0.dylib", "libfreetype.6.dylib", "libharfbuzz.0.dylib", "libsharpyuv.0.dylib", "libopenjp2.2.5.4.dylib",
    "libavif.16.4.2.dylib", "libwebp.7.dylib", "libXau.6.dylib", "libwebpmux.3.dylib", "libwebpdemux.2.dylib", "libtiff.6.dylib",
]  # fmt: skip


def test_a_wheel_library_moved_to_the_top_of_the_bundle_is_still_covered_by_its_wheel():
    moved = [f"_internal/{name}" for name in MACOS_PILLOW]
    with pytest.raises(SystemExit, match="no license notice covers _internal/libbrotlicommon.1.2.0.dylib"):
        build.components(moved, {})
    wheels = dict.fromkeys(MACOS_PILLOW, "pillow") | {"libscipy_openblas64_.dylib": "numpy", "libgfortran.5.dylib": "numpy"}
    found = build.components([*moved, "_internal/libgfortran.5.dylib", "_internal/numpy/.dylibs/libscipy_openblas64_.dylib"], wheels)
    assert {k: len(v) for k, v in found.items()} == {"pillow": len(MACOS_PILLOW), "numpy": 2}
    here = build.shipped()
    assert set(here.values()) == set(build.WHEELS) and all(build.NATIVE.search(name) for name in here)


@pytest.mark.skipif(os.name == "nt", reason="PyInstaller makes these links on Linux only")
def test_the_links_to_wheel_libraries_are_dropped_and_other_links_stay(tmp_path):
    inside = tmp_path / "_internal"
    (inside / "numpy.libs").mkdir(parents=True)
    (inside / "numpy.libs" / "libopenblas-1234abcd.so").write_bytes(b"lib")
    (inside / "libz.so.1").write_bytes(b"z")
    (inside / "libopenblas-1234abcd.so").symlink_to("numpy.libs/libopenblas-1234abcd.so")
    (inside / "libz.so").symlink_to("libz.so.1")
    assert build.drop_links(tmp_path) == 1
    assert sorted(p.name for p in inside.iterdir()) == ["libz.so", "libz.so.1", "numpy.libs"]


def test_the_built_zips_are_covered_by_their_notices():
    zips = sorted(build.ROOT.glob("dist/rwg2-migrator-*.zip"))
    if not zips:
        pytest.skip("no zip built in dist/ (run dev/build.py)")
    for path in zips:
        names = [n.split("/", 1)[1] for n in zipfile.ZipFile(path).namelist() if "/" in n]
        found = build.components(names)
        licenses = {n.split("/")[1] for n in names if n.startswith("licenses/") and n.count("/") > 1 and not n.endswith("/")}
        assert set(found) - {"numpy or pillow"} <= licenses | {"python"} and "python" in licenses, (path.name, found.keys(), licenses)
        if f"-{build.__version__}-" in path.name:
            assert {text.stem for text in build.TEXTS.glob("*.txt")} <= licenses, (path.name, licenses)


def test_a_linux_build_that_needs_a_newer_glibc_than_the_oldest_supported_is_refused(tmp_path, monkeypatch):
    app = tmp_path / build.APP
    (app / "_internal").mkdir(parents=True)
    (app / "_internal" / "libpython3.12.so.1.0").write_bytes(b"\x7fELF GLIBC_2.2.5\0GLIBC_2.17\0GLIBCXX_3.4.32\0GLIBC_PRIVATE\0")
    (app / "README.md").write_text("GLIBC_9.9 in a text file does not count")
    assert build.glibc_needed(app) == ((2, 17), "_internal/libpython3.12.so.1.0")
    monkeypatch.setattr(build.sys, "platform", "linux")
    build.check_glibc(app, any_glibc=False)
    (app / build.APP).write_bytes(b"\x7fELF GLIBC_2.38\0")
    assert build.glibc_needed(app) == ((2, 38), build.APP)
    with pytest.raises(SystemExit, match=r"needs glibc 2\.38 \(for rwg2-migrator\)"):
        build.check_glibc(app, any_glibc=False)
    build.check_glibc(app, any_glibc=True)
    monkeypatch.setattr(build.sys, "platform", "win32")
    build.check_glibc(app, any_glibc=False)
