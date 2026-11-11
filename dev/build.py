"""Builds the one-folder app with PyInstaller for this OS and zips it as dist/rwg2-migrator-<version>-<os>-<arch>.zip, with the
license texts of everything it bundles, checks that the built program starts, then lists every zip of dist/ in dist/SHA256SUMS.

  uv run --group build python dev/build.py"""

import argparse
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

from rwg2migrator import __version__
from rwg2migrator.jobs import file_hash

ROOT = Path(__file__).resolve().parent.parent
APP = "rwg2-migrator"
BUNDLED = ("numpy", "pillow", "psutil", "pyinstaller")
LICENSE_WORDS = ("LICEN", "COPYING", "NOTICE")
OS_NAMES = {"win32": "windows", "linux": "linux", "darwin": "macos"}
ARCHES = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}
TOP_FILES = ("LICENSE", "THIRD_PARTY_NOTICES", "README.md")
SUMS = "SHA256SUMS"
TEXTS = ROOT / "dev" / "licenses"
NATIVE = re.compile(r"\.(so(\.\d+)*|dll|pyd|dylib)$", re.IGNORECASE)
FROM_WHEELS = re.compile(r"-[0-9a-f]{8}[.-]")
WHEELS = ("numpy", "pillow", "psutil")
WHEEL_DIRS = {"numpy": "numpy", "numpy.libs": "numpy", "PIL": "pillow", "pillow.libs": "pillow", "psutil": "psutil"}
FORBIDDEN = re.compile(r"readline", re.IGNORECASE)
COMPONENTS = (
    (r"^(lib)?(ssl|crypto)\b|^_ssl\.|^_hashlib\.", "openssl"),
    (r"^libz\.|^zlib", "zlib"),
    (r"^libbz2|^_bz2\.", "bzip2"),
    (r"^liblzma|^_lzma\.", "xz"),
    (r"^libffi|^_ctypes\.", "libffi"),
    (r"^libexpat|^pyexpat\.|^_elementtree\.", "expat"),
    (r"^libmpdec|^_decimal\.", "mpdecimal"),
    (r"^(lib)?sqlite3|^_sqlite3\.", "sqlite"),
    (r"^libtinfo|^libncurses", "ncurses"),
    (r"^libstdc\+\+|^libgcc_s", "gcc-runtime"),
    (r"^libpython|^python3\d*\.dll|^vcruntime|^msvcp|^ucrtbase|^api-ms-win-|\.cpython-3\d+|^_?[a-z0-9_]+\.pyd$", "python"),
)
START_TIMEOUT = 120
GLIBC = re.compile(rb"GLIBC_(\d+)\.(\d+)")
OLDEST_GLIBC = (2, 35)


def target():
    return f"{OS_NAMES[sys.platform]}-{ARCHES[platform.machine().lower()]}"


def pyinstaller(work):
    import PyInstaller.__main__

    PyInstaller.__main__.run([
        str(ROOT / "dev" / "launcher.py"), "--name", APP, "--onedir", "--console", "--noconfirm", "--clean",
        "--distpath", str(work / "dist"), "--workpath", str(work / "build"), "--specpath", str(work),
        "--collect-data", "rwg2migrator", "--collect-submodules", "rwg2migrator",
        "--exclude-module", "tkinter", "--exclude-module", "readline", "--python-option", "X utf8",
    ])  # fmt: skip
    return work / "dist" / APP


@cache
def shipped():
    """{file name: distribution} of the native files the bundled wheels ship. PyInstaller moves some of them out of their wheel's
    folder (Pillow's libraries on macOS go to the top of the bundle), so the folder alone does not always name the wheel."""
    wheels = {name: importlib.metadata.distribution(name).files or [] for name in WHEELS}
    return {f.name: name for name, files in wheels.items() for f in files if NATIVE.search(f.name)}


def component(rel, wheels):
    """The licenses/ folder whose text covers one bundled native file (a path relative to the app folder, / separated); wheels:
    {file name: distribution} of what the bundled wheels ship."""
    parts, name = rel.split("/"), rel.rsplit("/", 1)[-1]
    if FORBIDDEN.search(name):
        raise SystemExit(f"{rel} is GNU readline (GPL-3.0-only) or links it: it must not be bundled")
    wheel = next((WHEEL_DIRS[p] for p in parts[:-1] if p in WHEEL_DIRS), None) or wheels.get(name)
    if wheel or FROM_WHEELS.search(name):
        return wheel or "numpy or pillow"
    return next((folder for pattern, folder in COMPONENTS if re.search(pattern, name, re.IGNORECASE)), None)


def components(rels, wheels=None):
    """{licenses/ folder: bundled native files} of every native file; refuses a file no notice covers. wheels: what the bundled
    wheels ship (default: as installed here)."""
    out, unknown = {}, []
    wheels = shipped() if wheels is None else wheels
    for rel in (r for r in rels if NATIVE.search(r)):
        folder = component(rel, wheels)
        if folder is None:
            unknown.append(rel)
        else:
            out.setdefault(folder, []).append(rel)
    if unknown:
        raise SystemExit(f"no license notice covers {', '.join(unknown)}: add its text to dev/licenses and a row to THIRD_PARTY_NOTICES")
    return out


def python_license():
    base, version = Path(sys.base_prefix), f"python{sys.version_info.major}.{sys.version_info.minor}"
    return next((p for p in (base / "LICENSE.txt", base / "lib" / version / "LICENSE.txt") if p.is_file()), None)


def copy_licenses(folder):
    """Each bundled distribution's license files, and CPython's, under folder/licenses/<name>/; refuses to ship without them."""
    dest = folder / "licenses"
    for name in BUNDLED:
        dist = importlib.metadata.distribution(name)
        files = [f for f in dist.files if any(w in f.name.upper() for w in LICENSE_WORDS)]
        if not files:
            raise SystemExit(f"no license file found in the {name} distribution")
        for f in files:
            out = dest / name / Path(*f.parts[1:])
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f.locate(), out)
    cpython = python_license()
    if cpython is None:
        raise SystemExit(f"CPython's LICENSE.txt was not found under {sys.base_prefix}")
    (dest / "python").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cpython, dest / "python" / "LICENSE.txt")
    components([p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()])
    for text in sorted(TEXTS.glob("*.txt")):
        (dest / text.stem).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(text, dest / text.stem / "LICENSE.txt")


def glibc_needed(folder):
    """((major, minor), file) of the newest glibc a bundled file asks for: the oldest Linux the app starts on. None when none asks.
    PyInstaller bundles this machine's libstdc++ and, with a distribution's Python, its libpython."""
    files = [p for p in folder.rglob("*") if p.is_file() and (NATIVE.search(p.name) or p.name == APP)]
    found = (((int(a), int(b)), p.relative_to(folder).as_posix()) for p in files for a, b in GLIBC.findall(p.read_bytes()))
    return max(found, default=None)


def check_glibc(folder, any_glibc):
    need = glibc_needed(folder) if sys.platform == "linux" else None
    if need and need[0] > OLDEST_GLIBC and not any_glibc:
        (major, minor), name = need
        raise SystemExit(
            f"this build needs glibc {major}.{minor} (for {name}), so it would not start on Ubuntu 22.04 or Debian 12, whose limit is "
            f"{OLDEST_GLIBC[0]}.{OLDEST_GLIBC[1]}. Build the release zip with the release workflow, or on a system that old; pass "
            "--any-glibc to build for systems as new as this one only"
        )


def drop_links(folder):
    """Remove the links PyInstaller puts at the top of _internal to the libraries numpy.libs and pillow.libs hold: every module
    finds those through its own search path, and the zip would store each library twice. Returns how many were removed."""
    links = [p for p in (folder / "_internal").iterdir() if p.is_symlink() and p.resolve().parent.name.endswith(".libs")]
    for link in links:
        link.unlink()
    return len(links)


def unresolved(folder):
    """Libraries a bundled file needs that the loader would not find, as ldd sees it with the app's own library path."""
    files = [str(p) for p in folder.rglob("*") if p.is_file() and NATIVE.search(p.name)]
    env = os.environ | {"LD_LIBRARY_PATH": str(folder / "_internal")}
    out = subprocess.run(["ldd", *files], capture_output=True, text=True, env=env, check=False).stdout
    return sorted({line.split()[0] for line in out.splitlines() if "not found" in line})


def slim(folder):
    """On Linux, with ldd at hand to prove it, ship each wheel library once."""
    if sys.platform != "linux" or not shutil.which("ldd"):
        return
    before = unresolved(folder)
    drop_links(folder)
    missing = [name for name in unresolved(folder) if name not in before]
    if missing:
        raise SystemExit(f"without PyInstaller's links these bundled libraries are no longer found: {', '.join(missing)}")


def starts(folder):
    """The built program answers --help with exit code 0."""
    exe = folder / (APP + (".exe" if os.name == "nt" else ""))
    r = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=START_TIMEOUT, check=False)
    if r.returncode != 0 or "rwg2-migrator" not in r.stdout:
        raise SystemExit(f"the built program did not start: exit {r.returncode}\n{r.stdout}\n{r.stderr}")


def write_sums(dist):
    lines = [f"{file_hash(z)}  {z.name}\n" for z in sorted(dist.glob(f"{APP}-*.zip"))]
    (dist / SUMS).write_text("".join(lines), encoding="ascii")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", default=str(ROOT / "dist"), help="where the zip and SHA256SUMS go")
    ap.add_argument("--work", default=str(ROOT / "build"), help="PyInstaller's work folder")
    ap.add_argument("--any-glibc", action="store_true", help="Linux: allow a build that only starts on systems as new as this one")
    opts = ap.parse_args()
    dist, work = Path(opts.dist), Path(opts.work)
    folder = pyinstaller(work)
    for name in TOP_FILES:
        shutil.copyfile(ROOT / name, folder / name)
    copy_licenses(folder)
    check_glibc(folder, opts.any_glibc)
    slim(folder)
    starts(folder)
    dist.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(dist / f"{APP}-{__version__}-{target()}"), "zip", root_dir=folder.parent, base_dir=APP)
    print(f"built {archive}")
    print("".join(write_sums(dist)), end="")


if __name__ == "__main__":
    main()
