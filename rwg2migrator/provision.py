"""Gets the official GTNH server pack and a Java runtime into the cache, each checked against pinned values."""

import ctypes
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from importlib import resources

from . import Refusal, __version__, world
from .jobs import file_hash
from .region import fsync_dir, rmtree, write_atomic

MB = 1 << 20
BLOCK = MB
PROGRESS_SHARE = 0.05
NET_TIMEOUT = 60
DOWNLOAD_RETRIES = 3
HTTP_PARTIAL = 206
WINDOWS_MAX_PATH = 260
UNPACK_EVENTS = 10
JAVA_PROBE_TIMEOUT = 60
LTS_JAVA = (25, 21, 17)
THIRTY_TWO_BIT = ("x86", "i386", "i486", "i586", "i686")
QUIET_JAVA = ("-XX:-UsePerfData", "-Djdk.disableLastUsageTracking=true")
ANSI = "mbcs" if os.name == "nt" else "utf-8"
PARTIAL = ".partial"
PRISTINE = "config.pristine"
CACHE_MARKER = "rwg2-migrator-cache.txt"
CACHE_NOTE = "rwg2-migrator keeps its cache here: the GTNH server pack, downloads, its own Java and world scans.\n"
SERVER_WRITTEN = frozenset({
    "server.properties",
    "eula.txt",
    "ops.json",
    "whitelist.json",
    "banned-ips.json",
    "banned-players.json",
    "usercache.json",
    "serverutilities/serverutilities.cfg",
    "serverutilities/aurora.cfg",
})
USER_AGENT = f"rwg2-migrator/{__version__}"
ADOPTIUM = "https://api.adoptium.net/v3/assets/latest/{feature}/hotspot?architecture={arch}&image_type=jre&os={os}&vendor=eclipse"
PROPERTY = re.compile(r"^ {4}([\w.]+) = (.*)$")
NOT_KNOWN = "{what} is not the GTNH {release} server zip this tool expects ({size:,} bytes, sha256 {sha})"
NOT_KNOWN_ACTION = "Either it is a different zip (the tool needs {name}), or GTNH re-uploaded it and this tool needs an update"
DEFENDER = "Windows Defender (or your antivirus) removed or changed {path}{more}. This is a known false positive (GTNH issue #26655)"
RESTORE_ACTION = "Run: rwg2-migrator generator setup (it restores them from the GTNH zip)"
DEFENDER_ACTION = (
    "Restore it from Protection history, or allow the folder {cache} in your antivirus; then run: rwg2-migrator generator setup"
)
AGAIN_ACTION = "Something here changes them, often an antivirus: allow the folder {cache} in it, then run: rwg2-migrator generator setup"


def _quiet(event):
    pass


def data(name):
    return resources.files("rwg2migrator").joinpath("data", name)


@dataclass(frozen=True)
class Release:
    name: str
    entry: dict


@dataclass(frozen=True)
class Download:
    url: str
    part: str
    size: int
    emit: object = _quiet


def release(name=None):
    table = world.releases()["releases"]
    name = name or max(table, key=world.version)
    if name not in table:
        raise Refusal(f"GTNH {name} is not a release this tool knows", f"Pass --release with one of: {', '.join(sorted(table))}")
    return Release(name, table[name])


def file_table(entry):
    """{relative path: [size, crc32]} for every file of the release's server zip."""
    return json.loads(data(entry["files"]).read_text())["files"]


def pack_dir(cache, name):
    return os.path.join(cache, "packs", name)


def own_cache(cache):
    return os.path.isfile(os.path.join(cache, CACHE_MARKER))


def claim_cache(cache):
    """The cache folder, new or marked as the tool's own; a folder that already holds other files is refused."""
    if own_cache(cache):
        return cache
    if os.path.isdir(cache) and os.listdir(cache):
        raise Refusal(
            f"the cache folder {cache} already holds other files, and the tool keeps and deletes its own files there",
            "Pass --cache a new or empty folder, or leave --cache out",
        )
    os.makedirs(cache, exist_ok=True)
    write_atomic(os.path.join(cache, CACHE_MARKER), CACHE_NOTE.encode())
    return cache


def crc32(path):
    crc = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            crc = zlib.crc32(block, crc)
    return crc


def _fetch_once(d):
    have = os.path.getsize(d.part) if os.path.exists(d.part) else 0
    if have > d.size:
        os.remove(d.part)
        have = 0
    if have == d.size:
        return
    headers = {"User-Agent": USER_AGENT} | ({"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(urllib.request.Request(d.url, headers=headers), timeout=NET_TIMEOUT) as r:
        resumed = have if r.status == HTTP_PARTIAL else 0
        d.emit({"step": "download", "event": "start", "url": d.url, "resume_from": resumed, "http": r.status, "size": d.size})
        _copy(d, r, resumed)


def _copy(d, src, have):
    step = max(int(d.size * PROGRESS_SHARE), 1)
    mark = have + step
    with open(d.part, "r+b" if have else "wb") as f:
        f.seek(have)
        f.truncate()
        while block := src.read(BLOCK):
            f.write(block)
            have += len(block)
            if have >= mark:
                d.emit({"step": "download", "event": "progress", "done": have, "total": d.size})
                mark += step
        f.flush()
        os.fsync(f.fileno())
    if have != d.size:
        raise OSError(f"the download stopped after {have:,} of {d.size:,} bytes")


def fetch(d):
    """Download d.url into d.part, continuing from what d.part already holds (HTTP Range)."""
    os.makedirs(os.path.dirname(d.part), exist_ok=True)
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            return _fetch_once(d)
        except (OSError, urllib.error.URLError) as e:
            d.emit({"step": "download", "event": "interrupted", "attempt": attempt, "error": str(e)})
            last = e
    raise Refusal(
        f"the download of {d.url} failed: {last}", "Check the internet connection, then run this again; it continues where it stopped"
    )


def verify_zip(path, rel, emit=_quiet):
    z = rel.entry["server_zip"]
    size = os.path.getsize(path)
    digest = file_hash(path) if size == z["size"] else None
    if digest != z["sha256"]:
        reason = NOT_KNOWN.format(what=path, release=rel.name, size=z["size"], sha=z["sha256"])
        raise Refusal(reason, NOT_KNOWN_ACTION.format(name=os.path.basename(z["url"])))
    emit({"step": "pack", "event": "verified", "path": path, "size": size, "sha256": digest})


def _need_space(folder, nbytes):
    os.makedirs(folder, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if free < nbytes:
        raise Refusal(
            f"{folder} has {free / MB:,.0f} MB free; the generator needs {nbytes / MB:,.0f} MB",
            "Free some space, or pass --cache a new, empty folder on a drive with more room",
        )


def obtain_zip(cache, rel, emit=_quiet):
    """The verified server zip in the cache, downloaded (or continued) when needed."""
    z = rel.entry["server_zip"]
    final = os.path.join(cache, "downloads", os.path.basename(z["url"]))
    if os.path.exists(final):
        verify_zip(final, rel, emit)
        return final
    part = final + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    _need_space(os.path.dirname(final), z["size"] - have + sum(s for s, _ in file_table(rel.entry).values()))
    fetch(Download(z["url"], part, z["size"], emit))
    try:
        verify_zip(part, rel, emit)
    except Refusal:
        os.remove(part)
        raise
    os.replace(part, final)
    return final


def zip_table(zip_path):
    """{name: [size, crc32]} of every file entry, as the zip's central directory states them."""
    with zipfile.ZipFile(zip_path) as z:
        return {i.filename: [i.file_size, i.CRC] for i in z.infolist() if not i.is_dir()}


def check_files(root, table):
    """Relative paths whose file is missing or differs from the table's size and CRC."""
    bad = []
    for rel, (size, crc) in table.items():
        path = os.path.join(root, *rel.split("/"))
        try:
            if os.path.getsize(path) != size or crc32(path) != crc:
                bad.append(rel)
        except OSError:
            bad.append(rel)
    return bad


def on_disk(name):
    """Where a zip entry lives in the unpacked pack: its config/ is kept as config.pristine/."""
    return PRISTINE + name[len("config") :] if name.startswith("config/") else name


def damaged_files(folder, table):
    """Zip entries whose unpacked file is missing or changed, leaving out the files each boot rewrites."""
    paths = {on_disk(n): n for n in table if n not in SERVER_WRITTEN}
    return [paths[p] for p in check_files(folder, {p: table[n] for p, n in paths.items()})]


def damaged(bad, cache, again=False):
    """The refusal for pack files that are missing or changed; again: they were just written from the verified zip."""
    more = {0: "", 1: " and 1 more file"}.get(len(bad) - 1, f" and {len(bad) - 1} more files")
    if os.name == "nt" and all(b.endswith(".jar") for b in bad):
        return Refusal(DEFENDER.format(path=bad[0], more=more), DEFENDER_ACTION.format(cache=cache))
    if again:
        reason = f"{bad[0]}{more} of the generator changed right after being written from the GTNH zip"
        return Refusal(reason, AGAIN_ACTION.format(cache=cache))
    return Refusal(f"{bad[0]}{more} of the generator are missing or changed", RESTORE_ACTION)


def check_pack(folder, entry, cache):
    if bad := damaged_files(folder, file_table(entry)):
        raise damaged(bad, cache)


def _check_path_length(base, table):
    longest = max(map(len, table))
    if os.name == "nt" and len(os.path.abspath(base)) + 1 + longest >= WINDOWS_MAX_PATH:
        raise Refusal(
            f"the cache folder {base} is too long: files in the pack add {longest} characters and Windows stops at {WINDOWS_MAX_PATH}",
            "Pass --cache a shorter folder, for example C:\\rwg2-cache",
        )


def _write_entry(z, info, path):
    """One zip entry, flushed to the disk before the pack counts as unpacked."""
    if info.is_dir():
        os.makedirs(path, exist_ok=True)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with z.open(info) as src, open(path, "wb") as f:
        shutil.copyfileobj(src, f, BLOCK)
        f.flush()
        os.fsync(f.fileno())


def _sync_dirs(root):
    for d, _, _ in os.walk(root):
        fsync_dir(d)


def unpack(zip_path, folder, table, emit=_quiet):
    """Extract into folder.partial, check every file against the table, then rename; config/ becomes config.pristine/."""
    partial = folder + PARTIAL
    _check_path_length(partial, table)
    rmtree(partial)
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        step = max(len(infos) // UNPACK_EVENTS, 1)
        for i, info in enumerate(infos, 1):
            _write_entry(z, info, os.path.join(partial, *info.filename.split("/")))
            if i % step == 0:
                emit({"step": "unpack", "event": "progress", "done": i, "total": len(infos)})
    if bad := check_files(partial, table):
        raise damaged(bad, os.path.dirname(os.path.dirname(folder)), again=True)
    os.rename(os.path.join(partial, "config"), os.path.join(partial, PRISTINE))
    _sync_dirs(partial)
    os.rename(partial, folder)
    fsync_dir(os.path.dirname(folder))
    emit({"step": "unpack", "event": "done", "files": len(table)})


def restore(zip_path, folder, names, emit=_quiet):
    """Extract the named entries again over a damaged pack."""
    paths = {name: os.path.join(folder, *on_disk(name).split("/")) for name in names}
    with zipfile.ZipFile(zip_path) as z:
        for name, path in paths.items():
            _write_entry(z, z.getinfo(name), path)
    for d in {os.path.dirname(p) for p in paths.values()}:
        fsync_dir(d)
    emit({"step": "pack", "event": "restored", "files": len(names)})


def _get_zip(cache, rel, user_zip, emit):
    if not user_zip:
        return obtain_zip(cache, rel, emit)
    verify_zip(user_zip, rel, emit)
    emit({"step": "pack", "event": "zip", "path": user_zip, "download": False})
    return user_zip


def install(cache, rel, user_zip=None, emit=_quiet):
    """The unpacked pack folder with every file checked; a damaged one is restored from the user's zip or a (resumed) download."""
    folder, table = pack_dir(cache, rel.name), file_table(rel.entry)
    if not os.path.isdir(folder):
        zip_path = _get_zip(cache, rel, user_zip, emit)
        unpack(zip_path, folder, table, emit)
    elif bad := damaged_files(folder, table):
        emit({"step": "pack", "event": "damaged", "files": len(bad), "first": bad[0]})
        zip_path = _get_zip(cache, rel, user_zip, emit)
        restore(zip_path, folder, bad, emit)
        if still := damaged_files(folder, table):
            raise damaged(still, cache, again=True)
    else:
        emit({"step": "pack", "event": "present", "folder": folder})
        return folder
    if zip_path != user_zip:
        os.remove(zip_path)
    return folder


@dataclass(frozen=True)
class Java:
    path: str
    version: str
    major: int
    arch: str
    bits: str


def _no_window():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _major(v):
    parts = v.split(".")
    return int(re.match(r"\d+", parts[1] if parts[0] == "1" and len(parts) > 1 else parts[0]).group())


def _encodes(path):
    try:
        path.encode(ANSI)
    except UnicodeEncodeError:
        return False
    return True


def _short(path):
    """The 8.3 short form of an existing path on Windows, or None (8.3 names can be turned off per drive)."""
    if os.name != "nt":
        return None
    size = ctypes.windll.kernel32.GetShortPathNameW(path, None, 0)
    buf = ctypes.create_unicode_buffer(size)
    return buf.value if size and ctypes.windll.kernel32.GetShortPathNameW(path, buf, size) else None


def java_safe(path):
    """path, or its 8.3 short form, in a form java.exe can be given; None when neither fits. Windows' Java launcher reads its own
    location and its arguments in the ANSI code page, where other characters (a Cyrillic or CJK account name on a Western
    Windows) arrive as '?', and the JVM then cannot find itself."""
    return next((p for p in (path, _short(path)) if p and _encodes(p)), None)


def java_info(path):
    """Version and architecture from `java -XshowSettings:properties -version`, or None when it does not run."""
    path = java_safe(path)
    if path is None:
        return None
    try:
        r = subprocess.run(
            [path, *QUIET_JAVA, "-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            timeout=JAVA_PROBE_TIMEOUT,
            **_no_window(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    props = dict(m.groups() for m in map(PROPERTY.match, r.stderr.splitlines()) if m)
    v = props.get("java.version")
    return Java(path, v, _major(v), props.get("os.arch", ""), props.get("sun.arch.data.model", "")) if v else None


def why_not(java, span):
    if java.bits == "32" or java.arch in THIRTY_TWO_BIT:
        return f"Java {java.version} at {java.path} is 32-bit; the generator needs a 64-bit Java"
    if not span["min"] <= java.major <= span["max"]:
        return f"Java {java.version} at {java.path} is outside the Java {span['min']}-{span['max']} this GTNH release runs on"
    return None


def _exe():
    return "java.exe" if os.name == "nt" else "java"


def _install_globs(env):
    if os.name == "nt":
        pf = env.get("ProgramFiles", r"C:\Program Files")
        vendors = ("Eclipse Adoptium", "Java", "Microsoft", "Zulu", "BellSoft", "Amazon Corretto")
        prism = os.path.join(env.get("APPDATA", ""), "PrismLauncher", "java", "*", "bin", _exe())
        return [os.path.join(pf, v, "*", "bin", _exe()) for v in vendors] + [prism]
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return [
            "/Library/Java/JavaVirtualMachines/*/Contents/Home/bin/java",
            f"{home}/Library/Application Support/PrismLauncher/java/*/bin/java",
        ]
    return ["/usr/lib/jvm/*/bin/java", f"{home}/.local/share/PrismLauncher/java/*/bin/java"]


def _cache_globs(cache):
    base = os.path.join(cache, "java", "*")
    return [os.path.join(base, "bin", _exe()), os.path.join(base, "Contents", "Home", "bin", _exe())]


def candidates(cache, search=True, env=None):
    """Java executables to try: JAVA_HOME, PATH and the usual install folders (unless search is off), then the cache."""
    env = os.environ if env is None else env
    found = []
    if search:
        found += [os.path.join(env["JAVA_HOME"], "bin", _exe())] if env.get("JAVA_HOME") else []
        found += [w] if (w := shutil.which("java", path=env.get("PATH"))) else []
        found += [p for g in _install_globs(env) for p in sorted(glob.glob(g))]
    found += [p for g in _cache_globs(cache) for p in sorted(glob.glob(g))]
    return list({os.path.normcase(os.path.realpath(p)): p for p in found if os.path.isfile(p)}.values())


@dataclass(frozen=True)
class JavaWanted:
    path: str | None = None
    search: bool = True


SEARCH = JavaWanted()


def choose_java(cache, span, want=SEARCH, emit=_quiet):
    """The given Java if it fits, else the newest fitting one found, else a downloaded Temurin JRE."""
    explicit = want.path
    if explicit:
        java = java_info(explicit)
        if java is None:
            raise Refusal(f"{explicit} does not run as Java", "Pass --java the full path of java or java.exe")
        if reason := why_not(java, span):
            raise Refusal(
                reason, f"Pass --java a 64-bit Java {span['min']}-{span['max']}, or leave --java out to let the tool find or download one"
            )
        return java
    fits = []
    for java in filter(None, map(java_info, candidates(cache, want.search))):
        reason = why_not(java, span)
        emit({"step": "java", "event": "rejected" if reason else "found", "path": java.path, "version": java.version, "reason": reason})
        fits += [] if reason else [java]
    return max(fits, key=lambda j: j.major) if fits else temurin(cache, span, emit)


def _platform():
    os_name = {"nt": "windows"}.get(os.name, "mac" if sys.platform == "darwin" else "linux")
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "aarch64", "aarch64": "aarch64"}.get(platform.machine().lower(), platform.machine())
    return os_name, arch


def _get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=NET_TIMEOUT) as r:
        return json.load(r)


def _extract(archive, dest):
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")


def temurin(cache, span, emit=_quiet):
    """Download a Temurin JRE into the cache, checked against the checksum the Adoptium API publishes."""
    feature = next((v for v in LTS_JAVA if span["min"] <= v <= span["max"]), None)
    if feature is None:
        raise Refusal(f"no long-term Java fits Java {span['min']}-{span['max']}", "Install a 64-bit Java in that range and pass --java")
    os.makedirs(cache, exist_ok=True)
    safe = java_safe(cache)
    if safe is None:
        raise Refusal(
            f"the cache folder {cache} has characters Windows cannot hand to Java with this system's language settings",
            "Install a 64-bit Java 21 or 25 (it is found by itself), or pass --cache a folder whose path has only plain letters, "
            "for example C:\\rwg2-cache",
        )
    os_name, arch = _platform()
    try:
        asset = _get_json(ADOPTIUM.format(feature=feature, arch=arch, os=os_name))[0]
    except (OSError, urllib.error.URLError, ValueError, IndexError) as e:
        raise Refusal(
            f"no Java {span['min']}-{span['max']} was found and the Java download list cannot be read ({e})",
            "Install a 64-bit Java 21 or 25, or check the internet connection",
        ) from e
    pkg, folder = asset["binary"]["package"], os.path.join(safe, "java", asset["release_name"])
    emit({"step": "java", "event": "download", "name": pkg["name"], "size": pkg["size"]})
    archive = os.path.join(cache, "downloads", pkg["name"])
    fetch(Download(pkg["link"], archive + ".part", pkg["size"], emit))
    if file_hash(archive + ".part") != pkg["checksum"]:
        os.remove(archive + ".part")
        raise Refusal(
            f"the Java download {pkg['name']} does not match the checksum Adoptium publishes", "Run this again to download it again"
        )
    os.replace(archive + ".part", archive)
    _unpack_java(archive, folder)
    java = java_info(_java_in(folder))
    if java is None:
        raise Refusal(f"the Java just installed in {folder} does not start", "Install a 64-bit Java 21 or 25 and pass --java")
    emit({"step": "java", "event": "installed", "path": java.path, "version": java.version, "checksum": pkg["checksum"]})
    return java


def _java_in(folder):
    """The java program of an unpacked JRE, or None when the archive held none where a JRE keeps it."""
    places = (os.path.join(folder, "bin", _exe()), os.path.join(folder, "Contents", "Home", "bin", _exe()))
    return next((p for p in places if os.path.isfile(p)), None)


def _unpack_java(archive, folder):
    partial = folder + PARTIAL
    rmtree(folder)
    rmtree(partial)
    _extract(archive, partial)
    inner = os.listdir(partial)
    os.rename(os.path.join(partial, inner[0]) if len(inner) == 1 else partial, folder)
    rmtree(partial)
    os.remove(archive)
