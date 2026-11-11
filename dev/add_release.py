"""Pin a GTNH server release: size, sha256, the file table, and (with --boot) the probe digests, mods and biome list.

Run with the project's Python from the repo root, for example:
  python -I dev/add_release.py 2.9.0 --url https://downloads.gtnewhorizons.com/ServerPacks/GT_New_Horizons_2.9.0_Server_Java_17-26.zip \
      --zip GT_New_Horizons_2.9.0_Server_Java_17-26.zip --cache CACHE --boot --accept-eula
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwg2migrator import generator, jobs, nbt, provision, world  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "rwg2migrator" / "data"
JAVA_RANGE = re.compile(r"Java_(\d+)-(\d+)")
PROBE_MOD = "rwg2probe"


def emit(event):
    print(json.dumps(event), flush=True)


def head_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": provision.USER_AGENT})
    with urllib.request.urlopen(req, timeout=provision.NET_TIMEOUT) as r:
        return int(r.headers["Content-Length"])


def get_zip(a):
    if a.zip:
        return a.zip
    part = os.path.join(a.cache, "downloads", os.path.basename(a.url) + ".part")
    provision.fetch(provision.Download(a.url, part, head_size(a.url), emit))
    return part


def java_range(a):
    m = JAVA_RANGE.search(os.path.basename(a.url))
    lo, hi = (a.java_min, a.java_max) if a.java_min else (int(m.group(1)), int(m.group(2)))
    return {"min": lo, "max": hi}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1) + "\n")


def write_table(path, release, zip_name, files):
    """One file per line, so a new release shows as a readable diff."""
    rows = ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in files.items())
    head = json.dumps({"release": release, "zip": zip_name})[:-1]
    path.write_text(f'{head}, "files": {{\n{rows}\n}}}}\n')


def boot_facts(a, entry):
    """Boot the sampler once on the unpacked pack: digests, the biome list and the mod list it really loads."""
    folder = provision.pack_dir(a.cache, a.release)
    if not os.path.isdir(folder):
        provision.unpack(a.zip_path, folder, provision.file_table(entry), emit)
    java = provision.choose_java(a.cache, entry["java"], provision.JavaWanted(a.java), emit)
    ctx = generator.Context(a.cache, a.release, entry, java, a.accept_eula, emit)
    with generator.lock(a.cache):
        gen = generator.start(ctx, generator.Boot())
        try:
            digests = generator.check(gen, {}, emit)
            biomes = gen.call("biomes")["biomes"]
        finally:
            gen.stop(emit=emit)
    level = nbt.read_file(os.path.join(folder, generator.SAMPLER_WORLD, world.LEVEL))
    mods = dict(sorted(((m, v) for m, v in world.mods(level).items() if m != PROBE_MOD), key=lambda kv: kv[0].lower()))
    return digests, biomes, mods


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("release")
    p.add_argument("--url", required=True)
    p.add_argument("--zip", help="use this zip instead of downloading the URL")
    p.add_argument("--cache", required=True)
    p.add_argument("--java", help="java executable for the boot")
    p.add_argument("--java-min", type=int)
    p.add_argument("--java-max", type=int)
    p.add_argument("--boot", action="store_true", help="also boot the probe to record digests, mods and biomes")
    p.add_argument("--accept-eula", action="store_true")
    a = p.parse_args(argv)
    a.zip_path = get_zip(a)
    files_name = f"files-{a.release}.json"
    write_table(DATA / files_name, a.release, os.path.basename(a.url), provision.zip_table(a.zip_path))
    table = json.loads((DATA / "releases.json").read_text())
    old = table["releases"].get(a.release, {})
    zip_pin = {"url": a.url, "size": os.path.getsize(a.zip_path), "sha256": jobs.file_hash(a.zip_path)}
    entry = {**old, "server_zip": zip_pin, "java": java_range(a), "files": files_name}
    if a.boot:
        digests, biomes, mods = boot_facts(a, {k: v for k, v in entry.items() if k not in ("probe_digests", "rwg2")})
        write_json(DATA / f"biomes-{a.release}.json", {"release": a.release, "biomes": biomes})
        entry |= {"dreamcraft": mods[world.CORE_MOD], "rwg2": mods["rwg2"], "probe_digests": digests, "mods": mods}
        table["dreamcraft"][mods[world.CORE_MOD]] = a.release
    changed = {k: (old.get(k), entry[k]) for k in entry if old.get(k) != entry[k] and k != "mods"}
    mods_changed = old.get("mods") != entry.get("mods")
    table["releases"][a.release] = entry
    write_json(DATA / "releases.json", table)
    print(json.dumps({"release": a.release, "changed": changed, "mods_changed": mods_changed}, indent=1))


if __name__ == "__main__":
    main()
