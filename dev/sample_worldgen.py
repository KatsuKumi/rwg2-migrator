"""Build rwg2migrator/data/worldgen-tes.json: the tile-entity types and owner names world generation writes.

  python -I dev/sample_worldgen.py generate --out WORLD --seed S [--cache C] [--radius R] [--centre X,Z] [--heap 24G]
  python -I dev/sample_worldgen.py collect WORLD... --out rwg2migrator/data/worldgen-tes.json

generate boots the tool's own generator (the GTNH server pack it sets up, with RWG2 Smooth and structures on) on a new world and
generates and decorates a square of chunks, as the preview does around the kept area; it runs on a machine with the generator
set up and RAM to spare. collect reads every chunk no player ever came near (InhabitedTime 0: only the generator wrote it) of the
given worlds and writes the types and owner names it finds, with where they came from. The structures of the file already at --out
(types a mod's structure code places that no sample held, each with its evidence) are kept.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from types import SimpleNamespace

from rwg2migrator import area, cli_search, generator, scan

NEVER_VISITED = 0
DEFAULT_RADIUS = 96
LEVEL_TYPE = "RWG2_CONTINENTS"
CHUNK = 16
ABOUT = (
    "Tile-entity types and owner names in chunks only the world generator wrote (InhabitedTime 0), and structures: types a mod's "
    "structure code places that no sampled chunk held, each with its evidence; dev/sample_worldgen.py"
)


def generate(args):
    """A new RWG2 world in the tool's pack folder, a square of 2*radius+1 chunks generated and decorated around centre."""
    ctx, _ = cli_search.Generators(args.cache or scan.cache_root(), SimpleNamespace(accept_eula=True), print).context()
    name = os.path.basename(os.path.abspath(args.out))
    props = {"generate-structures": "true", "level-type": LEVEL_TYPE, "level-seed": str(args.seed)}
    boot = generator.Boot(kind=generator.GEN, world=name, offsets=(0, 0, 0, 0), heap=args.heap, properties=props)
    cx, cz = (int(v) // CHUNK for v in args.centre.split(","))
    cells = area.dilate({(cx, cz)}, args.radius)
    t0 = time.monotonic()
    gen = generator.start(ctx, boot)
    try:
        totals = generator.generate(gen, boot.offsets, cells, lambda e: print(json.dumps(e), flush=True))
        gen.call("save")
    finally:
        code = gen.stop()
    folder = os.path.join(gen.record["folder"], name)
    print(json.dumps({"world": folder, "chunks": len(cells), "totals": totals, "exit": code, "seconds": round(time.monotonic() - t0)}))
    return 0 if code == 0 else 1


def sample(world_dir, cache):
    """(types, owners, chunks) of the chunks of world_dir no player came near."""
    sc = scan.scan(world_dir, cache)
    t = sc.chunk_table()
    fresh = {c for c, ticks in zip(area.cells(t), t["inhabited"].tolist(), strict=True) if ticks == NEVER_VISITED}
    types = Counter()
    for cx, cz, name, n in sc.tile_entities():
        if (cx, cz) in fresh:
            types[name] += n
    owners = Counter(name for cx, cz, _, name in sc.owners() if (cx, cz) in fresh)
    return types, owners, len(fresh)


def _structures(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("structures", {})
    except FileNotFoundError:
        return {}


def collect(args):
    types, owners, sources = Counter(), Counter(), []
    for w, label in zip(args.worlds, args.label + [""] * len(args.worlds), strict=False):
        t, o, n = sample(w, args.cache or scan.cache_root())
        types.update(t)
        owners.update(o)
        name = label or os.path.basename(os.path.dirname(os.path.abspath(w)))
        sources.append({"world": name, "chunks": n, "tile_entities": sum(t.values())})
        print(f"{w}: {n:,} chunks no player came near, {sum(t.values()):,} tile entities of {len(t)} types", file=sys.stderr)
    out = {
        "about": ABOUT,
        "sources": sources,
        "types": dict(sorted(types.items())),
        "owners": dict(sorted(owners.items())),
        "structures": _structures(args.out),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True, help="the new world's name (a folder in the generator's pack folder)")
    g.add_argument("--cache")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--radius", type=int, default=DEFAULT_RADIUS, help="chunks each side of the centre")
    g.add_argument("--centre", default="0,0", help="block X,Z")
    g.add_argument("--heap", default="24G")
    g.set_defaults(fn=generate)
    c = sub.add_parser("collect")
    c.add_argument("worlds", nargs="+")
    c.add_argument("--label", action="append", default=[], help="a name for each world in the file (default: its parent folder)")
    c.add_argument("--cache")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=collect)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
