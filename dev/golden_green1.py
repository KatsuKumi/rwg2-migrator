"""Evaluate a config directly on a world through the real generator, and pin the result (the test world's green #1).

  python -I dev/golden_green1.py --world <fixture World> --cache <generator cache> --out <dir> --config BX,BZ,LX,LZ [--threads N]
      [--config-dir DIR]

Core = every overworld claim, kept = core + 1 chunk, the fixture's own reference look and default bans. The generator boots on
the world's config folder, as the search does (--config-dir overrides it; "none" boots on the pack's own). Writes into --out:
  green1.json        the metrics, the candidate card and the nearest Tainted Land / ocean on the renders around kept
  green1-site.npz    the site (edge pairs with the real ground, grid, land points)
  green1-sample.npz  the generator's records for the config at the edge, grid and land points, and the palette names
  reference.json     the reference look
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwg2migrator import appearance, area, generator, jobs, provision, scan, score, search, world  # noqa: E402


def emit(event):
    print(json.dumps(event), file=sys.stderr, flush=True)


def context(cache):
    s = generator.settings(cache)
    rel = provision.release(s["release"])
    java = provision.choose_java(cache, rel.entry["java"], provision.JavaWanted(s["java"]))
    return generator.Context(cache, rel.name, rel.entry, java, False, emit)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    ap.add_argument("--config", required=True, help="the RWG2 offsets BX,BZ,LX,LZ")
    ap.add_argument("--config-dir")
    a = ap.parse_args(argv)
    overlay = world.locate(a.world).config if a.config_dir is None else None if a.config_dir == "none" else a.config_dir
    config = tuple(int(v) for v in a.config.split(","))
    os.makedirs(a.out, exist_ok=True)
    level = world.read_level(a.world)
    release, _ = world.release_for(world.mods(level), world.releases())
    core = set().union(*world.claims(a.world).values())
    kept = area.dilate(core, 1)
    sc = scan.scan(a.world, a.cache)
    site = score.build_site(lambda p: sc.at("ground", p), core, kept)
    ref = appearance.reference(sc, core, release)
    bans = appearance.default_bans(ref)
    judge = score.Judge(ref, bans, appearance.table(release))
    gen = generator.start(context(a.cache), generator.Boot(overlay=overlay))
    try:
        sampler = search.ProbeSampler(gen, level["Data"]["RandomSeed"], a.threads)
        ev = search.Evaluator(site, judge, sampler, a.out)
        t0 = time.monotonic()
        r = ev.run([config], "direct", search.LAND)[0]
        seconds = round(time.monotonic() - t0, 1)
        described = search.describe(ev, [config], ref)[0]
        recs, names = sampler.sample([config], np.r_[site.edge, site.grid, site.land])
    finally:
        gen.stop(emit=emit)
    described["card"] = search.verdicts.card(described, 1, True)
    site.save(os.path.join(a.out, "green1-site.npz"))
    np.savez_compressed(os.path.join(a.out, "green1-sample.npz"), rec=recs[0].view(np.uint8), names=np.array(names))
    search.dump(os.path.join(a.out, "reference.json"), asdict(ref) | {"bans": sorted(bans)})
    run = {"config": list(config), "record": r, "described": described, "seconds": seconds, "config_dir": overlay}
    search.dump(os.path.join(a.out, "green1.json"), run | {"config_key": jobs.tree_key(overlay)})
    shown = {"record": r, "card": described["card"], "taint": described["taint"], "ocean": described["ocean"]}
    print(json.dumps(shown, indent=1, default=float))


if __name__ == "__main__":
    main()
