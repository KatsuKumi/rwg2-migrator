"""Extract poc-seam.npz from the POC artefacts into the golden folder next to the fixture (once; they may be cleaned up later).

  python -I dev/make_golden.py --world <fixture World> --team TEAM --poc-ground current_ground.npy --window X0,Z0,SIZE
      --render BX,BZ,LX,LZ=PATTERN... --renders DIR... [--out DIR]

--out defaults to RWG2_FIXTURE/golden, where the tests read it; the data comes from the test world, so it stays out of the repo.

The POC measured the seam on its stale ground grid (tall grass counted as ground) with a mask-roll walk of the claim mask over its
square --window (the block window of current_ground.npy). This keeps, for the border of TEAM's claims: the core chunks, the inner
and outer column of every edge pair found that way, the stale ground at the inner column, and the RWG2 height at the outer column
from the POC's own renders, one --render per config (the file name pattern of its render in --renders); the configs' rows are in
search-green-rows.csv in the same folder. The renders' heights equal point mode (POC check_dense).
"""

import argparse
import glob
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwg2migrator import world  # noqa: E402

DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def ints(text):
    return tuple(int(v) for v in text.split(","))


def render_arg(text):
    config, _, pattern = text.partition("=")
    return ints(config), pattern


def mask(core, window):
    x0, z0, size = window
    m = np.zeros((size, size), bool)
    for cx, cz in core:
        m[cz * 16 - z0 : cz * 16 - z0 + 16, cx * 16 - x0 : cx * 16 - x0 + 16] = True
    return m


def ring(cur, inside, window):
    """The POC's rwg2_match.ring: inner and outer columns of every step that leaves the mask, inner ground finite."""
    x0, z0, size = window
    inner, outer = [], []
    for dz, dx in DIRS:
        a = inside & ~np.roll(np.roll(inside, -dz, 0), -dx, 1)
        zz, xx = np.nonzero(a)
        zo, xo = zz + dz, xx + dx
        keep = (zo >= 0) & (zo < size) & (xo >= 0) & (xo < size) & np.isfinite(cur[zz, xx])
        inner.append(np.c_[xx[keep] + x0, zz[keep] + z0])
        outer.append(np.c_[xo[keep] + x0, zo[keep] + z0])
    return np.concatenate(inner), np.concatenate(outer)


def height(path):
    data = Path(path + ".height.bin").read_bytes()
    res = struct.unpack(">i", data[:4])[0]
    return np.frombuffer(data[4:], ">f4").reshape(res, res).astype(np.float32)


def find(dirs, pattern):
    hits = [p for d in dirs for p in glob.glob(os.path.join(d, pattern)) if not p.endswith((".bin", ".txt"))]
    if not hits:
        raise SystemExit(f"no render {pattern} in {dirs}")
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    fixture = os.environ.get("RWG2_FIXTURE")
    ap.add_argument("--world", required=True)
    ap.add_argument("--team", required=True, help="the ServerUtilities team whose overworld claims are the core")
    ap.add_argument("--poc-ground", required=True)
    ap.add_argument("--window", type=ints, required=True, help="X0,Z0,SIZE: the block window of --poc-ground")
    ap.add_argument("--render", type=render_arg, action="append", required=True, help="BX,BZ,LX,LZ=PATTERN of a render in --renders")
    ap.add_argument("--renders", nargs="+", required=True)
    ap.add_argument("--out", default=fixture and os.path.join(fixture, "golden"), required=not fixture, help="default RWG2_FIXTURE/golden")
    a = ap.parse_args(argv)
    core = sorted(world.claims(a.world)[a.team])
    cur = np.load(a.poc_ground)
    x0, z0, _ = a.window
    inner, outer = ring(cur, mask(core, a.window), a.window)
    configs = np.array([c for c, _ in a.render], np.int64)
    h_out = np.stack([height(find(a.renders, p))[outer[:, 1] - z0, outer[:, 0] - x0] for _, p in a.render])
    h_in = cur[inner[:, 1] - z0, inner[:, 0] - x0]
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "poc-seam.npz")
    np.savez_compressed(out, core=np.array(core), inner=inner, outer=outer, h_in=h_in, configs=configs, h_out=h_out)
    print(f"{len(core)} core chunks, {len(inner)} pairs, {len(np.unique(outer, axis=0))} edge columns, {len(configs)} configs")


if __name__ == "__main__":
    main()
