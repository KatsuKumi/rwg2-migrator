"""A deterministic stand-in for the generator's sample and render ops: hills, four biomes and an ocean, shifted by the offsets."""

import time

import numpy as np

from rwg2migrator.generator import RECORD

NAMES = ["Temperate Forest", "Jade Cliffs", "Hot Desert", "Tainted Land"]


class FakeSampler:
    def __init__(self, delay=0.0):
        self.delay, self.calls = delay, 0

    def columns(self, config, xz):
        bx, bz, lx, lz = config
        x, z = xz[:, 0].astype(float), xz[:, 1].astype(float)
        rec = np.zeros(len(xz), RECORD)
        rec["h"] = 66.5 + 2 * np.sin((x + bx) / 37.0) + 2 * np.cos((z + bz) / 53.0)
        pick = (np.sin((x + bx) / 211.0) + np.cos((z + bz) / 173.0) + 2) * 1.2
        rec["palette"] = np.minimum(pick.astype(int), len(NAMES) - 1)
        ocean = np.sin((x + lx) / 1500.0) + np.sin((z + lz) / 1700.0) < -1.7
        rec["ocean"] = ocean
        rec["meta"] = np.where(ocean, 0, 1)
        return rec

    def sample(self, configs, points):
        self.calls += 1
        time.sleep(self.delay)
        pts = np.asarray(points)
        return np.stack([self.columns(tuple(c), pts) for c in configs]), list(NAMES)

    def render(self, config, window, bpp, continental=True):
        x0, z0, size = window
        res = size // bpp
        zz, xx = np.mgrid[0:res, 0:res]
        rec = self.columns(tuple(config), np.c_[x0 + xx.ravel() * bpp, z0 + zz.ravel() * bpp])
        return rec.reshape(res, res), list(NAMES)
