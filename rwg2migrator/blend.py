"""Biome blend at the old kept border. In the margin band each column takes RWG2's biome where a weight rising from
core to the kept border passes a smooth noise threshold, so the grass-colour edge becomes a wavy line inside the band instead of
the straight border; core and the new terrain keep their biomes. The colour step across the kept border is measured with and
without it."""

from collections import Counter

import numpy as np

from . import Refusal, appearance, area, chunks, score

SCALES = (8, 3)
WEIGHTS = (0.65, 0.35)
SPREAD = 0.4
SALT = 0x5EED3000
HASH_X, HASH_Z, HASH_MIX = 0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D
U32 = (1 << 32) - 1
OVER_DE = appearance.SAME_LOOK_DE
NO_BIOME = -1
PCT = 100


def _lattice(ix, iz, octave):
    h = (ix.astype(np.int64) * HASH_X ^ iz.astype(np.int64) * HASH_Z ^ (SALT + octave)) & U32
    h = ((h ^ (h >> 15)) * HASH_MIX) & U32
    return (h ^ (h >> 13)).astype(np.float64) / U32


def noise(x, z):
    """Smooth value noise in [0, 1] at block columns: two octaves on hashed lattices, the same for any run."""
    out = np.zeros(np.shape(x))
    for octave, (scale, weight) in enumerate(zip(SCALES, WEIGHTS, strict=True)):
        fx, fz = np.asarray(x, float) / scale, np.asarray(z, float) / scale
        ix, iz = np.floor(fx).astype(np.int64), np.floor(fz).astype(np.int64)
        tx, tz = fx - ix, fz - iz
        tx, tz = tx * tx * (3 - 2 * tx), tz * tz * (3 - 2 * tz)
        top = _lattice(ix, iz, octave) * (1 - tx) + _lattice(ix + 1, iz, octave) * tx
        bottom = _lattice(ix, iz + 1, octave) * (1 - tx) + _lattice(ix + 1, iz + 1, octave) * tx
        out += weight * (top * (1 - tz) + bottom * tz)
    return out


def depth(core, kept):
    """How many chunks the band reaches out from core at most."""
    k, band = 1, kept - core
    while not band <= area.dilate(core, k):
        k += 1
    return k


def weight(lay, kept, core, chunks_deep):
    """0 next to core, 1 next to the new terrain, smoothstepped across the band; 0 outside it. chunks_deep = depth(core, kept)
    of the part, so both distances are exact everywhere in the band."""
    limit = chunks.SIDE * (chunks_deep + 1)
    dk, dc = np.minimum(lay.chebyshev(~kept, limit), limit), np.minimum(lay.chebyshev(core, limit), limit)
    f = np.clip((dc - 0.5) / np.maximum(dk + dc - 1, 1), 0, 1)
    return np.where(kept & ~core, f * f * (3 - 2 * f), 0.0)


def rwg2_ids(names, biomes, outside):
    """{render biome name index: the biome id RWG2's chunks hold most often under it}, learned from the generated terrain:
    biome names repeat across mods, ids do not."""
    seen = Counter(zip(names[outside].tolist(), biomes[outside].tolist(), strict=True))
    best = {}
    for (name, biome), n in sorted(seen.items()):
        if name != NO_BIOME and biome != NO_BIOME and n > best.get(name, (0, 0))[0]:
            best[name] = (n, biome)
    return {name: biome for name, (_, biome) in best.items()}


def blended(field, f):
    """The band columns that take RWG2's biome, and that biome; f = (lay, masks (kept, core, outside), render names, xz, the
    band's depth in chunks)."""
    lay, (kept, core, outside), names, (x, z), deep = f
    ids = rwg2_ids(names, field, outside)
    lut = np.full(max(ids, default=0) + 2, NO_BIOME, np.int64)
    lut[list(ids)] = list(ids.values())
    rwg2 = np.where(names >= 0, lut[np.clip(names, 0, lut.size - 1)], NO_BIOME)
    take = (weight(lay, kept, core, deep) > 0.5 + SPREAD * (2 * noise(x, z) - 1)) & (rwg2 != NO_BIOME) & (field != NO_BIOME)
    return take, np.where(take, rwg2, field), int(np.sum(kept & ~core & (names >= 0) & (rwg2 == NO_BIOME)))


def refuse_bytes(chunk_biomes, names):
    """A chunk whose old format stores biome ids in bytes cannot take an id above 255; chunk_biomes = {pos: (format, ids)}."""
    for (cx, cz), (fmt, ids) in sorted(chunk_biomes.items()):
        if fmt == chunks.NEID and ids.max(initial=0) > chunks.BYTE_MAX:
            big = int(ids.max())
            raise Refusal(
                f"the biome blend would write biome {big} ({names.get(big, 'unknown')}) into chunk {cx},{cz}, whose old chunk "
                f"format holds biome ids up to {chunks.BYTE_MAX} only",
                "Run preview again without --blend-biomes",
            )


class Colours:
    """The grass colour (CIE Lab) of a biome id at a block height, from the shipped appearance table; NaN for an unknown id."""

    def __init__(self, release):
        self.names, self.table, self.cache = appearance.biome_names(release), appearance.table(release), {}

    def _column(self, biome):
        if biome not in self.cache:
            entry = self.table.get(appearance.fold(self.names.get(biome, "")))
            self.cache[biome] = (
                np.array([appearance.lab(appearance.colour_at(entry, y, biome)) for y in range(chunks.HEIGHT)])
                if entry
                else np.full((chunks.HEIGHT, 3), np.nan)
            )
        return self.cache[biome]

    def lab(self, biomes, ys):
        biomes, ys = np.asarray(biomes, np.int64).ravel(), np.clip(np.nan_to_num(ys, nan=0), 0, chunks.HEIGHT - 1).astype(int).ravel()
        out = np.full((biomes.size, 3), np.nan)
        for b in np.unique(biomes):
            sel = biomes == b
            out[sel] = self._column(int(b))[ys[sel]]
        return out


def _over(a, b):
    de = np.linalg.norm(a - b, axis=1)
    good = np.isfinite(de)
    return int(good.sum()), int(np.sum(de[good] > OVER_DE))


def border_steps(kept, at, colours):
    """Pairs across the kept border, and how many differ in grass colour by more than OVER_DE, for the biomes at(xz) gives as
    (biome ids, ground)."""
    inner, outer = score.edge_pairs(kept)
    (bi, gi), (bo, go) = at(inner), at(outer)
    return _over(colours.lab(bi, gi - 1), colours.lab(bo, go - 1))


def zone_steps(lay, field, ground, f):
    """Neighbouring columns that both lie in the zone and differ in grass colour by more than OVER_DE: the length of every colour
    edge there, the one blending moves included. f = (zone mask, colours)."""
    zone, colours = f
    lab = colours.lab(field, ground - 1).reshape(*field.shape, 3)
    total, over = 0, 0
    for ax in (0, 1):
        nb_lab = np.stack([lay.shift(lab[..., i], 1, ax, np.nan) for i in range(3)], -1)
        both = zone & lay.shift(zone, 1, ax, False)
        n, o = _over(lab[both], nb_lab[both])
        total, over = total + n, over + o
    return total, over


def share(pairs, over):
    return round(PCT * over / pairs, 2) if pairs else 0.0
