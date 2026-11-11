"""How biomes look from above: the shipped table, the base's reference look, and the looks a candidate is judged by."""

import json
import re
from dataclasses import dataclass, field, fields

import numpy as np

from . import Refusal, area
from .ground import TOPS
from .provision import data

LOOKS = ("green", "dull", "sand", "snow", "wet", "water", "other")
LOOK_LABELS = {
    "green": "Looks like your base",
    "dull": "Grass, off colour",
    "sand": "Sand / desert / mesa",
    "snow": "Snow / ice",
    "wet": "Marsh / bog / swamp",
    "water": "Ocean / water",
    "other": "Taint / volcano / odd",
}
NEVER_BANNED = ("green", "dull")
SURFACES = ("grass", "dirt", "rock", "sand", "mesa", "snow", "mud", "marsh", "lake", "other")
BAD = ("sand", "mesa", "snow", "mud", "marsh")
BAD_LOOKS = {"sand": ("sand", "mesa"), "snow": ("snow",), "wet": ("marsh", "mud")}
NAME_RULES = (("sand", r"desert|mesa"), ("snow", r"snow|ice|tundra"), ("wet", r"marsh|bog|swamp"), ("water", r"ocean"))
GOOD = 0.6
BAD_SHARE = 0.10
BAD_ZERO = 0.25
DELTA_E_ZERO = 30.0
SAME_LOOK_DE = 10.0
PARTIAL = 0.5
MARSH_DEPTH = 2
RING = 256
CHUNK = 16
PREF_KEYS = {"base", "banned", "tolerance"}
D65 = np.array([0.95047, 1.0, 1.08883])
SRGB_KNEE, SRGB_OFFSET, SRGB_SCALE, SRGB_GAMMA, SRGB_LINEAR = 0.04045, 0.055, 1.055, 2.4, 12.92
LAB_EPSILON, LAB_KAPPA = 216 / 24389, 24389 / 27
SRGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])


def fold(name):
    return name.casefold()


def table(release):
    return json.loads(data(f"appearance-{release}.json").read_text())["biomes"]


def biome_names(release):
    return {b["id"]: b["name"] for b in json.loads(data(f"biomes-{release}.json").read_text())["biomes"]}


def rgb(hexcolour):
    return np.array([int(hexcolour[i : i + 2], 16) for i in (1, 3, 5)], float)


def colour_at(entry, y, biome_id=None):
    """The rendered top colour of a biome's grass at block height y."""
    variant = next((v for v in entry["variants"] if biome_id in v["ids"]), entry["variants"][0])
    runs = variant["grass"]
    return rgb(runs[max(i for i, (start, _) in enumerate(runs) if start <= max(y, 0))][1])


def lab(colour):
    c = np.asarray(colour, float) / 255
    c = np.where(c > SRGB_KNEE, ((c + SRGB_OFFSET) / SRGB_SCALE) ** SRGB_GAMMA, c / SRGB_LINEAR)
    xyz = SRGB_TO_XYZ @ c / D65
    f = np.where(xyz > LAB_EPSILON, np.cbrt(xyz), (LAB_KAPPA * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a, b):
    return float(np.linalg.norm(lab(a) - lab(b)))


def shares(top, water):
    """Shares of SURFACES over columns with ground or water: water up to MARSH_DEPTH deep is marsh, deeper is lake."""
    top, water = np.asarray(top).ravel(), np.asarray(water).ravel()
    counts = dict.fromkeys(SURFACES, 0)
    wet = water > 0
    counts["marsh"] = int(np.sum(wet & (water <= MARSH_DEPTH)))
    counts["lake"] = int(np.sum(water > MARSH_DEPTH))
    kinds = np.bincount(top[~wet], minlength=len(TOPS))
    for i, name in enumerate(TOPS[1:], 1):
        counts[name] += int(kinds[i])
    total = sum(counts.values())
    return {k: round(v / total, 4) if total else 0.0 for k, v in counts.items()}


def surface_value(sh):
    return sh["grass"] + PARTIAL * (sh["dirt"] + sh["rock"])


def bad_value(sh):
    return sum(sh[k] for k in BAD)


@dataclass(frozen=True)
class Reference:
    name: str
    biome_id: int
    colour: tuple
    shares: dict
    median_y: int
    columns: int
    look: str = "green"
    surface: float = field(init=False)
    bad: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "surface", surface_value(self.shares))
        object.__setattr__(self, "bad", bad_value(self.shares))

    @classmethod
    def from_dict(cls, d):
        """A Reference read back from JSON: the computed fields are dropped and the colour is a tuple again."""
        names = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in names} | {"colour": tuple(d["colour"])})


def score(sh, colour, ref):
    """1 looks like the base, 0 does not. A green base uses the POC formula (biome_appearance.score); any other base compares
    the top-block shares directly, with the grass colour counting as much as the base is grass."""
    de = delta_e(colour, ref.colour)
    if ref.look != "green":
        overlap = sum(min(sh[k], ref.shares[k]) for k in SURFACES)
        return round(overlap * (1 - ref.shares["grass"] * min(1.0, de / DELTA_E_ZERO)), 3)
    excess = max(0.0, bad_value(sh) - ref.bad)
    grassy = min(1.0, surface_value(sh) / ref.surface) if ref.surface else 0.0
    return round(grassy * max(0.0, 1 - excess / BAD_ZERO) * max(0.0, 1 - de / DELTA_E_ZERO), 3)


def same_look(look, value, de, ref):
    """Looks like the base: the same green within SAME_LOOK_DE, or for another base the same look with a good score."""
    if ref.look == "green":
        return look == "green" and de <= SAME_LOOK_DE
    return look == ref.look and value >= GOOD


def look_of(name, sh, ocean, value):
    """One of LOOKS: a bad surface share first, then the name, then ocean, then how close it is to the base."""
    bad = {look: sum(sh[k] for k in keys) for look, keys in BAD_LOOKS.items()} | {"water": sh["lake"] if ocean else 0.0}
    worst = max(bad, key=bad.get)
    if bad[worst] >= BAD_SHARE:
        return worst
    named = next((look for look, pattern in NAME_RULES if re.search(pattern, name, re.I)), None)
    if named:
        return named
    if ocean:
        return "water"
    if value == 0:
        return "other"
    return "green" if value >= GOOD else "dull"


def default_bans(ref):
    """Every look except the base's own; off-colour grass is never banned (the soft score handles it)."""
    return frozenset(look for look in LOOKS if look not in NEVER_BANNED and look != ref.look)


def incompat(look, value, bans):
    return 1.0 if look in bans else max(0.0, 1.0 - value / GOOD)


@dataclass(frozen=True)
class Look:
    look: str
    value: float
    incompat: float
    same: bool


def judge(name, ref, bans, tab):
    """How one RWG2 biome (a sampler palette name) looks next to the base."""
    e = tab.get(fold(name))
    s = (e or {}).get("sampled")
    if s is None:
        look = next((look for look, pattern in NAME_RULES if re.search(pattern, name, re.I)), "dull")
        look = "water" if e and e["ocean"] and look == "dull" else look
        return Look(look, 0.0, incompat(look, 0.0, bans), False)
    colour = rgb(s["colour"])
    value = 0.0 if s["ocean"] else score(s["shares"], colour, ref)
    look = look_of(name, s["shares"], s["ocean"], value)
    return Look(look, value, incompat(look, value, bans), same_look(look, value, delta_e(colour, ref.colour), ref))


def mode(values):
    v, n = np.unique(np.asarray(values).ravel(), return_counts=True)
    return int(v[np.argmax(n)])


def _zone_grid(sc, chunks, field_name):
    cs = np.array(sorted(chunks))
    x0, z0 = cs.min(0) * CHUNK
    x1, z1 = (cs.max(0) + 1) * CHUNK
    grid = sc.grid(field_name, (int(x0), int(z0), int(x1), int(z1)))
    h, w = grid.shape
    mask = np.zeros((h // CHUNK, w // CHUNK), bool)
    mask[cs[:, 1] - z0 // CHUNK, cs[:, 0] - x0 // CHUNK] = True
    return grid, np.kron(mask, np.ones((CHUNK, CHUNK), bool))


def reference(sc, core, release, pick=None):
    """The base's look: its main real biome, top-block shares in the ring outside core, colour at the median ground height."""
    names, tab = biome_names(release), table(release)
    core = set(map(tuple, core))
    biome, inside = _zone_grid(sc, core, "biome")
    explored = inside & np.isfinite(_zone_grid(sc, core, "ground")[0])
    if pick is None and not explored.any():
        raise Refusal("no chunk of the area to keep was ever explored", "Keep chunks the world already has")
    bid = mode(biome[explored]) if pick is None else _id_for(pick, names)
    ring = area.near(core, RING) - core
    cols = {k: _zone_grid(sc, ring, k) for k in ("biome", "top", "water", "ground")}
    sel = cols["biome"][1] & (cols["biome"][0] == bid)
    ground = cols["ground"][0][sel]
    if not sel.any() or not np.isfinite(ground).any():
        raise Refusal(f"no columns of {names.get(bid, bid)} lie within {RING} blocks of the area", "Pick the biome your base looks like")
    sh = shares(cols["top"][0][sel], cols["water"][0][sel])
    median_y = int(np.median(ground[np.isfinite(ground)])) - 1
    entry = tab.get(fold(names.get(bid, "")))
    if entry is None:
        raise Refusal(f"the biome with id {bid} is not in the appearance table", "Pick the biome your base looks like")
    colour = tuple(float(v) for v in colour_at(entry, median_y, bid))
    ref = Reference(entry["name"], bid, colour, sh, median_y, int(sel.sum()))
    own = look_of(ref.name, sh, entry["ocean"], score(sh, np.array(colour), ref))
    return Reference(ref.name, bid, colour, sh, median_y, ref.columns, own)


def _id_for(name, names):
    ids = [i for i, n in names.items() if fold(n) == fold(name)]
    if not ids:
        raise Refusal(f"{name!r} is not a biome of this GTNH release", "Pick a biome name from the map legend")
    return min(ids)


def load_prefs(path):
    """{'base': biome name or None, 'banned': list of looks or None, 'tolerance': scale of the tolerances}."""
    prefs = {"base": None, "banned": None, "tolerance": 1.0}
    if not path:
        return prefs
    try:
        with open(path, encoding="utf-8") as f:
            got = json.load(f)
    except (OSError, ValueError) as e:
        raise Refusal(f"the preferences file {path} cannot be read ({e})", "Fix the file or leave --prefs out") from e
    unknown = set(got) - PREF_KEYS if isinstance(got, dict) else {"(not an object)"}
    bad_looks = set(got.get("banned") or ()) - set(LOOKS) if not unknown else set()
    if unknown or bad_looks:
        raise Refusal(
            f"the preferences file {path} has unknown entries: {', '.join(sorted(unknown | bad_looks))}",
            f"Use only {', '.join(sorted(PREF_KEYS))}; banned looks are {', '.join(LOOKS[2:])}",
        )
    return prefs | got


def bans_for(ref, prefs):
    return frozenset(prefs["banned"]) - set(NEVER_BANNED) if prefs.get("banned") is not None else default_bans(ref)
