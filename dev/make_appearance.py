"""Build rwg2migrator/data/appearance-<release>.json: how each biome the pack registers looks from above.

  python -I dev/make_appearance.py --biomes rwg2migrator/data/biomes-2.9.0.json --poc biome_appearance.json \\
      --dump dump_<sig>.json --assets <minecraft textures folder> --pack <unpacked server pack> --out <json>

Inputs: the probe's `biomes` list (ids, names, temperature, rainfall from the real pack); the POC's biome_appearance.json
(top-block shares and colours sampled through RWG2's own surface code, one entry per RWG2 palette); the POC's
BiomeAppearanceDump output (grass colours by height from the client code, for biomes that override them); and the
Mojang grass colormap and grass_top texture from a client jar or a decompiled workspace. Only derived colour numbers
are written; no Mojang file is copied. The pack is read to confirm that its jars carry the client grass-colour overrides.
"""

import argparse
import json
import re
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SEA_Y = 64
LAPSE = 0.05 / 30.0
COLORMAP_MAX = 255.0
HEIGHT = 256
GRASS_COLOUR_METHOD = b"func_150558_b"
GRASS_COLOUR_DESC = b"(III)I"
CONSTANT_SIZES = {3: 4, 4: 4, 5: 8, 6: 8, 7: 2, 8: 2, 9: 4, 10: 4, 11: 4, 12: 4, 15: 3, 16: 2, 17: 4, 18: 4, 19: 2, 20: 2}
UTF8, LONG_KINDS = 1, (5, 6)
OCEAN = re.compile(r"ocean|coast|kelp forest|coral reef", re.I)
TC_DUMPED = ("BiomeGenEerie", "BiomeGenMagicalForest", "BiomeGenTaint")
VARIANT_KEYS = ("temperature", "rainfall", "grass")


def fold(name):
    return name.casefold()


def formula(cmap, t, r, y):
    """ColorizerGrass with the biome temperature lapse above sea level, in float32 like the game."""
    f32 = np.float32
    t = f32(t) - (f32(y) - f32(SEA_Y)) * f32(LAPSE) if y > SEA_Y else f32(t)
    t, r = float(min(max(t, f32(0)), f32(1))), float(min(max(f32(r), f32(0)), f32(1)))
    i, j = int((1.0 - t) * COLORMAP_MAX), int((1.0 - r * t) * COLORMAP_MAX)
    return cmap[j << 8 | i].astype(float)


def unpack(c):
    return np.array([(c >> 16) & 255, (c >> 8) & 255, c & 255], float)


def hexrgb(rgb):
    return "#" + "".join(f"{round(v):02X}" for v in rgb)


def runs(colours):
    """[[y, '#RRGGBB'], ...]: the colour from y up to the next entry."""
    out = []
    for y, c in enumerate(colours):
        if not out or out[-1][1] != c:
            out.append([y, c])
    return out


def texture_mean(path):
    a = np.asarray(Image.open(path).convert("RGBA")).astype(float)
    return a[..., :3][a[..., 3] > 0].mean(0)


def dump_colour(by_y, y):
    probes = sorted(int(k) for k in by_y)
    return unpack(by_y[str(max([p for p in probes if p <= y] or probes[:1]))])


def overrides(base, cmap):
    t, r = base["temperature"], base["rainfall"]
    return any(not np.array_equal(unpack(v), formula(cmap, t, r, int(y))) for y, v in base["grass_colour_by_y"].items())


def declared_methods(data):
    """(name, descriptor) of every method a class file declares."""
    count, pos, pool = struct.unpack_from(">H", data, 8)[0], 10, {}
    i = 1
    while i < count:
        kind = data[pos]
        if kind == UTF8:
            n = struct.unpack_from(">H", data, pos + 1)[0]
            pool[i] = data[pos + 3 : pos + 3 + n]
            pos += 3 + n
        else:
            pos += 1 + CONSTANT_SIZES[kind]
        i += 2 if kind in LONG_KINDS else 1
    pos += 6
    pos += 2 + 2 * struct.unpack_from(">H", data, pos)[0]
    fields, pos = struct.unpack_from(">H", data, pos)[0], pos + 2
    for _ in range(fields):
        pos = _skip_member(data, pos)
    methods, pos = struct.unpack_from(">H", data, pos)[0], pos + 2
    out = []
    for _ in range(methods):
        name, desc = struct.unpack_from(">HH", data, pos + 2)
        out.append((pool.get(name), pool.get(desc)))
        pos = _skip_member(data, pos)
    return out


def _skip_member(data, pos):
    """Past one field or method: access, name, descriptor, then its attributes."""
    attributes, pos = struct.unpack_from(">H", data, pos + 6)[0], pos + 8
    for _ in range(attributes):
        pos += 6 + struct.unpack_from(">I", data, pos + 2)[0]
    return pos


def _jar_overriders(jar):
    with zipfile.ZipFile(jar) as z:
        classes = [(n, z.read(n)) for n in z.namelist() if n.endswith(".class")]
    method = (GRASS_COLOUR_METHOD, GRASS_COLOUR_DESC)
    return {
        n[: -len(".class")].replace("/", ".") for n, data in classes if GRASS_COLOUR_METHOD in data and method in declared_methods(data)
    }


def pack_overriders(pack):
    """Class names (dotted) in the pack's jars that declare getBiomeGrassColor (SRG func_150558_b)."""
    return set().union(*(_jar_overriders(j) for j in sorted(Path(pack, "mods").rglob("*.jar"))))


def pack_check(pack, dump, cmap):
    """Which grass-colour overrides in the pack's mod jars the client dump covers (by class, or by the RWG2 class of a legacy one)."""
    dumped = {b["class"] for b in dump["base"] if overrides(b, cmap)} | {f"thaumcraft.common.lib.world.biomes.{c}" for c in TC_DUMPED}
    found = sorted(pack_overriders(pack))
    covered = [c for c in found if c in dumped or c.replace("rwg.", "rwg2.", 1) in dumped]
    return {
        "mod_classes_declaring_getBiomeGrassColor": len(found),
        "covered_by_client_dump": len(covered),
        "not_covered": [c for c in found if c not in covered],
        "vanilla": "the dedicated server jar has no client code, so vanilla overrides (swamp, mesa, roofed forest) come from the dump only",
    }


def grass_runs(b, dumped, cmap, tex):
    """(runs of the rendered top colour by height, where they come from) for one registered biome."""
    d = dumped.get(b["id"])
    same_name = d is None or d["name"] != b["name"]
    if same_name:
        key = (fold(b["name"]), b["temperature"], b["rainfall"])
        d = next((x for x in dumped.values() if (fold(x["name"]), x["temperature"], x["rainfall"]) == key), None)
    if d is not None and overrides(d, cmap):
        colours = [hexrgb(tex * dump_colour(d["grass_colour_by_y"], y) / 255) for y in range(HEIGHT)]
        return runs(colours), "client code of the biome of the same name" if same_name else "client code"
    colours = [hexrgb(tex * formula(cmap, b["temperature"], b["rainfall"], y) / 255) for y in range(HEIGHT)]
    return runs(colours), "formula" if d is not None else "formula (client code not dumped)"


def sampled(entry):
    s = entry.get("sampled")
    if not s:
        return None
    return {"columns": s["columns"], "shares": s["shares"], "colour": s["rendered_top_colour"], "ocean": bool(entry.get("ocean_or_coast"))}


def build(a):
    biomes = json.loads(Path(a.biomes).read_text())
    poc = json.loads(Path(a.poc).read_text())["rwg2"]
    dump = json.loads(Path(a.dump).read_text())
    cmap = np.asarray(Image.open(Path(a.assets, "colormap", "grass.png")).convert("RGB")).reshape(-1, 3)
    tex = texture_mean(Path(a.assets, "blocks", "grass_top.png"))
    dumped = {b["id"]: b for b in dump["base"]}
    out = {}
    for b in biomes["biomes"]:
        grass, source = grass_runs(b, dumped, cmap, tex)
        variant = {"ids": [b["id"]], "temperature": b["temperature"], "rainfall": b["rainfall"], "top": b["top"]}
        variant |= {"grass": grass, "source": source}
        e = out.setdefault(fold(b["name"]), {"name": b["name"], "variants": [], "ocean": bool(OCEAN.search(b["name"]))})
        same = next((v for v in e["variants"] if all(v[k] == variant[k] for k in VARIANT_KEYS)), None)
        if same:
            same["ids"] = sorted(set(same["ids"]) | {b["id"]})
        else:
            e["variants"].append(variant)
    missing = []
    for name, entry in poc.items():
        e = out.get(fold(name))
        if e is None:
            missing.append(name)
            continue
        e["sampled"] = sampled(entry)
        e["ocean"] = e["ocean"] or bool(entry.get("ocean_or_coast"))
    return {
        "release": biomes["release"],
        "made_by": "dev/make_appearance.py",
        "colour": "grass_top texture mean x grass tint; tint from the Mojang colormap (lapse above y 64) or the biome's client code",
        "texture_mean": round(float(tex[0]), 3),
        "pack_check": pack_check(a.pack, dump, cmap) if a.pack else None,
        "palettes_without_registered_biome": missing,
        "biomes": dict(sorted(out.items())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--biomes", "--poc", "--dump", "--assets", "--out"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--pack", help="unpacked GTNH server pack (its mods/ is read)")
    a = ap.parse_args(argv)
    table = build(a)
    Path(a.out).write_text(json.dumps(table, indent=1) + "\n")
    srcs = {}
    for e in table["biomes"].values():
        for v in e["variants"]:
            srcs[v["source"]] = srcs.get(v["source"], 0) + len(v["ids"])
    sampled_n = sum("sampled" in e for e in table["biomes"].values())
    print(
        json.dumps(
            {
                "biomes": len(table["biomes"]),
                "sampled": sampled_n,
                "sources": srcs,
                "pack_check": table["pack_check"],
                "palettes_without_registered_biome": table["palettes_without_registered_biome"],
            },
            indent=1,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
