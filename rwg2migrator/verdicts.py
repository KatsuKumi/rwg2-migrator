"""Plain-language lines and badges for a candidate; the thresholds are named here and pinned by tests."""

import math

VERY_CLOSE_MEDIAN = 1
VERY_CLOSE_OVER10 = 0.10
CLOSE_MEDIAN = 3
CLOSE_OVER30 = 0.02
EXCELLENT_SAME = 0.70
WATER_ROUND = 100
POOLS = 0.01
TAINT_WARN = 2048
REACH = 3584
PCT = 100
COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")
SECTOR = 360 / len(COMPASS)
OFF_LOOK = {
    "green": "a slightly different green",
    "dull": "duller grass",
    "sand": "sand",
    "snow": "snow",
    "wet": "marsh",
    "water": "water",
    "other": "odd ground (taint, volcano)",
}
RECOMMENDED = "Recommended"
EXCELLENT = "Excellent match"
PENDING = "Water and Tainted Land: looked at when the search ends"
SPREAD = " It spreads and ruins land; turn off spreading at Apply (reversible)."


def compass(dx, dz):
    """Direction of a block offset; north is -z in Minecraft."""
    angle = math.degrees(math.atan2(-dz, dx)) % 360
    return COMPASS[int((angle + SECTOR / 2) // SECTOR) % len(COMPASS)]


def border(r):
    if r["median"] <= VERY_CLOSE_MEDIAN and r["over10"] <= VERY_CLOSE_OVER10:
        return "very close", f"Border: very close (most steps {_blocks(r['median'])}; smoothing evens the rest)"
    if r["median"] <= CLOSE_MEDIAN and r["over30"] <= CLOSE_OVER30:
        return "close", f"Border: close (most steps {_blocks(r['median'])}; smoothing ramps the bigger ones)"
    return "rough", f"Border: rough (steps up to {_blocks(r['max'])}; smoothing ramps them, but some cliffs may show)"


def _blocks(v):
    v = int(round(v))
    return f"{v} block" + ("" if v == 1 else "s")


def excellent(r):
    """The border is very close, enough of the edge looks like the base, and the land rule passes."""
    return (
        bool(r.get("feasible"))
        and r["median"] <= VERY_CLOSE_MEDIAN
        and r["over10"] <= VERY_CLOSE_OVER10
        and r["same_edge"] >= EXCELLENT_SAME
    )


def looks(same, base, off):
    """off: {'side': ..., 'look': ...} for the side with the most off-look edge columns, or None."""
    line = f"Looks: {round(PCT * same)}% the same as your base ({base}) right at the edge"
    return line + (f"; {OFF_LOOK[off['look']]} on the {off['side']} side" if off else "")


def water(ocean, below_sea):
    if ocean is None:
        line = f"Water: no ocean within {_reach()} blocks"
    elif ocean >= WATER_ROUND:
        line = f"Water: no ocean within {int(ocean // WATER_ROUND) * WATER_ROUND} blocks"
    else:
        line = f"Water: ocean within {WATER_ROUND} blocks"
    return line + ("; some pools near the border" if below_sea > POOLS else "")


def _reach():
    """How far the search looked for ocean and Tainted Land around every kept part, rounded down like the distances."""
    return REACH // WATER_ROUND * WATER_ROUND


def taint(t):
    if t is None:
        return f"Tainted Land: none within {_reach()} blocks"
    line = f"Tainted Land: {round(t['blocks'])} blocks {t['direction']}."
    return line + (SPREAD if t["blocks"] < TAINT_WARN else "")


def fair(r):
    """Land-checked and the border is not rough."""
    return bool(r.get("feasible")) and border(r)[0] != "rough"


def recommended(candidates):
    """The index of the candidate to recommend: the best ranked excellent one, else the best ranked fair one; None when there is
    none, so a search stopped early never badges a rough border (the unchanged world) Recommended."""
    for ok in (excellent, fair):
        i = next((i for i, c in enumerate(candidates) if ok(c)), None)
        if i is not None:
            return i
    return None


def part_border(i, p):
    """The border line of one kept part of several, so the player sees which part fits badly."""
    x, z = p["centre"]
    return f"Border of part {i} ({p['chunks']} chunks around {x}, {z}): " + border(p)[1].removeprefix("Border: ")


def card(c, rank, recommend):
    """c: the candidate's metrics (seam stats, same_edge, off, ocean, below_sea, taint, feasible)."""
    badges = [RECOMMENDED] if recommend else []
    if excellent(c):
        badges.append(EXCELLENT)
    quality, border_line = border(c)
    lines = [border_line, looks(c["same_edge"], c["base"], c["off"]), water(c["ocean"], c["below_sea"]), taint(c["taint"])]
    lines += [part_border(i, p) for i, p in enumerate(c.get("parts", []), 1)]
    return {"rank": rank, "badges": badges, "border": quality, "lines": lines}
