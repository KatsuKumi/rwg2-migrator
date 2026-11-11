import dataclasses
import glob
import json
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from . import Refusal, nbt

LEVEL = "level.dat"
SERVER_PROPERTIES = "server.properties"
OREGEN = os.path.join("data", "GregTech_OregenPattern.dat")
RWG_DATA = os.path.join("data", "rwg_data.dat")
CLAIMS = os.path.join("serverutilities", "teams", "claimedchunks")
LEGACY_GENERATOR = "RWG"
MIGRATED = ("RWG2", "RWG2_CONTINENTS")
PERLIN = ("UNKNOWN", "PERLIN", "DYNAMICPERLIN")
BLOCK_PREFIX = "\x01"
OVERWORLD = "0"
CHUNK_SHIFT = 4
QUOTES = "\"'"
REQUIRED_MODS = ("RWG", "rwg2")
CORE_MOD = "dreamcraft"
GAME_DIRS = (".minecraft", "minecraft")
INSTANCE_CFG = {"PrismLauncher": "prismlauncher.cfg", "multimc": "multimc.cfg", "MultiMC": "multimc.cfg"}
LEVEL_SHAPE = (
    (("Data",), dict),
    (("FML",), dict),
    (("FML", "ModList"), list),
    (("FML", "ItemData"), list),
    (("Data", "RandomSeed"), int),
    (("Data", "SpawnX"), int),
    (("Data", "SpawnZ"), int),
)
MALFORMED = (OSError, ValueError, KeyError, TypeError, AttributeError)
SKIPPED = "the game skips it, so that player's position is unknown"
JAVA_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)")
JAVA_CONTROLS = {"t": "\t", "n": "\n", "r": "\r", "f": "\f"}
SAVES_HELP = "Point the tool at a world folder (it holds level.dat), its server folder, or its single-player save folder"
CONFIG = "config"
RWG2_CFG = "rwg2.cfg"
CONFIG_FROM = {"server": "the server's", "singleplayer": "the game instance's", "folder": "found next to the world"}
CHOSEN = "chosen by you"
NO_CONFIG = (
    "no config folder of its own was found next to the world, so the search and the preview use GTNH's default RWG2 settings, and Apply "
    "writes no config file: it shows the 4 offset lines to add by hand"
)


@dataclass(frozen=True)
class Place:
    world: str
    root: str
    config: str | None
    kind: str


def normalise(text):
    """A path as the user may paste it: surrounding quotes and spaces removed, ~ expanded, made absolute."""
    p = text.strip().strip(QUOTES).strip()
    return os.path.abspath(os.path.expanduser(p))


def same(a, b):
    """Whether two paths name the same folder."""
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def properties(path, encoding="utf-8"):
    out = {}
    with open(path, encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(("#", "!")) and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def java_unescape(text):
    """A value as Java reads it from a .properties file: \\uXXXX is that character, \\t \\n \\r \\f are controls, and a
    backslash before anything else (: # ! = or a space) is dropped."""
    return JAVA_ESCAPE.sub(lambda m: chr(int(m[1][1:], 16)) if len(m[1]) > 1 else JAVA_CONTROLS.get(m[1], m[1]), text)


def _level_name(root):
    """The world folder's name; Java writes server.properties in Latin-1 with every other character escaped."""
    name = properties(os.path.join(root, SERVER_PROPERTIES), "latin-1").get("level-name")
    return java_unescape(name) if name else "world"


def _server(root, exists=True):
    name = _level_name(root)
    world = os.path.join(root, name)
    if exists and not os.path.isdir(world):
        raise Refusal(
            f"the server folder {root} has no world folder {name!r}", "Start the server once, or point the tool at the world folder"
        )
    return Place(world, root, _holding_rwg2(os.path.join(root, CONFIG)), "server")


def locate(text, exists=True):
    """Resolve level.dat, region/, a server folder, a save folder or a world folder to the same world. With exists False a world
    folder that is missing for now (Apply or Rollback cut off between their two renames) is placed by its parent."""
    p = normalise(text)
    if os.path.basename(p).lower() == LEVEL or os.path.basename(p) == "region":
        p = os.path.dirname(p)
    if exists and not os.path.isdir(p):
        raise Refusal(f"{p} does not exist or is not a folder", SAVES_HELP)
    if os.path.isfile(os.path.join(p, SERVER_PROPERTIES)):
        return _server(p, exists)
    parent = os.path.dirname(p)
    if os.path.isfile(os.path.join(parent, SERVER_PROPERTIES)) and _level_name(parent) == os.path.basename(p):
        return Place(p, parent, _holding_rwg2(os.path.join(parent, CONFIG)), "server")
    if os.path.basename(parent) == "saves":
        game = os.path.dirname(parent)
        return Place(p, game, _holding_rwg2(os.path.join(game, CONFIG)), "singleplayer")
    beside = _holding_rwg2(os.path.join(parent, CONFIG))
    own = beside and all(same(w, p) for w in worlds_on(beside))
    return Place(p, parent, beside if own else None, "folder")


def holds_rwg2(folder):
    """A config folder the tool uses holds rwg2.cfg: Apply would otherwise create one with only the 4 offset lines, which an upload
    would put over the host's full file."""
    return os.path.isfile(os.path.join(folder, RWG2_CFG))


def _holding_rwg2(folder):
    return folder if holds_rwg2(folder) else None


def worlds_on(config):
    """The worlds that run on a config folder: the world of its server, else the saves of its game instance and the worlds next to
    it."""
    root = os.path.dirname(config)
    if os.path.isfile(os.path.join(root, SERVER_PROPERTIES)):
        found = [os.path.join(root, _level_name(root))]
    else:
        found = files(root, "*") + files(os.path.join(root, "saves"), "*")
    return [w for w in found if os.path.isfile(os.path.join(w, LEVEL))]


def config_dir(given):
    """A config folder the user named, as an absolute path; refused unless it holds rwg2.cfg."""
    path = normalise(given)
    if not holds_rwg2(path):
        raise Refusal(
            f"{path} is not a config folder (it holds no {RWG2_CFG})",
            "Pick the config folder of the server or game instance that runs this world (it holds rwg2.cfg)",
        )
    return path


def with_config(place, given):
    """place with the config folder the user named instead of the one found for it."""
    return dataclasses.replace(place, config=config_dir(given)) if given else place


def config_from(place, given=None):
    """Where the config folder in use comes from, in words; None without one."""
    if given:
        return CHOSEN
    return CONFIG_FROM[place.kind] if place.config else None


def _tag(level, path):
    v = level
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def read_level(world):
    path = os.path.join(world, LEVEL)
    if not os.path.isfile(path):
        raise Refusal(f"there is no level.dat in {world}", SAVES_HELP)
    try:
        level = nbt.read_file(path)
    except (OSError, nbt.NBTError) as e:
        raise Refusal(f"level.dat in {world} cannot be read ({e})", "Restore level.dat from level.dat_old or a backup") from e
    missing = [" ".join(path) for path, kind in LEVEL_SHAPE if not isinstance(_tag(level, path), kind)]
    if missing:
        raise Refusal(
            f"level.dat in {world} has no {missing[0]} tag, so it is damaged or was not saved by a Forge game",
            "Open the world once in GTNH 2.9 or newer, close it, then try again",
        )
    return level


def registry(level):
    return {e["V"]: e["K"][1:] for e in level["FML"]["ItemData"] if e["K"].startswith(BLOCK_PREFIX)}


def mods(level):
    return {m["ModId"]: m["ModVersion"] for m in level["FML"]["ModList"]}


def releases():
    return json.loads(resources.files("rwg2migrator").joinpath("data", "releases.json").read_text())


def version(text):
    return tuple(int(p) if p.isdigit() else 0 for p in text.replace("-", ".").split("."))


def release_for(world_mods, table):
    core = world_mods.get(CORE_MOD)
    if core is None:
        raise Refusal(f"this world has no GT New Horizons core mod ({CORE_MOD}) in its mod list", "This tool only migrates GTNH worlds")
    name = table["dreamcraft"].get(core)
    newest = max(table["releases"], key=version)
    if name is None:
        newer = version(core) > version(table["releases"][newest]["dreamcraft"])
        action = (
            "Update rwg2-migrator to a version that knows this GTNH release"
            if newer
            else "Open it once in GTNH 2.9 or newer, then try again"
        )
        raise Refusal(f"this world's GTNH version is unknown (core mod version {core})", action)
    if version(name) < version(min(table["releases"], key=version)):
        raise Refusal(
            f"this world was last opened in GTNH {name} (core mod version {core})",
            "Open it once in GTNH 2.9 or newer, let it load, close the game or server, then try again",
        )
    if name not in table["releases"]:
        raise Refusal(f"GTNH {name} is not supported by this version of the tool", "Update rwg2-migrator")
    return name, table["releases"][name]


def check_generator(level):
    name = level["Data"].get("generatorName", "")
    if name in MIGRATED:
        raise Refusal("this world already uses RWG2", "There is nothing to migrate; pick a world that still uses legacy RWG")
    if name.lower() != LEGACY_GENERATOR.lower():
        raise Refusal(
            f"this world uses the {name!r} generator, not legacy RWG",
            "Only worlds made with legacy RWG (the RWG world type) can be migrated",
        )
    return name


def check_files(world):
    if not os.path.isfile(os.path.join(world, OREGEN)):
        raise Refusal(
            f"{OREGEN} is missing from {world}",
            "Restore it from a backup of this world, then try again",
        )


def check(world_dir):
    """(level, report) of a world the tool can migrate; anything else is refused with the next action."""
    level = read_level(world_dir)
    world_mods = mods(level)
    name, release = release_for(world_mods, releases())
    generator = check_generator(level)
    check_files(world_dir)
    diff = mod_diff(world_mods, release)
    report = {"gtnh": name, "dreamcraft": world_mods[CORE_MOD], "generator": generator, "mods": len(world_mods), "mod_diff": diff}
    return level, report


def rwg_data(world):
    path = os.path.join(world, RWG_DATA)
    if not os.path.isfile(path):
        return {"biomeListVersion": None, "noiseImplementation": None}
    try:
        data = nbt.read_file(path).get("data", {})
        version, noise = data.get("biomeListVersion"), data.get("noiseImplementation")
        if not isinstance(version, int | None) or not isinstance(noise, str | None):
            raise TypeError("biomeListVersion or noiseImplementation has an unexpected tag type")
        return {"biomeListVersion": version, "noiseImplementation": noise}
    except MALFORMED as e:
        return {"biomeListVersion": None, "noiseImplementation": None, "error": f"{RWG_DATA} cannot be read ({e}); the game skips it"}


def mod_diff(world_mods, release):
    pack = release["mods"]
    missing = [m for m in REQUIRED_MODS if m not in world_mods]
    if missing:
        raise Refusal(f"this world's mod list has no {' or '.join(missing)}", "Open the world once with the full GTNH 2.9 pack installed")
    if world_mods["rwg2"] != release["rwg2"]:
        raise Refusal(
            f"this world has rwg2 {world_mods['rwg2']}, but GTNH ships rwg2 {release['rwg2']}",
            "Open the world once with the unmodified GTNH pack, then try again",
        )
    return {
        "beyond_pack": {m: v for m, v in sorted(world_mods.items()) if m not in pack},
        "not_in_world": sorted(m for m in pack if m not in world_mods),
        "other_version": {m: [v, pack[m]] for m, v in sorted(world_mods.items()) if m in pack and pack[m] != v},
    }


def files(folder, pattern):
    """Paths in folder that match a glob pattern; folder is taken literally even when its name holds [ ] * or ?."""
    return sorted(glob.glob(os.path.join(glob.escape(folder), pattern)))


def _why(e):
    return f"no {e} tag" if isinstance(e, KeyError) else str(e)


def claims(world):
    """{team: {(cx, cz), ...}} of overworld ServerUtilities claims."""
    out = {}
    for p in files(os.path.join(world, CLAIMS), "*.dat"):
        try:
            out[Path(p).stem] = {(c["x"], c["z"]) for c in nbt.read_file(p).get("ClaimedChunks", {}).get(OVERWORLD, [])}
        except MALFORMED as e:
            reason = f"the claims file {p} cannot be read ({_why(e)})"
            raise Refusal(reason, "Restore it from a backup, or remove it to ignore that team") from e
    return out


def player_names(root):
    path = os.path.join(root, "usercache.json")
    try:
        with open(path) as f:
            return {e["uuid"]: e["name"] for e in json.load(f)}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _player(name, data):
    x, y, z = data["Pos"]
    bed = (data["SpawnX"] >> CHUNK_SHIFT, data["SpawnZ"] >> CHUNK_SHIFT) if "SpawnX" in data and "SpawnZ" in data else None
    return {
        "name": name,
        "dim": data.get("Dimension", 0),
        "pos": [round(x, 1), round(y, 1), round(z, 1)],
        "chunk": (int(x // 16), int(z // 16)),
        "bed_chunk": bed,
    }


def _player_or_problem(name, source, load):
    try:
        return _player(name, load()), None
    except MALFORMED as e:
        return None, f"{source} cannot be read ({_why(e)}); {SKIPPED}"


def players(world, root, level):
    """(players, problems): a player file the game would skip is named in problems instead of stopping the tool."""
    names = player_names(root)
    found = [
        _player_or_problem(names.get(Path(p).stem, Path(p).stem), f"the player file {p}", lambda p=p: nbt.read_file(p))
        for p in files(os.path.join(world, "playerdata"), "*.dat")
    ]
    if "Player" in level["Data"]:
        found.append(_player_or_problem("single-player player", "the Player tag of level.dat", lambda: level["Data"]["Player"]))
    return [f for f, _ in found if f], [w for _, w in found if w]


def spawn(level):
    return level["Data"]["SpawnX"], level["Data"]["SpawnZ"]


def on_onedrive(path):
    return any("onedrive" in part.lower() for part in Path(path).parts)


def _cfg_instance_dir(data_dir, cfg_name):
    try:
        value = properties(os.path.join(data_dir, cfg_name)).get("InstanceDir", "instances")
    except OSError:
        value = "instances"
    return os.path.join(data_dir, value) if not os.path.isabs(value) else value


def _launcher_data_dirs(env, home):
    roam = env.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
    share = env.get("XDG_DATA_HOME", os.path.join(home, ".local", "share"))
    mac = os.path.join(home, "Library", "Application Support")
    flatpak = os.path.join(home, ".var", "app", "org.prismlauncher.PrismLauncher", "data")
    return [os.path.join(base, name) for base in (roam, share, mac, flatpak) for name in INSTANCE_CFG]


def game_dirs(env=None, home=None):
    """Every game folder (the one holding saves/) of the usual launchers, instance folders read from their cfg."""
    env = os.environ if env is None else env
    home = str(Path.home()) if home is None else home
    roam = env.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
    found = [
        os.path.join(roam, ".minecraft"),
        os.path.join(home, ".minecraft"),
        os.path.join(home, "Library", "Application Support", "minecraft"),
    ]
    instance_roots = [_cfg_instance_dir(d, INSTANCE_CFG[os.path.basename(d)]) for d in _launcher_data_dirs(env, home) if os.path.isdir(d)]
    for inst in instance_roots:
        found += [os.path.join(i, g) for i in files(inst, "*") for g in GAME_DIRS]
    found += files(os.path.join(home, "curseforge", "minecraft", "Instances"), "*")
    found += files(os.path.join(roam, "ATLauncher", "instances"), "*")
    return list(dict.fromkeys(d for d in found if os.path.isdir(os.path.join(d, "saves"))))


def _save(game, path):
    save = os.path.dirname(path)
    icon = os.path.join(save, "icon.png")
    try:
        data = nbt.read_file(path)["Data"]
        name, last, error = str(data.get("LevelName", os.path.basename(save))), int(data.get("LastPlayed", 0)), None
    except MALFORMED as e:
        name, last, error = os.path.basename(save), 0, f"level.dat cannot be read ({_why(e)})"
    return {
        "name": name,
        "path": save,
        "game_dir": game,
        "last_played": last,
        "icon": icon if os.path.isfile(icon) else None,
        "error": error,
    }


def saves(env=None, home=None):
    """Single-player saves, newest first; a save whose level.dat does not read is listed with its error."""
    found = [_save(game, path) for game in game_dirs(env, home) for path in files(os.path.join(game, "saves"), os.path.join("*", LEVEL))]
    return sorted(found, key=lambda s: -s["last_played"])
