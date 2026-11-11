import os
import shutil

import pytest
from conftest import level_dat, make_world, player_dat, tag, write_nbt, zero_body

from rwg2migrator import Refusal, nbt, world


def test_normalise_strips_quotes_and_spaces(tmp_path):
    assert world.normalise(f'  "{tmp_path}"  ') == str(tmp_path)
    assert world.normalise(f"'{tmp_path}'") == str(tmp_path)


def test_every_path_form_resolves_to_the_same_world(server):
    w = os.path.join(server, "World")
    forms = [server, w, os.path.join(w, "level.dat"), os.path.join(w, "region"), f'"{w}"', server + os.sep]
    assert {world.locate(f) for f in forms} == {world.Place(w, server, None, "server")}
    os.makedirs(os.path.join(server, "config"))
    assert world.locate(server).config is None
    open(os.path.join(server, "config", "rwg2.cfg"), "w").close()
    assert {world.locate(f) for f in forms} == {world.Place(w, server, os.path.join(server, "config"), "server")}


def test_single_player_save_and_plain_folder(tmp_path):
    save = tmp_path / ".minecraft" / "saves" / "My World"
    write_nbt(str(save / "level.dat"), level_dat())
    assert world.locate(str(save)) == world.Place(str(save), str(tmp_path / ".minecraft"), None, "singleplayer")
    (tmp_path / ".minecraft" / "config").mkdir()
    (tmp_path / ".minecraft" / "config" / "rwg2.cfg").write_text("")
    p = world.locate(str(save / "level.dat"))
    assert p == world.Place(str(save), str(tmp_path / ".minecraft"), str(tmp_path / ".minecraft" / "config"), "singleplayer")
    loose = tmp_path / "copied" / "World"
    write_nbt(str(loose / "level.dat"), level_dat())
    assert world.locate(str(loose)).kind == "folder"


def test_a_config_folder_next_to_a_plain_world_folder_is_used(tmp_path):
    w = tmp_path / "download" / "World"
    write_nbt(str(w / "level.dat"), level_dat())
    assert world.locate(str(w)).config is None
    (tmp_path / "download" / "config").mkdir()
    (tmp_path / "download" / "config" / "Thaumcraft.cfg").write_text("")
    assert world.locate(str(w)).config is None
    (tmp_path / "download" / "config" / "rwg2.cfg").write_text("")
    p = world.locate(str(w))
    assert p == world.Place(str(w), str(tmp_path / "download"), str(tmp_path / "download" / "config"), "folder")
    assert world.config_from(p) == "found next to the world" and world.config_from(p, "x") == world.CHOSEN


def test_a_config_folder_another_world_runs_on_is_not_used_for_a_world_next_to_it(tmp_path, server):
    shutil.copytree(os.path.join(server, "World"), os.path.join(server, "World-copy"))
    os.makedirs(os.path.join(server, "config"))
    open(os.path.join(server, "config", "rwg2.cfg"), "w").close()
    p = world.locate(os.path.join(server, "World-copy"))
    assert (p.kind, p.config, world.config_from(p)) == ("folder", None, None)
    assert world.worlds_on(os.path.join(server, "config")) == [os.path.join(server, "World")]
    game = tmp_path / "inst" / ".minecraft"
    write_nbt(str(game / "saves" / "Save" / "level.dat"), level_dat())
    write_nbt(str(game / "Loose" / "level.dat"), level_dat())
    (game / "config").mkdir()
    (game / "config" / "rwg2.cfg").write_text("")
    assert world.locate(str(game / "Loose")).config is None
    shutil.rmtree(game / "saves")
    assert world.locate(str(game / "Loose")).config == str(game / "config")
    write_nbt(str(game / "Sibling" / "level.dat"), level_dat())
    assert world.locate(str(game / "Loose")).config is None


def test_a_named_config_folder_must_hold_rwg2_cfg(tmp_path, server):
    place = world.locate(server)
    (tmp_path / "Thaumcraft.cfg").write_text("")
    with pytest.raises(Refusal, match="is not a config folder .it holds no rwg2.cfg."):
        world.with_config(place, str(tmp_path))
    (tmp_path / "rwg2.cfg").write_text("")
    assert world.with_config(place, f'"{tmp_path}"').config == str(tmp_path) and world.with_config(place, None) == place


def test_missing_folder_and_level_dat_refused(tmp_path):
    with pytest.raises(Refusal, match="does not exist"):
        world.locate(str(tmp_path / "nope"))
    with pytest.raises(Refusal, match="no level.dat"):
        world.read_level(str(tmp_path))


def test_server_without_its_world_folder_refused(tmp_path):
    (tmp_path / "server.properties").write_text("level-name=Gone\n")
    with pytest.raises(Refusal, match="no world folder 'Gone'"):
        world.locate(str(tmp_path))


def _mods(dreamcraft):
    return {"dreamcraft": dreamcraft}


def test_release_lookup():
    table = world.releases()
    assert world.release_for(_mods("2.9.99"), table)[0] == "2.9.0"
    with pytest.raises(Refusal, match=r"GTNH 2\.8\.4 \(core mod version 2\.7\.268\)") as e:
        world.release_for(_mods("2.7.268"), table)
    assert "2.9 or newer" in e.value.action
    with pytest.raises(Refusal, match="unknown") as e:
        world.release_for(_mods("9.9.9"), table)
    assert "Update rwg2-migrator" in e.value.action
    with pytest.raises(Refusal, match="no GT New Horizons core mod"):
        world.release_for({}, table)


@pytest.mark.parametrize(
    ("name", "words"), [("RWG2_CONTINENTS", "already uses RWG2"), ("RWG2", "already uses RWG2"), ("default", "not legacy RWG")]
)
def test_generator_refusals(name, words):
    with pytest.raises(Refusal, match=words):
        world.check_generator({"Data": {"generatorName": name}})
    assert world.check_generator({"Data": {"generatorName": "RWG"}}) == "RWG"


def test_mod_diff():
    release = world.releases()["releases"]["2.9.0"]
    mods = dict(release["mods"]) | {"extramod": "1.0"}
    mods.pop("JourneyMapServer")
    mods["gregtech"] = "0.0.1"
    diff = world.mod_diff(mods, release)
    assert diff == {
        "beyond_pack": {"extramod": "1.0"},
        "not_in_world": ["JourneyMapServer"],
        "other_version": {"gregtech": ["0.0.1", "5.09.54.265"]},
    }
    with pytest.raises(Refusal, match="rwg2 1.0"):
        world.mod_diff(mods | {"rwg2": "1.0"}, release)
    with pytest.raises(Refusal, match="no RWG"):
        world.mod_diff({k: v for k, v in mods.items() if k != "RWG"}, release)


def test_claims_players_and_spawn(server):
    w = os.path.join(server, "World")
    assert world.claims(w) == {"alpha": {(1, 1), (2, 1)}}
    level = world.read_level(w)
    alice = {"name": "Alice", "dim": 0, "pos": [24.0, 64.0, 24.0], "chunk": (1, 1), "bed_chunk": (1, 1)}
    assert world.players(w, server, level) == ([alice], [])
    assert world.spawn(level) == (8, 8)
    assert world.registry(level)[90] == "minecraft:portal"


def test_single_player_data_player(tmp_path):
    save = tmp_path / "saves" / "S"
    write_nbt(str(save / "level.dat"), level_dat(player=player_dat(-40, 300)))
    level = world.read_level(str(save))
    assert world.players(str(save), str(tmp_path), level)[0][0]["chunk"] == (-3, 18)


def _save(game, name, last):
    lvl = level_dat()
    lvl["Data"][1] |= {"LastPlayed": [nbt.LONG, last], "LevelName": [nbt.STRING, name]}
    write_nbt(os.path.join(game, "saves", name, "level.dat"), lvl)


def test_saves_listing_reads_prism_instance_dir(tmp_path):
    home = str(tmp_path / "home")
    share = os.path.join(home, ".local", "share")
    prism = os.path.join(share, "PrismLauncher")
    os.makedirs(prism)
    moved = str(tmp_path / "big-disk" / "instances")
    with open(os.path.join(prism, "prismlauncher.cfg"), "w") as f:
        f.write(f"[General]\nInstanceDir={moved}\n")
    _save(os.path.join(moved, "GTNH", ".minecraft"), "Base", 2000)
    _save(os.path.join(home, ".minecraft"), "Vanilla", 1000)
    _save(os.path.join(home, "curseforge", "minecraft", "Instances", "Pack"), "Curse", 3000)
    found = world.saves(env={"XDG_DATA_HOME": share}, home=home)
    assert [s["name"] for s in found] == ["Curse", "Base", "Vanilla"]
    assert found[1]["path"] == os.path.join(moved, "GTNH", ".minecraft", "saves", "Base")


def test_onedrive_warning():
    assert world.on_onedrive("C:/Users/x/OneDrive/Documents/server/World")
    assert not world.on_onedrive("/srv/gtnh/World")


def test_folder_names_with_glob_characters_are_literal(tmp_path):
    root = str(tmp_path / "GTNH server [2.9]")
    w = make_world(root)
    assert world.claims(w) == {"alpha": {(1, 1), (2, 1)}}
    assert [p["name"] for p in world.players(w, root, world.read_level(w))[0]] == ["Alice"]
    home = str(tmp_path / "home [x]")
    _save(os.path.join(home, ".minecraft"), "Base", 1)
    _save(os.path.join(home, "curseforge", "minecraft", "Instances", "Pack [1]"), "Curse", 2)
    assert [s["name"] for s in world.saves(env={}, home=home)] == ["Curse", "Base"]


def test_unreadable_nbt_names_the_file(tmp_path):
    empty, deflate = tmp_path / "empty.dat", tmp_path / "deflate.dat"
    empty.write_bytes(b"")
    write_nbt(str(deflate), level_dat())
    zero_body(deflate)
    for p in (empty, deflate):
        with pytest.raises(nbt.NBTError, match=p.name):
            nbt.read_file(p)


def test_level_dat_that_does_not_decode_or_lacks_fml_is_refused(server):
    w = os.path.join(server, "World")
    zero_body(os.path.join(w, "level.dat"))
    with pytest.raises(Refusal, match="cannot be read") as e:
        world.read_level(w)
    assert "level.dat_old" in e.value.action
    for drop in (("FML",), ("FML", "ItemData"), ("Data",), ("Data", "RandomSeed")):
        lvl = level_dat()
        holder = lvl if len(drop) == 1 else lvl[drop[0]][1]
        del holder[drop[-1]]
        write_nbt(os.path.join(w, "level.dat"), lvl)
        with pytest.raises(Refusal, match=f"no {' '.join(drop)} tag") as e:
            world.read_level(w)
        assert "Open the world once" in e.value.action
    lvl = level_dat()
    lvl["FML"][1]["ModList"] = tag(nbt.INT, 0)
    write_nbt(os.path.join(w, "level.dat"), lvl)
    with pytest.raises(Refusal, match="no FML ModList tag"):
        world.read_level(w)


def test_player_files_the_game_skips_are_named_not_fatal(server):
    w = os.path.join(server, "World")
    open(os.path.join(w, "playerdata", "dead.dat"), "wb").close()
    write_nbt(os.path.join(w, "playerdata", "nopos.dat"), {"Dimension": tag(nbt.INT, 0)})
    level = level_dat(player={"Dimension": tag(nbt.INT, 0)})
    found, problems = world.players(w, server, nbt.plain(nbt.COMPOUND, level))
    assert [p["name"] for p in found] == ["Alice"] and len(problems) == 3
    assert "dead.dat" in problems[0] and "nopos.dat" in problems[1] and "no 'Pos' tag" in problems[1]
    assert "Player tag of level.dat" in problems[2] and all("the game skips it" in p for p in problems)


def test_unreadable_rwg_data_and_claims(server):
    w = os.path.join(server, "World")
    open(os.path.join(w, "data", "rwg_data.dat"), "wb").close()
    rd = world.rwg_data(w)
    assert rd["noiseImplementation"] is None and "rwg_data.dat cannot be read" in rd["error"]
    write_nbt(os.path.join(w, "data", "rwg_data.dat"), {"data": tag(nbt.COMPOUND, {"noiseImplementation": tag(nbt.BYTE_ARRAY, b"x")})})
    assert "unexpected tag type" in world.rwg_data(w)["error"]
    zero_body(os.path.join(w, world.CLAIMS, "alpha.dat"))
    with pytest.raises(Refusal, match="alpha.dat cannot be read") as e:
        world.claims(w)
    assert "remove it to ignore that team" in e.value.action


def test_saves_lists_a_broken_save_with_its_error(tmp_path):
    home = str(tmp_path / "home")
    game = os.path.join(home, ".minecraft")
    _save(game, "Good", 5)
    _save(game, "Broken", 9)
    zero_body(os.path.join(game, "saves", "Broken", "level.dat"))
    write_nbt(os.path.join(game, "saves", "Odd", "level.dat"), {"Data": tag(nbt.INT, 1)})
    found = world.saves(env={}, home=home)
    assert [(s["name"], s["error"] is None) for s in found] == [("Good", True), ("Broken", False), ("Odd", False)]
    assert "cannot be read" in found[1]["error"]


def test_a_world_name_java_wrote_with_escapes_is_found(tmp_path):
    name = "Monde été!"
    (tmp_path / "server.properties").write_text("#Minecraft server properties\nlevel-name=Monde \\u00E9t\\u00E9\\!\n", encoding="latin-1")
    write_nbt(str(tmp_path / name / "level.dat"), level_dat())
    assert world.java_unescape("a\\:b\\ c\\\\d\\u00e9") == "a:b c\\dé"
    for path in (tmp_path, tmp_path / name):
        place = world.locate(str(path))
        assert (place.world, place.kind) == (str(tmp_path / name), "server")
