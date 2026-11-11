import pytest

from rwg2migrator import forgecfg

SU = """# Configuration file\r
\r
auto_shutdown {\r
    B:enabled=true\r
    S:times <\r
        04:00\r
     >\r
}\r
\r
backups {\r
    S:backup_folder_path=/srv/backups/server/\r
    B:enable_backups=true\r
    B:enabled=true\r
}\r
"""


def test_set_keys_changes_only_the_named_values(tmp_path):
    path = tmp_path / "su.cfg"
    path.write_bytes(SU.encode())
    forgecfg.set_keys(str(path), {("backups", "enable_backups"): ("B", "false"), ("auto_shutdown", "enabled"): ("B", "false")})
    after = path.read_bytes().decode()
    assert after == SU.replace("    B:enabled=true\r\n    S:times", "    B:enabled=false\r\n    S:times").replace(
        "B:enable_backups=true", "B:enable_backups=false"
    )
    assert forgecfg.get(str(path), ("backups", "enabled")) == "true"
    assert forgecfg.get(str(path), ("backups", "backup_folder_path")) == "/srv/backups/server/"


def test_a_missing_key_is_an_error_and_writes_nothing(tmp_path):
    path = tmp_path / "su.cfg"
    path.write_bytes(SU.encode())
    with pytest.raises(forgecfg.CfgError, match="backups/nope"):
        forgecfg.set_keys(str(path), {("backups", "nope"): ("B", "false")})
    assert path.read_bytes() == SU.encode()


def test_a_missing_file_gets_only_the_keys(tmp_path):
    path = tmp_path / "rwg2.cfg"
    edits = {
        ("expert / worldgen migration", "Biome Placement Offset X"): ("I", "-12000"),
        ("expert / worldgen migration", "Landmass Placement Offset Z"): ("I", "6144"),
    }
    forgecfg.set_keys(str(path), edits)
    text = path.read_text()
    assert text.count("{") == 1 and '"expert / worldgen migration" {' in text
    assert forgecfg.get(str(path), ("expert / worldgen migration", "Biome Placement Offset X")) == "-12000"
    assert forgecfg.get(str(path), ("expert / worldgen migration", "Landmass Placement Offset Z")) == "6144"
    forgecfg.set_keys(str(path), {("expert / worldgen migration", "Biome Placement Offset X"): ("I", "5")})
    assert forgecfg.get(str(path), ("expert / worldgen migration", "Biome Placement Offset X")) == "5"
    assert forgecfg.get(str(tmp_path / "none.cfg"), ("a", "b")) is None


def test_a_file_that_is_not_utf_8_keeps_every_other_byte(tmp_path):
    path = tmp_path / "rwg2.cfg"
    raw = b"# r\xe9glages \x81\r\ngeneral {\r\n    I:offset=0\r\n    S:nom=\xe9t\xe9\r\n}\r\n"
    path.write_bytes(raw)
    assert forgecfg.get(str(path), ("general", "offset")) == "0"
    forgecfg.set_keys(str(path), {("general", "offset"): ("I", "42")})
    assert path.read_bytes() == raw.replace(b"I:offset=0", b"I:offset=42")
