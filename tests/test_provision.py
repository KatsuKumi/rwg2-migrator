import hashlib
import http.server
import io
import json
import os
import stat
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

from rwg2migrator import Refusal, provision

BLOB = os.urandom(300_000)
JAVA25 = "    java.version = 25.0.5\n    os.arch = amd64\n    sun.arch.data.model = 64\n"


class Files(http.server.BaseHTTPRequestHandler):
    """Serves server.files; honours Range unless server.ignore_range; closes after server.cut bytes once."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = self.server.files[self.path.split("?")[0]]
        start = 0
        rng = self.headers.get("Range")
        if rng and not self.server.ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.server.requests.append(rng)
        end = len(body)
        if self.server.cut:
            end, self.server.cut = start + self.server.cut, 0
        self.wfile.write(body[start:end])


@pytest.fixture
def web():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Files)
    server.files, server.requests, server.cut, server.ignore_range = {"/pack.zip": BLOB}, [], 0, False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield server
    server.shutdown()


def download(web, part, events):
    provision.fetch(provision.Download(web.url + "/pack.zip", str(part), len(BLOB), events.append))


def test_interrupted_download_continues_with_range(web, tmp_path):
    part, events = tmp_path / "d" / "pack.zip.part", []
    web.cut = len(BLOB) // 2
    download(web, part, events)
    assert part.read_bytes() == BLOB
    starts = [e for e in events if e["event"] == "start"]
    assert [e["event"] for e in events if e["event"] != "progress"] == ["start", "interrupted", "start"]
    assert starts[0]["resume_from"] == 0 and starts[1] == starts[1] | {"resume_from": len(BLOB) // 2, "http": 206}
    assert web.requests == [None, f"bytes={len(BLOB) // 2}-"]


def test_a_server_that_ignores_range_restarts_from_zero(web, tmp_path):
    part, events = tmp_path / "pack.zip.part", []
    part.write_bytes(b"x" * 1000)
    web.ignore_range = True
    download(web, part, events)
    assert part.read_bytes() == BLOB and events[0]["resume_from"] == 0 and events[0]["http"] == 200


def rel_for(url, blob=BLOB):
    zip_pin = {"url": url, "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    return provision.Release("9.9.9", {"server_zip": zip_pin, "files": "none.json"})


def test_obtain_zip_verifies_and_keeps_only_a_matching_download(web, tmp_path, monkeypatch):
    monkeypatch.setattr(provision, "file_table", lambda entry: {})
    rel = rel_for(web.url + "/pack.zip")
    path = provision.obtain_zip(str(tmp_path), rel)
    assert Path(path).read_bytes() == BLOB and not os.path.exists(path + ".part")
    web.files["/other.zip"] = BLOB[::-1]
    other = provision.Release("9.9.9", rel.entry | {"server_zip": rel.entry["server_zip"] | {"url": web.url + "/other.zip"}})
    with pytest.raises(Refusal, match="server zip this tool expects") as e:
        provision.obtain_zip(str(tmp_path), other)
    assert "re-uploaded" in e.value.action
    assert os.listdir(tmp_path / "downloads") == ["pack.zip"]


def test_a_user_zip_that_differs_is_refused(tmp_path):
    wrong = tmp_path / "GT_New_Horizons_2.9.0_Server_Java_17-26.zip"
    wrong.write_bytes(BLOB[:-1])
    with pytest.raises(Refusal, match=r"is not the GTNH 9.9.9 server zip this tool expects \(300,000 bytes"):
        provision.verify_zip(str(wrong), rel_for("https://x/GT.zip"))


def make_zip(path, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in files.items():
            z.writestr(name, body)
    return provision.zip_table(path)


PACK = {"config/a.cfg": b"a=1\n", "mods/x.jar": b"jar bytes", "server.properties": b"level-name=World\n"}


def test_unpack_checks_every_file_then_renames(tmp_path):
    table = make_zip(tmp_path / "p.zip", PACK)
    folder = str(tmp_path / "cache" / "packs" / "9.9.9")
    events = []
    provision.unpack(str(tmp_path / "p.zip"), folder, table, events.append)
    assert Path(folder, "config.pristine", "a.cfg").read_bytes() == b"a=1\n"
    assert not os.path.exists(folder + provision.PARTIAL) and events[-1] == {"step": "unpack", "event": "done", "files": 3}
    jars = {"mods/x.jar": table["mods/x.jar"]}
    assert provision.check_files(folder, jars) == []
    with open(os.path.join(folder, "mods", "x.jar"), "ab") as f:
        f.write(b"!")
    assert provision.check_files(folder, jars) == ["mods/x.jar"]
    assert provision.damaged_files(folder, table) == ["mods/x.jar"]


def test_the_damage_message_fits_the_case(monkeypatch):
    monkeypatch.setattr(provision.os, "name", "nt")
    e = provision.damaged(["mods/x.jar"], "C:/cache")
    assert e.reason.startswith("Windows Defender (or your antivirus) removed or changed mods/x.jar") and "#26655" in e.reason
    assert "allow the folder C:/cache" in e.action and e.action.endswith("rwg2-migrator generator setup")
    many = provision.damaged(["mods/x.jar", "config/a.cfg", "java9args.txt"], "C:/cache")
    assert many.reason == "mods/x.jar and 2 more files of the generator are missing or changed" and "Defender" not in many.reason
    assert many.action == provision.RESTORE_ACTION
    monkeypatch.setattr(provision.os, "name", "posix")
    assert "Defender" not in provision.damaged(["mods/x.jar"], "/c").reason
    assert provision.damaged(["mods/x.jar", "a"], "/c").reason.startswith("mods/x.jar and 1 more file of")
    again = provision.damaged(["mods/x.jar"], "/c", again=True)
    assert "right after being written" in again.reason and "allow the folder /c" in again.action


def test_unpack_refuses_a_file_that_does_not_match(tmp_path):
    table = make_zip(tmp_path / "p.zip", PACK)
    table["mods/x.jar"] = [table["mods/x.jar"][0], 1]
    folder = str(tmp_path / "packs" / "9.9.9")
    with pytest.raises(Refusal, match="mods/x.jar"):
        provision.unpack(str(tmp_path / "p.zip"), folder, table)
    assert not os.path.exists(folder)


def test_install_from_a_user_zip_skips_the_download(tmp_path, monkeypatch):
    blob_path = tmp_path / "user.zip"
    table = make_zip(blob_path, PACK)
    monkeypatch.setattr(provision, "file_table", lambda entry: table)
    monkeypatch.setattr(provision, "fetch", lambda d: pytest.fail("no download expected"))
    rel = rel_for("https://x/GT.zip", blob_path.read_bytes())
    events = []
    folder = provision.install(str(tmp_path / "cache"), rel, str(blob_path), events.append)
    assert [e["event"] for e in events if e["event"] != "progress"] == ["verified", "zip", "done"] and events[0]["sha256"] == rel.entry[
        "server_zip"
    ]["sha256"]
    assert events[1] == {"step": "pack", "event": "zip", "path": str(blob_path), "download": False}
    assert os.path.isfile(os.path.join(folder, "mods", "x.jar")) and blob_path.exists()
    assert provision.install(str(tmp_path / "cache"), rel, None, events.append) == folder and events[-1]["event"] == "present"


def damage(folder):
    os.remove(os.path.join(folder, "mods", "x.jar"))
    Path(folder, provision.PRISTINE, "a.cfg").write_text("changed by a power cut")
    Path(folder, "server.properties").write_text("rewritten by every boot\n")
    Path(folder, "gen-1").mkdir()


def test_a_damaged_pack_is_restored_from_the_zip(tmp_path, monkeypatch):
    blob_path = tmp_path / "user.zip"
    table = make_zip(blob_path, PACK)
    monkeypatch.setattr(provision, "file_table", lambda entry: table)
    rel = rel_for("https://x/GT.zip", blob_path.read_bytes())
    folder = provision.install(str(tmp_path / "cache"), rel, str(blob_path))
    damage(folder)
    with pytest.raises(Refusal, match="are missing or changed"):
        provision.check_pack(folder, rel.entry, str(tmp_path / "cache"))
    events = []
    monkeypatch.setattr(provision, "fetch", lambda d: pytest.fail("the user's zip is used"))
    assert provision.install(str(tmp_path / "cache"), rel, str(blob_path), events.append) == folder
    damaged = next(e for e in events if e["event"] == "damaged")
    assert damaged == {"step": "pack", "event": "damaged", "files": 2, "first": "config/a.cfg"}
    assert events[-1] == {"step": "pack", "event": "restored", "files": 2} and blob_path.exists()
    assert provision.damaged_files(folder, table) == [] and Path(folder, "gen-1").is_dir()
    assert Path(folder, "server.properties").read_text() == "rewritten by every boot\n"


def test_a_pack_restored_from_a_download_drops_the_zip_and_a_file_that_changes_again_is_named(web, tmp_path, monkeypatch):
    blob_path = tmp_path / "pack.zip"
    table = make_zip(blob_path, PACK)
    web.files["/pack.zip"] = blob_path.read_bytes()
    monkeypatch.setattr(provision, "file_table", lambda entry: table)
    rel = rel_for(web.url + "/pack.zip", blob_path.read_bytes())
    cache = str(tmp_path / "cache")
    folder = provision.install(cache, rel)
    assert os.listdir(os.path.join(cache, "downloads")) == []
    damage(folder)
    real_restore = provision.restore
    monkeypatch.setattr(provision, "restore", lambda *a: None)
    with pytest.raises(Refusal, match="changed right after being written") as e:
        provision.install(cache, rel)
    assert "allow the folder" in e.value.action and os.listdir(os.path.join(cache, "downloads")) == ["pack.zip"]
    monkeypatch.setattr(provision, "restore", real_restore)
    provision.install(cache, rel)
    assert provision.damaged_files(folder, table) == [] and os.listdir(os.path.join(cache, "downloads")) == []


def test_the_cache_folder_must_be_new_empty_or_the_tools_own(tmp_path):
    new = tmp_path / "new" / "cache"
    assert provision.claim_cache(str(new)) == str(new) and (new / provision.CACHE_MARKER).is_file()
    (new / "packs").mkdir()
    assert provision.claim_cache(str(new)) == str(new)
    (tmp_path / "empty").mkdir()
    assert provision.own_cache(provision.claim_cache(str(tmp_path / "empty")))
    (tmp_path / "home" / "Downloads").mkdir(parents=True)
    with pytest.raises(Refusal, match="already holds other files"):
        provision.claim_cache(str(tmp_path / "home"))
    assert os.listdir(tmp_path / "home") == ["Downloads"]


def fake_java(path, settings):
    if os.name == "nt":
        pytest.skip("the stand-in for Java is a shell script")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "$@" > "$(dirname "$0")/args"\ncat >&2 <<\'EOF\'\nProperty settings:\n{settings}EOF\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


SPAN = {"min": 17, "max": 26}


def test_java_settings_are_read_and_checked(tmp_path):
    j = provision.java_info(fake_java(tmp_path / "a" / "java", JAVA25))
    assert (j.major, j.arch, j.bits) == (25, "amd64", "64") and provision.why_not(j, SPAN) is None
    args = Path(tmp_path / "a" / "args").read_text().split()
    assert args == ["-XX:-UsePerfData", "-Djdk.disableLastUsageTracking=true", "-XshowSettings:properties", "-version"]
    j8 = provision.java_info(fake_java(tmp_path / "b" / "java", "    java.version = 1.8.0_462\n    os.arch = amd64\n"))
    assert j8.major == 8 and "outside the Java 17-26" in provision.why_not(j8, SPAN)
    j32 = provision.java_info(
        fake_java(tmp_path / "c" / "java", "    java.version = 21.0.2\n    os.arch = x86\n    sun.arch.data.model = 32\n")
    )
    assert "32-bit" in provision.why_not(j32, SPAN)
    assert provision.java_info(str(tmp_path / "missing")) is None


def test_choose_java_prefers_the_newest_fitting_one_and_rejects_others(tmp_path, monkeypatch):
    home = tmp_path / "home21"
    fake_java(home / "bin" / "java", JAVA25.replace("25.0.5", "21.0.4"))
    on_path = fake_java(tmp_path / "path" / "java", JAVA25)
    old = fake_java(tmp_path / "old" / "bin" / "java", "    java.version = 1.8.0_462\n    os.arch = amd64\n")
    monkeypatch.setenv("JAVA_HOME", str(home))
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "path"), "/bin", "/usr/bin"]))
    monkeypatch.setattr(provision, "_install_globs", lambda env: [old])
    monkeypatch.setattr(provision, "temurin", lambda *a: pytest.fail("no download expected"))
    events = []
    assert provision.choose_java(str(tmp_path / "cache"), SPAN, provision.SEARCH, events.append).path == on_path
    assert sorted(e["event"] for e in events) == ["found", "found", "rejected"]
    with pytest.raises(Refusal, match="outside the Java 17-26"):
        provision.choose_java(str(tmp_path / "cache"), SPAN, provision.JavaWanted(old))


def jre_archive(settings, program="bin/java"):
    script = f"#!/bin/sh\ncat >&2 <<'EOF'\nProperty settings:\n{settings}EOF\n".encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        info = tarfile.TarInfo(f"jdk-25.0.1+8-jre/{program}")
        info.size, info.mode = len(script), 0o755
        t.addfile(info, io.BytesIO(script))
    return buf.getvalue()


def adoptium(web, archive, checksum):
    pkg = {"checksum": checksum, "link": web.url + "/jre.tar.gz", "name": "OpenJDK25U-jre.tar.gz", "size": len(archive)}
    web.files["/jre.tar.gz"] = archive
    web.files["/api"] = json.dumps([{"binary": {"package": pkg}, "release_name": "jdk-25.0.1+8-jre"}]).encode()


def test_a_downloaded_jre_that_holds_no_java_program_is_refused(web, tmp_path, monkeypatch):
    archive = jre_archive(JAVA25, program="release")
    adoptium(web, archive, hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(provision, "ADOPTIUM", web.url + "/api?{feature}{arch}{os}")
    with pytest.raises(Refusal, match="the Java just installed in .* does not start"):
        provision.choose_java(str(tmp_path / "cache"), SPAN, provision.JavaWanted(search=False))


@pytest.mark.skipif(os.name == "nt", reason="the stand-in for Java is a shell script")
def test_without_java_a_checked_temurin_jre_is_downloaded(web, tmp_path, monkeypatch):
    archive = jre_archive(JAVA25)
    adoptium(web, archive, hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(provision, "ADOPTIUM", web.url + "/api?{feature}{arch}{os}")
    events = []
    cache = str(tmp_path / "cache")
    java = provision.choose_java(cache, SPAN, provision.JavaWanted(search=False), events.append)
    assert java.major == 25 and java.path == os.path.join(cache, "java", "jdk-25.0.1+8-jre", "bin", "java")
    assert events[-1]["event"] == "installed" and events[-1]["checksum"] == hashlib.sha256(archive).hexdigest()
    assert not os.listdir(os.path.join(cache, "downloads"))
    assert provision.choose_java(cache, SPAN, provision.JavaWanted(search=False)).path == java.path


def test_a_jre_with_the_wrong_checksum_is_refused(web, tmp_path, monkeypatch):
    adoptium(web, jre_archive(JAVA25), "0" * 64)
    monkeypatch.setattr(provision, "ADOPTIUM", web.url + "/api?{feature}{arch}{os}")
    with pytest.raises(Refusal, match="does not match the checksum"):
        provision.choose_java(str(tmp_path / "cache"), SPAN, provision.JavaWanted(search=False))
    assert not os.path.exists(tmp_path / "cache" / "java")


UNICODE_CACHE = "D:\\gtnh\\Кэш-日本"


def test_a_java_path_outside_the_ansi_code_page_uses_its_short_form_or_is_refused(tmp_path, monkeypatch, web):
    monkeypatch.setattr(provision, "ANSI", "cp1252")
    shorts = {UNICODE_CACHE: "D:\\gtnh\\KE5F~1"}
    monkeypatch.setattr(provision, "_short", shorts.get)
    assert provision.java_safe("C:\\Program Files\\Java\\café\\java.exe") == "C:\\Program Files\\Java\\café\\java.exe"
    assert provision.java_safe(UNICODE_CACHE) == "D:\\gtnh\\KE5F~1" and provision.java_safe("D:\\Мир\\java.exe") is None
    assert provision.java_info("D:\\Мир\\java.exe") is None
    cache = tmp_path / "Мир"
    monkeypatch.setattr(provision, "ADOPTIUM", web.url + "/never-asked")
    with pytest.raises(Refusal, match="characters Windows cannot hand to Java") as e:
        provision.temurin(str(cache), SPAN)
    assert "--cache a folder whose path has only plain letters" in e.value.action and web.requests == []
