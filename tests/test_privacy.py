"""No tracked file, nor any file inside a tracked jar or zip, holds a string of the private deny-list that RWG2_PRIVACY_DENYLIST
names (one string per line, kept outside the repo)."""

import os
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DENYLIST = os.environ.get("RWG2_PRIVACY_DENYLIST")


def _token(s):
    head = rb"\b" if re.match(r"\w", s[0]) else b""
    tail = rb"\b" if re.match(r"\w", s[-1]) else b""
    return head + re.escape(s.encode()) + tail


def denied(strings):
    """One case-blind pattern; an end that is a letter or digit must be a word edge, so "ab" misses "abc" but "1.2." hits "1.2.3"."""
    return re.compile(b"|".join(_token(s) for s in strings), re.IGNORECASE)


def blobs(rel):
    """(where, bytes) of a tracked file: its path and content, then each member of it when it is a jar or zip."""
    path = ROOT / rel
    yield rel, rel.encode() + b"\n" + path.read_bytes()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                yield f"{rel}!{name}", name.encode() + b"\n" + z.read(name)


def test_a_denied_string_matches_as_a_whole_token_at_its_word_ends():
    rx = denied(["ab", "1.2.", "x@"])
    hit = [bool(rx.search(t)) for t in (b"an AB.", b"1.2.34", b"to x@y", b"abc", b"cab", b"21.2.3")]
    assert hit == [True, True, True, False, False, False]


@pytest.mark.skipif(not DENYLIST, reason="set RWG2_PRIVACY_DENYLIST to the private deny-list file, one string per line")
def test_no_tracked_file_holds_a_denied_string():
    rx = denied([s for s in Path(DENYLIST).read_text(encoding="utf-8").splitlines() if s.strip()])
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    hits = [where for rel in files if rel and (ROOT / rel).is_file() for where, data in blobs(rel) if rx.search(data)]
    assert hits == []
