"""Key-level edits of Forge .cfg files: only the named values change, every other byte stays as it was."""

import os
import re
from itertools import groupby

from . import Refusal
from .region import write_atomic

LINE = re.compile(r'^(\s*)([A-Z]):("?)([^"=]+)\3=(.*?)(\r?\n?)$')
OPEN = re.compile(r'^\s*"?([^"{]+?)"?\s*\{\s*$')
CLOSE = re.compile(r"^\s*\}\s*$")
INDENT = "    "
KEEP_BYTES = "surrogateescape"


class CfgError(ValueError):
    pass


def _walk(lines):
    """Yield (line, key path or None, value match or None) for every line."""
    path = []
    for ln in lines:
        m = None
        if OPEN.match(ln):
            path.append(OPEN.match(ln).group(1).strip())
        elif CLOSE.match(ln) and path:
            path.pop()
        else:
            m = LINE.match(ln)
        yield ln, (*path, m.group(4)) if m else None, m


def apply(lines, edits):
    """edits: {(category, ..., key): (type letter, value)}; each key must appear exactly once."""
    hits = dict.fromkeys(edits, 0)
    out = []
    for ln, key, m in _walk(lines):
        if key in edits:
            hits[key] += 1
            ln = f"{m.group(1)}{m.group(2)}:{m.group(3)}{m.group(4)}{m.group(3)}={edits[key][1]}{m.group(6)}"
        out.append(ln)
    bad = {"/".join(k): n for k, n in hits.items() if n != 1}
    if bad:
        raise CfgError(f"keys not found exactly once: {bad}")
    return out


def _quoted(name):
    return f'"{name}"' if " " in name else name


def minimal(edits):
    """A new file holding only these keys (one category level); Forge adds every other key with its default."""
    out = ["# Configuration file\n"]
    for cat, keys in groupby(sorted(edits.items()), key=lambda kv: kv[0][0]):
        out.append(f"{_quoted(cat)} {{\n")
        out += [f"{INDENT}{kind}:{_quoted(key[-1])}={value}\n" for key, (kind, value) in keys]
        out.append("}\n\n")
    return out


def _lines(path):
    with open(path, encoding="utf-8", errors=KEEP_BYTES, newline="") as f:
        return f.readlines()


def set_keys(path, edits):
    lines = apply(_lines(path), edits) if os.path.exists(path) else minimal(edits)
    write_atomic(path, "".join(lines).encode("utf-8", KEEP_BYTES))


def edit(path, edits, shown=None):
    """set_keys, with a file that holds a setting twice or lacks one refused in plain words; shown: the file to name, when path
    is a copy of it."""
    try:
        set_keys(path, edits)
    except CfgError as e:
        raise Refusal(
            f"{shown or path} has a setting twice, or is missing one the tool writes ({e})",
            "Start the server or game once so it rewrites the file, then run this again",
        ) from e


def get(path, key):
    """The value of one key, or None when the file or the key is missing."""
    if not os.path.exists(path):
        return None
    return next((m.group(5) for _, k, m in _walk(_lines(path)) if k == tuple(key)), None)
