"""Every refusal a step can end with must read in on-screen terms once the page has rewritten it: no command, no flag."""

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "rwg2migrator"
REFUSALS = {"Refusal", "GoAhead", "LowMemory"}
RULE = re.compile(r"""\[\s*/((?:[^/\\\n]|\\.)+)/([a-z]*),\s*(['"])((?:\\.|(?!\3).)*)\3\s*\]""")
COMMANDS = "ui|jobs|area|search|preview|apply|rollback|verify|candidates|generator|inspect|map|clean"
COMMAND_LINE = re.compile(rf"rwg2-migrator (?:{COMMANDS})\b|(?<![\w-])--[a-z]")
COMMAND_LINE_ONLY = (
    "Apply needs your go-ahead",
    "Rollback loses all play since Apply",
    "a Roll back of {place.world} stopped before it finished",
    "job {preview.id} is a {d.get('kind')} job, not a preview",
    "job {preview.id} is not a preview of {named.world}",
    "job {searched.id} is not a finished search",
    "no job was named",
    "job {job.id} is running",
    "the preferences file {path} cannot be read ({e})",
    "a margin of {margin} chunks leaves no safety band",
    "the area to keep is empty",
    "the cache folder {cache} is inside the world folder",
    "the folder for --out {path} does not exist",
    "--out {path} is a folder",
    "--out {path} is inside the world folder",
    "nothing to clean was named",
    "{cache} is not a cache folder of rwg2-migrator",
    "a ring of {args.ring} chunks leaves the chunks next to the kept area undecorated",
    "--trees and --blend-biomes work on the smoothed border",
    "--bpp {bpp} is not one of",
    "--window {text} is not X0,Z0,SIZE",
    "GTNH {name} is not a release this tool knows",
    "the cache folder {cache} already holds other files",
    "{explicit} does not run as Java",
    "there is no suggested extra part",
    "{reason}",
)
NO_PLACE_ON_THE_PAGE = ("the cache folder {base} is too long",)


def _text(node, names):
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):
        return "".join(str(v.value) if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _text(node.left, names) + _text(node.right, names)
    if isinstance(node, ast.IfExp):
        return _text(node.body, names) + " / " + _text(node.orelse, names)
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    return "{" + ast.unparse(node) + "}"


def refusals():
    """(where, reason, action) of every refusal the package raises, with each placeholder left in braces."""
    out = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        text = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant | ast.JoinedStr | ast.BinOp)]
        names = {n.targets[0].id: _text(n.value, {}) for n in text if isinstance(n.targets[0], ast.Name)}
        for n in ast.walk(tree):
            called = n.func if isinstance(n, ast.Call) else None
            name = getattr(called, "id", getattr(called, "attr", None))
            if name in REFUSALS and len(n.args) > 1:
                out.append((f"{path.name}:{n.lineno}", _text(n.args[0], names), _text(n.args[1], names)))
    return out


def page_rules(table):
    """The page's rewrite rules of one table (ACTIONS or REASONS) whose replacement is plain text."""
    script = (PACKAGE / "web" / "static" / "app.js").read_text(encoding="utf-8")
    block = script.split(f"const {table} = [", 1)[1].split("\n];", 1)[0]
    return [(re.compile(p, re.I if "i" in flags else 0), text.replace("\\'", "'")) for p, flags, _, text in RULE.findall(block)]


def on_page(text, rules):
    """text after the page's rules, each applied once as the browser applies them ($1, $2 are the rule's groups)."""
    for pattern, plain in rules:
        text = pattern.sub(lambda m, plain=plain: re.sub(r"\$(\d)", lambda g: m.group(int(g[1])) or "", plain), text, count=1)
    return text


def test_no_refusal_reaches_the_page_as_a_command_or_a_flag():
    actions = page_rules("ACTIONS")
    assert len(actions) > 15, "the page's ACTIONS table was not read"
    found = refusals()
    assert len(found) > 150, "the package's refusals were not read"
    left = [
        f"{where}: {reason} | {on_page(action, actions)}"
        for where, reason, action in found
        if not reason.startswith(COMMAND_LINE_ONLY + NO_PLACE_ON_THE_PAGE)
        and (COMMAND_LINE.search(reason) or COMMAND_LINE.search(on_page(action, actions)))
    ]
    assert not left, "\n".join(left)
    unused = [p for p in COMMAND_LINE_ONLY + NO_PLACE_ON_THE_PAGE if not any(reason.startswith(p) for _, reason, _ in found)]
    assert not unused, unused


def test_the_rules_turn_command_line_advice_into_what_the_page_offers():
    actions = page_rules("ACTIONS")
    said = {
        "Roll back (this loses the play since Apply): rwg2-migrator rollback /srv/World": "click Roll back on the Verify screen (this "
        "loses the play since Apply)",
        "Roll back: rwg2-migrator rollback /srv/World": "click Roll back on the Verify screen",
        "Free some space, or pass --cache a new, empty folder on a drive with more room": "Free some space on that drive",
        "Free some space on that drive, then run apply again": "Free some space on that drive, then click Apply now again",
        "Stop the server or close the game, then run this again": "Stop the server or close the game, then try again",
        "Run preview again and apply the new preview": "build the preview again and apply the new preview",
        "Pass --cache a shorter folder, for example C:\\rwg2-cache": "close the app, then start it again from a console with a shorter "
        "cache folder: rwg2-migrator ui --cache C:\\rwg2-cache",
    }
    assert {text: on_page(text, actions) for text in said} == said
