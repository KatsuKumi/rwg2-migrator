import pytest

from rwg2migrator import verdicts

GREEN1 = {"median": 1.0, "p90": 9.0, "over10": 0.0772, "over30": 0.0, "max": 21.0, "feasible": True, "same_edge": 0.73}


def test_border_thresholds():
    assert verdicts.border(GREEN1) == ("very close", "Border: very close (most steps 1 block; smoothing evens the rest)")
    assert verdicts.border(GREEN1 | {"over10": 0.11})[0] == "close"
    assert verdicts.border(GREEN1 | {"median": 3.0, "over10": 0.3, "over30": 0.02})[0] == "close"
    rough = verdicts.border(GREEN1 | {"median": 4.0, "over30": 0.1, "max": 37.0})
    assert rough == ("rough", "Border: rough (steps up to 37 blocks; smoothing ramps them, but some cliffs may show)")


def test_excellent_needs_border_look_and_land():
    assert verdicts.excellent(GREEN1)
    assert not verdicts.excellent(GREEN1 | {"same_edge": 0.69})
    assert not verdicts.excellent(GREEN1 | {"feasible": False})
    assert not verdicts.excellent(GREEN1 | {"median": 2.0})


@pytest.mark.parametrize(
    ("ocean", "below", "line"),
    [
        (524.8, 0.009, "Water: no ocean within 500 blocks"),
        (None, 0.0, "Water: no ocean within 3500 blocks"),
        (50.0, 0.0, "Water: ocean within 100 blocks"),
        (816.0, 0.0122, "Water: no ocean within 800 blocks; some pools near the border"),
    ],
)
def test_water(ocean, below, line):
    assert verdicts.water(ocean, below) == line


def test_taint_and_compass():
    assert verdicts.compass(0, -330) == "north" and verdicts.compass(300, -10) == "east"
    assert verdicts.compass(-100, -100) == "north-west" and verdicts.compass(5, 400) == "south"
    near = verdicts.taint({"blocks": 330.1, "direction": "north"})
    assert near == "Tainted Land: 330 blocks north." + verdicts.SPREAD
    assert verdicts.taint({"blocks": 2500, "direction": "east"}) == "Tainted Land: 2500 blocks east."
    assert verdicts.taint(None) == "Tainted Land: none within 3500 blocks"


def test_card():
    c = GREEN1 | {"base": "Temperate Forest", "off": {"side": "west", "look": "dull"}, "ocean": 524.8, "below_sea": 0.009}
    c["taint"] = {"blocks": 330.1, "direction": "north"}
    card = verdicts.card(c, 1, True)
    assert card["badges"] == [verdicts.RECOMMENDED, verdicts.EXCELLENT] and card["border"] == "very close"
    assert card["lines"][1] == "Looks: 73% the same as your base (Temperate Forest) right at the edge; duller grass on the west side"
    assert verdicts.card(c | {"same_edge": 0.5}, 2, False)["badges"] == []
    dull = c | {"same_edge": 0.66}
    assert verdicts.recommended([dull, c, c]) == 1 and verdicts.recommended([dull, dull]) == 0 and verdicts.recommended([]) is None


def test_a_card_says_the_border_of_each_part_when_there_are_several():
    c = GREEN1 | {"base": "Temperate Forest", "off": None, "ocean": None, "below_sea": 0, "taint": None}
    assert len(verdicts.card(c, 1, False)["lines"]) == 4
    rough = {"chunks": 700, "centre": [-2400, 1200], "median": 9, "over10": 0.4, "over30": 0.1, "max": 41}
    lines = verdicts.card(c | {"parts": [GREEN1 | {"chunks": 1000, "centre": [640, -320]}, rough]}, 1, False)["lines"]
    assert lines[4].startswith("Border of part 1 (1000 chunks around 640, -320): very close")
    assert lines[5] == "Border of part 2 (700 chunks around -2400, 1200): " + verdicts.border(rough)[1].removeprefix("Border: ")
    assert lines[5].endswith(": rough (steps up to 41 blocks; smoothing ramps them, but some cliffs may show)")


def test_stopped_search_never_recommends_the_rough_unchanged_world():
    unchanged = GREEN1 | {"config": [1, 0, 0, 0, 0], "median": 7.0, "over10": 0.33, "over30": 0.08, "max": 61.0, "same_edge": 0.5}
    unchecked = GREEN1 | {"median": 2.0, "feasible": False}
    assert verdicts.recommended([unchanged]) is None and verdicts.recommended([unchecked, unchanged]) is None
    shown = unchanged | {"base": "Temperate Forest", "off": None, "ocean": None, "below_sea": 0.0, "taint": None}
    assert verdicts.card(shown, 1, verdicts.recommended([shown]) == 0)["badges"] == []
