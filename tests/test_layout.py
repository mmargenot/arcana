"""Pip-layout algorithm tests: correct count, in-bounds, and symmetry.

Symmetry is the load-bearing property for pip cards read in a spread — every
layout except `bend` (a deliberately diagonal heraldic ordinary) must be
bilaterally symmetric about the art window's vertical axis, for every count.
"""
from pathlib import Path
import pytest

from arcana import layout
from arcana.geometry import Geometry

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "decks" / "configs" / "vaporwave-rws"

RANKS = range(1, 11)
ASYMMETRIC = {"bend"}


@pytest.fixture(scope="session")
def geo():
    return Geometry.load(CONFIG / "deck.yaml")


@pytest.mark.parametrize("name", layout.names())
@pytest.mark.parametrize("count", RANKS)
def test_returns_exact_count(name, count, geo):
    assert len(layout.place(name, count, geo)) == count


@pytest.mark.parametrize("name", layout.names())
@pytest.mark.parametrize("count", RANKS)
def test_pips_stay_inside_art_window(name, count, geo):
    """Centres must keep the 16px pip clear of the frame band."""
    for cx, cy in layout.place(name, count, geo):
        assert 8 <= cx <= geo.art_w - 8
        assert 8 <= cy <= geo.art_h - 8


@pytest.mark.parametrize("name", sorted(set(layout.names()) - ASYMMETRIC))
@pytest.mark.parametrize("count", RANKS)
def test_bilaterally_symmetric(name, count, geo):
    """Reflecting every pip across the vertical axis maps the set to itself."""
    pts = set(layout.place(name, count, geo))
    mirrored = {(geo.art_w - cx, cy) for cx, cy in pts}
    assert pts == mirrored


def test_bend_is_diagonal(geo):
    """bend is intentionally NOT mirror-symmetric — the whole point of the
    ordinary is the x/y correlation, so reflection changes the pip set."""
    pts = set(layout.place("bend", 5, geo))
    mirrored = {(geo.art_w - cx, cy) for cx, cy in pts}
    assert pts != mirrored


def test_unknown_layout_rejected(geo):
    with pytest.raises(KeyError, match="unknown pip layout"):
        layout.place("nonesuch", 3, geo)
