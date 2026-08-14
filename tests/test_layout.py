"""Pip placement tests: the buffer (countability), the inner border, folding,
and layout validity.

The load-bearing property is now that pips are always **countable** — never
closer than a pip + gap, and always inside the card. Where a layout can't do
that at the minimum size it is invalid, and says so.
"""
from pathlib import Path
import pytest

from arcana import layout
from arcana.geometry import Geometry, load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "decks" / "configs" / "vaporwave-rws"

ASYMMETRIC = {"bend", "bend-sinister"}          # diagonal ordinaries


@pytest.fixture(scope="session")
def geo():
    return Geometry.load(CONFIG / "deck.yaml")


# Pips scale continuously and centres round to whole pixels, so the buffer and
# border invariants can each be off by up to ~1px of rounding. Allow for it.
TOL = 1.5


def _assert_placement(centres, k, geo, gap=layout.PIP_GAP):
    pip = layout.PIP_BASE * k
    ix, iy = layout._pip_inset(geo)
    # buffer: every pair at least a pip+gap apart (Chebyshev; rounding tolerance)
    for i, (ax, ay) in enumerate(centres):
        for bx, by in centres[i + 1:]:
            assert max(abs(ax - bx), abs(ay - by)) >= pip + gap - TOL
    # inner border: every pip box stays a gap inside the invisible border
    for cx, cy in centres:
        assert ix + gap - TOL <= cx - pip / 2 and cx + pip / 2 <= geo.art_w - ix - gap + TOL
        assert iy + gap - TOL <= cy - pip / 2 and cy + pip / 2 <= geo.art_h - iy - gap + TOL


def test_valid_placements_are_countable(geo):
    """Every arrangement that IS valid keeps its buffer and stays in bounds."""
    seen = 0
    for name in layout.names():
        for n in range(1, 11):
            try:
                centres, k = layout.arrange(name, n, geo)
            except layout.InvalidPipLayout:
                continue
            seen += 1
            assert len(centres) == n
            assert layout.PIP_MIN_SCALE <= k <= layout.PIP_MAX_SCALE
            _assert_placement(centres, k, geo)
    assert seen > 0


def test_every_layout_works_at_every_rank(geo):
    """The load-bearing goal: no layout/rank combination fails at the default
    config. Foldable ordinaries fold; 2-D ordinaries that can't hold a high count
    fall back to the compact grid. Either way a valid buffered placement comes
    back for all names at all ranks 1..10."""
    for name in layout.names():
        for n in range(1, 11):
            centres, k = layout.arrange(name, n, geo)   # must not raise
            assert len(centres) == n
            _assert_placement(centres, k, geo)


def test_unfittable_shape_falls_back_to_grid(geo):
    """A 2-D ordinary that can't hold a high count in its own shape still returns
    a valid, buffered placement — the compact grid — instead of raising."""
    centres, k = layout.arrange("cross", 10, geo)
    assert len(centres) == 10
    assert layout.PIP_MIN_SCALE <= k <= layout.PIP_MAX_SCALE
    _assert_placement(centres, k, geo)


def test_symmetry_where_feasible(geo):
    for name in sorted(set(layout.names()) - ASYMMETRIC):
        for n in range(1, 11):
            try:
                centres, _ = layout.arrange(name, n, geo)
            except layout.InvalidPipLayout:
                continue
            xs = sorted(x for x, _ in centres)
            mirror = sorted(geo.art_w - x for x, _ in centres)
            assert all(abs(a - b) <= 1 for a, b in zip(xs, mirror)), (name, n)


def test_invalid_layout_is_informative(geo):
    """A layout is only invalid when even the grid fallback can't reach the
    minimum size — here forced with an absurd min_scale. The error names the
    best achievable size and the min it fell short of."""
    with pytest.raises(layout.InvalidPipLayout, match="min_scale"):
        layout.arrange("cross", 6, geo, min_scale=6.0)


def test_unknown_layout_rejected(geo):
    with pytest.raises(KeyError, match="unknown pip layout"):
        layout.arrange("nonesuch", 3, geo)


def test_deck_mapping_validates(geo):
    cfg = load_config(CONFIG / "deck.yaml")
    layout.validate_pip_layouts(cfg, geo)         # the shipped mapping is valid


def test_bad_mapping_rejected(geo):
    """The validator surfaces a mapping the deck can't honour. An unknown layout
    name can't be placed at all, so it's rejected with the offending name."""
    cfg = load_config(CONFIG / "deck.yaml")
    bad = dict(cfg)
    bad["pip_layouts"] = {"default": "square", "by_rank": {6: "nonesuch"}}
    with pytest.raises(ValueError, match="nonesuch"):
        layout.validate_pip_layouts(bad, geo)


def test_oversize_min_scale_rejected(geo):
    """An impossible size floor is caught for the whole mapping, not per card —
    at 6x even the grid terminal can't place the high ranks with a buffer."""
    cfg = load_config(CONFIG / "deck.yaml")
    bad = dict(cfg)
    bad["pip"] = {"min_scale": 6.0}
    with pytest.raises(ValueError, match="min_scale"):
        layout.validate_pip_layouts(bad, geo)


def test_pip_config_reads_deck(geo):
    assert layout.pip_config({"pip": {"gap": 10, "min_scale": 1}}) == \
        {"gap": 10, "min_scale": 1, "max_scale": layout.PIP_MAX_SCALE}
    assert layout.pip_config(None)["gap"] == layout.PIP_GAP
