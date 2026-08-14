"""Heraldic field-design tests: shape, tone set, the invisible border, and
bilateral symmetry for the symmetric designs.

A field is geometry in local tone-space (ground vs device); these tests pin the
structural guarantees, not the exact pattern. The invisible-border check is the
load-bearing one — the design must never run into the frame.
"""
from pathlib import Path
import numpy as np
import pytest

from arcana import field
from arcana.geometry import Geometry
from arcana.palette import LIGHT, MID, DARK

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "decks" / "configs" / "vaporwave-rws"

# The designs that are mirror-symmetric about the vertical axis. The rest are
# intentionally asymmetric (diagonals like per-bend/bendy, the per-pale split,
# the diagonal quarterly, and even-count stripe/parity patterns).
SYMMETRIC = {"barry", "base", "bordure", "chevron", "chevronny", "chief",
             "cross", "fess", "pale", "per-chevron", "per-fess", "plain",
             "saltire"}


@pytest.fixture(scope="session")
def geo():
    return Geometry.load(CONFIG / "deck.yaml")


@pytest.mark.parametrize("name", field.names())
def test_shape_tones_opaque(name, geo):
    m = field.build(name, geo)
    assert m.shape == (geo.art_h, geo.art_w)
    assert set(np.unique(m).tolist()) <= {LIGHT, MID}     # ground + device only
    assert (m != 0).all()                                  # fully opaque, no holes


@pytest.mark.parametrize("name", field.names())
def test_invisible_border(name, geo):
    """The outer inset ring is uniform ground, so no design touches the frame."""
    m = field.build(name, geo)
    ix, iy = field.insets(geo)
    assert (m[:iy, :] == LIGHT).all() and (m[-iy:, :] == LIGHT).all()
    assert (m[:, :ix] == LIGHT).all() and (m[:, -ix:] == LIGHT).all()


@pytest.mark.parametrize("name", sorted(SYMMETRIC))
def test_symmetric_designs(name, geo):
    m = field.build(name, geo)
    assert np.array_equal(m, m[:, ::-1])


def test_tone_roles_are_parameters(geo):
    """A design only picks ground vs device; the caller supplies which tones."""
    m = field.build("per-fess", geo, ground=LIGHT, device=DARK)
    assert set(np.unique(m).tolist()) <= {LIGHT, DARK}
    assert DARK in np.unique(m)


def test_unknown_field_rejected(geo):
    with pytest.raises(KeyError, match="unknown field design"):
        field.build("nonesuch", geo)
