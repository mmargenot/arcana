"""Regression tests. Every one guards a bug that is invisible at 1x.

Tiles are not committed: the fixture seeds a full set of placeholder tiles into
a temp dir via `arcana.seed`, and loads elements from there. Config comes from
the committed deck under `decks/configs/vaporwave-rws/`.
"""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from arcana import compose, tileio, seed
from arcana.palette import Palette
from arcana.geometry import Geometry, load_config
from arcana.elements import load_all, AssetError, read_tile, write_tile

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "decks" / "configs" / "vaporwave-rws"


@pytest.fixture(scope="session")
def ctx(tmp_path_factory):
    assets = tmp_path_factory.mktemp("assets")
    seed.seed_deck(assets)
    pal = Palette.load(CONFIG / "palette.yaml")
    geo = Geometry.load(CONFIG / "deck.yaml")
    cfg = load_config(CONFIG / "deck.yaml")
    return pal, geo, cfg, load_all(assets, cfg["elements"])


# --- geometry -----------------------------------------------------------
def test_card_equals_art_plus_bands(ctx):
    _, geo, _, _ = ctx
    assert geo.card_w == geo.art_w + 2 * geo.margin
    assert geo.card_h == geo.art_h + geo.band_numeral + geo.band_title


def test_border_runs_divide_evenly(ctx):
    _, geo, _, _ = ctx
    geo.validate()


# --- palette ------------------------------------------------------------
def test_every_bank_on_its_rung(ctx):
    pal, *_ = ctx
    assert pal.validate(strict=False) == []


def test_index_matrix_is_suit_invariant(ctx):
    """The whole point of index space: suits differ only in the LUT."""
    pal, geo, cfg, els = ctx
    base = compose.render_border(pal, geo, els["corner"], els["edge"])
    for suit in cfg["suit_pips"]:
        other = compose.render_border(pal.for_suit(suit), geo, els["corner"], els["edge"])
        assert np.array_equal(base, other)


def test_unquoted_hex_rejected(tmp_path):
    """'#RRGGBB' unquoted is a YAML comment and parses as null."""
    p = tmp_path / "bad.yaml"
    p.write_text((CONFIG / "palette.yaml").read_text()
                 .replace('line: "#221E1A"', 'line: #221E1A'))
    with pytest.raises(ValueError, match="null"):
        Palette.load(p)


# --- frame --------------------------------------------------------------
def test_frame_rules_contiguous(ctx):
    """A gap is nearly invisible on screen and glaring in print."""
    _, geo, _, els = ctx
    f, _ = compose.build_border(geo, els["corner"], els["edge"])
    assert all(compose.check_contiguous(f, geo).values())


def test_frame_symmetric(ctx):
    _, geo, _, els = ctx
    f, _ = compose.build_border(geo, els["corner"], els["edge"])
    assert all(compose.check_symmetry(f).values())


def test_frame_is_one_ring(ctx):
    """Broken assembly scored 120 fragments; correct is one large component."""
    _, geo, _, els = ctx
    f, _ = compose.build_border(geo, els["corner"], els["edge"])
    from scipy import ndimage
    lab, _ = ndimage.label((f != 0).astype(np.uint8),
                           structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    assert np.bincount(lab.ravel())[1:].max() > 4000


def test_side_edge_keeps_outer_rule(ctx):
    """A side edge is the top edge TRANSPOSED. An extra flip moves the rule
    to the inner side of the band, losing it for the whole length."""
    _, _, _, els = ctx
    assert (compose.orient(els["edge"].layers["border"])[:, 0] != 0).all()


def test_cartouche_opaque(ctx):
    """Not opaque -> the frame shows through and bisects thin pips."""
    _, _, _, els = ctx
    assert els["cartouche"].is_opaque()


def test_role_mismatch_rejected(ctx):
    _, geo, _, els = ctx
    with pytest.raises(ValueError, match="role"):
        compose.build_border(geo, els["pip_cups"], els["edge"])


# --- asset io -----------------------------------------------------------
def test_rgb_png_rejected(tmp_path, ctx):
    """Build the fixture from a loaded element — assets may be stored as
    ASCII, so globbing for a .png makes the test depend on storage format."""
    _, _, _, els = ctx
    a = els["corner"].layers["border"]
    write_tile(a, tmp_path / "indexed.png")
    Image.open(tmp_path / "indexed.png").convert("RGB").save(tmp_path / "rgb.png")
    with pytest.raises(AssetError, match="not indexed"):
        read_tile(tmp_path / "rgb.png")


def test_size_mismatch_rejected(tmp_path, ctx):
    _, _, _, els = ctx
    write_tile(els["corner"].layers["border"], tmp_path / "t.png")
    with pytest.raises(AssetError, match="manifest declares"):
        read_tile(tmp_path / "t.png", expect=(99, 99))


def test_ascii_round_trip(ctx):
    _, _, _, els = ctx
    a = els["pip_cups"].layers["motif"]
    assert np.array_equal(a, tileio.from_ascii(tileio.to_ascii(a)))


def test_rgb_round_trip(tmp_path, ctx):
    """Draw in any editor; snap back to indices on import."""
    _, _, _, els = ctx
    a = els["corner"].layers["border"]
    tileio.write_rgb(a, tmp_path / "t.png")
    assert np.array_equal(a, tileio.read_rgb(tmp_path / "t.png"))


def test_antialiasing_rejected(tmp_path, ctx):
    _, _, _, els = ctx
    a = els["corner"].layers["border"]
    tileio.write_rgb(a, tmp_path / "t.png")
    rgba = np.array(Image.open(tmp_path / "t.png").convert("RGBA"))
    rgba[5, 5, :3] = [100, 90, 110]
    Image.fromarray(rgba, "RGBA").save(tmp_path / "aa.png")
    with pytest.raises(AssetError, match="off the authoring palette"):
        tileio.read_rgb(tmp_path / "aa.png")
