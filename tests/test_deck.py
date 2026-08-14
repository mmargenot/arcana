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
    """The outer rule must be one continuous ring around the card, not a heap of
    fragments (a broken assembly once scored 120 pieces). The beveled frame is
    two concentric rule rings plus loose motif marks, so assert the largest
    component rings the whole card — spanning top-to-bottom and left-to-right."""
    _, geo, _, els = ctx
    f, _ = compose.build_border(geo, els["corner"], els["edge"])
    from scipy import ndimage
    lab, _ = ndimage.label((f != 0).astype(np.uint8),
                           structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    biggest = np.bincount(lab.ravel())[1:].argmax() + 1
    ys, xs = np.where(lab == biggest)
    assert ys.min() == 0 and ys.max() == geo.card_h - 1      # touches top and bottom
    assert xs.min() == 0 and xs.max() == geo.card_w - 1      # touches left and right
    assert (lab == biggest).sum() > 2000                     # a substantial ring


def test_side_edge_keeps_outer_rule(ctx):
    """The outer LINE rule is structural now — drawn by build_border, not carried
    by the edge tile — so it must stay unbroken along all four sides. A gap here
    is the classic 'rule on the wrong side of the band' bug."""
    from arcana.palette import LINE
    _, geo, _, els = ctx
    f, _ = compose.build_border(geo, els["corner"], els["edge"])
    W, H = geo.card_w, geo.card_h
    assert (f[0, :] == LINE).all() and (f[H - 1, :] == LINE).all()
    assert (f[:, 0] == LINE).all() and (f[:, W - 1] == LINE).all()


def test_corner_beveled(ctx):
    """The corner mitres instead of butting square. The old square cut showed a
    vertical stripe at the corner (f[0,1]=MID, f[0,2]=LIGHT); a bevel wraps the
    outer LINE around the corner and steps the rules along the diagonal."""
    from arcana.palette import LINE, MID, LIGHT
    _, geo, _, els = ctx
    f, _ = compose.build_border(geo, els["corner"], els["edge"])
    assert f[0, 0] == f[0, 1] == f[0, 2] == LINE          # outer rule wraps corner
    assert (f[0, 0], f[1, 1], f[2, 2]) == (LINE, MID, LIGHT)   # mitre along diagonal
    assert f[1, 2] == f[2, 1] == MID                      # symmetric about diagonal


def test_frame_independent_of_motif(ctx):
    """The beveled frame is structural: swapping the corner/edge motif must not
    move a single rule pixel. Build once with the real motifs and once with blank
    ones; every pixel of the bare rule frame must be identical in both."""
    from arcana.elements import Element
    _, geo, _, els = ctx
    blank_corner = Element(name="blank", role="corner", size=(16, 16),
                           layers={"border": np.zeros((16, 16), np.uint8)})
    blank_edge = Element(name="blank", role="edge", size=(16, 8),
                         layers={"border": np.zeros((16, 8), np.uint8)})
    real, _ = compose.build_border(geo, els["corner"], els["edge"])
    bare, _ = compose.build_border(geo, blank_corner, blank_edge)
    rule = bare != 0                          # every non-empty bare pixel is a rule
    assert np.array_equal(real[rule], bare[rule])


def test_cartouche_opaque(ctx):
    """Not opaque -> the frame shows through and bisects thin pips."""
    _, _, _, els = ctx
    assert els["cartouche"].is_opaque()


def test_role_mismatch_rejected(ctx):
    _, geo, _, els = ctx
    with pytest.raises(ValueError, match="role"):
        compose.build_border(geo, els["pip_cups"], els["edge"])


# --- pip cards ----------------------------------------------------------
def test_build_pip_card_shape_and_indices(ctx):
    """A pip card is one card-sized matrix in global index space."""
    pal, geo, cfg, els = ctx
    m = compose.build_pip_card(pal, geo, els, 5, "saltire", cfg["suit_pips"]["cups"])
    assert m.shape == (geo.card_h, geo.card_w)
    assert int(m.max()) < len(pal.colors)


def test_pip_card_is_suit_invariant(ctx):
    """Like the border, the pip-card index matrix is the same for every suit —
    colour is a LUT swap at render time, not a re-composition."""
    pal, geo, cfg, els = ctx
    base = compose.build_pip_card(pal, geo, els, 7, "cross", cfg["suit_pips"]["cups"])
    for suit in ("wands", "swords", "pentacles"):
        pal_s = pal.for_suit(suit)
        other = compose.build_pip_card(pal_s, geo, els, 7, "cross", cfg["suit_pips"]["cups"])
        assert np.array_equal(base, other)


def test_pip_scale_auto_fit(ctx):
    """Pip size is auto-fit from centre spacing: within [MIN, MAX] for every
    arrangement × count, MAX for a lone pip, and never larger for a crowded ten
    than a lone ace."""
    from arcana import layout
    from arcana.compose import _pip_scale, PIP_MIN_SCALE, PIP_MAX_SCALE
    _, geo, _, _ = ctx
    assert _pip_scale(layout.place("single", 1, geo)) == PIP_MAX_SCALE
    for name in layout.names():
        for n in range(1, 11):
            assert PIP_MIN_SCALE <= _pip_scale(layout.place(name, n, geo)) <= PIP_MAX_SCALE
    assert _pip_scale(layout.place("pile", 10, geo)) <= _pip_scale(layout.place("single", 1, geo))


def test_pip_card_field_design_applied(ctx):
    """The field design is an independent axis: a non-plain field changes the
    background matrix without touching the pips or frame."""
    pal, geo, cfg, els = ctx
    plain = compose.build_pip_card(pal, geo, els, 4, "square", cfg["suit_pips"]["cups"], "plain")
    checky = compose.build_pip_card(pal, geo, els, 4, "square", cfg["suit_pips"]["cups"], "checky")
    assert plain.shape == checky.shape
    assert not np.array_equal(plain, checky)


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
    rgba[8, 8, :3] = [100, 90, 110]      # (8,8) is an opaque motif pixel
    Image.fromarray(rgba, "RGBA").save(tmp_path / "aa.png")
    with pytest.raises(AssetError, match="off the authoring palette"):
        tileio.read_rgb(tmp_path / "aa.png")
