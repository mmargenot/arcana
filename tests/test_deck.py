"""Regression tests. Every one guards a bug that is invisible at 1x.

Tiles are not committed: the fixture seeds a full set of placeholder tiles into
a temp dir via `arcana.seed`, and loads elements from there. Config comes from
the committed deck under `decks/configs/vaporwave-rws/`.
"""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from arcana import compose, tileio, seed, data
from arcana.seed import placeholder_font
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
    base = compose.build_pip_card(pal, geo, els, 8, "square", cfg["suit_pips"]["cups"])
    for suit in ("wands", "swords", "pentacles"):
        pal_s = pal.for_suit(suit)
        other = compose.build_pip_card(pal_s, geo, els, 8, "square", cfg["suit_pips"]["cups"])
        assert np.array_equal(base, other)


def test_pip_card_invalid_layout_raises(ctx):
    """An impossible (layout, count) is surfaced, not rendered broken. Every
    layout works at every rank at the normal size, so the impossible case is
    forced with an absurd minimum the grid terminal can't meet either."""
    from arcana.layout import InvalidPipLayout
    pal, geo, cfg, els = ctx
    with pytest.raises(InvalidPipLayout):
        compose.build_pip_card(pal, geo, els, 6, "cross", cfg["suit_pips"]["cups"],
                               pip_cfg={"min_scale": 6.0})


def test_pip_card_unfittable_shape_diamonds(ctx):
    """A 2-D ordinary that can't hold a high count in its own shape renders by
    folding into a diamond rather than raising — every layout works at every
    rank."""
    pal, geo, cfg, els = ctx
    m = compose.build_pip_card(pal, geo, els, 10, "cross", cfg["suit_pips"]["cups"])
    assert m.shape == (geo.card_h, geo.card_w)
    assert int(m.max()) < len(pal.colors)


def test_pip_card_field_design_applied(ctx):
    """The field design is an independent axis: a non-plain field changes the
    background matrix without touching the pips or frame."""
    pal, geo, cfg, els = ctx
    plain = compose.build_pip_card(pal, geo, els, 4, "square", cfg["suit_pips"]["cups"], "plain")
    checky = compose.build_pip_card(pal, geo, els, 4, "square", cfg["suit_pips"]["cups"], "checky")
    assert plain.shape == checky.shape
    assert not np.array_equal(plain, checky)


def test_minor_field_runs_under_the_title_band(ctx):
    """The field is FULL-BLEED for minors too (compose.content_field): the
    same `geo.margin` depth on all four sides, so the frame band's ornament
    sits on field top and bottom just like on the sides, and the title band is
    field colour, never bare paper. Outside `margin` stays clear — that ring
    belongs to the frame's rules and their paper gap."""
    pal, geo, cfg, els = ctx
    m = compose.build_development(pal, geo, els, 4, "square", cfg["suit_pips"]["cups"])
    ox, oy = geo.art_origin
    g = geo.margin
    title_row = oy + geo.art_h + 1                       # inside the title band
    assert (m[title_row, ox:ox + geo.art_w] != 0).all()  # field runs under it
    assert (m[g, ox:ox + geo.art_w] != 0).all()          # under the top band too
    assert (m[:g, :] == 0).all() and (m[geo.card_h - g:, :] == 0).all()


# --- labels -------------------------------------------------------------
FONT = placeholder_font()


def test_labelled_card_shape_and_indices(ctx):
    """A labelled card is still one card-sized global-index matrix — labels are
    composited in, not a second pass at render."""
    pal, geo, cfg, els = ctx
    top, bottom = data.minor_label(3, "cups")
    m = compose.build_pip_card(pal, geo, els, 3, "chevron", cfg["suit_pips"]["cups"],
                               font=FONT, top=top, bottom=bottom)
    assert m.shape == (geo.card_h, geo.card_w)
    assert int(m.max()) < len(pal.colors)


def test_labels_off_by_default_leaves_base_untouched(ctx):
    """No font -> no labels: the base card is byte-identical, so the existing
    suit-invariance guarantee is preserved."""
    pal, geo, cfg, els = ctx
    base = compose.build_pip_card(pal, geo, els, 5, "saltire", cfg["suit_pips"]["cups"])
    same = compose.build_pip_card(pal, geo, els, 5, "saltire", cfg["suit_pips"]["cups"],
                                  font=FONT, top=None, bottom=None)
    assert np.array_equal(base, same)


def test_rank_numeral_is_suit_invariant_but_suit_title_is_not(ctx):
    """The suit-invariance boundary: a rank numeral is the SAME matrix for every
    suit (LUT swap only), but a suit-name title deliberately differs — it names
    the suit, so it cannot be suit-invariant."""
    pal, geo, cfg, els = ctx
    numeral = [compose.build_pip_card(pal.for_suit(s), geo, els, 3, "chevron",
                                      cfg["suit_pips"]["cups"], font=FONT, top="3")
               for s in ("cups", "wands", "swords")]
    assert all(np.array_equal(numeral[0], x) for x in numeral[1:])

    titled = [compose.build_pip_card(pal.for_suit(s), geo, els, 3, "chevron",
                                     cfg["suit_pips"][s], font=FONT,
                                     bottom=data.minor_label(3, s)[1])
              for s in ("cups", "wands")]
    assert not np.array_equal(titled[0], titled[1])


def test_frame_not_clipped_by_labels(ctx):
    """Minor labels paint UNDER the border, so the frame rule ring stays
    contiguous — a label bleeding into the frame is exactly this bug."""
    pal, geo, cfg, els = ctx
    m = compose.build_pip_card(pal, geo, els, 10, "square", cfg["suit_pips"]["pentacles"],
                               font=FONT, top="10", bottom="TEN OF PENTACLES")
    # the composed card's outer LINE ring is unbroken on all four sides
    from arcana.palette import LINE
    W, H = geo.card_w, geo.card_h
    assert (m[0, :] == LINE).all() and (m[H - 1, :] == LINE).all()
    assert (m[:, 0] == LINE).all() and (m[:, W - 1] == LINE).all()


def test_build_major_card_shape_and_indices(ctx):
    """The major builder produces a card-sized global-index matrix with its label
    floating over the mural (figure image is a later seam)."""
    pal, geo, cfg, els = ctx
    top, bottom = data.major_label(0, split=True)
    m = compose.build_major_card(pal, geo, els, FONT, top=top, bottom=bottom,
                                 pip_key=cfg["suit_pips"]["majors"])
    assert m.shape == (geo.card_h, geo.card_w)
    assert int(m.max()) < len(pal.colors)


def test_band_rects_stay_inside_the_bands(ctx):
    """Usable label rects are derived from geo and sit within their bands, inset
    from the card sides — never spilling into the art window or off-card."""
    _, geo, _, _ = ctx
    r = compose.band_rects(geo)
    nx0, ny0, nx1, ny1 = r["numeral"]
    tx0, ty0, tx1, ty1 = r["title"]
    assert 0 <= ny0 < ny1 <= geo.band_numeral                       # within top band
    assert geo.band_numeral + geo.art_h <= ty0 < ty1 <= geo.card_h  # within bottom band
    assert nx0 == tx0 == geo.corner and nx1 == tx1 == geo.card_w - geo.corner


# --- medallions ---------------------------------------------------------
def test_default_medallion_matches_mount(ctx):
    """The default (suit, scale 1.0) reproduces the old `mount(pip, cartouche)`
    medallion byte-for-byte — so every pre-existing frame/suit-invariance test
    still describes the shipped output."""
    _, _, cfg, els = ctx
    made = compose.build_medallion(els, cfg["suit_pips"]["cups"])
    mounted = compose.mount(els[cfg["suit_pips"]["cups"]], els["cartouche"])
    assert np.array_equal(made.layers["motif"], mounted.layers["motif"])


def test_medallion_styles_and_scale(ctx):
    """suit and lozenge build card-legal local-index medallions; none omits it;
    a scale < 1 yields a smaller emblem (the knob that shrinks it)."""
    from arcana.palette import MAX_LOCAL
    _, _, cfg, els = ctx
    pk = cfg["suit_pips"]["cups"]
    assert compose.build_medallion(els, pk, style="none") is None
    full = compose.build_medallion(els, pk, style="suit", scale=1.0)
    small = compose.build_medallion(els, pk, style="suit", scale=0.5)
    loz = compose.build_medallion(els, pk, style="lozenge", scale=0.5)
    for m in (full, small, loz):
        assert int(m.layers["motif"].max()) <= MAX_LOCAL
    assert small.layers["motif"].shape[0] < full.layers["motif"].shape[0]


def test_lozenge_is_a_solid_suit_gem(ctx):
    """The lozenge is a solid diamond — LINE outline over a MID fill (motif bank
    => suit colour), transparent box corners. A hollow or off-slot gem is the bug."""
    from arcana.palette import T, LINE, MID
    _, _, cfg, els = ctx
    loz = compose.build_medallion(els, cfg["suit_pips"]["cups"], style="lozenge").layers["motif"]
    assert set(np.unique(loz).tolist()) <= {T, LINE, MID}
    S = loz.shape[0]
    assert loz[S // 2, S // 2] == MID           # solid centre
    assert loz[0, 0] == T                        # transparent corner


def test_no_medallion_continues_the_border(ctx):
    """style=none continues the edge ornament through the medallion slot, and the
    frame stays one contiguous, symmetric ring — a broken border from the fill is
    exactly this bug."""
    from arcana.palette import DARK
    _, geo, _, els = ctx
    f, _ = compose.build_border(geo, els["corner"], els["edge"], None)
    assert all(compose.check_contiguous(f, geo).values())
    assert all(compose.check_symmetry(f).values())
    # dentil ornament now reaches the top-edge centre (rows below the rule, which
    # were bare before the fill) — proves the slot is filled, not left blank
    cx = geo.card_w // 2
    assert (f[8:12, cx - 6:cx + 6] == DARK).any()


def test_no_medallion_vertical_slot_has_no_gap(ctx):
    """The vertical slot's half-height is not tile-divisible (the remainder is why
    the slot exists), so tiling the dentils outward alone truncated the last tab
    and left a ~4px hole at the left/right edge midpoints. The anchored final tab
    must close it: no dentil column may have a longer transparent run straddling
    the vertical centre than it does in the clean rhythm near the corner."""
    from arcana.palette import T
    _, geo, _, els = ctx
    f, _ = compose.build_border(geo, els["corner"], els["edge"], None)
    C, H = geo.corner, geo.card_h

    def max_run(mask):
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    mid = H // 2
    # dentil columns carry both marks AND gaps along the run; the structural rule
    # columns are solid and the bare columns never mark, so this picks the ornament.
    dentil_cols = [c for c in range(C)
                   if (f[C:H - C, c] != T).any() and (f[C:H - C, c] == T).any()]
    assert dentil_cols, "no dentil ornament found on the vertical edge"
    for c in dentil_cols:
        centre_gap = max_run(f[mid - 16:mid + 16, c] == T)   # straddles the axis
        rhythm_gap = max_run(f[C:C + 32, c] == T)            # clean run near corner
        assert centre_gap <= rhythm_gap, (
            f"column {c}: centre gap {centre_gap} exceeds rhythm gap {rhythm_gap}")


def test_medallion_size_keywords_resolve(ctx):
    """Medallion scale accepts named sizes (`small`, `tiny`, …) or a number, and
    the deck ships the `small` keyword — a typo'd keyword is rejected loudly."""
    from arcana.cli import _resolve_scale, _medallion_opts, MEDALLION_SIZES
    _, _, cfg, _ = ctx
    assert _resolve_scale("small") == MEDALLION_SIZES["small"] == 0.5
    assert _resolve_scale("0.5") == 0.5 and _resolve_scale(0.5) == 0.5
    assert _medallion_opts(cfg) == ("suit", 0.5)                 # deck default
    assert _medallion_opts(cfg, scale_override="tiny")[1] == MEDALLION_SIZES["tiny"]
    with pytest.raises(SystemExit, match="unknown medallion size"):
        _resolve_scale("smallish")


def test_lozenge_card_is_suit_invariant(ctx):
    """A lozenge card's index matrix is identical across suits (motif binding is
    index-level) — the suit is a LUT swap, same as everything else."""
    pal, geo, cfg, els = ctx
    base = compose.build_pip_card(pal, geo, els, 5, "saltire", cfg["suit_pips"]["cups"],
                                  med_style="lozenge", med_scale=0.5)
    for suit in ("wands", "swords", "pentacles"):
        other = compose.build_pip_card(pal.for_suit(suit), geo, els, 5, "saltire",
                                       cfg["suit_pips"]["cups"], med_style="lozenge",
                                       med_scale=0.5)
        assert np.array_equal(base, other)


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


def test_seed_erases_the_printed_numeral():
    """RWS prints a rank inside the picture — `0` over the Fool, `XVIII` over
    the Moon — and arcana renders its own from `arcana.data`. Left in the seed
    it survives generation and the card carries two numerals in two typefaces.
    Erasing is GUARDED: where art sits behind the numeral, filling would smear
    the scene, so the numeral stays and the prompt is the remaining defence."""
    import numpy as np
    from arcana.pixelate import erase_numeral, NUMERAL_BOX

    sky = np.full((400, 300, 3), 200.0)
    h, w, _ = sky.shape
    x0, y0 = int(NUMERAL_BOX[0] * w) + 2, int(NUMERAL_BOX[1] * h) + 2
    sky[y0:y0 + 6, x0:x0 + 6] = 20.0                 # the numeral, on flat ground
    assert erase_numeral(sky).max() == 200.0         # gone

    busy = np.random.default_rng(0).uniform(0, 255, (400, 300, 3))
    assert np.array_equal(erase_numeral(busy), busy)  # art behind it: untouched
