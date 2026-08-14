"""Regression tests for the bitmap font, text renderers, and label data. Every
one guards a bug invisible at 1x: an off-palette glyph edge, a label that wraps
or overflows the band, a title that drifts off-centre, or a numeral table
that's silently wrong.
"""
import numpy as np
import pytest

from arcana import text, data
from arcana.seed import placeholder_font
from arcana.palette import T, DARK, MAX_LOCAL
from arcana.text import CELL_H, CELL_W, ADVANCE, INK, line_width, render_line, fit_line, render_band

FONT = placeholder_font()

# every combined label the deck can produce — the "one line no matter what" set.
MINOR_TITLES = [data.minor_label(r, s)[1]
                for r in range(1, 11)
                for s in ("wands", "cups", "swords", "pentacles")]
MAJOR_TITLES = [data.major_label(n)[1] for n in range(len(data.MAJOR_NAMES))]


# --- font ---------------------------------------------------------------
def test_font_covers_inventory():
    """Digits, A-Z, hyphen and space — the whole label alphabet, one cell each."""
    for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ":
        g = FONT.glyph(ch)
        assert g.shape == (CELL_H, CELL_W), ch


def test_glyphs_are_single_ink_local():
    """A glyph is monochrome ink on a transparent field — only {0, INK} indices,
    so binding to one bank slot is lossless and nothing lands off-palette."""
    for ch, g in FONT.glyphs.items():
        assert set(np.unique(g).tolist()) <= {T, INK}, ch
        assert INK <= MAX_LOCAL


def test_ascii_round_trip():
    """A glyph tile is an index matrix like any other — it survives ASCII IO."""
    from arcana import tileio
    g = FONT.glyph("A")
    assert np.array_equal(g, tileio.from_ascii(tileio.to_ascii(g)))


def test_rgb_round_trip(tmp_path):
    """And RGB IO, so authored override glyphs snap back to indices on import."""
    from arcana import tileio
    g = FONT.glyph("R")
    tileio.write_rgb(g, tmp_path / "g.png")
    assert np.array_equal(g, tileio.read_rgb(tmp_path / "g.png"))


# --- line ---------------------------------------------------------------
def test_render_line_width_is_sum_of_advances():
    """The strip is exactly as wide as the advances, so band centring is exact."""
    for s in ("A", "AB", "THREE OF CUPS"):
        assert render_line(FONT, s).shape[1] == line_width(FONT, s)
        assert line_width(FONT, s) == (len(s) - 1) * ADVANCE + CELL_W


def test_render_line_is_one_strip_of_local_ink():
    strip = render_line(FONT, "ACE OF WANDS")
    assert strip.shape[0] == CELL_H
    assert set(np.unique(strip).tolist()) <= {T, INK}


def test_space_and_unknown_are_blank_advance():
    """Space and any unknown char take their cell width but leave no ink — so a
    missing glyph never crashes and never prints garbage."""
    assert not FONT.glyph(" ").any()
    assert not FONT.glyph("¡").any()          # unknown -> blank cell
    assert render_line(FONT, "A A").shape[1] == line_width(FONT, "A A")


# --- fit (one line, always) --------------------------------------------
@pytest.mark.parametrize("s", MINOR_TITLES + MAJOR_TITLES)
def test_fit_line_never_exceeds_width_and_stays_one_line(s):
    """Every deck label fits the title band's usable width on a single line —
    no wrapping, ever (long majors like THE WHEEL OF FORTUNE included)."""
    max_w = 128                                    # 160 card - 2x16 corner inset
    strip = fit_line(FONT, s, max_w)
    assert strip.shape[0] == CELL_H                # one line
    assert strip.shape[1] <= max_w


def test_fit_line_condenses_before_squeezing():
    """A line that fits at full advance is returned untouched (no lossy squeeze
    when it isn't needed)."""
    s = "ACE OF CUPS"
    assert np.array_equal(fit_line(FONT, s, 200), render_line(FONT, s, ADVANCE))


# --- band (centred) -----------------------------------------------------
def test_render_band_is_band_sized_and_centred():
    """The label block's bbox centre is the band centre (±1px), the same centring
    discipline the pip layouts assert — a title that drifts is a bug."""
    bw, bh = 128, CELL_H + 2
    band = render_band(FONT, "SIX OF SWORDS", bw, bh)
    assert band.shape == (bh, bw)
    ys, xs = np.nonzero(band != T)
    assert abs((xs.min() + xs.max()) / 2 - bw / 2) <= 1
    assert abs((ys.min() + ys.max()) / 2 - bh / 2) <= 1


def test_render_band_empty_string_is_blank():
    assert not render_band(FONT, "", 128, 12).any()


# --- data ---------------------------------------------------------------
def test_roman_covers_majors():
    assert [data.roman(n) for n in (0, 1, 4, 5, 9, 10, 13, 21)] == \
        ["0", "I", "IV", "V", "IX", "X", "XIII", "XXI"]


def test_arabic_rank_is_compact():
    assert [data.arabic_rank(n) for n in (1, 2, 10)] == ["A", "2", "10"]


def test_minor_label_combined_vs_split():
    assert data.minor_label(3, "cups") == (None, "THREE OF CUPS")
    assert data.minor_label(3, "cups", split=True, style="arabic") == ("3", "CUPS")
    assert data.minor_label(3, "cups", split=True, style="words") == ("THREE", "CUPS")


def test_major_label_combined_vs_split():
    assert data.major_label(0) == (None, "0 THE FOOL")
    assert data.major_label(13, split=True) == ("XIII", "DEATH")


def test_deck_overrides_names():
    """A deck renames suits and majors through the labels: block — engine ships
    defaults, deck customises, same split as pip_layouts/field_designs."""
    cfg = {"labels": {"suit_titles": {"cups": "CHALICES"},
                      "major_titles": ["THE JESTER"]}}
    assert data.suit_title("cups", cfg) == "CHALICES"
    assert data.minor_label(2, "cups", cfg=cfg)[1] == "TWO OF CHALICES"
    assert data.major_name(0, cfg) == "THE JESTER"
    assert data.major_name(1, cfg) == "THE MAGICIAN"       # unlisted -> canonical


def test_label_options_defaults():
    assert data.label_options(None) == {"enabled": True, "style": "arabic", "split": False}
    assert data.label_options({"labels": {"enabled": False}})["enabled"] is False
