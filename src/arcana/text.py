"""
Bitmap text: plaintext labels -> local-index strips, the same way `arcana.layout`
turns a count into pip centres and `arcana.field` turns a name into a background.
Letterforms are AUTHORED pixel art (a fixed-cell font); labels are CHOSEN by
string, never redrawn per card.

Everything here is pure index-matrix work in LOCAL space (`0` transparent, ink
in one slot — `INK` below). No palette, no RGB: colour is bound to a bank and
resolved at render, exactly like every other tile. Glyph ink lives in a single
slot so a label is monochrome; `compose.build_label` binds it to the `border`
bank so titles match the frame and vary with the palette.

Metrics (cell, tracking, the one-line fit) are named constants here, not
scattered literals — they get tuned on renders.

No mirroring, ever. Text is not bilaterally symmetric; the `compose` LAST/mirror
discipline is for pips, never glyphs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_uppercase
import numpy as np

from arcana.palette import T, DARK

# --- metrics ------------------------------------------------------------
# A 6x10 glyph box with 1px tracking -> 7px advance. At 144px art width that is
# ~20 chars/line; a ~7px cap centres in the 16px numeral band. `fit_line`
# condenses the advance to CELL_W (tracking 0) before it ever squeezes pixels.
CELL_W, CELL_H = 6, 10
TRACKING = 1
ADVANCE = CELL_W + TRACKING          # 7
INK = DARK                            # local slot glyphs are authored in (3)

# stems for authored per-glyph overrides on disk (chars that aren't filename-safe
# get a word). Mirrors the tile-override story: an authored file wins.
_STEM = {**{c: c for c in "0123456789"}, **{c: c for c in ascii_uppercase},
         "-": "hyphen", " ": "space"}


@dataclass(frozen=True, slots=True)
class Font:
    """A fixed-cell bitmap font: `char -> local-index cell tile`, loaded once."""
    glyphs: dict[str, np.ndarray] = field(default_factory=dict)
    cell_w: int = CELL_W
    cell_h: int = CELL_H

    def glyph(self, ch: str) -> np.ndarray:
        """The cell tile for `ch`. Space/unknown -> a blank cell (advance, no ink)."""
        g = self.glyphs.get(ch.upper())
        return g if g is not None else np.zeros((self.cell_h, self.cell_w), np.uint8)


def _blit(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Clip-safe alpha copy (index 0 transparent). Local to avoid a compose
    import cycle — text is a leaf module."""
    h, w = src.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = src[y0 - y:y1 - y, x0 - x:x1 - x]
    win = dst[y0:y1, x0:x1]
    win[sub != T] = sub[sub != T]


# --- rendering ----------------------------------------------------------
def line_width(font: Font, s: str, advance: int = ADVANCE) -> int:
    return 0 if not s else (len(s) - 1) * advance + font.cell_w


def render_line(font: Font, s: str, advance: int = ADVANCE) -> np.ndarray:
    """Lay glyphs left-to-right at `advance`, one CELL_H-tall local-index strip."""
    strip = np.zeros((font.cell_h, line_width(font, s, advance)), np.uint8)
    for i, ch in enumerate(s):
        _blit(strip, font.glyph(ch), i * advance, 0)
    return strip


def _squeeze(strip: np.ndarray, max_w: int) -> np.ndarray:
    """Nearest-neighbour horizontal squeeze to <= max_w (index-preserving)."""
    w = strip.shape[1]
    if w <= max_w or max_w <= 0:
        return strip
    xs = (np.arange(max_w) * w) // max_w
    return strip[:, xs]


def fit_line(font: Font, s: str, max_w: int) -> np.ndarray:
    """One line, ALWAYS: render at full advance, condense tracking to fit, and
    only as a last resort squeeze pixels. Result width is always <= max_w."""
    for advance in (ADVANCE, CELL_W):
        strip = render_line(font, s, advance)
        if strip.shape[1] <= max_w:
            return strip
    return _squeeze(strip, max_w)


def render_band(font: Font, s: str, band_w: int, band_h: int) -> np.ndarray:
    """Fit `s` to one line and centre it (H and V) in a band-sized local-index
    matrix. Reuses the layout centring discipline: block bbox centre == band
    centre."""
    band = np.zeros((band_h, band_w), np.uint8)
    if s:
        strip = fit_line(font, s, band_w)
        sh, sw = strip.shape
        _blit(band, strip, (band_w - sw) // 2, (band_h - sh) // 2)
    return band


# --- loading ------------------------------------------------------------
def load_font(font_dir: str | Path | None = None) -> Font:
    """The engine placeholder font, with any authored glyph tiles in `font_dir`
    overriding per glyph (same first-file-wins override story as tiles)."""
    from arcana.seed import placeholder_font  # lazy: seed imports Font from here
    from arcana.tileio import read_any
    glyphs = dict(placeholder_font().glyphs)
    if font_dir and Path(font_dir).is_dir():
        d = Path(font_dir)
        for ch, stem in _STEM.items():
            if any((d / (stem + ext)).exists() for ext in (".txt", ".png")):
                glyphs[ch] = read_any(d / stem, expect=(CELL_H, CELL_W))
    return Font(glyphs=glyphs)
