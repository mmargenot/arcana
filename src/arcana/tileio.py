"""
Tile IO that does not depend on any particular editor.

Indexed PNG is the canonical format, but it is a convenience, not a
requirement. Two escape hatches:

  RGB IMPORT  draw in any editor at all — GIMP, Krita, Piskel, Photoshop,
              even MS Paint — using the six authoring colors. `read_rgb`
              snaps each pixel to the nearest authoring swatch and recovers
              the index. Lossless as long as you stay on-palette, and it
              reports anything that was off.

  ASCII       a tile is a small grid of integers 0-5, so it is perfectly
              representable as text. 16 lines of 16 characters. Editable in
              any text editor, diffable in git (a PNG diff tells you nothing;
              an ASCII diff shows you exactly which pixels moved), and
              generatable by script without an image library.

Glyph weight roughly tracks value, so an ASCII tile is legible as a picture:

    .  transparent      @  line
    %  dark             +  mid
    -  light            '  paper
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

from arcana.palette import MAX_LOCAL, LOCAL_SLOTS
from arcana.elements import AUTHORING, AssetError

# index -> glyph. Heavier glyph = darker value.
GLYPHS = ".@'%+-"
#         012345
#         0 transparent  1 line  2 paper  3 dark  4 mid  5 light
_FROM_GLYPH = {g: i for i, g in enumerate(GLYPHS)}


# ---------------------------------------------------------------- RGB import
def read_rgb(path: str | Path, expect: tuple[int, int] | None = None,
             tolerance: float = 24.0) -> np.ndarray:
    """
    Import a tile drawn in ANY editor, in RGB, using the authoring colors.
    Snaps each pixel to the nearest authoring swatch.

    tolerance is the max allowed squared-ish distance before a pixel is
    reported as off-palette — anti-aliasing and stray colors get caught here
    rather than silently becoming the wrong index.
    """
    path = Path(path)
    img = Image.open(path)
    if img.mode == "P":
        a = np.array(img, np.uint8)
        if a.max(initial=0) <= MAX_LOCAL:
            return a
    rgba = np.array(img.convert("RGBA"), np.int32)
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    ref = np.array(AUTHORING, np.int32)

    d = ((rgb[:, :, None, :] - ref[None, None, :, :]) ** 2).sum(-1)
    idx = d.argmin(-1).astype(np.uint8)
    dist = np.sqrt(d.min(-1))
    idx[alpha < 128] = 0                      # real transparency wins

    off = (dist > tolerance) & (alpha >= 128)
    if off.any():
        n = int(off.sum())
        ys, xs = np.nonzero(off)
        sample = rgb[ys[0], xs[0]]
        raise AssetError(
            f"{path.name}: {n} pixel(s) off the authoring palette, first at "
            f"({xs[0]},{ys[0]}) = #{sample[0]:02X}{sample[1]:02X}{sample[2]:02X}. "
            "Anti-aliasing is the usual cause — turn it off in your editor.")
    if expect is not None and idx.shape != expect:
        raise AssetError(f"{path.name} is {idx.shape[1]}x{idx.shape[0]}, "
                         f"expected {expect[1]}x{expect[0]}")
    return idx


def quantize_rgb_global(path: str | Path, palette, tolerance: float = 24.0,
                        force: bool = False) -> np.ndarray:
    """
    Import ANY RGB image — external pixel art, a generative model's output, a
    re-edited export — as a GLOBAL index matrix (0-14) by snapping each pixel
    to the nearest of the palette's 14 drawable colors. The `read_rgb` of the
    mural world: same distance discipline (int32 — squared RGB distance maxes
    at 195,075 and int16 wraps), same off-palette report naming the first bad
    pixel, but against a deck's rendered colors instead of the six authoring
    greys. `force=True` snaps off-palette pixels to their nearest slot instead
    of raising (ties break to the lowest index, deterministically).

    Real transparency (alpha < 128) becomes index 0 — on a card that shows the
    field behind the mural image. The result is still deck-portable: it stores
    SLOTS, not the hexes it was quantized against.
    """
    path = Path(path)
    rgba = np.array(Image.open(path).convert("RGBA"), np.int32)
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    ref = palette.rgb_lut().astype(np.int32)[1:]      # the 14 drawable colors

    d = ((rgb[:, :, None, :] - ref[None, None, :, :]) ** 2).sum(-1)
    idx = (d.argmin(-1) + 1).astype(np.uint8)         # +1: ref[0] is global 1
    dist = np.sqrt(d.min(-1))
    idx[alpha < 128] = 0                              # real transparency wins

    off = (dist > tolerance) & (alpha >= 128)
    if off.any() and not force:
        n = int(off.sum())
        ys, xs = np.nonzero(off)
        sample = rgb[ys[0], xs[0]]
        raise AssetError(
            f"{path.name}: {n} pixel(s) off the deck palette, first at "
            f"({xs[0]},{ys[0]}) = #{sample[0]:02X}{sample[1]:02X}{sample[2]:02X}. "
            "Anti-aliasing or an unquantized source is the usual cause — "
            "re-export with a hard palette, or pass --force to snap.")
    return idx


def write_rgb(art: np.ndarray, path: str | Path) -> None:
    """Export as plain RGB PNG for editing in a non-indexed tool."""
    ref = np.array(AUTHORING, np.uint8)
    rgb = ref[art]
    rgba = np.dstack([rgb, np.where(art == 0, 0, 255).astype(np.uint8)])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path)


# ---------------------------------------------------------------- ASCII
def to_ascii(art: np.ndarray, name: str = "") -> str:
    head = [f"# {name}" if name else "# tile",
            f"# {art.shape[1]}x{art.shape[0]}",
            "# " + "  ".join(f"{g}={s}" for g, s in zip(GLYPHS, LOCAL_SLOTS))]
    body = ["".join(GLYPHS[v] for v in row) for row in art]
    return "\n".join(head + body) + "\n"


def from_ascii(text: str, expect: tuple[int, int] | None = None) -> np.ndarray:
    rows = [ln.rstrip("\n") for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        raise AssetError("no tile rows found")
    widths = {len(r) for r in rows}
    if len(widths) > 1:
        raise AssetError(f"ragged rows, widths {sorted(widths)} — every line "
                         "must be the same length")
    out = np.zeros((len(rows), len(rows[0])), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in _FROM_GLYPH:
                raise AssetError(
                    f"unknown glyph {ch!r} at ({x},{y}). Valid: {GLYPHS}")
            out[y, x] = _FROM_GLYPH[ch]
    if expect is not None and out.shape != expect:
        raise AssetError(f"tile is {out.shape[1]}x{out.shape[0]}, "
                         f"expected {expect[1]}x{expect[0]}")
    return out


def write_ascii(art: np.ndarray, path: str | Path, name: str = "") -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_ascii(art, name or p.stem))


def read_ascii(path: str | Path, expect: tuple[int, int] | None = None) -> np.ndarray:
    return from_ascii(Path(path).read_text(), expect)


# ---------------------------------------------------------------- dispatch
def read_any(stem: str | Path, expect: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load a tile by stem, trying each format. Lets different elements in the
    same project come from different tools.
    """
    stem = Path(stem)
    # NB: never with_suffix here — element paths have dotted stems like
    # 'corner.border', and with_suffix would eat the bank name.
    for suffix, fn in ((".txt", read_ascii), (".png", None)):
        p = stem.parent / (stem.name + suffix)
        if p.exists():
            if suffix == ".txt":
                return fn(p, expect)
            from arcana.elements import read_tile
            try:
                return read_tile(p, expect)      # indexed
            except AssetError as e:
                if "not indexed" not in str(e):
                    raise
                return read_rgb(p, expect)       # RGB fallback
    raise AssetError(f"no tile found at {stem}.txt or {stem}.png")
