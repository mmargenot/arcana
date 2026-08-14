"""
Composition: elements + geometry + palette -> a global index matrix.

The frame line is STRUCTURAL and the tiles are DECORATION. `build_border` draws
the beveled rule frame itself — straight rules between the corners, mitered at
each corner — and then overlays the corner and edge tiles as pure ornament. That
split is deliberate: swap a corner or edge motif and the frame line does not
move a pixel.

Four rules earned the hard way, each guarding a bug that is invisible at 1x:

  ORIENT   a side edge is the top edge TRANSPOSED, nothing more. An extra
           horizontal flip puts the outer rule on the inner side of the band,
           so both side edges lose their rule for their whole length.

  FRAME    the rule frame is drawn from PROFILE, never read off a tile. The
           corner mitres by sampling PROFILE at depth min(row, col), so every
           rule turns 45 degrees instead of butting square.

  MOTIF    corner and edge tiles carry ornament on a transparent field. They are
           pasted over the frame, so their transparent pixels leave the rules
           intact and only their marks show.

  LAST     place medallions after mirroring, never before. Mirroring a cup
           gives you a cup lying on its side.
"""
from __future__ import annotations
import numpy as np

from arcana.palette import T, LINE, PAPER, DARK, MID, LIGHT, Palette
from arcana.geometry import Geometry
from arcana.elements import Element
from arcana.layout import place
from arcana.field import build as build_field

# frame profile: (start_row, end_row, local_slot), outside in.
# Shared by the corner, the edge and the backing strip so rules line up by
# construction rather than by hand.
PROFILE = ((0, 1, LINE), (1, 2, MID), (2, 3, LIGHT), (5, 7, DARK))


def _require_role(el: Element, role: str) -> None:
    """Roles are declared in deck.yaml; enforce them where pieces are consumed
    so a manifest edit that swaps two elements fails loudly."""
    if el.role != role:
        raise ValueError(f"{el.name!r} has role {el.role!r}, expected {role!r}")


def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Clip-safe alpha paste. Index 0 in src leaves dst untouched."""
    h, w = src.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = src[y0 - y:y1 - y, x0 - x:x1 - x]
    win = dst[y0:y1, x0:x1]
    win[sub != T] = sub[sub != T]


def paste_centered(dst: np.ndarray, src: np.ndarray, cx: int, cy: int) -> None:
    paste(dst, src, cx - src.shape[1] // 2, cy - src.shape[0] // 2)


def profile_strip(thickness: int, length: int) -> np.ndarray:
    s = np.zeros((thickness, length), np.uint8)
    for a, b, v in PROFILE:
        s[a:b, :] = v
    return s


def corner_rule(C: int) -> np.ndarray:
    """The frame corner as a 45-degree mitre. Colour at (r, c) is PROFILE sampled
    at depth min(r, c), so every rule — the LINE/MID/LIGHT triple and the inner
    DARK rule alike — turns the corner along the diagonal instead of cutting
    square. Symmetric about the diagonal, so it meets the transposed side edge
    with no seam."""
    prof = np.zeros(C, np.uint8)
    for a, b, v in PROFILE:
        prof[a:b] = v
    rr, cc = np.mgrid[0:C, 0:C]
    return prof[np.minimum(rr, cc)]


def orient(tile: np.ndarray) -> np.ndarray:
    """Top-edge piece -> left-edge piece. Transpose only. No flip."""
    return tile.T


# ---------------------------------------------------------------- border
def build_border(geo: Geometry, corner: Element, edge: Element,
                 medallion: Element | None = None,
                 frame_bank: str = "border") -> tuple[np.ndarray, np.ndarray]:
    """Returns (frame_local, medallion_local) — both in local index space."""
    _require_role(corner, "corner")
    _require_role(edge, "edge")
    if medallion is not None:
        _require_role(medallion, "medallion")
    nh, nv = geo.runs
    W, H, C, E = geo.card_w, geo.card_h, geo.corner, geo.edge
    frame = np.zeros((H, W), np.uint8)
    med = np.zeros((H, W), np.uint8)

    ct = corner.layers[frame_bank]      # corner MOTIF, transparent field
    et = edge.layers[frame_bank]        # edge MOTIF, transparent field
    evt = orient(et)

    # structural frame: straight rules between the corners, mitred at the corner.
    # Drawn from PROFILE, so the frame is identical whatever motifs are pasted on.
    paste(frame, profile_strip(C, W - 2 * C), C, 0)          # top rules
    paste(frame, orient(profile_strip(C, H - 2 * C)), 0, C)  # left rules
    paste(frame, corner_rule(C), 0, 0)                       # beveled corner

    # ornament overlay: dentils along each run, staircase in the corner. Only the
    # top-left quadrant is drawn; the mirror below fills the other three.
    x = C
    for _ in range(nh):
        paste(frame, et, x, 0); x += E
    x_med = x
    y = C
    for _ in range(nv):
        paste(frame, evt, 0, y); y += E
    y_med = y
    paste(frame, ct, 0, 0)

    frame[:, W // 2:] = frame[:, :W // 2][:, ::-1]
    frame[H - H // 2:] = frame[:H // 2][::-1, :]

    # medallions last, flush with the card edge, never mirrored
    if medallion is not None:
        m = next(iter(medallion.layers.values()))
        mh, mw = m.shape
        cx = x_med + geo.med_h // 2 - mw // 2
        cy = y_med + geo.med_v // 2 - mh // 2
        paste(med, m, cx, 0)
        paste(med, m, cx, H - mh)
        paste(med, m, 0, cy)
        paste(med, m, W - mw, cy)
    return frame, med


def render_border(p: Palette, geo: Geometry, corner: Element, edge: Element,
                  medallion: Element | None = None,
                  frame_bank: str = "border",
                  med_bank: str = "motif") -> np.ndarray:
    frame, med = build_border(geo, corner, edge, medallion, frame_bank)
    out = p.bind(frame, frame_bank)
    if medallion is not None:
        g = p.bind(med, med_bank)
        out[g != T] = g[g != T]
    return out


# Pip size (integer multiples of the 16px motif tile) is auto-fit per card: the
# largest factor whose square doesn't overlap its neighbour. Depends jointly on
# the arrangement and the count, since both set the centre spacing.
PIP_BASE = 16
PIP_MIN_SCALE = 2
PIP_MAX_SCALE = 3


def _pip_scale(centres: list[tuple[int, int]]) -> int:
    """Largest integer pip factor that fits `centres` without overlap. Uses the
    minimum Chebyshev separation — axis-aligned squares of side `s` overlap iff
    both |dx| < s and |dy| < s — clamped to [PIP_MIN_SCALE, PIP_MAX_SCALE]. A
    lone pip (no neighbour) gets the max."""
    if len(centres) <= 1:
        return PIP_MAX_SCALE
    sep = min(max(abs(ax - bx), abs(ay - by))
              for i, (ax, ay) in enumerate(centres)
              for (bx, by) in centres[i + 1:])
    return max(PIP_MIN_SCALE, min(PIP_MAX_SCALE, sep // PIP_BASE))


def build_pip_card(palette: Palette, geo: Geometry, els: dict[str, Element],
                   count: int, layout_name: str, pip_key: str,
                   field_design: str = "plain") -> np.ndarray:
    """A minor-arcana pip card as one global index matrix: a heraldic field
    background (the named design, in the `field` bank), `count` pips placed by
    the named layout, then the border (with the suit's pip mounted in the
    cartouche) composited on top. Suit-invariant — colour is applied by
    rendering with `palette.for_suit(...)`. The field varies only inside an
    inset margin, so it never runs into the frame (an invisible border)."""
    card = np.zeros((geo.card_h, geo.card_w), np.uint8)
    ox, oy = geo.art_origin

    field = build_field(field_design, geo)
    paste(card, palette.bind(field, "field"), ox, oy)

    centres = place(layout_name, count, geo)
    k = _pip_scale(centres)
    pip = palette.bind(els[pip_key].layers["motif"], "motif")
    if k > 1:
        pip = np.repeat(np.repeat(pip, k, axis=0), k, axis=1)
    for cx, cy in centres:
        paste_centered(card, pip, ox + cx, oy + cy)

    med = mount(els[pip_key], els["cartouche"])
    border = render_border(palette, geo, els["corner"], els["edge"], med)
    card[border != T] = border[border != T]
    return card


def mount(pip: Element, cartouche: Element) -> Element:
    """Drop a pip into the centre of a cartouche, returning a new Element."""
    c = next(iter(cartouche.layers.values())).copy()
    a = next(iter(pip.layers.values()))
    oy = (c.shape[0] - a.shape[0]) // 2
    ox = (c.shape[1] - a.shape[1]) // 2
    win = c[oy:oy + a.shape[0], ox:ox + a.shape[1]]
    win[a != T] = a[a != T]
    bank = next(iter(cartouche.layers))
    return Element(name=f"{pip.name}@{cartouche.name}", role="medallion",
                   size=c.shape, layers={bank: c})


# ---------------------------------------------------------------- checks
def check_symmetry(m: np.ndarray) -> dict[str, bool]:
    h, w = m.shape
    return {"horizontal": bool(np.array_equal(m[:, :w // 2], m[:, w // 2:][:, ::-1])),
            "vertical": bool(np.array_equal(m[:h // 2], m[h - h // 2:][::-1]))}


def check_contiguous(frame: np.ndarray, geo: Geometry) -> dict[str, bool]:
    """Each rule must run unbroken between the corners. A gap is nearly
    invisible on screen and obvious in print."""
    C, W, H = geo.corner, geo.card_w, geo.card_h
    out = {}
    for a, b, _ in PROFILE:
        for r in range(a, b):
            out[f"top@{r}"] = bool((frame[r, C:W - C] != T).all())
            out[f"left@{r}"] = bool((frame[C:H - C, r] != T).all())
    return out


def components(frame: np.ndarray) -> int:
    """Connected-component count. The frame ring should be one large piece."""
    from scipy import ndimage
    lab, n = ndimage.label((frame != T).astype(np.uint8),
                           structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    return int(n)
