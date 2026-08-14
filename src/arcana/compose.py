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

from arcana.palette import T, LINE, DARK, MID, LIGHT, Palette
from arcana.geometry import Geometry
from arcana.elements import Element, overlay, blit
from arcana.layout import arrange
from arcana.field import build as build_field
from arcana.text import Font, render_band, CELL_H

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
    blit(dst, src, x, y)


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
    # top-left quadrant is drawn; the mirror below fills the other three. With no
    # medallion, the dentils continue through the medallion slot: whole tiles up
    # to the mirror axis, then a final tab ANCHORED flush to the axis. The
    # vertical slot's half-height is not tile-divisible (that remainder is why the
    # slot exists), so tiling outward alone would truncate the last tab against
    # the mirror and leave a ~4px hole; the anchored tab lands its inter-tab gap
    # on the axis instead, so the tab meets its mirror with the normal seam.
    x = C
    for _ in range(nh):
        paste(frame, et, x, 0); x += E
    x_med = x
    if medallion is None:
        while x + E <= W // 2:
            paste(frame, et, x, 0); x += E
        if x < W // 2:                       # sub-tile remainder: flush the axis
            paste(frame, et, W // 2 - E, 0)
    y = C
    for _ in range(nv):
        paste(frame, evt, 0, y); y += E
    y_med = y
    if medallion is None:
        while y + E <= H // 2:
            paste(frame, evt, 0, y); y += E
        if y < H // 2:
            paste(frame, evt, 0, H // 2 - E)
    paste(frame, ct, 0, 0)

    frame[:, W // 2:] = frame[:, :W // 2][:, ::-1]
    frame[H - H // 2:] = frame[:H // 2][::-1, :]

    # medallions last, flush with the card edge, never mirrored
    if medallion is not None:
        m = medallion.matrix
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
        overlay(out, p.bind(med, med_bank))
    return out


def _scale_tile(tile: np.ndarray, scale: float) -> np.ndarray:
    """Nearest-neighbour resize of an index tile by a (possibly fractional)
    factor — preserves palette indices exactly (no blending)."""
    h, w = tile.shape
    sh, sw = max(1, round(h * scale)), max(1, round(w * scale))
    if (sh, sw) == (h, w):
        return tile
    ys = (np.arange(sh) * h) // sh
    xs = (np.arange(sw) * w) // sw
    return tile[ys][:, xs]


# ---------------------------------------------------------------- content
# A card is a stack of independent, composable pieces (see `assemble`): one
# CONTENT object fills the art area, the BORDER rings it, and an optional LABEL
# names it. `build_development` and `build_mural` are the two content kinds —
# same slot, interchangeable.
def build_development(palette: Palette, geo: Geometry, els: dict[str, Element],
                      count: int, layout_name: str, pip_key: str,
                      field_design: str = "plain",
                      pip_cfg: dict | None = None) -> np.ndarray:
    """The minor-arcana CONTENT: a heraldic field background (the named design,
    in the `field` bank) with `count` pips placed by the named layout, as one
    card-sized global-index object (bands left blank). Both field and pips vary
    only inside an inset margin, so nothing runs into the frame; pip size is
    auto-fit by `layout.arrange`, which raises if the layout can't fit."""
    card = np.zeros((geo.card_h, geo.card_w), np.uint8)
    ox, oy = geo.art_origin

    field = build_field(field_design, geo)
    paste(card, palette.bind(field, "field"), ox, oy)

    centres, scale = arrange(layout_name, count, geo, **(pip_cfg or {}))
    pip = _scale_tile(palette.bind(els[pip_key].layers["motif"], "motif"), scale)
    for cx, cy in centres:
        paste_centered(card, pip, ox + cx, oy + cy)
    return card


def build_mural(palette: Palette, geo: Geometry, els: dict[str, Element],
                field_design: str = "plain") -> np.ndarray:
    """The major-arcana CONTENT: the whole scene as one object. For now it is
    the field background filling the art window — a documented seam where the
    `figure`-bank image will paste later (that roadmap item plugs in here)."""
    card = np.zeros((geo.card_h, geo.card_w), np.uint8)
    ox, oy = geo.art_origin
    field = build_field(field_design, geo)
    paste(card, palette.bind(field, "field"), ox, oy)
    return card


# ---------------------------------------------------------------- labels
def band_rects(geo: Geometry) -> dict[str, tuple[int, int, int, int]]:
    """The USABLE (x0,y0,x1,y1) rectangles for the numeral and title bands.
    Labels HUG THE ART WINDOW — a `CELL_H`-tall strip on the art-side of each
    band, inset by `corner` on the sides — so they sit clear of the frame's
    outer rule and the edge medallions that ring the card. Derived from `geo`,
    never hard-coded; the top strip is still cramped where the top medallion
    intrudes (a documented tuning seam for the opt-in numeral)."""
    C = geo.corner
    strip = CELL_H + 2                       # glyph height + 1px breathing room
    top = geo.band_numeral                   # art top edge
    bottom = geo.band_numeral + geo.art_h    # art bottom edge
    return {
        "numeral": (C, top - strip, geo.card_w - C, top),
        "title": (C, bottom, geo.card_w - C, bottom + strip),
    }


def build_label(geo: Geometry, font: Font, *, top: str | None = None,
                bottom: str | None = None) -> np.ndarray:
    """A card-sized LOCAL-index matrix with `top` centred in the numeral-band
    rect and `bottom` in the title-band rect (rects from `band_rects`). Ink is a
    single slot; the caller binds it (to the `border` bank) and composites it.
    Top-slot placement is centred for now — nudging it clear of the top
    medallion is a documented tuning seam."""
    card = np.zeros((geo.card_h, geo.card_w), np.uint8)
    rects = band_rects(geo)
    for text, key in ((top, "numeral"), (bottom, "title")):
        if not text:
            continue
        x0, y0, x1, y1 = rects[key]
        paste(card, render_band(font, text, x1 - x0, y1 - y0), x0, y0)
    return card


# ---------------------------------------------------------------- assembly
def assemble(geo: Geometry, *, content: np.ndarray, border: np.ndarray,
             label: np.ndarray | None = None,
             label_over_border: bool = False) -> np.ndarray:
    """Stack the pieces into one global-index card. Content first, then the
    border and label in the order set by `label_over_border`: a minor's title
    sits UNDER the border (between pips and frame, so the frame rule is never
    clipped); a major's title floats OVER the mural. The one place stacking
    lives — card builders only choose which pieces to pass."""
    card = content.copy()
    if label is None:
        order = [border]
    elif label_over_border:
        order = [border, label]             # major: title floats over the art
    else:
        order = [label, border]             # minor: frame paints over the title
    for layer in order:
        overlay(card, layer)
    return card


def build_pip_card(palette: Palette, geo: Geometry, els: dict[str, Element],
                   count: int, layout_name: str, pip_key: str,
                   field_design: str = "plain", pip_cfg: dict | None = None,
                   *, font: Font | None = None, top: str | None = None,
                   bottom: str | None = None, med_style: str = "suit",
                   med_scale: float = 1.0) -> np.ndarray:
    """A minor-arcana pip card as one global index matrix: the development
    (field + pips) with the border (the suit medallion mounted in the cartouche)
    on top, plus optional labels UNDER the border. Suit-invariant when unlabelled
    — colour is applied by rendering with `palette.for_suit(...)`; a suit-name
    title legitimately varies per suit. Defaults (`font=None`, `med_style="suit"`,
    `med_scale=1.0`) reproduce the base card byte-for-byte."""
    content = build_development(palette, geo, els, count, layout_name, pip_key,
                                field_design, pip_cfg)
    med = build_medallion(els, pip_key, style=med_style, scale=med_scale)
    border = render_border(palette, geo, els["corner"], els["edge"], med)
    label = (palette.bind(build_label(geo, font, top=top, bottom=bottom), "border")
             if font is not None and (top or bottom) else None)
    return assemble(geo, content=content, border=border, label=label,
                    label_over_border=False)


def build_major_card(palette: Palette, geo: Geometry, els: dict[str, Element],
                     font: Font, *, top: str | None = None,
                     bottom: str | None = None, field_design: str = "plain",
                     pip_key: str | None = None, med_style: str = "suit",
                     med_scale: float = 1.0) -> np.ndarray:
    """A major-arcana card (labeling half): the mural + border + label floating
    ON TOP of the art, same place as a minor's title. The `figure`-bank image is
    a later roadmap item — it pastes into the mural (see `build_mural`)."""
    content = build_mural(palette, geo, els, field_design)
    med = build_medallion(els, pip_key, style=med_style, scale=med_scale)
    border = render_border(palette, geo, els["corner"], els["edge"], med)
    label = (palette.bind(build_label(geo, font, top=top, bottom=bottom), "border")
             if top or bottom else None)
    return assemble(geo, content=content, border=border, label=label,
                    label_over_border=True)


def mount(pip: Element, cartouche: Element) -> Element:
    """Drop a pip into the centre of a cartouche, returning a new Element."""
    c = cartouche.matrix.copy()
    a = pip.matrix
    oy = (c.shape[0] - a.shape[0]) // 2
    ox = (c.shape[1] - a.shape[1]) // 2
    overlay(c[oy:oy + a.shape[0], ox:ox + a.shape[1]], a)
    return Element(name=f"{pip.name}@{cartouche.name}", role="medallion",
                   size=c.shape, layers={cartouche.sole_bank: c})


def build_medallion(els: dict[str, Element], pip_key: str | None, *,
                    style: str = "suit", scale: float = 1.0) -> Element | None:
    """The edge medallion as a `motif`-bound Element (suit-coloured), or None.
    `suit` is the pip mounted in the cartouche; `lozenge` is an abstract diamond;
    `none` omits it (and `build_border` continues the ornament through the slot).
    `scale` resizes the emblem with the index-preserving NN resize — the knob for
    dialing the medallion down so it no longer competes with the title."""
    if style == "none" or (style == "suit" and pip_key is None):
        return None
    if style == "suit":
        layer = mount(els[pip_key], els["cartouche"]).matrix
    elif style == "lozenge":
        from arcana.seed import lozenge
        base = els["cartouche"].layers["motif"].shape[0]     # match the cartouche size
        layer = lozenge(base)
    else:
        raise ValueError(f"unknown medallion style {style!r}; "
                         "expected suit | lozenge | none")
    if scale != 1.0:
        layer = _scale_tile(layer, scale)
    return Element(name=f"medallion.{style}", role="medallion",
                   size=layer.shape, layers={"motif": layer})


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
