"""
Major-arcana murals: the image that fills a major's frame.

A mural is authored EXACTLY like a pip: per-bank layer files in the same
6-glyph local ASCII alphabet (`.@'%+-`, see tileio) or indexed/RGB PNG — a
mural is simply the deck's biggest element. Because every stored file is
local index space, a palette swap recolours the whole set without touching a
pixel; because one mural carries a layer per bank, the composed image can use
all 14 drawable colours even though no single file exceeds six slots.

    decks/configs/<deck>/murals/major_16.border.txt
                                major_16.motif.txt
                                major_16.figure.txt
                                major_16.field.txt

THE IMAGE BOX (`image_box`): a mural is NOT art-window-sized. The art window
overlaps the side frame bands by (corner - margin), so a window-wide image
would run under the side dentils — top and bottom sit clear, making the
overlap one-axis and easy to miss. The mural's canonical size is the box
INSIDE the frame band on the sides: (card_w - 2*corner) x art_h. The
full-bleed background field still runs under the band; the image never does.
A bonus of the narrower box: the foreground-inset convention becomes a
uniform `margin` on all four sides instead of sides-vs-verticals asymmetry.

Murals live in the deck CONFIG dir, committed, unlike tiles: they are the
deck's actual art, not regenerable placeholders, and ASCII text diffs like any
other source. (`arcana import-mural` writes this same format from an external
RGB image, so generated art and hand-authored art are indistinguishable on
disk.)

OPT-IN IS PRESENCE, ALL-OR-NOTHING: a deck with no murals/ layers renders the
bare field; once ANY major has layers, every major must — a partial set fails
loudly naming the missing stem, because a silently bare card in a printed
deck is exactly the bug (`arcana majors --no-murals` renders bare deliberately,
for comparison).

COMPOSITION MODEL: an image laid on a field. `compose.build_mural` paints the
field-bank background first — full-bleed, running under the frame band's
ornament on every side so the title floats on it — and overlays the mural image
at the art origin; the image's transparent pixels show the field, so a card's
"sky" is the field colour and recolours with the palette like everything else.
A `field` LAYER is still allowed — a scene that needs dark strata (night sky, a
pool) paints them in field-bank tones so they stay in the background's hue
family.

Bank semantics inside a mural (a convention, not a mechanism):

    line    outlines and black elements       paper   whites — roses, bones,
                                                      stars, the white horse
    field   background strata                 border  architecture, thrones,
                                                      robes (ties art to frame)
    motif   the key props and accents         figure  flesh and bodies; also
                                                      the warm accent ramp
                                                      (wood, gold, flame)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from arcana.elements import AssetError, Element
from arcana.geometry import Geometry
from arcana.palette import BANKS, DARK, LINE, PAPER


def image_box(geo: Geometry) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the mural image box on the card: flush with the frame
    band's inner edge on the sides (the art window would run 8px under the
    side dentils), and with the art window's top and bottom (the horizontal
    bands already stop there)."""
    return (geo.corner, geo.art_origin[1],
            geo.card_w - 2 * geo.corner, geo.art_h)


def stem(number: int) -> str:
    """The dotted-stem base for a major's layer files: `major_07` -> the
    on-disk family `major_07.<bank>.txt|png`. Zero-padded so a directory
    listing sorts in arcana order."""
    return f"major_{number:02d}"


def has_murals(murals_dir: str | Path) -> bool:
    """Does this deck ship ANY mural layers? Presence is the opt-in switch:
    once true, every major is required (see `load_mural(required=True)`), so a
    partial art set fails loudly instead of printing silently bare cards."""
    d = Path(murals_dir)
    return d.is_dir() and any(d.glob("major_*.*"))


def load_mural(murals_dir: str | Path, number: int, geo: Geometry,
               *, required: bool = False) -> Element | None:
    """Load one major's mural as an Element (role `mural`), gathering every
    per-bank layer present under `murals_dir`. Layer files use dotted stems
    (`major_16.border`), so existence is probed per format suffix — NEVER
    `Path.with_suffix`, which eats the bank name (the documented bug).

    Returns None when the mural has no layers at all, unless `required` — a
    deck that ships murals deserves an error naming the expected files, not a
    silently bare card."""
    d = Path(murals_dir)
    _, _, bw, bh = image_box(geo)
    expect = (bh, bw)
    from arcana.tileio import read_any
    layers: dict[str, np.ndarray] = {}
    for bank in BANKS:
        base = d / f"{stem(number)}.{bank}"
        if any((d / (base.name + s)).exists() for s in (".txt", ".png")):
            layers[bank] = read_any(base, expect=expect)
    if not layers:
        if required:
            raise AssetError(
                f"no mural for {stem(number)} under {d} — expected at least "
                f"one layer file {stem(number)}.<bank>.txt (bank in "
                f"{'/'.join(BANKS)})")
        return None
    return Element(name=stem(number), role="mural", size=expect, layers=layers)


def split_global(m: np.ndarray) -> dict[str, np.ndarray]:
    """The inverse of `Element.bind`: factor a global-index matrix (0-14) into
    per-bank LOCAL layers, the authored mural format. Each bank's three global
    slots map back to DARK/MID/LIGHT; universal LINE/PAPER pixels — which every
    layer can carry, since `bind` maps them onto themselves — land in the
    `figure` layer by convention (they are mostly outlines and highlights on
    the foreground). This is what lets `import-mural` write quantized external
    art in the exact same format a hand would author.

    Round trip: `element.bind(palette)` of the returned layers reproduces `m`
    exactly — the layers are disjoint by construction, so overlay order can't
    matter."""
    hi = int(m.max(initial=0))
    if hi > 14:
        raise AssetError(f"index {hi} outside global space (max 14)")
    layers: dict[str, np.ndarray] = {}
    for i, bank in enumerate(BANKS):
        base = 3 + 3 * i
        mask = (m >= base) & (m < base + 3)
        if mask.any():
            loc = np.zeros(m.shape, np.uint8)
            loc[mask] = m[mask] - base + DARK
            layers[bank] = loc
    universal = (m == LINE) | (m == PAPER)
    if universal.any():
        host = layers.setdefault("figure", np.zeros(m.shape, np.uint8))
        host[universal] = m[universal]
    return {b: layers[b] for b in BANKS if b in layers}
