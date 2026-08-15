"""
Major-arcana murals: the art-window image that fills a major's frame.

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
                                court_cups_queen.figure.txt   (any face key)

Murals live in the deck CONFIG dir, committed, unlike tiles: they are the
deck's actual art, not regenerable placeholders, and ASCII text diffs like any
other source. (`arcana import-mural` writes this same format from an external
RGB image, so generated art and hand-authored art are indistinguishable on
disk.)

COMMITTED ART WINS, PLACEHOLDERS FILL: a face with layers under the deck's
config murals/ uses them; a face without falls back to the placeholder
`arcana.seed` writes into the generated assets dir. So the deck always renders
and art can land ONE CARD AT A TIME, which authoring by hand or by model both
need. The rule this replaces — all-or-nothing — existed to stop a silently bare
card reaching print, and that intent now lives in an explicit gate:
`arcana majors --strict` fails, naming every face without committed art.
(`arcana majors --no-murals` still renders the bare field, for comparison.)

A FACE CARD is any card whose art is an authored art-window image rather than an
algorithmic pip lattice. Keys are free-form strings — `major_07` for a tarot
major, `court_cups_queen` for a court, `wizard` for a game's one-off — so the
same seam serves a tarot deck, a JQKA deck, and anything else with special
cards. Nothing below knows about tarot except the default face set.

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


def major_key(number: int) -> str:
    """The face key for a major: `7` -> `major_07`, whose layer files are the
    on-disk family `major_07.<bank>.txt|png`. Zero-padded so a directory
    listing sorts in arcana order."""
    return f"major_{number:02d}"


# The engine's default face set. A deck overrides it with a `faces:` list in
# deck.yaml -- the same "engine ships defaults, deck customises" split as
# pip_layouts / field_designs / labels.major_titles.
MAJOR_KEYS: tuple[str, ...] = tuple(major_key(n) for n in range(22))


def face_keys(cfg: dict | None = None) -> tuple[str, ...]:
    """Which face cards this deck has, in render order.

    A FACE CARD is one whose art is an authored art-window image rather than an
    algorithmic pip lattice -- the tarot majors, a traditional deck's JQKA
    courts, a game's one-off specials (`wizard`, `jester`). They differ only in
    which cards exist and what they are called, so the key is a free-form
    string and nothing here knows about tarot beyond the default."""
    keys = (cfg or {}).get("faces")
    if not keys:
        return MAJOR_KEYS
    if not all(isinstance(k, str) and k for k in keys):
        raise AssetError(f"faces: must be a list of non-empty strings, got {keys!r}")
    return tuple(keys)


def has_murals(murals_dir: str | Path, keys: tuple[str, ...] = ()) -> bool:
    """Does this deck COMMIT any face art? True once any face key has a layer
    file in the config dir. This is no longer an all-or-nothing gate -- seeded
    placeholders (arcana.seed) keep every other face renderable -- it just
    reports whether real art exists."""
    d = Path(murals_dir)
    if not d.is_dir():
        return False
    return any(any(d.glob(f"{k}.*")) for k in (keys or MAJOR_KEYS))


def _gather(d: Path, key: str, expect: tuple[int, int]) -> dict[str, np.ndarray]:
    """Every per-bank layer present for `key` under `d`. Layer files use dotted
    stems (`major_16.border`), so existence is probed per format suffix — NEVER
    `Path.with_suffix`, which eats the bank name (the documented bug)."""
    from arcana.tileio import read_any
    layers: dict[str, np.ndarray] = {}
    for bank in BANKS:
        base = d / f"{key}.{bank}"
        if any((d / (base.name + s)).exists() for s in (".txt", ".png")):
            layers[bank] = read_any(base, expect=expect)
    return layers


def load_mural(murals_dir: str | Path, key: str, geo: Geometry,
               *, fallback_dir: str | Path | None = None,
               required: bool = False) -> Element | None:
    """Load one face's mural as an Element (role `mural`).

    COMMITTED ART WINS. `murals_dir` is the deck's config dir, holding real art;
    `fallback_dir` is the generated assets dir, holding seeded placeholders. A
    face with committed layers uses them; one without falls back to its
    placeholder, so a deck renders while art lands one card at a time. The
    print gate is `arcana majors --strict`, not a render-time explosion.

    Returns None when neither exists, unless `required`."""
    expect = (geo.art_h, geo.art_w)
    layers = _gather(Path(murals_dir), key, expect)
    if not layers and fallback_dir is not None:
        layers = _gather(Path(fallback_dir), key, expect)
    if not layers:
        if required:
            raise AssetError(
                f"no mural for {key} under {murals_dir} — expected at least "
                f"one layer file {key}.<bank>.txt (bank in {'/'.join(BANKS)}), "
                f"or run `arcana seed` to write a placeholder")
        return None
    return Element(name=key, role="mural", size=expect, layers=layers)


def is_committed(murals_dir: str | Path, key: str) -> bool:
    """Does this face have real, committed art (not just a placeholder)? The
    question `--strict` asks before a deck goes to print."""
    return any(Path(murals_dir).glob(f"{key}.*"))


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
