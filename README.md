# arcana

Pixel-art card generation. A modular tile/palette engine plus per-deck configs,
built from tiles, a bank palette, and YAML geometry. See **AGENTS.md** for the
architecture and design rationale.

```
src/arcana/
  palette.py     Bank, Palette, local/global index space
  geometry.py    Geometry + tiling validation
  elements.py    Element, indexed-PNG loader, audit
  tileio.py      RGB import, ASCII format, format dispatch
  compose.py     border assembly + structural checks
  layout.py      pip arrangement algorithms (square, diamond, ordinaries…)
  mural.py       major-arcana murals (layer loader, import seam, opt-in probe)
  seed.py        placeholder-tile generators (procedural + ASCII pips)
  deck.py        deck resolution (config + artifacts paths)
  cli.py         thin `arcana` CLI
decks/
  configs/<name>/   palette.yaml + deck.yaml + murals/  (committed)
  artifacts/<name>/ generated tiles + renders           (git-ignored)
tests/
  test_deck.py   regression tests, tiles seeded into tmp
```

Config and code are committed; **tiles and renders are not**. A deck's config
lives under `decks/configs/<name>/`; running the engine regenerates its
placeholder tiles and PNGs under `decks/artifacts/<name>/`, which is git-ignored.

## Quickstart

```bash
uv sync
uv run arcana generate vaporwave-rws     # per-suit borders
uv run arcana cards vaporwave-rws        # minor-arcana pip cards (ranks 1-10)
uv run arcana majors vaporwave-rws       # the 22 major arcana (mural tooling ready)
uv run pytest
```

Everything is written under `decks/artifacts/<deck>/` (git-ignored).

- `arcana generate <deck>` seeds placeholder tiles (if missing) and renders the
  per-suit borders.
- `arcana cards <deck>` renders the minor arcana on two independent axes:
  - **pip arrangement** per rank (`pip_layouts` in `deck.yaml`); `--layout <name>`
    forces one across all ranks. Layouts: `single`, `pale`, `fess`, `bend`,
    `square`, `pile`, `chevron`, `cross`, `saltire`, `diamond`, `lozenge`,
    `pall`, `seme`.
  - **field design** per suit (`field_designs` in `deck.yaml`); `--field <name>`
    forces one across all suits. Designs: divisions (`per-pale`, `per-fess`,
    `per-bend`, `per-bend-sinister`, `per-chevron`, `per-saltire`, `quarterly`),
    ordinary bands (`chief`, `base`, `pale`, `fess`, `bend`, `chevron`, `cross`,
    `saltire`, `pile`, `bordure`), patterns (`barry`, `paly`, `bendy`,
    `chevronny`, `checky`, `lozengy`), and `plain`.

  Both are chosen by **name** (algorithms, not authored tiles); field colours come
  from the suit's `field` bank in `palette.yaml`. `--suit <s>` / `--all-suits`
  pick which suits. See `docs/examples/` — [field-gallery](docs/examples/field-gallery.png)
  (one card under every field design), [minor-arcana](docs/examples/minor-arcana.png)
  (the per-suit defaults across ranks 1–10, with numeral/title labels), and
  [major-arcana](docs/examples/major-arcana.png) (the 22 majors — frame and
  label; mural art is pending).

  Both `cards` and `majors` label each card with a **bitmap font** (`arcana.text`):
  a rank/roman numeral and the card name, chosen by plaintext (`arcana.data`),
  inked in the `border` bank so titles match the frame and vary with the palette.
  Content comes from engine defaults overridable in a `labels:` block; a combined
  title sits in the bottom band by default (`split: true` splits it across the
  numeral and title bands). `--no-labels` renders bare cards for comparison.
  - **medallion** — the suit emblem at the edge midpoints is tunable
    (`border.medallion` in `deck.yaml`; `--medallion suit|lozenge|none` and
    `--medallion-scale SIZE`): a scaled suit cartouche, an abstract suit-colored
    lozenge, or none (the border ornament then continues through the space). Size
    takes a keyword (`full`/`large`/`small`/`smaller`/`tiny`) or a number.
- `arcana majors <deck>` renders the 22 major arcana: frame, field, and label.
  When a deck ships mural art — a **mural** is an image laid on the field —
  it is composited into the art window automatically. Murals are per-bank
  ASCII layers under `decks/configs/<deck>/murals/`, authored in the same
  local index space as any tile — so a palette swap recolors them; their
  presence is the switch (all-or-nothing: a partial set fails loudly), and
  `--no-murals` renders bare for comparison. No deck currently commits mural
  art, so the majors render the bare field.
- `arcana import-mural <deck> <png> --major N` maps pixel art into the glyph
  representation: it quantizes any art-window-sized RGB image to the deck's
  14 drawable colors (`--force` snaps off-palette pixels) and writes the
  per-bank ASCII layer files. `arcana export-mural` is the exact inverse —
  it renders a mural's art window back to an RGB PNG (for external editors,
  or as a generation init image). The round trip is lossless, which is how
  externally generated pixel art (e.g. Retro Diffusion) enters the deck.
- `arcana seed <deck>` just (re)writes the placeholder tiles.
- `--scale N` is a NEAREST zoom applied only to the preview contact sheet so the
  tiny pixel-art cards are legible; individual PNGs are always native resolution.

## Authoring a tile

Tiles are authored in **local index space** (value structure, not color):

```
.  transparent   @  line    '  paper
%  dark          +  mid     -  light
```

`tileio.read_any` takes whichever format is present:

- **Indexed PNG** — Aseprite, Pixelorama (indexed mode), LibreSprite. Load the
  authoring palette (`seed.py` writes `authoring.gpl`).
- **RGB PNG** — any editor; use the six authoring colors, anti-aliasing off.
  Off-palette pixels are rejected with coordinates.
- **ASCII `.txt`** — a tile is a grid of `0-5`, so it is text; diffs usefully in
  git and needs no image library.
