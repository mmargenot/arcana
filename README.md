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
  mural.py       face-card murals (keys, loader, safe-area fitting, ground)
  pixelate.py    an RWS scan -> a generation seed (crop, de-screen, decimate)
  retro.py       Retro Diffusion client + scan fetch (stdlib urllib)
  seed.py        placeholder generators (tiles, pips, and a face per key)
  deck.py        deck resolution (config + artifacts paths)
  cli.py         thin `arcana` CLI
decks/
  configs/<name>/   palette.yaml + deck.yaml + generation.yaml + murals/
  artifacts/<name>/ generated tiles, seeds, candidates, renders (git-ignored)
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
uv run arcana majors vaporwave-rws       # the face cards (the 22 majors by default)
uv run pytest
```

Art for a face card takes **two commands and one decision**:

```bash
export RD_API_KEY=rdpk-...                              # your machine, never a config
uv run arcana rd vaporwave-rws --face major_00          # fetches the RWS scan, generates
#   ...look at the candidates, pick one...
uv run arcana import-mural vaporwave-rws \
    decks/artifacts/vaporwave-rws/rd/major_00_1.png --face major_00 --force
```

Choosing a candidate is the only step that needs a human, so it is the only one
left: `rd` fetches the source scan itself, and `import-mural` renders the
finished card (and its print copy) once it has written the layers.

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
  label; predates the generation path, so the art windows are bare).

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
- `arcana majors <deck>` renders the deck's **face cards** — the cards whose art
  is an authored image rather than an algorithmic pip lattice. Face cards are
  keyed by a free-form string, so the same path serves the 22 tarot majors
  (`major_00`…, the default), a traditional deck's courts
  (`court_cups_queen`), and a game's one-off specials (`wizard`); a deck
  declares its own set with `faces:` in `deck.yaml`. `--face KEY` renders one.

  A **mural** is the image laid on the field: per-bank ASCII layers under
  `decks/configs/<deck>/murals/`, in the same local index space as any tile, so
  a palette swap recolors them. Committed art always wins; a face without it
  falls back to the placeholder `arcana seed` writes, so **the deck always
  renders and art can land one card at a time**. `--strict` is the print gate —
  it fails, naming every face still on a placeholder. `--no-murals` renders the
  bare field for comparison.
- `arcana rd <deck> --face KEY` generates candidate art. With no `--init` it
  **fetches the face's public-domain RWS scan** (the Wikimedia Commons path is
  computable from the key), crops the card's own title band and paper margin
  off, and fits it to the visible safe area as the generation seed. Candidates
  land in `decks/artifacts/<deck>/rd/` for a human to pick from — nothing is
  imported automatically.

  The model is there for **semantic colour**, not pixelation: it knows a sky is
  a sky and should take the field bank, where a nearest-colour mapping only
  knows the sky is warm and drops it in the flesh ramp. Style, strength and
  candidate count live in `generation.yaml` as deck identity, alongside the
  per-face prompts. `RD_API_KEY` is read from the environment only.
- `arcana import-mural <deck> <png> --face KEY` maps pixel art into the glyph
  representation: it quantizes to the deck's 14 drawable colors (`--force` snaps
  off-palette pixels), seats the art inside the frame-safe rectangle by scaling
  (`--bleed` fills the whole window instead), writes the per-bank ASCII layers,
  and renders the finished card. It prints a colour histogram — the charter
  check — and warns when a bank went unused or when the margin carries the line
  work of a generator-drawn frame. `arcana export-mural` is the exact inverse,
  and the round trip is lossless.
- `arcana seed <deck>` (re)writes the placeholder tiles and a placeholder mural
  per face.
- `--scale N` is a NEAREST zoom applied only to the preview contact sheet so the
  tiny pixel-art cards are legible; individual PNGs are always native
  resolution. For print, set `print_scale` in `deck.yaml`: `majors` then also
  writes cards under `print/` at that integer multiple with the physical size in
  the PNG's DPI metadata (6× is 960×1656, ≈349 DPI at a true 2.75×4.75in card).
  Integer NEAREST is the only correct scaling — a fractional ratio breaks the
  pixel grid the deck is made of.

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
