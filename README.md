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
  seed.py        placeholder-tile generators (procedural + ASCII pips)
  deck.py        deck resolution (config + artifacts paths)
  cli.py         thin `arcana` CLI
decks/
  configs/<name>/   palette.yaml + deck.yaml   (committed)
  artifacts/<name>/ generated tiles + renders  (git-ignored)
tests/
  test_deck.py   regression tests, tiles seeded into tmp
```

Config and code are committed; **tiles and renders are not**. A deck's config
lives under `decks/configs/<name>/`; running the engine regenerates its
placeholder tiles and PNGs under `decks/artifacts/<name>/`, which is git-ignored.

## Quickstart

```bash
uv sync
uv run arcana generate vaporwave-rws     # -> decks/artifacts/vaporwave-rws/
uv run pytest
```

`arcana generate <deck>` seeds placeholder tiles (if missing) and renders the
per-suit borders. `arcana seed <deck>` just (re)writes the placeholder tiles.
`--scale N` on `generate` is a NEAREST zoom applied only to the preview contact
sheet (`all_suits.png`) so the tiny pixel-art cards are legible; the per-suit
PNGs are always native resolution.

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
