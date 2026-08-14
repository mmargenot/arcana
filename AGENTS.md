# AGENTS.md

Guidance and design rationale for anyone — human or agent — working on
**arcana**, a pixel-art card generator. This is the single reference for the
project: how to run it, how the engine is shaped, *why* it is shaped that way,
what was tried and rejected, and the invariants that must not regress.

The first deck is `vaporwave-rws`: a vaporwave palette over public-domain
Rider-Waite-Smith composition.

---

## Working here

```bash
uv sync
uv run arcana generate vaporwave-rws     # -> decks/artifacts/vaporwave-rws/
uv run arcana seed vaporwave-rws         # (re)write placeholder tiles only
uv run pytest                            # 16 regression tests
```

### Layout

```
src/arcana/
  palette.py   Bank, Palette, local/global index space
  geometry.py  Geometry + tiling validation
  elements.py  Element, indexed-PNG loader, audit
  tileio.py    RGB import, ASCII format, format dispatch
  compose.py   border assembly + structural checks
  seed.py      placeholder-tile generators (procedural + ASCII pips)
  deck.py      deck resolution (config + artifacts paths)
  cli.py       thin `arcana` CLI (entry point: arcana.cli:main)
decks/
  configs/<name>/   palette.yaml + deck.yaml   (committed)
  artifacts/<name>/ generated tiles + renders  (git-ignored)
tests/
  test_deck.py      regression tests, tiles seeded into tmp
```

The package `__init__.py` is intentionally minimal — import from submodules
directly (`from arcana.palette import Palette`), not from the package root.

- Use **uv** for everything (`uv add`, `uv run`); Python 3.12; hatchling build.
- The engine in `src/arcana/` is **deck-agnostic** — it must never hard-code a
  particular deck's colors, art, or names.
- **All intra-package imports are absolute** (`from arcana.x import ...`), never
  relative.
- Per-deck **config** lives under `decks/configs/<name>/` and is committed.
  Generated **tiles and renders** live under `decks/artifacts/<name>/` and are
  git-ignored — never commit them. Placeholder tile art lives in code
  (`seed.py`), not as committed binaries.

## Invariants — do not break

Each of these is guarded by a test in `tests/test_deck.py`; every one hides a
bug that is invisible at 1× and glaring in print.

- Composition emits **palette indices, never RGB**. `Palette` is the only RGB
  boundary (colors resolve once, at export).
- **Index 0 is transparent** everywhere.
- All intra-package imports are **absolute**.
- Never `Path.with_suffix` on an element path — stems are dotted
  (`corner.border`) and `with_suffix` eats the bank name.
- Squared RGB distances need **int32**; int16 overflows at 3×255².
- Every hex value in YAML must be **quoted** — unquoted `#RRGGBB` is a comment
  and parses as null.
- Every bank sits on the **L = 30 / 50 / 74** value rungs, ±6.
- **Transpose, never flip** when turning a top edge into a side edge.
- Lay the **backing strip before ornament**; place **medallions after
  mirroring** (a mirrored cup is a cup on its side).
- Suit motifs should be **self-symmetric** — odd pip counts straddle the mirror
  axis, so an asymmetric motif makes half the pip cards asymmetric.
- The **field varies only inside an invisible border** — a plain-`ground` margin
  rings every field design so the pattern never runs into the frame.

---

## Architecture

### Two index spaces

Author in **local space (0–5)**: `transparent, line, paper, dark, mid, light`.
An element never knows its colors. `palette.bind(art, "motif")` maps 3/4/5 onto
a bank, giving **global space (0–14)**:

```
0 transparent | 1 line  2 paper | 3-5 border | 6-8 field | 9-11 motif | 12-14 figure
```

A composed card is one `uint8` matrix. Four suits render from it by swapping the
lookup table, never the pixels — the load-bearing property, tested by
`test_index_matrix_is_suit_invariant`. The schema (which bank at which index) is
fixed; only hex values swap, so a stored matrix survives a palette change.

**Why indices, not RGB:** the palette was revised three times mid-design;
`assert m.max() < len(PALETTE)` catches off-palette color at introduction; suit
recolor becomes a LUT swap with geometry guaranteed identical; alpha comes free.

### Banks, after the NES PPU

Two universal colors + four 3-color banks = **14 drawable**. Rejected a flat
17-slot palette (it kept growing, and "75% shared / 25% per suit" slices along
the wrong axis).

| bank | carries | varies by |
|---|---|---|
| `border` | frame, corner, edge, **labels** | nothing |
| `field` | card background | suit |
| `motif` | pips, ornament | suit |
| `figure` | skin (major figure image) | nothing |

Labels bind to `border` (not `figure`): a title matches the frame and is uniform
across a deck, per the maintainer's call. The `figure` bank is reserved for the
major-arcana figure image, still unexercised.

A card may use all four — same as an NES background where each tile picks a
palette but the screen uses every one. Multi-layer elements get >3 colors by
binding layers to different banks.

### Value rungs L = 30 / 50 / 74

Every bank sits on the same three HLS-lightness rungs, ±6, so suits differ only
in *hue*. This is not cosmetic: tarot is read in spreads, so four suits sit on
the table at once and a value mismatch reads as four decks shuffled together.
`Palette.validate` enforces it — it caught Swords sitting 14 points light at the
mid rung, and a skin-light color at L=67 instead of 74.

### Geometry — every number is a multiple of 4

| | px | why |
|---|---|---|
| card | 160 × 276 | 1:1.725 vs true tarot 1:1.727 |
| art window | 144 × 224 | 1:1.556; measured RWS scans are 1:1.561 |
| margin / numeral band / title band | 8 / 16 / 36 | |
| corner tile | 16 × 16 | also the border band thickness |
| edge tile | 16 × 8 | authored horizontal, transposed for sides |
| medallion slot | 16 h / 20 v | absorbs the tiling remainder |
| cartouche | 24 × 24 | bulges 8px past the band, deliberately |
| pip | 16 × 16 | must fit the band to double as a medallion |

**The medallion exists partly for arithmetic.** On a 160×276 card, *no* corner
size lets 8px edge tiles divide both runs evenly. The medallion slot absorbs the
remainder (`MED_H=16` → 7 tiles/side, `MED_V=20` → 14 tiles/side).
`Geometry.validate()` enforces this and suggests a fix.

### Tiles, not drawings

Author one corner + one edge; mirror into all four sides. ~2,176 authored pixels
cover a 44,160-pixel card. Symmetry is structural, not maintained by hand. Three
assembly rules, each guarding an invisible-at-1× bug:

- **Transpose, never flip.** An extra `[:, ::-1]` moves the outer rule to the
  inner side of the band, losing it for the whole length.
- **Back before ornament.** A plain profile strip across the full run first, or
  the medallion slot leaves a hole in every edge.
- **Medallions last.** Place after mirroring.

### The cartouche

The medallion *is* the suit pip, so the border identifies the suit even when a
card is half-covered in a spread. It sits in an **opaque** 24px roundel: a bare
pip let the frame's inner rule show through and bisect thin motifs (wand shaft,
sword blade). Opacity is the fix; the ring is what makes it pop. Sized at 24
(tested against 20, which clipped pip corners and doubled the pentacle's ring).
Asserted at load via `opaque: true`.

**Configurable** (`compose.build_medallion`, `border.medallion` in deck.yaml,
`--medallion`/`--medallion-scale`). Once titles carry a card's identity the full
medallion competes with them, so it's tunable:
- `style: suit | lozenge | none` — the pip cartouche, an abstract suit-coloured
  diamond (`seed.lozenge`, bound to `motif`), or nothing.
- `scale` — a size multiplier via the NN `_scale_tile`; the deck ships `0.5`.
- `none` — `build_border` continues the edge dentils through the medallion slot
  (horizontal fills exactly; the vertical run's non-tile-divisible remainder —
  the reason the slot exists — leaves a ~4px centre gap, rule still continuous).

The `medallion.horizontal/vertical` slot stays in `geometry` regardless — it's
load-bearing for edge-tile divisibility. Defaults (`suit`, `scale 1.0`)
reproduce the original medallion byte-for-byte.

### Assets: YAML holds structure, tiles hold pixels

`deck.yaml` holds sizes, roles, bank bindings, the pip lattice. Pixels do not
belong in a config file. Three interchangeable formats via `tileio.read_any`:

1. **Indexed PNG** — Aseprite, Pixelorama (indexed mode), LibreSprite.
2. **RGB PNG** — any editor; snapped to the authoring palette on import,
   off-palette pixels rejected (anti-aliasing is the usual culprit).
3. **ASCII `.txt`** — a tile is integers 0–5, so it is text; diffs usefully in
   git and needs no image library.

The authoring swatches are **grey on purpose** — local space means drawing value
structure; a pink authoring palette would have you composing for pink.

### Pip layouts (minor arcana)

`arcana.layout` is a registry of arrangement algorithms; each maps a pip
**count** and the geometry to a list of pip centres in art-window space.
`compose.build_pip_card` fills the field, places the pips via a chosen
algorithm, and composites the border with the suit medallion — one
suit-invariant index matrix, coloured by the suit LUT at render time (same
property as the border).

Selection is **per rank**, because arrangement follows the count, not the suit.
`deck.yaml`'s `pip_layouts` maps each rank (Ace–Ten) to an algorithm, with a
`default`; `arcana cards --layout NAME` overrides every rank at once to compare
algorithms.

Algorithms, grouped: lines (`pale` │, `fess` ─, `bend` ╲), grids (`square`,
`pile` ▽), ordinaries (`chevron` ∧, `cross` +, `saltire` ✕, `diamond` ◇ outline,
`lozenge` ◆ filled, `pall` Y), the strewn `seme`, plus `single`.

**Odd/even is explicit.** Every layout except `bend` is bilaterally symmetric
about the vertical axis: odd counts put one pip *on* the axis, even counts use
mirror pairs, so a card is never lopsided (a regression test asserts this for
all layouts × counts 1–10). `bend` is the deliberate exception — heraldry's bend
is a diagonal ordinary, so its x/y correlation is the point and reflection
changes the pip set.

**Sizing is auto-fit, centred, and every pip is countable.** `layout.arrange`
returns both the centres *and* a pip scale. Every candidate arrangement is first
**recentred** — its bounding box is moved to the card centre — because a raw
chevron sits entirely in the top half and would otherwise render tiny and high.
Then the scale is the LARGEST (continuous, `min_scale`..`max_scale`) that keeps
every pip a `gap` from its neighbours AND a `gap` inside the invisible inner
border. Size is therefore a function of the distribution *and* the rank — three
pips `in pale` come out bigger than three `in chevron`. The buffer is the
load-bearing property: pips never touch or overlap, so a reader can always count
the rank. The knobs (`gap`, `min_scale`, `max_scale`) live in `deck.yaml`'s
`pip:` block ("the tank"); engine defaults are `gap=6`, `min_scale=1.2` (≈19px —
1× reads too small), `max_scale=3`.

**Every layout works at every rank, in character.** A shape offers a *family* of
in-character variants and `arrange` keeps the biggest-pip one:

- **Ordinaries fold.** A line (`fess`/`pale`/`bend`) or arm ordinary
  (`chevron`/`pall`) that won't fit as a single copy splits into F *parallel*
  copies, each holding fewer (wider-spaced) pips. Copies are TRANSLATED, never
  scaled — scaling a copy inward crushes its own spacing, the bug that once made
  a double-chevron worse than a single. Folding both keeps the count fitting
  *and* keeps pips large (fewer pips per copy ⇒ more room each).
- **Grids search their column count.** `square` tries every column count and
  keeps the largest-pip one — in a tall tarot window that is reliably two
  columns, but the search stays correct for any deck geometry instead of a
  hard-coded rule (a `sqrt(n)` guess picked too many columns and shrank the
  upper ranks badly).
- **Everything else folds into a diamond.** A genuinely-2D shape that can't hold
  a high count in its own form (a `cross` of 8+ is wider than the window allows
  at a countable size) folds into a **diamond** — a rhombus *outline*, which
  reaches the minimum size for every rank, never a rectangular grid. This is the
  one shape-changing step, and it is deliberate ("a cross folds into a diamond").

`layout.arrange` raises `InvalidPipLayout` only when even the diamond can't be
placed — e.g. an absurd `pip.min_scale` — and the error names the best achievable
size and the fixes. `validate_pip_layouts` (called at `deck.load_deck`) checks
the whole shipped mapping up front.

### Field designs (minor arcana)

The field (card background) is a SECOND axis, independent of the pips.
`arcana.field` is a registry of heraldic designs; each maps the geometry to a
`field`-bank matrix for the art window. `compose.build_pip_card` fills the field
with a chosen design instead of a flat tone, then places pips and border as
before — still one suit-invariant matrix.

A design is **geometry in local tone-space** — it only decides which cells are
`ground` vs `device` (`LIGHT`/`MID`/`DARK` slots); the hues come from the suit's
`field` bank at render, so `cups: barry` renders in cups' teal tones with **no
per-design colour authoring**. Designs are chosen by **name** (a string in
`deck.yaml`), never authored as tiles — that's the point: naming a field is
strictly easier than drawing a pip. Two tone-roles with defaults (`ground=light`,
`device=mid`) keep pips readable; changing a suit's field colours stays a
`palette.yaml` edit.

Selection is **per suit** (the suit's identity; the rank drives the pips).
`deck.yaml`'s `field_designs` maps each suit to a design with a `default`;
`arcana cards --field NAME` overrides every suit at once. Three families, after
heraldry: **divisions** (`per-pale`, `per-fess`, `per-bend`[`-sinister`],
`per-chevron`, `per-saltire`, `quarterly`), **ordinary bands** (`chief`, `base`,
`pale`, `fess`, `bend`, `chevron`, `cross`, `saltire`, `pile`, `bordure`), and
**patterns** (`barry`, `paly`, `bendy`, `chevronny`, `checky`, `lozengy`), plus
`plain`.

**Invisible border.** A design varies only in an inner rectangle; a plain
`ground` margin rings it (inset larger on the sides, where the frame overlaps the
art window) so the pattern never runs into the frame. A regression test asserts
the outer ring is uniform ground for every design.

### Labels (numerals & titles)

A card names itself with a **bitmap font**, the same "authored art, chosen by
string" model as pips and fields — letterforms are drawn once; a label is picked
by plaintext, never redrawn per card.

- **Font** (`arcana.text`) — a fixed 6×10 cell, 7px advance, one ink slot on a
  transparent field. Metrics are named constants; the placeholder glyphs live in
  code (`seed.placeholder_font`, like the pips), and a deck overrides them by
  dropping tiles in its `assets/font/` (same first-file-wins story as tiles).
  `render_line` → `fit_line` (**one line, always**: condense the advance, then a
  last-resort NN squeeze) → `render_band` (centre in a band). No mirroring —
  text is not bilaterally symmetric.
- **Content** (`arcana.data`) — canonical RWS major names, rank words, and the
  Roman/Arabic helpers, as engine defaults a deck overrides in a `labels:` block
  (`numeral_style`, `split`, `suit_titles`, `major_titles`).
- **Composition** (`arcana.compose`) — a card is a stack of composable pieces:
  a **content** object (`build_development` for a minor's field+pips, or
  `build_mural` for a major's scene), the **border**, and an optional **label**
  (`build_label`), combined by one `assemble` compositor. `build_label` inks the
  glyphs in the **`border` bank** (uniform per deck, matches the frame). Labels
  **hug the art window** (`band_rects`) — a `CELL_H` strip on the art-side of
  each band — to clear the outer rule and edge medallions.
- **Z-order** — a minor's title paints UNDER the border (between pips and frame,
  so the rule ring is never clipped); a major's title floats OVER the mural
  (`label_over_border`). By default everything is one combined title in the
  bottom band, which never fights the top medallion; `split: true` is the
  two-band look (numeral top, name bottom) and is where the cramped top band
  gets tuned on renders.
- **Suit-invariance boundary** — a rank numeral is identical across suits (LUT
  swap only), but a suit-name title deliberately differs; both are asserted.
  `build_pip_card` with no `font` is byte-identical to before, so the base
  suit-invariance guarantee is untouched.

---

## Design rationale & the experiments behind it

Real measurements from the design work that produced this engine. They explain
constraints that otherwise look arbitrary.

### Source material & licensing

RWS (1909) is public domain: **US since 1966**, **UK after 31 Dec 2012**. Work
from 1909–1911 scans, **not** a modern colorized edition — the 1971 US Games
recolor is a separate copyrighted variant. Licensing contemporary art was
considered and rejected: it costs money and negotiation to reach a place the
public domain gives free. (US Games has historically sent takedowns over
RWS-derived decks; most attorneys think a judge would side with the public
domain, and it only matters at sales volume.)

### The source scans are stipple, not flat color

Measured on four RWS scans (Fool, Hanged Man, World, Judgement): the line plate
is warm brown `#473F2D` (not black), paper is warm cream `#EDE6D8`, and a "flat"
yellow sky spans luminance 39–217 within one apparently-solid color. **Naive
downscaling averages the stipple back toward paper and produces mud** — which
killed both the "just pixelate the scans" plan and the low-strength img2img plan
(low strength preserves exactly the stipple color that fights a stark palette).

### Detail collapse — segment first, quantize second

| method | result |
|---|---|
| Lanczos → palettize | muddy, 8/12 colors |
| block-mode + relining | rejected — detail not collapsed |
| mean-shift + morphological close | the figure dissolved (RWS has no closed silhouette) |
| **SLIC superpixels** | **worked** — 479,210 px → ~150 regions, shapes survive |

SLIC is k-means in 5D (RGB + xy); the spatial term keeps clusters contiguous, so
it finds *regions* rather than scattered color classes. Params: `n_segments=220,
compactness=9, sigma=2` after `bilateralFilter(15,90,90)` + `medianBlur(9)`.
**Resolution finding:** at 96×150 SLIC loses the dog and sun rays; **144×224 is
where this source starts working.**

### Color assignment is semantic

Mapping the palette by value-rank + hue-temperature went **entirely pink** (the
Fool's yellow sky reads warm → maps to the pink ramp) and sliced the flat sky
into value tiers that shouldn't exist. Lesson: SLIC gives shape for free, but
knowing a sky should be teal requires knowing what a sky *is*. Let the SLIC
output be a near-monochrome value study and let a generative model supply color.

### Generation plan (not yet implemented)

```
RWS scan
  -> bilateral + median (kill stipple)
  -> SLIC ~220 segments, flatten to region medians
  -> downscale to 144x224
  -> generative model, input_image at strength ~0.6, force_palette,
     bypass_prompt_expansion
```

The collapse carries composition; the model supplies semantic color and pixel
rendering. A test generation without a composition constraint produced a forest
scene with a deer instead of the intended figure — **prompt expansion** is the
consistency killer for a 78-card deck, so `bypass_prompt_expansion` is
non-negotiable. Sequence the 22 majors first (Fool / Tower / Moon cover figure,
architecture, landscape); Marseille-style pip minors collapse 40 cards into a
lattice + 4 sprites, leaving ~38 pieces of real art instead of 78.

---

## Bug catalog — each now has a regression test

| bug | symptom | why it was invisible |
|---|---|---|
| **Side-edge flip** | `f.T[:, ::-1]` put the outer rule on the inner side — both side edges lost their rule for the whole length | reads as a stray inset line |
| **Medallion hole** | swapping the medallion for a bare pip removed the frame rules across its span | 16px gap mid-edge |
| **Frame fragmentation** | the two above gave 120 disconnected components; correct is one ~4,244px ring | invisible at 1×, glaring in print |
| **Wrong LUT** | rendered four suits identically — built four palettes, applied the base one | *proved* the index matrix is suit-invariant |
| **int16 overflow** | squared RGB distance maxes at 195,075; int16 caps at 32,767 → NaN under sqrt | silent wraparound |
| **`Path.with_suffix`** | on dotted stems it eats the bank name → `corner.txt` | file-not-found, far from the cause |
| **Unquoted YAML hex** | `#RRGGBB` is a comment; parses as null | no error until render |
| **Off-rung Swords** | +14 L at the mid rung | only visible with cards side by side |
| **Label clips the frame** | a title bleeding into the frame rule breaks the ring | minor labels paint UNDER the border, so the rule wins — `test_frame_not_clipped_by_labels` |
| **Anti-aliased glyph** | a soft glyph edge lands off the authoring palette | glyphs are single-slot local ink, asserted `⊆ {0, INK}` |
| **Title wraps/overflows** | a long major spills the band or wraps to two lines | `fit_line` guarantees one line ≤ width for every deck label |

---

## Roadmap

The **border pipeline**, the **minor-arcana pip pipeline**, and the **labeling
pipeline** are complete and tested: two independent card axes — a heraldic
**field design** (per suit, `arcana.field`) and a **pip arrangement** (per rank,
`arcana.layout`, 13 algorithms) — plus a **bitmap-font label** (`arcana.text` +
`arcana.data`), composited with the border via `compose.build_pip_card` /
`build_major_card` and `arcana cards` / `arcana majors`. The **`figure` bank is
the only unexercised bank**. Still open:

1. **Major-arcana image incorporation** — feed SLIC-collapsed RWS scans to a
   generative model (see the generation plan above) and composite the result
   into the mural (`compose.build_mural` leaves the seam). Lights up the
   `figure` bank.
2. **Court cards** — the 16 figures (page/knight/queen/king), likely sharing the
   major-arcana image path.

## Open questions

- Does pip-on-border-and-in-field read as too busy? At 10 pips it holds; a
  10-pip card puts 14 on screen. Fallbacks: medallions on vertical edges only,
  or a simplified outline pip for the border.
- The pentacle is already a roundel, so mounting it in a cartouche is redundant;
  the clean fix is drawing the star directly onto the cartouche interior for
  that one suit.
- `force_palette` is still untested against a real generation.
- The placeholder swords pip nearly vanishes at 16×16 — a drawing problem, not a
  system one, but the busy-ness judgment is provisional until real pips exist.
