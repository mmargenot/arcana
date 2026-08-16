# Reference images

`major_NN.png` are what `arcana rd` sends as `reference_images`. RD Pro takes up
to nine and is the only tier that accepts them at all — which is the reason the
deck is on Pro. They matter more than the prompt does: the service applies
`input_palette` *after* generating (hence `return_pre_palette`), so the palette
re-maps colour but can never make the model draw flat. Flatness has to be shown.

Committed under `configs/`, not `artifacts/`, because `arcana` does not
regenerate them — they are hand-supplied input, like `palette.yaml`.

## The nine, and why these

| | teaches |
|---|---|
| `major_17` Star, `major_10` Wheel, `major_19` Sun | big flat radial masses, bbox filling the frame |
| `major_13` Death, `major_15` Devil, `major_03` Empress | one bold silhouette, thick outline, no interior detail |
| `major_16` Tower, `major_18` Moon, `major_02` Priestess | violet doing architecture rather than accent |

Wiry cards — Justice, Hierophant, Magician — were deliberately left out. Thin
linear implements are low-mass and fragment easily, and they are the least
representative of the flat look.

## Derived from `image-1786898094229.png`

That sheet is the source; the crops are reproducible from it.

- It is a **3× nearest-neighbour upscale** (3000×3384 → native 1000×1128).
  Decimate with `[::3, ::3]` first. A reference left at 3× teaches the model that
  a pixel is a 3×3 block, which is the chunky look we are not going for.
- Native grid: **160×276 cards with 8px gaps** — the deck's own geometry. Columns
  start at 0/168/336/504/672/840, rows at 0/284/568/852, face `r * 6 + c`.
- Within a card, take `art_origin` + `field.insets` — the **112×208 safe area**,
  which drops the frame band the art window overlaps.
- Rows **176–186** are the plinth, identical in every card. Cut above it: it is
  engine-drawn furniture, and a reference containing it teaches the model to
  draw one.
- Then crop to the emblem's bounding box + 4px. Tight crops are the point —
  they teach an emblem that fills its frame, which is the same lesson
  `style_suffix` asks for in words.

The other file, `image-1786898062071.png`, is the Star card alone. It is a
**JPEG** despite the extension, at a non-integer ~6.6× scale, so its pixel grid
cannot be recovered — every column differs by compression noise. Unused: the
sheet carries the Star at a clean 3×.
