"""
An RWS scan -> a SEED for generation. Not the artwork.

This module prepares an init image; it does not produce a finished card. The
deck's face art comes from a generative model that takes this seed and returns
pixel art constrained to the deck's palette (`arcana.retro`). The seed's whole
job is to carry COMPOSITION -- who is standing where, holding what -- at the
art window's size and aspect. Judge it on whether the scene survives, not on
whether it looks like finished pixel art.

Two things still have to be right, and both are why this is a module rather
than one call to `Image.resize`:

    1. crop        the printed picture area only. A scan includes paper margin
                   and the card's own TITLE BAND; arcana renders its own title
                   from `arcana.data`, so importing RWS's would put two on the
                   card.
    2. downscale   without dissolving the line work. Pamela Colman Smith's ink
                   hatching is high-frequency LINE detail, and a mean-based
                   resize averages it against paper into mud -- which is what
                   makes a seed the model cannot read. Outline expansion plus
                   a median decimation keeps the figure legible at 144x224.

Palette is NOT applied here. Quantisation to the deck's 14 slots happens on
import (`tileio.quantize_rgb_global`), after generation. Doing it to a scan
directly does not work: mapped by RGB distance, the Fool's warm aged sky lands
in the flesh ramp and ~80% of the card collapses into one bank, leaving the
border bank with three pixels. Knowing a sky should be teal requires knowing it
is a sky, which is the model's job, not a distance metric's.

MEASURED, NOT ASSUMED. On the Wikimedia Commons 1909 Fool the "flat" yellow sky
carries a high-frequency sigma of ~4 grey levels, not the wild stipple a raw
plate scan would show. The bilateral(15,90,90)+median(9) an earlier note
prescribed would erode hatching to no purpose here, so `descreen` defaults to a
3px median and nothing else. Turn it up for a noisier scan; do not turn it up by
reflex.

WHY NO LIBRARY. PixelOE is the obvious dependency for outline-aware
pixelisation and it is the wrong trade: it requires torch, kornia and
opencv-python, gigabytes of transitive weight for an engine whose whole list is
numpy/pillow/pyyaml/scipy. Its palette features are redundant besides, since
quantisation here targets the deck's fixed 14 slots. The idea below is
PixelOE's contrast-aware outline expansion; the code is ours and uses
scipy.ndimage morphology already on hand.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as nd

# A printed card is ink or colour on paper; paper is neither dark nor saturated.
DARK = 120          # luma below this is ink
SAT = 45            # max-min channel spread above this is colour
RULE = 0.70         # fraction of a row/col that must be ink to be a frame rule


def _content_mask(rgb: np.ndarray) -> np.ndarray:
    lum = rgb.mean(2)
    sat = rgb.max(2) - rgb.min(2)
    return (lum < DARK) | (sat > SAT)


def find_picture(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """The printed picture area as (x0, y0, x1, y1), half-open.

    A tarot scan is paper margin, then a ruled frame, then the picture, then a
    TITLE BAND at the bottom carrying the card's name. The band must go: arcana
    renders its own title from `arcana.data`, and importing the printed one
    would put two titles on the card.

    Frame rules are found as rows/columns that are almost entirely ink. The
    band's top edge is the LAST such horizontal rule above the bottom frame --
    every RWS major shares this layout, so one rule works for all 22.
    """
    dark = rgb.mean(2) < DARK
    h, w = dark.shape
    rows, cols = dark.mean(1), dark.mean(0)

    def rules(profile: np.ndarray) -> list[int]:
        return [int(i) for i in np.nonzero(profile > RULE)[0]]

    vr, hr = rules(cols), rules(rows)
    if not vr or not hr:
        raise ValueError("no frame rules found; pass an explicit crop")

    def run(marks: list[int], seed: int) -> tuple[int, int]:
        """The contiguous rule BAND around `seed`. A printed rule is several
        pixels thick, so cropping one pixel past its first row leaves a black
        sliver down the edge of the art -- visible as a stray line once the
        card is composed."""
        s = e = seed
        m = set(marks)
        while s - 1 in m:
            s -= 1
        while e + 1 in m:
            e += 1
        return s, e

    x0 = run(vr, vr[0])[1] + 1
    x1 = run(vr, vr[-1])[0]
    y0 = run(hr, hr[0])[1] + 1
    y1 = run(hr, hr[-1])[0]

    # the title band's rule: the last near-solid horizontal rule that still
    # leaves most of the card above it. Below 55% we are looking at the band's
    # own lettering, not its edge.
    inner = [y for y in hr if y0 + 0.55 * (y1 - y0) < y < y1 - 2]
    if inner:
        y1 = run(hr, min(inner))[0]
    return x0, y0, x1, y1


def crop_to_aspect(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Centre-crop to the target aspect, keeping as much art as possible.

    The picture area of an RWS card is taller than arcana's art window (1:1.62
    against 1:1.56 for this deck), so something has to give. Cover-and-crop
    keeps the figure's proportions -- letterboxing instead would shrink the
    figure and hand the difference to blank field, which is worse on a card
    whose whole subject is one standing figure.
    """
    ih, iw, _ = rgb.shape
    want = w / h
    have = iw / ih
    if have > want:                       # too wide: trim left/right
        keep = int(round(ih * want))
        off = (iw - keep) // 2
        return rgb[:, off:off + keep]
    keep = int(round(iw / want))           # too tall: trim top/bottom
    off = (ih - keep) // 2
    return rgb[off:off + keep, :]


def descreen(rgb: np.ndarray, size: int = 3) -> np.ndarray:
    """Kill halftone dots without touching hatching. A small median is the
    right tool: it removes isolated dots but preserves an edge, where a
    gaussian would smear the line work this whole module exists to protect.
    `size <= 1` disables it."""
    if size <= 1:
        return rgb
    return np.stack([nd.median_filter(rgb[..., c], size=size) for c in range(3)], -1)


def outline_expand(rgb: np.ndarray, k: int, strength: float = 0.9) -> np.ndarray:
    """Thicken locally-extreme detail so it survives decimation.

    Decimation samples one value per k x k block. A one-pixel ink line covering
    an eighth of its block loses every vote and disappears -- which is how the
    sun's rays and the hatching die. So before decimating, push each pixel
    toward its local minimum where dark detail dominates the neighbourhood and
    toward its local maximum where light does, weighted by how one-sided the
    neighbourhood is. Lines thicken to roughly the block size and win their
    blocks; flat areas, where min and max agree, are left alone.
    """
    lum = rgb.mean(2)
    lo = nd.grey_erosion(lum, size=k)
    hi = nd.grey_dilation(lum, size=k)
    span = np.maximum(hi - lo, 1e-6)
    # 0 = pixel sits at the local floor, 1 = at the ceiling
    pos = np.clip((lum - lo) / span, 0, 1)
    # pull toward whichever extreme is nearer, scaled by local contrast
    weight = strength * np.clip(span / 128.0, 0, 1)
    target = np.where(pos < 0.5, lo, hi)
    shift = (target - lum) * weight
    out = rgb + shift[..., None]
    return np.clip(out, 0, 255)


def decimate(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Block-representative downscale to exactly (h, w).

    NEVER the mean -- averaging hatching against paper is precisely what
    produces mud. The median of a block keeps whichever of ink or ground
    actually dominates it, so a line that owns its block stays a line and one
    that does not disappears cleanly instead of greying everything.

    The source is first resized to an exact multiple of the target so every
    output pixel draws on the same number of input pixels; a fractional ratio
    (1144x1919 -> 144x224 is 7.5 x 8.1) otherwise makes some output pixels
    weigh more than others and shimmers along straight rules.
    """
    ih, iw, _ = rgb.shape
    k = max(1, min(ih // h, iw // w))
    im = Image.fromarray(rgb.astype(np.uint8)).resize((w * k, h * k), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    blocks = a.reshape(h, k, w, k, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, k * k, 3)
    return np.median(blocks, axis=2).astype(np.uint8)


def pixelate(src: str | Path, w: int, h: int, *,
             crop: tuple[int, int, int, int] | None = None,
             descreen_size: int = 3, strength: float = 0.9) -> np.ndarray:
    """A scan -> an (h, w, 3) RGB array on the art-window grid.

    `crop` overrides `find_picture` for a scan that is already trimmed, or one
    whose frame the rule detector cannot see."""
    rgb = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
    x0, y0, x1, y1 = crop if crop else find_picture(rgb)
    rgb = rgb[y0:y1, x0:x1]
    if rgb.size == 0:
        raise ValueError(f"empty crop {(x0, y0, x1, y1)}")
    rgb = crop_to_aspect(rgb, w, h)
    rgb = descreen(rgb, descreen_size)
    k = max(1, min(rgb.shape[0] // h, rgb.shape[1] // w))
    rgb = outline_expand(rgb, k, strength)
    return decimate(rgb, w, h)
