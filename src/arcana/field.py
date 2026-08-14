"""
Heraldic field designs for minor-arcana cards — the geometric background, chosen
independently of the pip arrangement.

A design is pure geometry in LOCAL TONE SPACE: it only decides which cells are
`ground` vs `device`, using the field-bank slots (`DARK`/`MID`/`LIGHT` = 3/4/5).
The actual hues come from the suit's `field` bank at render time, so a design
needs no colour authoring — `cups: barry` renders in cups' teal tones and matches
every other cups card by construction. Designs are selected by NAME (a plaintext
string in `deck.yaml`), never authored as tiles.

Three families, after heraldry:
- **divisions** — split the field into two tincture areas (per-pale, per-fess,
  per-bend[-sinister], per-chevron, per-saltire, quarterly)
- **ordinary bands** — one device band over a plain ground (chief, base, pale,
  fess, bend, chevron, cross, saltire, pile, bordure)
- **patterns** — an all-over repeat (barry, paly, bendy, chevronny, checky,
  lozengy)
plus `plain` (flat ground — the default).

**Invisible border.** The design only varies in an inner rectangle; a plain
`ground` margin rings it so the pattern never runs into the frame. `build`
fills the whole art window with `ground` and pastes the design inset by
`insets(geo, margin)` — larger on the sides, where the frame overlaps the art
window, so the visible ground margin is even on all four sides.
"""
from __future__ import annotations
from collections.abc import Callable

import numpy as np

from arcana.geometry import Geometry
from arcana.palette import MID, LIGHT

DesignFn = Callable[[int, int, int, int], np.ndarray]  # (h, w, ground, device)

_REGISTRY: dict[str, DesignFn] = {}


def register(name: str) -> Callable[[DesignFn], DesignFn]:
    def deco(fn: DesignFn) -> DesignFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def names() -> list[str]:
    return sorted(_REGISTRY)


def field_for_suit(deck_cfg: dict, suit: str, override: str | None = None) -> str:
    """Resolve the field design for a suit from a deck's `field_designs` block:
    an explicit `--field` override wins, else `by_suit`, else `default`
    (falling back to `plain`)."""
    if override:
        return override
    spec = deck_cfg.get("field_designs", {})
    return spec.get("by_suit", {}).get(suit) or spec.get("default", "plain")


def insets(geo: Geometry, margin: int = 8) -> tuple[int, int]:
    """(ix, iy): how far the design is inset from the art-window edges. The frame
    band overlaps the art window horizontally by (corner - card margin) but not
    vertically, so the horizontal inset is larger — the visible ground margin
    then comes out even on every side."""
    return margin + (geo.corner - geo.margin), margin


def build(name: str, geo: Geometry, ground: int = LIGHT, device: int = MID,
          margin: int = 8) -> np.ndarray:
    """A field as a local-index matrix shape (art_h, art_w), fully opaque: the
    named design in an inner rectangle, ringed by a plain `ground` margin."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown field design {name!r}; have {names()}")
    h, w = geo.art_h, geo.art_w
    out = np.full((h, w), ground, np.uint8)
    ix, iy = insets(geo, margin)
    ih, iw = h - 2 * iy, w - 2 * ix
    if ih > 0 and iw > 0:
        out[iy:iy + ih, ix:ix + iw] = _REGISTRY[name](ih, iw, ground, device)
    return out


def _uv(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised coordinates in [0, 1]: u across, v down."""
    rr, cc = np.mgrid[0:h, 0:w]
    return cc / max(w - 1, 1), rr / max(h - 1, 1)


def _paint(mask: np.ndarray, ground: int, device: int) -> np.ndarray:
    return np.where(mask, np.uint8(device), np.uint8(ground)).astype(np.uint8)


# ---------------------------------------------------------------- plain
@register("plain")
def _plain(h, w, ground, device):
    return np.full((h, w), ground, np.uint8)


# ---------------------------------------------------------------- divisions
@register("per-pale")
def _per_pale(h, w, ground, device):
    u, _ = _uv(h, w)
    return _paint(u < 0.5, ground, device)


@register("per-fess")
def _per_fess(h, w, ground, device):
    _, v = _uv(h, w)
    return _paint(v < 0.5, ground, device)


@register("per-bend")
def _per_bend(h, w, ground, device):
    u, v = _uv(h, w)
    return _paint(v > u, ground, device)                      # lower-left triangle


@register("per-bend-sinister")
def _per_bend_sinister(h, w, ground, device):
    u, v = _uv(h, w)
    return _paint(v > 1 - u, ground, device)


@register("per-chevron")
def _per_chevron(h, w, ground, device):
    u, v = _uv(h, w)
    boundary = 0.4 + 0.6 * np.abs(u - 0.5) / 0.5              # apex up, arms to base
    return _paint(v > boundary, ground, device)


@register("per-saltire")
def _per_saltire(h, w, ground, device):
    u, v = _uv(h, w)
    top_bottom = ((v < u) & (v < 1 - u)) | ((v > u) & (v > 1 - u))
    return _paint(top_bottom, ground, device)


@register("quarterly")
def _quarterly(h, w, ground, device):
    u, v = _uv(h, w)
    return _paint((u < 0.5) == (v < 0.5), ground, device)     # TL & BR


# ---------------------------------------------------------------- ordinary bands
@register("chief")
def _chief(h, w, ground, device):
    _, v = _uv(h, w)
    return _paint(v < 0.28, ground, device)


@register("base")
def _base(h, w, ground, device):
    _, v = _uv(h, w)
    return _paint(v > 0.72, ground, device)


@register("pale")
def _pale(h, w, ground, device):
    u, _ = _uv(h, w)
    return _paint(np.abs(u - 0.5) < 0.15, ground, device)


@register("fess")
def _fess(h, w, ground, device):
    _, v = _uv(h, w)
    return _paint(np.abs(v - 0.5) < 0.13, ground, device)


@register("bend")
def _bend(h, w, ground, device):
    u, v = _uv(h, w)
    return _paint(np.abs(v - u) < 0.16, ground, device)


@register("chevron")
def _chevron(h, w, ground, device):
    u, v = _uv(h, w)
    boundary = 0.4 + 0.6 * np.abs(u - 0.5) / 0.5
    return _paint(np.abs(v - boundary) < 0.13, ground, device)


@register("cross")
def _cross(h, w, ground, device):
    u, v = _uv(h, w)
    return _paint((np.abs(u - 0.5) < 0.15) | (np.abs(v - 0.5) < 0.13), ground, device)


@register("saltire")
def _saltire(h, w, ground, device):
    u, v = _uv(h, w)
    return _paint((np.abs(v - u) < 0.14) | (np.abs(v - (1 - u)) < 0.14), ground, device)


@register("pile")
def _pile(h, w, ground, device):
    u, v = _uv(h, w)
    return _paint(np.abs(u - 0.5) < 0.5 * (1 - v), ground, device)   # wedge, point down


@register("bordure")
def _bordure(h, w, ground, device):
    u, v = _uv(h, w)
    b = 0.12
    return _paint((u < b) | (u > 1 - b) | (v < b) | (v > 1 - b), ground, device)


# ---------------------------------------------------------------- patterns
@register("barry")
def _barry(h, w, ground, device, bands=6):
    rr = np.mgrid[0:h, 0:w][0]
    return _paint((rr * bands // h) % 2 == 0, ground, device)


@register("paly")
def _paly(h, w, ground, device, bands=6):
    cc = np.mgrid[0:h, 0:w][1]
    return _paint((cc * bands // w) % 2 == 0, ground, device)


@register("bendy")
def _bendy(h, w, ground, device, bands=6):
    u, v = _uv(h, w)
    return _paint(((u + v) * bands).astype(int) % 2 == 0, ground, device)


@register("chevronny")
def _chevronny(h, w, ground, device, bands=5):
    u, v = _uv(h, w)
    m = v + np.abs(u - 0.5)
    return _paint((m * bands).astype(int) % 2 == 0, ground, device)


@register("checky")
def _checky(h, w, ground, device, cols=6):
    rr, cc = np.mgrid[0:h, 0:w]
    rows = max(1, round(cols * h / w))
    return _paint(((cc * cols // w) + (rr * rows // h)) % 2 == 0, ground, device)


@register("lozengy")
def _lozengy(h, w, ground, device, n=5):
    u, v = _uv(h, w)
    a = ((u + v) * n).astype(int) % 2
    b = ((u - v + 1) * n).astype(int) % 2
    return _paint(a == b, ground, device)
