"""
Pip layout algorithms for minor-arcana cards.

Each algorithm maps a pip COUNT and the deck GEOMETRY to a list of `(cx, cy)`
pip centres in art-window space (0..art_w, 0..art_h). The card composer offsets
them by the art origin. Algorithms are registered by name; a card picks one per
rank (see `pip_layouts` in a deck's `deck.yaml`).

Odd/even is handled explicitly per family. The bilaterally-symmetric layouts
place a single pip ON the mirror axis for odd counts and mirror pairs for even
counts, so a card is never lopsided. `bend` is the exception: heraldry's bend is
a diagonal ordinary, deliberately not left–right symmetric.

Algorithm functions work in normalised coordinates `(u, v)` in [-1, 1] (u right,
v down); `place()` maps them into the padded art window and rounds to pixels.
"""
from __future__ import annotations
import math
from collections.abc import Callable

from arcana.geometry import Geometry

Point = tuple[int, int]
NPoint = tuple[float, float]
LayoutFn = Callable[[int, Geometry], list[NPoint]]

# Keep pip centres this far inside the art window so pips clear the frame band.
PAD = 20

_REGISTRY: dict[str, LayoutFn] = {}


def register(name: str) -> Callable[[LayoutFn], LayoutFn]:
    def deco(fn: LayoutFn) -> LayoutFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def names() -> list[str]:
    return sorted(_REGISTRY)


def place(name: str, count: int, geo: Geometry) -> list[Point]:
    """Pip centres in art-window pixels for `count` pips under `name`."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown pip layout {name!r}; have {names()}")
    if count < 1:
        return []
    x0, y0 = geo.art_w / 2, geo.art_h / 2
    hx, hy = x0 - PAD, y0 - PAD
    return [(int(round(x0 + u * hx)), int(round(y0 + v * hy)))
            for u, v in _REGISTRY[name](count, geo)]


def _lin(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [(a + b) / 2]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# ---------------------------------------------------------------- lines
@register("single")
def _single(n: int, geo: Geometry) -> list[NPoint]:
    # One centred pip; extras (if ever asked) stack vertically.
    return [(0.0, v) for v in _lin(-1.0, 1.0, n)] if n > 1 else [(0.0, 0.0)]


@register("pale")
def _pale(n: int, geo: Geometry) -> list[NPoint]:
    """Vertical band — all pips on the mirror axis."""
    return [(0.0, v) for v in _lin(-1.0, 1.0, n)]


@register("fess")
def _fess(n: int, geo: Geometry) -> list[NPoint]:
    """Horizontal band — one row, centred pip when odd."""
    return [(u, 0.0) for u in _lin(-1.0, 1.0, n)]


@register("bend")
def _bend(n: int, geo: Geometry) -> list[NPoint]:
    """Diagonal band. Deliberately NOT left–right symmetric (heraldic bend)."""
    return [(t, t) for t in _lin(-1.0, 1.0, n)]


# ---------------------------------------------------------------- grids
@register("square")
def _square(n: int, geo: Geometry) -> list[NPoint]:
    """Near-square grid; a short final row is centred so the card stays
    symmetric for odd counts."""
    if n == 1:
        return [(0.0, 0.0)]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    xs = _lin(-1.0, 1.0, cols)
    ys = _lin(-1.0, 1.0, rows)
    pts: list[NPoint] = []
    placed = 0
    for r in range(rows):
        in_row = min(cols, n - placed)
        if in_row == cols:
            row_xs = xs
        elif in_row == 1:
            row_xs = [0.0]
        else:
            span = (in_row - 1) / (cols - 1)      # centre the short row
            row_xs = _lin(-span, span, in_row)
        pts += [(x, ys[r]) for x in row_xs]
        placed += in_row
        if placed >= n:
            break
    return pts


@register("pile")
def _pile(n: int, geo: Geometry) -> list[NPoint]:
    """Point-down triangle (heraldic pile): widest row on top, narrowing down."""
    if n == 1:
        return [(0.0, 0.0)]
    rows = math.ceil((-1 + math.sqrt(1 + 8 * n)) / 2)   # rows·(rows+1)/2 >= n
    ys = _lin(-1.0, 1.0, rows)
    pts: list[NPoint] = []
    placed = 0
    for i in range(rows):
        width = rows - i                                # top row widest
        in_row = min(width, n - placed)
        if in_row <= 0:
            break
        if rows == 1 or in_row == 1:
            row_xs = [0.0]
        else:
            full = (width - 1) / (rows - 1)             # this row's extent
            span = full * (in_row - 1) / (width - 1) if width > 1 else 0.0
            row_xs = _lin(-span, span, in_row)
        pts += [(x, ys[i]) for x in row_xs]
        placed += in_row
        if placed >= n:
            break
    return pts


# ---------------------------------------------------------------- ordinaries
@register("chevron")
def _chevron(n: int, geo: Geometry) -> list[NPoint]:
    """Inverted V: apex up, pips split evenly between the two arms; the apex
    pip appears only for odd counts."""
    apex = (0.0, -1.0)
    end = (1.0, 1.0)                                    # right arm end
    pts: list[NPoint] = []
    if n % 2 == 1:
        pts.append(apex)
        k = (n - 1) // 2
    else:
        k = n // 2
    for j in range(1, k + 1):
        t = j / k
        u = apex[0] + (end[0] - apex[0]) * t
        v = apex[1] + (end[1] - apex[1]) * t
        pts += [(u, v), (-u, v)]
    return pts


@register("cross")
def _cross(n: int, geo: Geometry) -> list[NPoint]:
    """Greek cross (+): a central pip for odd counts, then alternating vertical
    and horizontal pairs marching outward."""
    if n == 1:
        return [(0.0, 0.0)]
    pts: list[NPoint] = []
    rem = n
    if n % 2 == 1:
        pts.append((0.0, 0.0))
        rem -= 1
    pairs = rem // 2
    for i in range(pairs):
        r = (i + 1) / pairs
        if i % 2 == 0:
            pts += [(0.0, -r), (0.0, r)]               # vertical arm
        else:
            pts += [(-r, 0.0), (r, 0.0)]               # horizontal arm
    return pts


@register("saltire")
def _saltire(n: int, geo: Geometry) -> list[NPoint]:
    """St Andrew's cross (X): a central pip for odd counts, then rings of four
    on the two diagonals; a leftover two sit as the top pair."""
    if n == 1:
        return [(0.0, 0.0)]
    pts: list[NPoint] = []
    rem = n
    if n % 2 == 1:
        pts.append((0.0, 0.0))
        rem -= 1
    rings = rem // 4
    extra = (rem % 4) // 2                              # 0 or 1 leftover pair
    slots = rings + (1 if extra else 0)
    for i in range(rings):
        r = (i + 1) / slots
        pts += [(-r, -r), (r, -r), (-r, r), (r, r)]
    if extra:
        pts += [(-1.0, -1.0), (1.0, -1.0)]             # top pair
    return pts


@register("diamond")
def _diamond(n: int, geo: Geometry) -> list[NPoint]:
    """Rhombus outline: top and bottom on the axis, remaining pips as mirror
    pairs along the edges; odd counts add a centre pip."""
    if n == 1:
        return [(0.0, 0.0)]
    pts: list[NPoint] = []
    if n % 2 == 1:
        pts += [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
        k = (n - 3) // 2
    else:
        pts += [(0.0, -1.0), (0.0, 1.0)]
        k = (n - 2) // 2
    for j in range(1, k + 1):
        s = j / (k + 1)                                # along T->R->B
        if s <= 0.5:
            t = s / 0.5
            u, v = t, -1.0 + t                         # T(0,-1) -> R(1,0)
        else:
            t = (s - 0.5) / 0.5
            u, v = 1.0 - t, t                          # R(1,0) -> B(0,1)
        pts += [(u, v), (-u, v)]
    return pts


@register("lozenge")
def _lozenge(n: int, geo: Geometry) -> list[NPoint]:
    """Filled diamond: concentric diamond rings from the centre outward,
    each ring built axis-first then in mirror pairs."""
    if n == 1:
        return [(0.0, 0.0)]
    jmax = 1
    while 1 + 2 * jmax * (jmax + 1) < n:               # capacity 1 + Σ 4j
        jmax += 1
    pts: list[NPoint] = [(0.0, 0.0)]
    remaining = n - 1
    for j in range(1, jmax + 1):
        take = min(4 * j, remaining)
        pts += _diamond_ring(j, jmax, take)
        remaining -= take
        if remaining <= 0:
            break
    return pts


@register("pall")
def _pall(n: int, geo: Geometry) -> list[NPoint]:
    """Y arrangement (heraldic pall): two mirrored upper arms rising to the top
    corners and one lower stem on the axis. Count is split across the three
    arms; symmetric about the vertical axis."""
    if n == 1:
        return [(0.0, 0.0)]
    per_arm = n // 3
    stem = n - 2 * per_arm
    pts: list[NPoint] = []
    for j in range(1, per_arm + 1):
        t = j / per_arm
        pts += [(t, -t), (-t, -t)]                     # upper-right, upper-left
    for k in range(1, stem + 1):
        pts.append((0.0, k / stem))                    # lower stem, on the axis
    return pts


@register("seme")
def _seme(n: int, geo: Geometry) -> list[NPoint]:
    """Strewn: pips spread across the window in centred staggered rows (widths
    differ by a pip so rows nestle). Bilaterally symmetric and deterministic —
    no RNG, so it renders the same every time."""
    if n == 1:
        return [(0.0, 0.0)]
    rows = min(n, 3 if n <= 6 else 4)
    ys = _lin(-0.85, 0.85, rows)
    widths: list[int] = []
    remaining = n
    for i in range(rows):
        w = round(remaining / (rows - i))
        widths.append(w)
        remaining -= w
    widest = max(widths)
    pts: list[NPoint] = []
    for i, w in enumerate(widths):
        if w <= 0:
            continue
        if w == 1:
            xs = [0.0]
        else:
            span = 0.85 * (w - 1) / (widest - 1) if widest > 1 else 0.0
            xs = _lin(-span, span, w)
        pts += [(x, ys[i]) for x in xs]
    return pts


def _diamond_ring(j: int, jmax: int, count: int) -> list[NPoint]:
    """`count` symmetric points on diamond ring `j` (radius j/jmax)."""
    rj = j / jmax
    axis = [(0.0, -rj), (0.0, rj)]                      # top, bottom
    pairs: list[tuple[NPoint, NPoint]] = []
    for a in range(1, j + 1):
        u, v = a / jmax, (j - a) / jmax
        pairs.append(((u, -v), (-u, -v)))              # upper edge
        if v != 0:                                     # avoid duplicating L/R
            pairs.append(((u, v), (-u, v)))            # lower edge
    out: list[NPoint] = []
    if count % 2 == 1:
        out.append(axis[0])
    elif count >= 2:
        out += axis
    for pair in pairs:
        if len(out) >= count:
            break
        out += pair
    return out[:count]
