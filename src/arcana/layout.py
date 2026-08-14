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

# Pip tiles are 16px, scaled by an integer factor. Defaults; a deck may override
# them (see `pip:` in deck.yaml, threaded through `arrange`).
PIP_BASE = 16
PIP_GAP = 6           # min pixels between pips and from the inner border
PIP_MIN_SCALE = 1.2   # smallest pip factor (×16px); below this a layout is invalid
PIP_MAX_SCALE = 3.0   # largest
_FIT_FLOOR = 0.5      # scale search never probes below this

# Foldable ordinaries: split into parallel/nested copies when one won't fit — a
# horizontal line becomes two rows, a diagonal two parallel diagonals, a chevron
# two nested chevrons, a pall stacked Ys. axis: 'h' horizontal line, 'v' vertical
# line, 'd' ╲, 'a' ╱, 'chevron', 'pall'.
_LINEAR = {"fess": "h", "pale": "v", "bend": "d", "bend-sinister": "a",
           "chevron": "chevron", "pall": "pall"}

# Grid layouts whose column count is searched for the biggest pips (not authored
# as a generator, since the arrangement is the search — see `_candidates`).
_GRID = {"square"}

_REGISTRY: dict[str, LayoutFn] = {}


def register(name: str) -> Callable[[LayoutFn], LayoutFn]:
    def deco(fn: LayoutFn) -> LayoutFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def names() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LINEAR) | _GRID)


class InvalidPipLayout(ValueError):
    """A (layout, count) that cannot place its pips at the minimum size with the
    buffer inside the card. Message states the best achievable size and fixes."""

    def __init__(self, name: str, count: int, best_scale: float, gap: int,
                 min_scale: float, geo: Geometry):
        best = (f"{best_scale:.1f}x ({round(PIP_BASE * best_scale)}px)"
                if best_scale else "nothing")
        super().__init__(
            f"pip layout {name!r} can't place {count} pips at the minimum size in "
            f"the {geo.art_w}x{geo.art_h} art window — not even folded or as a "
            f"diamond. Best fit is {best} with a {gap}px gap, but min_scale is "
            f"{min_scale:.1f}x ({round(PIP_BASE * min_scale)}px). Fix: lower "
            f"pip.min_scale or pip.gap in deck.yaml, or reduce the count.")
        self.name, self.count = name, count


def _pip_inset(geo: Geometry) -> tuple[int, int]:
    """Inner boundary pips stay within — the same rectangle the field varies in
    (mirrors `field.insets`), so pips never run into the frame."""
    return geo.margin + (geo.corner - geo.margin), geo.margin


def _normalize(pts: list[NPoint]) -> list[NPoint]:
    m = max((max(abs(u), abs(v)) for u, v in pts), default=1.0)
    return pts if m <= 0 else [(u / m, v / m) for u, v in pts]


def _recenter(pts: list[NPoint]) -> list[NPoint]:
    """Centre a candidate's bounding box on the origin, then normalise so the
    limiting axis reaches ±1. Centring is what puts an arrangement in the MIDDLE
    of the card — a raw chevron sits entirely in the top half, so without this it
    renders tiny and high; normalising after lets the fit spread it to the frame."""
    if not pts:
        return pts
    xs = [u for u, _ in pts]
    ys = [v for _, v in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return _normalize([(u - cx, v - cy) for u, v in pts])


def _grid(count: int, cols: int) -> list[NPoint]:
    """Symmetric row-major grid; a short final row is centred on the axis."""
    cols = max(1, min(cols, count))
    rows = math.ceil(count / cols)
    ys = _lin(-1.0, 1.0, rows)
    pts: list[NPoint] = []
    placed = 0
    for r in range(rows):
        in_row = min(cols, count - placed)
        if in_row == cols:
            xs = _lin(-1.0, 1.0, cols)
        elif in_row == 1:
            xs = [0.0]
        else:
            span = (in_row - 1) / (cols - 1)
            xs = _lin(-span, span, in_row)
        pts += [(x, ys[r]) for x in xs]
        placed += in_row
        if placed >= count:
            break
    return pts


def _terminal(count: int, geo: Geometry):
    """Overflow arrangement for a shape that can't hold `count` pips in its own
    form: fold into a DIAMOND (a rhombus outline), never a rectangular grid.
    The outline — unlike a filled diamond, whose packing has an inherent tight
    pair at some counts — spaces its pips around the perimeter, so it reaches the
    minimum size for every rank 1..10 while staying bilaterally symmetric."""
    yield _recenter(_diamond(count, geo))


def _split(count: int, fold: int) -> list[int]:
    """Distribute `count` pips as evenly as possible across `fold` copies."""
    return [count // fold + (1 if i < count % fold else 0) for i in range(fold)]


def _chevronel(size: int, apex_v: float) -> list[NPoint]:
    """One chevron (inverted V): apex on the axis at height `apex_v`, arms at
    slope 1 stepping down-and-out at a CONSTANT pitch. The pitch is independent
    of `size`, so parallel copies keep their spacing instead of crushing it — the
    fix for the old nested-and-scaled fold."""
    if size <= 1:
        return [(0.0, apex_v)]
    pts: list[NPoint] = []
    if size % 2 == 1:
        pts.append((0.0, apex_v))
        arm = (size - 1) // 2
    else:
        arm = size // 2
    for j in range(1, arm + 1):
        d = j * 0.5
        pts += [(d, apex_v + d), (-d, apex_v + d)]
    return pts


def _pallel(size: int, apex_v: float) -> list[NPoint]:
    """One pall (Y): two upper arms rising out to the corners and a stem dropping
    on the axis from the junction at `apex_v`, constant pitch (see `_chevronel`)."""
    if size <= 1:
        return [(0.0, apex_v)]
    per = size // 3
    stem = size - 2 * per
    pts: list[NPoint] = []
    for j in range(1, per + 1):
        d = j * 0.5
        pts += [(d, apex_v - d), (-d, apex_v - d)]     # upper arms rise out
    for k in range(1, stem + 1):
        pts.append((0.0, apex_v + k * 0.5))            # stem drops on the axis
    return pts


def _fold(axis: str, count: int, fold: int) -> list[NPoint]:
    """`count` pips folded into `fold` PARALLEL copies of the ordinary `axis`.
    Every fold translates a same-size copy (constant internal pitch) and offsets
    it, so the pip+gap buffer holds within and between copies — more folds give
    each copy fewer, wider-spaced pips, which both fits the count and keeps the
    pips large."""
    fold = max(1, min(fold, count))
    if axis == "h":                                    # horizontal rows
        return _grid(count, math.ceil(count / fold))
    if axis == "v":                                    # vertical columns
        return _grid(count, fold)
    if axis in ("d", "a"):                             # parallel diagonals
        perp = _lin(-1.0, 1.0, fold)
        pts: list[NPoint] = []
        for o, sz in zip(perp, _split(count, fold)):
            for a in _lin(-1.0, 1.0, sz):
                pts.append((a + o, a - o) if axis == "d" else (a + o, -a + o))
        return _normalize(pts)
    # arm-ordinaries: parallel chevronels / palls, apexes spread over the top so
    # copies nest without overlapping. Constant pitch keeps each copy countable.
    make = _chevronel if axis == "chevron" else _pallel
    top = -1.0 if axis == "chevron" else -0.6
    apex = _lin(top, top + 0.9 * (fold - 1) / fold, fold) if fold > 1 else [top]
    pts = []
    for sz, av in zip(_split(count, fold), apex):
        pts += make(sz, av)
    return _normalize(pts)


def _candidates(name: str, count: int, geo: Geometry):
    """In-character centre-sets to try, all recentred on the card. `arrange` picks
    the one with the biggest pips: ordinaries offer their fold variants, the grid
    offers every column count (2 usually wins in a tall window). No rectangular
    fallback — a shape that can't hold the count in its own form is handled by
    `_terminal` (a diamond) in `arrange`."""
    if name in _LINEAR:
        # A fold only reads as the ordinary if its LARGEST copy still holds the
        # shape: an arm-ordinary (chevron/pall) needs ≥3 pips (apex + two arms),
        # a line ≥2. Without this, "biggest pip wins" over-folds a chevron into a
        # column of single dots — a vertical line, not a ∧.
        floor = min(3 if name in ("chevron", "pall") else 2, count)
        for f in range(1, count + 1):
            if max(_split(count, f)) >= floor:
                yield _recenter(_fold(_LINEAR[name], count, f))
    elif name in _GRID:
        for cols in range(1, count + 1):
            yield _recenter(_grid(count, cols))
    elif name in _REGISTRY:
        yield _recenter(_REGISTRY[name](count, geo))
    else:
        raise KeyError(f"unknown pip layout {name!r}; have {names()}")


def _min_sep(cent: list[NPoint], rx: float, ry: float) -> float:
    """Min pixel Chebyshev separation between centres scaled by (rx, ry)."""
    return min((max(abs(u1 - u2) * rx, abs(v1 - v2) * ry)
                for i, (u1, v1) in enumerate(cent) for (u2, v2) in cent[i + 1:]),
               default=float("inf"))


def _fit_scale(cent: list[NPoint], geo: Geometry, gap: int,
               min_scale: float, max_scale: float) -> tuple[float, list[Point] | None]:
    """Largest pip scale (≤ max) that places `cent` with the buffer inside the
    inner border (spread ≤ 1). Returns (scale, centres) or (best_scale, None)."""
    ix, iy = _pip_inset(geo)
    x0, y0 = geo.art_w / 2, geo.art_h / 2

    def place_at(scale: float) -> list[Point] | None:
        pip = PIP_BASE * scale
        rx, ry = x0 - ix - gap - pip / 2, y0 - iy - gap - pip / 2
        if rx <= 0 or ry <= 0:
            return None
        sep = _min_sep(cent, rx, ry)
        if sep == float("inf"):
            s = 0.0
        elif pip + gap > sep:
            return None
        else:
            s = (pip + gap) / sep
            if s > 1:
                return None
        return [(int(round(x0 + u * rx * s)), int(round(y0 + v * ry * s)))
                for u, v in cent]

    if place_at(max_scale) is not None:
        best = max_scale
    elif place_at(_FIT_FLOOR) is None:
        return _FIT_FLOOR, None                        # doesn't fit even tiny
    else:
        lo, hi = _FIT_FLOOR, max_scale
        for _ in range(24):
            mid = (lo + hi) / 2
            if place_at(mid) is not None:
                lo = mid
            else:
                hi = mid
        best = lo
    return best, (place_at(best) if best >= min_scale else None)


def arrange(name: str, count: int, geo: Geometry, *, gap: int = PIP_GAP,
            min_scale: float = PIP_MIN_SCALE,
            max_scale: float = PIP_MAX_SCALE) -> tuple[list[Point], float]:
    """Place `count` pips for `name`: return (centres_px, scale). Every candidate
    is centred on the card, and the pip size is continuous (nearest-scaled at
    render), chosen as the LARGEST that keeps every pip a `gap` from its
    neighbours and the inner border. A shape offers in-character variants (fold
    counts, grid columns) and the biggest-pip one wins; if none reaches
    `min_scale`, the pips fold into a DIAMOND (`_terminal`) rather than a grid.
    Raises `InvalidPipLayout` only if even the diamond can't be placed."""
    if count < 1:
        return [], 0.0

    def best_of(cands):
        bs, bp = 0.0, None
        for cent in cands:
            scale, pts = _fit_scale(cent, geo, gap, min_scale, max_scale)
            nonlocal reach
            reach = max(reach, scale)
            if pts is not None and scale > bs:
                bs, bp = scale, pts
        return bs, bp

    reach = 0.0
    scale, pts = best_of(_candidates(name, count, geo))   # in-character forms
    if pts is None:
        scale, pts = best_of(_terminal(count, geo))       # fold into a diamond
    if pts is not None:
        return pts, scale
    raise InvalidPipLayout(name, count, reach, gap, min_scale, geo)


def place(name: str, count: int, geo: Geometry) -> list[Point]:
    """Centres only, at the default pip config (back-compat / tests)."""
    return arrange(name, count, geo)[0]


def pip_config(deck_cfg: dict | None) -> dict:
    """Pip sizing knobs from a deck's `pip:` block, with engine defaults."""
    p = (deck_cfg or {}).get("pip", {}) or {}
    return {"gap": int(p.get("gap", PIP_GAP)),
            "min_scale": float(p.get("min_scale", PIP_MIN_SCALE)),
            "max_scale": float(p.get("max_scale", PIP_MAX_SCALE))}


def validate_pip_layouts(deck_cfg: dict, geo: Geometry) -> None:
    """Every rank's resolved layout must place its pips at ≥ min_scale with the
    buffer inside the card. Raises a single ValueError listing all offenders."""
    pip = pip_config(deck_cfg)
    spec = deck_cfg.get("pip_layouts", {})
    by_rank, default = spec.get("by_rank", {}), spec.get("default", "square")
    errs: list[str] = []
    for rank in range(1, 11):
        name = by_rank.get(rank) or by_rank.get(str(rank)) or default
        try:
            arrange(name, rank, geo, **pip)
        except (InvalidPipLayout, KeyError) as e:
            errs.append(str(e))
    if errs:
        raise ValueError("invalid pip_layouts:\n  " + "\n  ".join(errs))


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
# `square` is a grid whose COLUMN COUNT is searched, not fixed: `_candidates`
# offers every column count and `arrange` keeps the biggest-pip one. In a tall
# tarot window that is reliably two columns (measured), but the search keeps it
# correct for any deck geometry instead of hard-coding a rule of thumb.


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
