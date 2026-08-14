"""
Palette as banks, after the NES PPU.

LOCAL INDEX SPACE (0-5) is what you draw in. An element never knows its colors:

    0 transparent   1 line   2 paper   3 dark   4 mid   5 light

GLOBAL INDEX SPACE (0-14) is what a composed card holds. Binding a local matrix
to a bank maps 3/4/5 onto that bank's three slots.

    0 transparent
    1 line          2 paper                     (universal)
    3 4 5   border      6  7  8   field
    9 10 11 motif      12 13 14   figure

The schema — which bank sits at which index — is fixed. Only the hex values
swap. That is what makes a stored index matrix survive a palette change: the
same array renders four suits by swapping the lookup table, never the pixels.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import colorsys
import numpy as np
import yaml

# local index space
T, LINE, PAPER, DARK, MID, LIGHT = range(6)
LOCAL_SLOTS = ("transparent", "line", "paper", "dark", "mid", "light")
MAX_LOCAL = LIGHT

BANKS = ("border", "field", "motif", "figure")


def lum(hex_color: str) -> float:
    """HLS lightness, 0-100. The value rung a color sits on."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[1] * 100


@dataclass(frozen=True, slots=True)
class Bank:
    dark: str
    mid: str
    light: str

    def __iter__(self):
        yield from (self.dark, self.mid, self.light)

    def check(self, label: str, rungs: dict, tol: float) -> list[str]:
        return [f"{label}.{k} {c} at L={lum(c):.0f}, rung {rungs[k]}"
                for k, c in zip(("dark", "mid", "light"), self)
                if abs(lum(c) - rungs[k]) > tol]


@dataclass(frozen=True, slots=True)
class Palette:
    name: str
    line: str
    paper: str
    banks: dict[str, Bank]
    suits: dict[str, dict[str, Bank]]
    rungs: dict[str, float]
    tolerance: float

    # ------------------------------------------------------------ indices
    @property
    def colors(self) -> tuple[str | None, ...]:
        out: list[str | None] = [None, self.line, self.paper]
        for b in BANKS:
            out.extend(self.banks[b])
        return tuple(out)

    def __getitem__(self, i: int) -> str | None:
        return self.colors[i]

    def bind(self, art: np.ndarray, bank: str) -> np.ndarray:
        """Local index matrix -> global index matrix."""
        hi = int(art.max(initial=0))
        if hi > MAX_LOCAL:
            raise ValueError(f"index {hi} outside local space (max {MAX_LOCAL})")
        base = 3 + 3 * BANKS.index(bank)
        lut = np.array([T, LINE, PAPER, base, base + 1, base + 2], np.uint8)
        return lut[art]

    # ------------------------------------------------------------ swapping
    def for_suit(self, suit: str) -> "Palette":
        """Same index space, different lookup table."""
        if suit not in self.suits:
            return self
        merged = dict(self.banks)
        merged.update(self.suits[suit])
        return replace(self, banks=merged)

    def rgb_lut(self, bg: str | None = None) -> np.ndarray:
        cols = [(bg or self.paper) if c is None else c for c in self.colors]
        return np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in cols],
                        np.uint8)

    def render(self, matrix: np.ndarray, bg: str | None = None) -> np.ndarray:
        return self.rgb_lut(bg)[matrix]

    # ------------------------------------------------------------ checks
    def validate(self, strict: bool = True) -> list[str]:
        errs: list[str] = []
        for name, b in self.banks.items():
            errs += b.check(name, self.rungs, self.tolerance)
        for suit, over in self.suits.items():
            for name, b in over.items():
                errs += b.check(f"{suit}.{name}", self.rungs, self.tolerance)
        if lum(self.line) > 25:
            errs.append(f"line {self.line} too light (L={lum(self.line):.0f})")
        if lum(self.paper) < 75:
            errs.append(f"paper {self.paper} too dark (L={lum(self.paper):.0f})")
        drawable = [c for c in self.colors if c]
        if len(set(drawable)) != len(drawable):
            errs.append("duplicate colors across banks")
        if strict and errs:
            raise ValueError(f"palette {self.name!r}:\n  " + "\n  ".join(errs))
        return errs

    # ------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path, strict: bool = True) -> "Palette":
        d = yaml.safe_load(Path(path).read_text())
        _reject_nulls(d)
        p = cls(
            name=d.get("name", "unnamed"),
            line=d["universal"]["line"],
            paper=d["universal"]["paper"],
            banks={k: Bank(**v) for k, v in d["banks"].items()},
            suits={s: {k: Bank(**v) for k, v in o.items()}
                   for s, o in d.get("suits", {}).items()},
            rungs={k: float(v) for k, v in d["rungs"].items()},
            tolerance=float(d.get("tolerance", 6)),
        )
        missing = set(BANKS) - set(p.banks)
        if missing:
            raise ValueError(f"palette is missing banks: {sorted(missing)}")
        p.validate(strict=strict)
        return p


def _reject_nulls(node, path: str = "") -> None:
    """'#RRGGBB' unquoted is a YAML comment. Catch it here, not at render."""
    if isinstance(node, dict):
        for k, v in node.items():
            _reject_nulls(v, f"{path}.{k}" if path else str(k))
    elif node is None and path:
        raise ValueError(
            f"'{path}' parsed as null — an unquoted hex value is a YAML "
            "comment. Quote every color.")
