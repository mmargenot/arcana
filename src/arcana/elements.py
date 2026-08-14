"""
Elements: a deck's element manifest plus indexed PNGs on disk.

An element is one authored tile — a corner, an edge, a cartouche, a pip. All
four are the same kind of thing: an indexed PNG whose palette indices are local
index space. One format, one loader.

Multi-layer elements use `name.<bank>.png`, so `corner.border.png` and
`corner.motif.png` are the frame and inlay of one corner. In Aseprite, name
your layers after banks and export with:

    aseprite -b corner.aseprite --split-layers --save-as 'corner.{layer}.png'

The AUTHORING palette is deliberately grey. Drawing in local space means
drawing VALUE STRUCTURE; the bank supplies hue at render time. A pink authoring
palette would have you unconsciously composing for pink.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

from arcana.palette import LOCAL_SLOTS, MAX_LOCAL, T

AUTHORING = [
    (255, 0, 255),     # 0 transparent — magenta, unmistakable if it leaks
    (20, 20, 24),      # 1 line
    (240, 236, 228),   # 2 paper
    (64, 64, 72),      # 3 dark   L=30
    (136, 136, 148),   # 4 mid    L=50
    (210, 210, 222),   # 5 light  L=74
]


class AssetError(Exception):
    pass


def overlay(dst: np.ndarray, src: np.ndarray) -> None:
    """Composite `src` onto `dst` in place: non-transparent pixels of `src`
    (index != 0) overwrite `dst`, transparent pixels leave it untouched. Both
    must be the same shape. The engine's one blend primitive."""
    dst[src != T] = src[src != T]


def blit(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Clip-safe positioned `overlay`: paste `src` with its top-left at (x, y),
    trimming anything past the edges of `dst`. Index 0 is transparent."""
    h, w = src.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    overlay(dst[y0:y1, x0:x1], src[y0 - y:y1 - y, x0 - x:x1 - x])


# ---------------------------------------------------------------- io
def write_tile(art: np.ndarray, path: str | Path) -> None:
    if art.max(initial=0) > MAX_LOCAL:
        raise AssetError(f"index {art.max()} exceeds local space ({MAX_LOCAL})")
    img = Image.fromarray(art.astype(np.uint8), mode="P")
    img.putpalette([v for c in AUTHORING for v in c] + [0, 0, 0] * (256 - len(AUTHORING)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, transparency=0)


def read_tile(path: str | Path, expect: tuple[int, int] | None = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise AssetError(f"missing asset: {path}")
    img = Image.open(path)
    if img.mode != "P":
        raise AssetError(
            f"{path.name} is {img.mode}, not indexed — its indices are gone. "
            "Aseprite: Sprite > Color Mode > Indexed, then re-export.")
    a = np.array(img, dtype=np.uint8)
    hi = int(a.max(initial=0))
    if hi > MAX_LOCAL:
        raise AssetError(f"{path.name} uses index {hi}; local space is 0-{MAX_LOCAL}")
    if expect is not None and a.shape != expect:
        raise AssetError(f"{path.name} is {a.shape[1]}x{a.shape[0]}, "
                         f"manifest declares {expect[1]}x{expect[0]}")
    return a


def write_authoring_palette(dirpath: str | Path) -> None:
    d = Path(dirpath); d.mkdir(parents=True, exist_ok=True)
    with open(d / "authoring.gpl", "w") as f:
        f.write("GIMP Palette\nName: local-index-space\nColumns: 6\n#\n")
        for i, (r, g, b) in enumerate(AUTHORING):
            f.write(f"{r:3} {g:3} {b:3}\t{i} {LOCAL_SLOTS[i]}\n")
    (d / "authoring.hex").write_text(
        "".join("{:02X}{:02X}{:02X}\n".format(*c) for c in AUTHORING))


# ---------------------------------------------------------------- element
@dataclass(frozen=True, slots=True)
class Element:
    name: str
    role: str                       # corner | edge | medallion | pip
    size: tuple[int, int]           # (h, w)
    layers: dict[str, np.ndarray]   # bank name -> local matrix

    @property
    def matrix(self) -> np.ndarray:
        """The sole (or first) layer's local matrix. Most elements are
        single-layer, so this names the common 'give me its pixels' access."""
        return next(iter(self.layers.values()))

    @property
    def sole_bank(self) -> str:
        """The sole (or first) layer's bank name."""
        return next(iter(self.layers))

    def bind(self, palette) -> np.ndarray:
        """Composite every layer into one global index matrix."""
        out = None
        for bank, art in self.layers.items():
            g = palette.bind(art, bank)
            if out is None:
                out = g.copy()
            else:
                overlay(out, g)
        return out

    def is_opaque(self, inset: int = 4) -> bool:
        """Interior fully covered? A medallion must be, or the frame shows
        through and bisects thin motifs."""
        a = self.matrix
        h, w = a.shape
        return bool((a[inset:h - inset, inset:w - inset] != T).all())


def load_element(root: str | Path, name: str, spec: dict) -> Element:
    """Loads indexed PNG, RGB PNG, or ASCII .txt — whichever is present."""
    from arcana.tileio import read_any
    root = Path(root)
    h, w = spec["size"]
    layers: dict[str, np.ndarray] = {}
    for bank in spec["layers"]:
        layers[bank] = read_any(root / f"{spec['path']}.{bank}", expect=(h, w))
    el = Element(name=name, role=spec["role"], size=(h, w), layers=layers)
    if spec.get("opaque") and not el.is_opaque():
        raise AssetError(
            f"{name} is declared opaque but its interior has holes — the frame "
            "will show through and bisect thin pips.")
    return el


def load_all(root: str | Path, manifest: dict) -> dict[str, Element]:
    return {n: load_element(root, n, s) for n, s in manifest.items()}


def audit(root: str | Path, manifest: dict) -> list[str]:
    out = []
    for name, spec in manifest.items():
        try:
            el = load_element(root, name, spec)
            used = sorted({int(v) for a in el.layers.values() for v in np.unique(a)})
            out.append(f"OK   {name:16} {el.size[1]}x{el.size[0]:<4} "
                       f"{list(el.layers)}  slots {[LOCAL_SLOTS[i] for i in used]}")
        except AssetError as e:
            out.append(f"FAIL {name:16} {e}")
    return out
