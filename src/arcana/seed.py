"""
Placeholder tile art for a deck.

A deck's *config* (palette + geometry + manifest) is committed under
`decks/configs/<name>/`; its *tiles* are not. This module regenerates a
runnable set of placeholder tiles so the pipeline builds before any real art
exists, and so a new deck can be bootstrapped from code.

Two kinds of source live here, and both stay in code rather than as committed
binary artifacts:

  - `corner`, `edge`, `roundel` are generated procedurally.
  - the five pips (`cups`, `wands`, `swords`, `pentacles`, `rose`) are the nicer
    hand-authored tiles, kept as ASCII string constants and parsed on demand.

`seed_deck(dest_root)` writes the border/cartouche tiles as indexed PNG and the
pips as ASCII `.txt` — deliberately mixed, to exercise `tileio.read_any`.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from arcana.palette import LINE, PAPER, DARK, MID
from arcana.elements import write_tile, write_authoring_palette
from arcana.tileio import from_ascii, write_ascii
from arcana.text import Font, CELL_W, CELL_H, INK


# ---------------------------------------------------------------- procedural
# The corner and edge tiles are DECORATION only: their marks sit on a transparent
# field and are pasted over the frame that compose.build_border draws. The frame
# rules do not live here, so either tile can be swapped without disturbing them.
def corner(C: int = 16) -> np.ndarray:
    """Corner motif — a diagonal staircase flourish, transparent elsewhere."""
    c = np.zeros((C, C), np.uint8)
    for r, k in ((8, 8), (9, 11), (11, 9), (12, 12), (10, 14), (14, 10)):
        c[r:r + 2, k:k + 2] = DARK
    return c


def edge(C: int = 16, E: int = 8) -> np.ndarray:
    """Edge motif — a dentil tab, transparent elsewhere."""
    e = np.zeros((C, E), np.uint8)
    e[8:10, 1:7] = DARK; e[10:12, 3:5] = DARK
    return e


def roundel(S: int = 24) -> np.ndarray:
    c = np.zeros((S, S), np.uint8)
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.hypot(xx - (S - 1) / 2, yy - (S - 1) / 2)
    c[r <= S / 2 - 0.4] = LINE
    c[r <= S / 2 - 1.2] = MID
    c[r <= S / 2 - 2.6] = PAPER
    return c


def lozenge(S: int = 24) -> np.ndarray:
    """A heraldic lozenge — a solid diamond, LINE outline over a MID fill, on a
    transparent field. An abstract medallion: bound to the `motif` bank it reads
    as a suit-coloured gem. The body is solid so the frame rule can't bisect it;
    the box corners stay transparent (the rule runs on, uninterrupted, behind)."""
    c = np.zeros((S, S), np.uint8)
    yy, xx = np.mgrid[0:S, 0:S]
    half = (S - 1) / 2
    d = (np.abs(xx - half) + np.abs(yy - half)) / (S / 2)   # L1 radius: 0 centre, 1 at edge
    c[d <= 1.0] = LINE
    c[d <= 1.0 - 1.5 / (S / 2)] = MID
    return c


# ---------------------------------------------------------------- pips (ASCII)
# Hand-authored, kept in code. Glyphs: . transparent  @ line  ' paper
# % dark  + mid  - light. Each is 16x16 and self-symmetric about x=7.5 so odd
# pip counts stay symmetric on the lattice.
PIPS_ASCII = {
    "cups": """
................
................
...@@@@@@@@@@...
...+----+++++...
...+----+++++...
...++++++++++...
...++++++++++...
...++++++++++...
.....%%%%%%.....
.......%%.......
.......%%.......
....%%%%%%%%....
...@@@@@@@@@@...
................
................
................
""",
    "wands": """
................
......@@@@......
......-++%......
......-++%......
....%%%%%%%%....
......-++%......
......-++%......
....%%%%%%%%....
......-++%......
......-++%......
....%%%%%%%%....
......-++%......
......-++%......
......@@@@......
................
................
""",
    "swords": """
................
.......@@.......
.......-+.......
.......-+.......
.......-+.......
.......-+.......
.......-+.......
.......-+.......
.......-+.......
...%%%%%%%%%%...
...%%%%%%%%%%...
.......++.......
.......++.......
.......++.......
......@@@@......
................
""",
    "pentacles": """
................
.....%%%%%%.....
....%%%++%%%....
...%%--%%--%%...
..%%---%%---%%..
.%%----%%----%%.
.%%%%%%%%%%%%%%.
.%+-%%%--%%%-+%.
.%+--%%--%%--+%.
.%%--%%%%%%--%%.
.%%--%%%%%%--%%.
..%%-%%--%%-%%..
...%%------%%...
....%%%++%%%....
.....%%%%%%.....
................
""",
    "rose": """
................
................
.....%%%%%%.....
...%%+++%++%%...
...%%---%--%%...
..%+-%--%-%-+%..
..%+--%%%%--+%..
..%+--%%%%--+%..
..%%%%%%%%%%%%..
..%+--%%%%--+%..
..%+-%--%-%-+%..
...%%---%--%%...
...%%+++%++%%...
.....%%%%%%.....
................
................
""",
}


def pip(name: str) -> np.ndarray:
    """The 16x16 local-index matrix for a suit pip."""
    return from_ascii(PIPS_ASCII[name])


# ---------------------------------------------------------------- font (5x7)
# The placeholder label font, kept in code like the pips. Each glyph is a 5x7
# cell of '#' (ink) / '.' (blank); `placeholder_font` centres it in the 6x10
# box `arcana.text` lays out. Uppercase-only titles keep the inventory small:
# digits, A-Z, and hyphen (space needs no ink). A real font drops in later via
# the deck override path (`text.load_font`).
FONT_5X7 = {
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": (".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": ("..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#.#.#", "#..##", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
}


def _glyph_tile(rows: tuple[str, ...]) -> np.ndarray:
    """A 5x7 pattern centred in the 6x10 cell text.py lays out (1px top pad)."""
    t = np.zeros((CELL_H, CELL_W), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                t[y + 1, x] = INK
    return t


def placeholder_font() -> Font:
    """The engine's built-in label font as a `text.Font`."""
    glyphs = {ch: _glyph_tile(rows) for ch, rows in FONT_5X7.items()}
    glyphs[" "] = np.zeros((CELL_H, CELL_W), np.uint8)
    return Font(glyphs=glyphs)


# ---------------------------------------------------------------- seed
def seed_deck(dest_root: str | Path) -> Path:
    """Write a full set of placeholder tiles (and the authoring palette) under
    dest_root, laid out to match the standard element manifest. Border and
    cartouche are indexed PNG; pips are ASCII .txt."""
    root = Path(dest_root)
    write_authoring_palette(root)
    write_tile(corner(), root / "border/corner.border.png")
    write_tile(edge(), root / "border/edge.border.png")
    write_tile(roundel(), root / "cartouche/roundel.motif.png")
    for name in PIPS_ASCII:
        write_ascii(pip(name), root / f"pips/{name}.motif.txt", name=f"{name}.motif")
    return root
