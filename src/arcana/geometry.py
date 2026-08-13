"""
Geometry from a deck's deck.yaml, validated at load.

The divisibility check matters more than it looks. On a 160x276 card, no corner
size lets 8px edge tiles fit both runs evenly — the medallion slot exists partly
to absorb that remainder. Get it wrong and tiles misalign by a few pixels
somewhere in the middle of an edge, which is nearly invisible on screen and
glaring in print.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True, slots=True)
class Geometry:
    card_w: int
    card_h: int
    art_w: int
    art_h: int
    margin: int
    band_numeral: int
    band_title: int
    corner: int
    edge: int
    med_h: int
    med_v: int

    @property
    def art_origin(self) -> tuple[int, int]:
        """Top-left of the art window in card space. For the card composer."""
        return (self.margin, self.band_numeral)

    @property
    def mirror_x(self) -> int:
        """Pip-lattice mirror axis, in art-window space."""
        return self.art_w // 2

    @property
    def runs(self) -> tuple[int, int]:
        """Edge tiles per half-run, horizontal and vertical."""
        h = (self.card_w - 2 * self.corner - self.med_h) // 2
        v = (self.card_h - 2 * self.corner - self.med_v) // 2
        return h // self.edge, v // self.edge

    def validate(self) -> None:
        want_w = self.art_w + 2 * self.margin
        want_h = self.art_h + self.band_numeral + self.band_title
        if (want_w, want_h) != (self.card_w, self.card_h):
            raise ValueError(
                f"card {self.card_w}x{self.card_h} != art+bands {want_w}x{want_h}")
        h = self.card_w - 2 * self.corner - self.med_h
        v = self.card_h - 2 * self.corner - self.med_v
        for label, run in (("horizontal", h), ("vertical", v)):
            if run % (2 * self.edge):
                raise ValueError(
                    f"{label} run {run} is not divisible by 2x{self.edge}. "
                    f"Adjust the medallion slot: try "
                    f"{run % (2*self.edge) + (self.med_h if label=='horizontal' else self.med_v)}.")

    @classmethod
    def load(cls, path: str | Path) -> "Geometry":
        d = yaml.safe_load(Path(path).read_text())
        g, b = d["geometry"], d["border"]
        geo = cls(
            card_w=g["card"][0], card_h=g["card"][1],
            art_w=g["art"][0], art_h=g["art"][1],
            margin=g["margin"],
            band_numeral=g["bands"]["numeral"], band_title=g["bands"]["title"],
            corner=b["corner"], edge=b["edge"],
            med_h=b["medallion"]["horizontal"], med_v=b["medallion"]["vertical"],
        )
        geo.validate()
        return geo


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
