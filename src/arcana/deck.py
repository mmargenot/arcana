"""
A deck ties together the parts the engine needs: a palette, a geometry, and an
element manifest. Config lives under `decks/configs/<name>/`; generated tiles
and renders live under `decks/artifacts/<name>/` (git-ignored) and are never
committed.

Paths are resolved relative to the current working directory by default (so
`uv run arcana ...` from the repo root just works); override the roots to point
elsewhere.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from arcana.palette import Palette
from arcana.geometry import Geometry, load_config

DECKS = Path("decks")
CONFIGS = DECKS / "configs"
ARTIFACTS = DECKS / "artifacts"


@dataclass(frozen=True, slots=True)
class Deck:
    name: str
    palette: Palette
    geometry: Geometry
    config: dict

    @property
    def elements(self) -> dict:
        return self.config["elements"]

    @property
    def suit_pips(self) -> dict:
        return self.config["suit_pips"]


def config_dir(name: str, configs_root: str | Path = CONFIGS) -> Path:
    return Path(configs_root) / name


def assets_dir(name: str, artifacts_root: str | Path = ARTIFACTS) -> Path:
    """Where placeholder/authored tiles for a deck are generated."""
    return Path(artifacts_root) / name / "assets"


def render_dir(name: str, artifacts_root: str | Path = ARTIFACTS) -> Path:
    """Where rendered PNGs for a deck are written."""
    return Path(artifacts_root) / name


def load_deck(name: str, configs_root: str | Path = CONFIGS) -> Deck:
    cdir = config_dir(name, configs_root)
    if not cdir.exists():
        raise FileNotFoundError(f"no deck config at {cdir}")
    palette = Palette.load(cdir / "palette.yaml")
    geometry = Geometry.load(cdir / "deck.yaml")
    config = load_config(cdir / "deck.yaml")
    return Deck(name=name, palette=palette, geometry=geometry, config=config)
