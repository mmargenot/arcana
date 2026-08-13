"""
Thin CLI for arcana. The engine lives in the package; this just wires a deck's
config to the render pipeline and writes the results.

    arcana generate <deck> [--scale N]   seed placeholder tiles if missing,
                                         render per-suit borders
    arcana seed <deck>                   (re)write placeholder tiles only

`--scale` is a NEAREST-neighbour zoom applied ONLY to the preview contact sheet
(all_suits.png), so the tiny pixel-art cards are legible on screen. The
per-suit PNGs are always written at native resolution; nothing in the deck
depends on it.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from arcana.deck import load_deck, assets_dir, render_dir, CONFIGS, ARTIFACTS
from arcana.elements import load_all, audit
from arcana.seed import seed_deck
from arcana.compose import build_border, render_border, mount, check_symmetry, check_contiguous


def cmd_seed(name: str, configs_root: Path, artifacts_root: Path) -> Path:
    dest = assets_dir(name, artifacts_root)
    seed_deck(dest)
    print(f"seeded placeholder tiles -> {dest}")
    return dest


def cmd_generate(name: str, scale: int, configs_root: Path, artifacts_root: Path) -> Path:
    deck = load_deck(name, configs_root)
    pal, geo, cfg = deck.palette, deck.geometry, deck.config

    assets = assets_dir(name, artifacts_root)
    if not assets.exists():
        seed_deck(assets)
        print(f"seeded placeholder tiles -> {assets}")
    els = load_all(assets, cfg["elements"])

    out = render_dir(name, artifacts_root)
    out.mkdir(parents=True, exist_ok=True)

    print(f"palette   {pal.name}: {len([c for c in pal.colors if c])} drawable")
    print(f"geometry  card {geo.card_w}x{geo.card_h}, art {geo.art_w}x{geo.art_h}, runs {geo.runs}")
    for line in audit(assets, cfg["elements"]):
        print(" ", line)

    frame, _ = build_border(geo, els["corner"], els["edge"])
    print("\nsymmetry ", check_symmetry(frame))
    print("broken   ", [k for k, v in check_contiguous(frame, geo).items() if not v] or "none")

    suits = list(cfg["suit_pips"])
    sheet = np.full((geo.card_h, (geo.card_w + 8) * len(suits), 3), 237, np.uint8)
    for i, suit in enumerate(suits):
        med = mount(els[cfg["suit_pips"][suit]], els["cartouche"])
        m = render_border(pal, geo, els["corner"], els["edge"], med)
        rgb = pal.for_suit(suit).render(m)
        sheet[:, i * (geo.card_w + 8):i * (geo.card_w + 8) + geo.card_w] = rgb
        Image.fromarray(rgb).save(out / f"border_{suit}.png")
    img = Image.fromarray(sheet)
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(out / "all_suits.png")
    print(f"\nwrote {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="arcana", description="Pixel-art card generation.")
    ap.add_argument("--configs-root", type=Path, default=CONFIGS,
                    help="directory of deck configs (default: decks/configs)")
    ap.add_argument("--artifacts-root", type=Path, default=ARTIFACTS,
                    help="directory for generated tiles/renders (default: decks/artifacts)")
    sub = ap.add_subparsers(dest="command")

    g = sub.add_parser("generate", help="render a deck's borders")
    g.add_argument("deck")
    g.add_argument("--scale", type=int, default=2,
                   help="NEAREST zoom for the preview sheet only (default: 2)")

    s = sub.add_parser("seed", help="(re)write a deck's placeholder tiles")
    s.add_argument("deck")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command == "generate":
        cmd_generate(args.deck, args.scale, args.configs_root, args.artifacts_root)
    elif args.command == "seed":
        cmd_seed(args.deck, args.configs_root, args.artifacts_root)
    else:
        ap.print_help()
        return 1
    return 0
