"""
Thin CLI for arcana. The engine lives in the package; this just wires a deck's
config to the render pipeline and writes the results.

    arcana generate <deck> [--scale N]   seed placeholder tiles if missing,
                                         render per-suit borders
    arcana cards <deck> [--layout NAME]  render minor-arcana pip cards (ranks
        [--suit S | --all-suits]         1-10) with numeral/title labels;
        [--no-labels]                    --layout forces one everywhere,
                                         --no-labels renders bare
    arcana majors <deck> [--no-labels]   render the 22 major arcana (labeling
                                         half; figure image is a later item)
    arcana seed <deck>                   (re)write placeholder tiles only

`--scale` is a NEAREST-neighbour zoom applied ONLY to the preview contact sheet,
so the tiny pixel-art cards are legible on screen. The individual PNGs are
always written at native resolution; nothing in the deck depends on it.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from arcana.deck import load_deck, assets_dir, render_dir, CONFIGS, ARTIFACTS
from arcana.elements import load_all, audit
from arcana.seed import seed_deck
from arcana import layout, field, data
from arcana.text import load_font
from arcana.compose import (build_border, render_border, build_medallion,
                            build_pip_card, build_major_card,
                            check_symmetry, check_contiguous)

MEDALLION_STYLES = ("suit", "lozenge", "none")


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
    med_style, med_scale = _medallion_opts(cfg)
    sheet = np.full((geo.card_h, (geo.card_w + 8) * len(suits), 3), 237, np.uint8)
    for i, suit in enumerate(suits):
        med = build_medallion(els, cfg["suit_pips"][suit], style=med_style, scale=med_scale)
        m = render_border(pal, geo, els["corner"], els["edge"], med)
        rgb = pal.for_suit(suit).render(m)
        sheet[:, i * (geo.card_w + 8):i * (geo.card_w + 8) + geo.card_w] = rgb
        Image.fromarray(rgb).save(out / f"border_{suit}.png")
    img = Image.fromarray(sheet)
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(out / "all_suits.png")
    print(f"\nwrote {out}")
    return out


def _layout_for_rank(cfg: dict, rank: int, override: str | None) -> str:
    if override:
        return override
    spec = cfg.get("pip_layouts", {})
    by_rank = spec.get("by_rank", {})
    return by_rank.get(rank) or by_rank.get(str(rank)) or spec.get("default", "square")


def _field_for_suit(cfg: dict, suit: str, override: str | None) -> str:
    if override:
        return override
    spec = cfg.get("field_designs", {})
    return spec.get("by_suit", {}).get(suit) or spec.get("default", "plain")


def _medallion_opts(cfg: dict, style_override: str | None = None,
                    scale_override: float | None = None) -> tuple[str, float]:
    """Medallion (style, scale) from `border.medallion`, CLI override winning —
    same default-vs-override pattern as layout/field."""
    spec = cfg.get("border", {}).get("medallion", {})
    style = style_override or spec.get("style", "suit")
    scale = scale_override if scale_override is not None else spec.get("scale", 1.0)
    return style, float(scale)


def cmd_cards(name: str, layout_override: str | None, field_override: str | None,
              suit: str | None, all_suits: bool, scale: int, no_labels: bool,
              med_override: str | None, med_scale_override: float | None,
              configs_root: Path, artifacts_root: Path) -> Path:
    deck = load_deck(name, configs_root)
    pal, geo, cfg = deck.palette, deck.geometry, deck.config
    if layout_override and layout_override not in layout.names():
        raise SystemExit(f"unknown layout {layout_override!r}; have {layout.names()}")
    if field_override and field_override not in field.names():
        raise SystemExit(f"unknown field {field_override!r}; have {field.names()}")
    med_style, med_scale = _medallion_opts(cfg, med_override, med_scale_override)

    assets = assets_dir(name, artifacts_root)
    if not assets.exists():
        seed_deck(assets)
        print(f"seeded placeholder tiles -> {assets}")
    els = load_all(assets, cfg["elements"])

    opts = data.label_options(cfg)
    labels_on = opts["enabled"] and not no_labels
    font = load_font(assets / "font") if labels_on else None

    out = render_dir(name, artifacts_root) / "cards"
    out.mkdir(parents=True, exist_ok=True)

    minors = [s for s in cfg["suit_pips"] if s != "majors"]
    if all_suits:
        suits = minors
    elif suit:
        suits = [suit]
    else:
        suits = minors                       # default: show every minor suit
    ranks = range(1, 11)

    pip_cfg = layout.pip_config(cfg)
    gap, W, H = 8, geo.card_w, geo.card_h
    sheet = np.full((len(suits) * (H + gap) - gap,
                     len(ranks) * (W + gap) - gap, 3), 237, np.uint8)
    for r, su in enumerate(suits):
        pip_key = cfg["suit_pips"][su]
        fname = _field_for_suit(cfg, su, field_override)
        for c, rank in enumerate(ranks):
            lname = _layout_for_rank(cfg, rank, layout_override)
            top, bottom = (data.minor_label(rank, su, style=opts["style"],
                                            split=opts["split"], cfg=cfg)
                           if labels_on else (None, None))
            try:
                m = build_pip_card(pal, geo, els, rank, lname, pip_key, fname,
                                   pip_cfg, font=font, top=top, bottom=bottom,
                                   med_style=med_style, med_scale=med_scale)
            except layout.InvalidPipLayout as e:
                raise SystemExit(str(e))
            rgb = pal.for_suit(su).render(m)
            Image.fromarray(rgb).save(out / f"{su}_{rank:02d}.png")
            y, x = r * (H + gap), c * (W + gap)
            sheet[y:y + H, x:x + W] = rgb
    tag = "-".join(t for t in (layout_override, field_override) if t) or "by-rank"
    img = Image.fromarray(sheet)
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(out / f"cards_{tag}.png")
    print(f"wrote {out}  ({len(suits)} suit(s) x 10 ranks, "
          f"layout: {layout_override or 'by-rank'}, field: {field_override or 'by-suit'}, "
          f"labels: {'on' if labels_on else 'off'})")
    return out


def cmd_majors(name: str, scale: int, no_labels: bool, med_override: str | None,
               med_scale_override: float | None, configs_root: Path,
               artifacts_root: Path) -> Path:
    """Render the 22 major arcana (labeling half — the figure image is a later
    roadmap item; `build_major_card` leaves the seam)."""
    deck = load_deck(name, configs_root)
    pal, geo, cfg = deck.palette, deck.geometry, deck.config

    assets = assets_dir(name, artifacts_root)
    if not assets.exists():
        seed_deck(assets)
        print(f"seeded placeholder tiles -> {assets}")
    els = load_all(assets, cfg["elements"])

    opts = data.label_options(cfg)
    labels_on = opts["enabled"] and not no_labels
    font = load_font(assets / "font") if labels_on else None
    pip_key = cfg["suit_pips"].get("majors")
    fname = _field_for_suit(cfg, "majors", None)
    med_style, med_scale = _medallion_opts(cfg, med_override, med_scale_override)

    out = render_dir(name, artifacts_root) / "majors"
    out.mkdir(parents=True, exist_ok=True)

    cols = 6
    rows = (len(data.MAJOR_NAMES) + cols - 1) // cols
    gap, W, H = 8, geo.card_w, geo.card_h
    sheet = np.full((rows * (H + gap) - gap, cols * (W + gap) - gap, 3), 237, np.uint8)
    for number in range(len(data.MAJOR_NAMES)):
        top, bottom = (data.major_label(number, split=opts["split"], cfg=cfg)
                       if labels_on else (None, None))
        m = build_major_card(pal, geo, els, font, top=top, bottom=bottom,
                             field_design=fname, pip_key=pip_key,
                             med_style=med_style, med_scale=med_scale)
        rgb = pal.for_suit("majors").render(m)
        Image.fromarray(rgb).save(out / f"major_{number:02d}.png")
        r, c = divmod(number, cols)
        y, x = r * (H + gap), c * (W + gap)
        sheet[y:y + H, x:x + W] = rgb
    img = Image.fromarray(sheet)
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(out / "majors.png")
    print(f"wrote {out}  (22 majors, labels: {'on' if labels_on else 'off'})")
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

    c = sub.add_parser("cards", help="render minor-arcana pip cards (ranks 1-10)")
    c.add_argument("deck")
    c.add_argument("--layout", choices=layout.names(), metavar="NAME",
                   help="force one pip arrangement for every rank (default: per-rank from deck.yaml)")
    c.add_argument("--field", choices=field.names(), metavar="NAME",
                   help="force one field design for every suit (default: per-suit from deck.yaml)")
    grp = c.add_mutually_exclusive_group()
    grp.add_argument("--suit", help="render a single suit (default: all minor suits)")
    grp.add_argument("--all-suits", action="store_true", help="render every minor suit")
    c.add_argument("--scale", type=int, default=2,
                   help="NEAREST zoom for the preview sheet only (default: 2)")
    c.add_argument("--no-labels", action="store_true",
                   help="render bare cards, without numeral/title labels")
    c.add_argument("--medallion", choices=MEDALLION_STYLES, metavar="STYLE",
                   help="edge medallion: suit | lozenge | none (default: from deck.yaml)")
    c.add_argument("--medallion-scale", type=float, metavar="F",
                   help="scale the medallion (e.g. 0.5); default from deck.yaml")

    m = sub.add_parser("majors", help="render the 22 major arcana (labeling half)")
    m.add_argument("deck")
    m.add_argument("--scale", type=int, default=2,
                   help="NEAREST zoom for the preview sheet only (default: 2)")
    m.add_argument("--no-labels", action="store_true",
                   help="render bare cards, without numeral/title labels")
    m.add_argument("--medallion", choices=MEDALLION_STYLES, metavar="STYLE",
                   help="edge medallion: suit | lozenge | none (default: from deck.yaml)")
    m.add_argument("--medallion-scale", type=float, metavar="F",
                   help="scale the medallion (e.g. 0.5); default from deck.yaml")

    s = sub.add_parser("seed", help="(re)write a deck's placeholder tiles")
    s.add_argument("deck")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command == "generate":
        cmd_generate(args.deck, args.scale, args.configs_root, args.artifacts_root)
    elif args.command == "cards":
        cmd_cards(args.deck, args.layout, args.field, args.suit, args.all_suits,
                  args.scale, args.no_labels, args.medallion, args.medallion_scale,
                  args.configs_root, args.artifacts_root)
    elif args.command == "majors":
        cmd_majors(args.deck, args.scale, args.no_labels, args.medallion,
                   args.medallion_scale, args.configs_root, args.artifacts_root)
    elif args.command == "seed":
        cmd_seed(args.deck, args.configs_root, args.artifacts_root)
    else:
        ap.print_help()
        return 1
    return 0
