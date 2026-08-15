"""
Thin CLI for arcana. The engine lives in the package; this just wires a deck's
config to the render pipeline and writes the results.

    arcana generate <deck> [--scale N]   seed placeholder tiles if missing,
                                         render per-suit borders
    arcana cards <deck> [--layout NAME]  render minor-arcana pip cards (ranks
        [--suit S | --all-suits]         1-10) with numeral/title labels;
        [--no-labels]                    --layout forces one everywhere,
                                         --no-labels renders bare
    arcana majors <deck> [--no-labels]   render the 22 major arcana: mural
        [--no-murals]                    (when the deck ships murals/ layers)
                                         + frame + label
    arcana import-mural <deck> <png>     quantize external pixel art (Retro
        --major N [--force]              Diffusion output, edited exports...)
                                         into a major's committed mural layers
    arcana export-mural <deck> --major N render a major's mural (art window
                                         only) to RGB PNG, for external tools;
                                         the exact inverse of import-mural
    arcana seed <deck>                   (re)write placeholder tiles only

`--scale` is a NEAREST-neighbour zoom applied ONLY to the preview contact sheet,
so the tiny pixel-art cards are legible on screen. The individual PNGs are
always written at native resolution; nothing in the deck depends on it.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

from arcana.deck import load_deck, assets_dir, config_dir, render_dir, CONFIGS, ARTIFACTS
from arcana.elements import AssetError, load_all, audit
from arcana.seed import seed_deck
from arcana import layout, field, data, mural
from arcana.text import load_font
from arcana.compose import (build_border, render_border, build_medallion,
                            build_pip_card, build_major_card,
                            check_symmetry, check_contiguous)

MEDALLION_STYLES = ("suit", "lozenge", "none")
# Named medallion sizes — keywords for the scale knob, so a deck (or the CLI) can
# say `small` instead of `0.5`. A bare number still works.
MEDALLION_SIZES = {"full": 1.0, "large": 0.75, "small": 0.5, "smaller": 0.4, "tiny": 0.3}


def _resolve_scale(value) -> float:
    """A medallion scale from a keyword (`small`, `tiny`, …) or a number."""
    if isinstance(value, (int, float)):
        return float(value)
    key = str(value).strip().lower()
    if key in MEDALLION_SIZES:
        return MEDALLION_SIZES[key]
    try:
        return float(key)
    except ValueError:
        raise SystemExit(f"unknown medallion size {value!r}; use a number or one "
                         f"of {', '.join(MEDALLION_SIZES)}")


def cmd_seed(name: str, configs_root: Path, artifacts_root: Path) -> Path:
    deck = load_deck(name, configs_root)
    dest = assets_dir(name, artifacts_root)
    seed_deck(dest, deck.geometry, mural.face_keys(deck.config))
    print(f"seeded placeholder tiles + {len(mural.face_keys(deck.config))} "
          f"placeholder face(s) -> {dest}")
    return dest


def _ensure_assets(name: str, artifacts_root: Path, cfg: dict,
                   geo=None) -> tuple[Path, dict]:
    """Seed placeholder tiles if the deck's assets dir is missing, then load
    every element from it. Shared by `generate`, `cards`, and `majors`."""
    assets = assets_dir(name, artifacts_root)
    if not assets.exists():
        seed_deck(assets, geo)
        print(f"seeded placeholder tiles -> {assets}")
    elif geo is not None and not (assets / "murals").exists():
        # a deck seeded before faces existed: fill them in rather than
        # rendering bare cards
        from arcana.seed import seed_faces
        seed_faces(assets, geo, mural.face_keys(cfg))
    return assets, load_all(assets, cfg["elements"])


def cmd_generate(name: str, scale: int, configs_root: Path, artifacts_root: Path) -> Path:
    deck = load_deck(name, configs_root)
    pal, geo, cfg = deck.palette, deck.geometry, deck.config

    assets, els = _ensure_assets(name, artifacts_root, cfg)

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


def _medallion_opts(cfg: dict, style_override: str | None = None,
                    scale_override: str | None = None) -> tuple[str, float]:
    """Medallion (style, scale) from `border.medallion`, CLI override winning —
    same default-vs-override pattern as layout/field."""
    spec = cfg.get("border", {}).get("medallion", {})
    style = style_override or spec.get("style", "suit")
    scale = scale_override if scale_override is not None else spec.get("scale", 1.0)
    return style, _resolve_scale(scale)


def cmd_cards(name: str, layout_override: str | None, field_override: str | None,
              suit: str | None, all_suits: bool, scale: int, no_labels: bool,
              med_override: str | None, med_scale_override: str | None,
              configs_root: Path, artifacts_root: Path) -> Path:
    deck = load_deck(name, configs_root)
    pal, geo, cfg = deck.palette, deck.geometry, deck.config
    if layout_override and layout_override not in layout.names():
        raise SystemExit(f"unknown layout {layout_override!r}; have {layout.names()}")
    if field_override and field_override not in field.names():
        raise SystemExit(f"unknown field {field_override!r}; have {field.names()}")
    med_style, med_scale = _medallion_opts(cfg, med_override, med_scale_override)

    assets, els = _ensure_assets(name, artifacts_root, cfg)

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
        fname = field.field_for_suit(cfg, su, field_override)
        for c, rank in enumerate(ranks):
            lname = layout.layout_for_rank(cfg, rank, layout_override)
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
               med_scale_override: str | None, no_murals: bool,
               configs_root: Path, artifacts_root: Path,
               face: str | None = None, strict: bool = False) -> Path:
    """Render the deck's face cards: mural (image laid on the field — see
    arcana.mural) + frame + label. Committed art wins; a face without it falls
    back to its seeded placeholder, so art can land one card at a time.
    `--no-murals` renders the bare field for comparison; `--strict` is the
    print gate, failing on any face that has no committed art."""
    deck = load_deck(name, configs_root)
    pal, geo, cfg = deck.palette, deck.geometry, deck.config

    assets, els = _ensure_assets(name, artifacts_root, cfg, geo)

    opts = data.label_options(cfg)
    labels_on = opts["enabled"] and not no_labels
    font = load_font(assets / "font") if labels_on else None
    pip_key = cfg["suit_pips"].get("majors")
    fname = field.field_for_suit(cfg, "majors", None)
    med_style, med_scale = _medallion_opts(cfg, med_override, med_scale_override)
    murals_dir = config_dir(name, configs_root) / "murals"
    keys = mural.face_keys(cfg)
    if face is not None:
        if face not in keys:
            raise SystemExit(f"unknown face {face!r}; deck has {', '.join(keys)}")
        keys = (face,)
    if strict:
        bare = [k for k in keys if not mural.is_committed(murals_dir, k)]
        if bare:
            raise SystemExit(
                f"--strict: {len(bare)} face(s) have no committed art under "
                f"{murals_dir}: {', '.join(bare)}")

    out = render_dir(name, artifacts_root) / "majors"
    out.mkdir(parents=True, exist_ok=True)

    cols = min(6, len(keys))
    rows = (len(keys) + cols - 1) // cols
    gap, W, H = 8, geo.card_w, geo.card_h
    sheet = np.full((rows * (H + gap) - gap, cols * (W + gap) - gap, 3), 237, np.uint8)
    committed = 0
    for i, key in enumerate(keys):
        top, bottom = (data.face_label(key, split=opts["split"], cfg=cfg)
                       if labels_on else (None, None))
        image = None
        if not no_murals:
            image = mural.load_mural(murals_dir, key, geo, fallback_dir=assets / "murals")
            committed += mural.is_committed(murals_dir, key)
        m = build_major_card(pal, geo, els, font, top=top, bottom=bottom,
                             field_design=fname, pip_key=pip_key,
                             med_style=med_style, med_scale=med_scale,
                             image=image)
        rgb = pal.for_suit("majors").render(m)
        Image.fromarray(rgb).save(out / f"{key}.png")
        r, c = divmod(i, cols)
        y, x = r * (H + gap), c * (W + gap)
        sheet[y:y + H, x:x + W] = rgb
    img = Image.fromarray(sheet)
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(out / "majors.png")
    art = "off" if no_murals else f"{committed}/{len(keys)} committed, rest placeholder"
    print(f"wrote {out}  ({len(keys)} face(s), art: {art}, "
          f"labels: {'on' if labels_on else 'off'})")
    return out


def cmd_import_mural(name: str, png: Path, face: str,
                     tolerance: float, force: bool, out: Path | None,
                     configs_root: Path, artifacts_root: Path) -> Path:
    """Externally generated pixel art -> committed mural layers: quantize the
    RGB to the deck's 14 drawable colors, factor global indices into per-bank
    local layers (mural.split_global), and write the same dotted-stem ASCII a
    hand would author. The exact inverse of `export-mural`, so external art
    and authored art are indistinguishable on disk."""
    deck = load_deck(name, configs_root)
    pal, geo = deck.palette, deck.geometry
    from arcana.tileio import quantize_rgb_global, write_ascii
    try:
        g = quantize_rgb_global(png, pal.for_suit("majors"), tolerance, force)
    except AssetError as e:
        raise SystemExit(str(e))
    if g.shape != (geo.art_h, geo.art_w):
        raise SystemExit(f"{png.name} is {g.shape[1]}x{g.shape[0]}, the art "
                         f"window is {geo.art_w}x{geo.art_h}")
    dest = out or config_dir(name, configs_root) / "murals"
    dest.mkdir(parents=True, exist_ok=True)
    s = face
    for stale in dest.glob(f"{s}.*.txt"):
        stale.unlink()          # a dropped bank's old layer must not linger
    layers = mural.split_global(g)
    for bank, layer in layers.items():
        write_ascii(layer, dest / f"{s}.{bank}.txt", name=f"{s}.{bank}")
    # the color histogram is the import's diagnostic: a slot you expected that
    # is missing (or one you didn't that is huge) means the quantize went wrong
    labels = ["transparent", "line", "paper"] + [
        f"{b}.{r}" for b in ("border", "field", "motif", "figure")
        for r in ("dark", "mid", "light")]
    counts = np.bincount(g.ravel(), minlength=15)
    print(f"wrote {len(layers)} layer(s) for {s} -> {dest}")
    for i, (lab, n) in enumerate(zip(labels, counts)):
        if n:
            print(f"  {i:2} {lab:14} {n:6} px")
    missing = [lab for lab, n in zip(labels[1:], counts[1:]) if not n]
    if missing:
        print(f"  ! unused: {', '.join(missing)} — a bank the art never touches "
              f"is a hue family missing from this card")
    ink = mural.margin_ink(g, geo)
    if ink > 0.12:
        print(f"  ! {ink:.0%} of the covered margin is line work — the generator "
              f"may have drawn its own frame, which the deck's frame will "
              f"collide with rather than cover")
    return dest


def cmd_export_mural(name: str, face: str, scale: int,
                     out: Path | None, configs_root: Path,
                     artifacts_root: Path) -> Path:
    """A face's mural as a plain RGB PNG — exactly the card's art window
    (image laid on the field), for editing in any external tool or feeding a
    generative model as an init image. Re-import the result with
    `import-mural`; a round trip is index-lossless.

    Falls back to the seeded placeholder when a face has no committed art,
    which is what makes the loop bootstrappable: there is always something to
    export before the first card is drawn."""
    deck = load_deck(name, configs_root)
    pal, geo, cfg = deck.palette, deck.geometry, deck.config
    murals_dir = config_dir(name, configs_root) / "murals"
    assets, _ = _ensure_assets(name, artifacts_root, cfg, geo)
    try:
        image = mural.load_mural(murals_dir, face, geo,
                                 fallback_dir=assets / "murals", required=True)
    except AssetError as e:
        raise SystemExit(str(e))
    fname = field.field_for_suit(cfg, "majors", None)
    from arcana.compose import build_mural as _build_mural
    m = _build_mural(pal, geo, {}, fname, image=image)
    ox, oy = geo.art_origin
    window = m[oy:oy + geo.art_h, ox:ox + geo.art_w]
    rgb = pal.for_suit("majors").render(window)
    dest = out or (render_dir(name, artifacts_root) / "murals" / f"{face}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(dest)
    print(f"wrote {dest}  ({geo.art_w}x{geo.art_h} at {scale}x)")
    return dest


def cmd_rd(name: str, face: str, seed: int | None, prompt: str | None,
           init: Path | None, check_cost: bool,
           configs_root: Path, artifacts_root: Path) -> Path:
    """Generate candidate art for one face and write it for review.

    The seed image is the point: `--init` takes an RWS scan (any size — it is
    cropped and fitted to the art window) or an already-art-window-sized PNG,
    used as-is. With no `--init`, the face's current art is used, which on a
    fresh deck is its seeded placeholder.

    Nothing is imported automatically. Candidates land in the artifacts dir so
    a human picks one; `import-mural` is the deliberate second step."""
    from arcana import retro
    deck = load_deck(name, configs_root)
    pal, geo, cfg = deck.palette, deck.geometry, deck.config
    if face not in mural.face_keys(cfg):
        raise SystemExit(f"unknown face {face!r}; deck has "
                         f"{', '.join(mural.face_keys(cfg))}")
    gen = retro.Generation.load(config_dir(name, configs_root) / "generation.yaml")
    try:
        text = prompt or gen.prompt_for(face)
    except AssetError as e:
        raise SystemExit(str(e))

    init_bytes = None
    out = render_dir(name, artifacts_root) / "rd"
    out.mkdir(parents=True, exist_ok=True)
    if init is not None:
        from arcana.pixelate import pixelate
        if not init.exists():
            raise SystemExit(
                f"no seed image at {init}. Source scans are inputs, not "
                f"deliverables, so they live under the git-ignored artifacts "
                f"dir and are not in the repo. For the RWS majors:\n"
                f"  mkdir -p {init.parent}\n"
                f"  curl -L -o {init} \\\n"
                f"    https://upload.wikimedia.org/wikipedia/commons/9/90/"
                f"RWS_Tarot_00_Fool.jpg")
        img = Image.open(init)
        if img.size != (geo.art_w, geo.art_h):
            print(f"preparing seed: {init.name} {img.width}x{img.height} -> "
                  f"{geo.art_w}x{geo.art_h}")
            arr = pixelate(init, geo.art_w, geo.art_h)
            img = Image.fromarray(arr)
        seed_png = out / f"{face}.seed.png"
        img.convert("RGB").save(seed_png)
        init_bytes = seed_png.read_bytes()
        print(f"seed -> {seed_png}")
    else:
        init_bytes = cmd_export_mural(name, face, 1, out / f"{face}.seed.png",
                                      configs_root, artifacts_root).read_bytes()

    payload = retro.build_payload(gen, geo, pal.for_suit("majors"), prompt=text,
                                  seed=seed, init=init_bytes, check_cost=check_cost)
    print(f"style {gen.style}  {geo.art_w}x{geo.art_h}  x{gen.candidates}  "
          f"strength {gen.strength}  ~${retro.estimate_cost(gen, geo):.2f}")
    if check_cost:
        print(f'prompt: "{text}"')
    try:
        response = retro.request(payload)
    except retro.GenerationError as e:
        raise SystemExit(str(e))
    if check_cost:
        print(json.dumps(response, indent=2)[:800])
        return out
    try:
        images = retro.decode_images(response)
    except retro.GenerationError as e:
        raise SystemExit(str(e))
    for i, png in enumerate(images, 1):
        (out / f"{face}_{i}.png").write_bytes(png)
    print(f"wrote {len(images)} candidate(s) -> {out}\n"
          f"pick one, then: arcana import-mural {name} {out}/{face}_1.png "
          f"--face {face} --force")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="arcana", description="Pixel-art card generation.")
    ap.add_argument("--configs-root", type=Path, default=CONFIGS,
                    help="directory of deck configs (default: decks/configs)")
    ap.add_argument("--artifacts-root", type=Path, default=ARTIFACTS,
                    help="directory for generated tiles/renders (default: decks/artifacts)")
    sub = ap.add_subparsers(dest="command")

    # Shared flags declared ONCE. `cards` and `majors` are the same card with a
    # different content layer, so their labelling and medallion knobs must not
    # drift: they did, and `--medallion-scale` ended up a float on `majors` and
    # a string on `cards`, so the documented keywords (small/tiny) worked on one
    # and were rejected by argparse on the other. `_resolve_scale` takes both.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scale", type=int, default=2,
                        help="NEAREST zoom for the preview sheet only (default: 2)")
    carded = argparse.ArgumentParser(add_help=False)
    carded.add_argument("--no-labels", action="store_true",
                        help="render bare cards, without numeral/title labels")
    carded.add_argument("--medallion", choices=MEDALLION_STYLES, metavar="STYLE",
                        help="edge medallion: suit | lozenge | none (default: from deck.yaml)")
    carded.add_argument("--medallion-scale", metavar="SIZE",
                        help="medallion size: a keyword (full/large/small/smaller/tiny) "
                             "or a number (e.g. 0.5); default from deck.yaml")

    g = sub.add_parser("generate", parents=[common], help="render a deck's borders")
    g.add_argument("deck")

    c = sub.add_parser("cards", parents=[common, carded],
                       help="render minor-arcana pip cards (ranks 1-10)")
    c.add_argument("deck")
    c.add_argument("--layout", choices=layout.names(), metavar="NAME",
                   help="force one pip arrangement for every rank (default: per-rank from deck.yaml)")
    c.add_argument("--field", choices=field.names(), metavar="NAME",
                   help="force one field design for every suit (default: per-suit from deck.yaml)")
    grp = c.add_mutually_exclusive_group()
    grp.add_argument("--suit", help="render a single suit (default: all minor suits)")
    grp.add_argument("--all-suits", action="store_true", help="render every minor suit")
    m = sub.add_parser("majors", parents=[common, carded],
                       help="render the deck's face cards (the 22 majors by default)")
    m.add_argument("deck")
    m.add_argument("--face", metavar="KEY",
                   help="render one face by key (e.g. major_00); default: all")
    m.add_argument("--no-murals", action="store_true",
                   help="render the bare field even where face art exists")
    m.add_argument("--strict", action="store_true",
                   help="the print gate: fail if any face has only a placeholder")

    im = sub.add_parser("import-mural",
                        help="quantize an external RGB image into a face's mural layers")
    im.add_argument("deck")
    im.add_argument("png", type=Path, help="RGB/RGBA image, art-window sized")
    im.add_argument("--face", required=True, metavar="KEY",
                    help="face key, e.g. major_00 / court_cups_queen / wizard")
    im.add_argument("--tolerance", type=float, default=24.0, metavar="F",
                    help="max color distance before a pixel is off-palette (default: 24)")
    im.add_argument("--force", action="store_true",
                    help="snap off-palette pixels to their nearest slot instead of failing")
    im.add_argument("--out", type=Path, metavar="DIR",
                    help="write layers here instead of the deck's murals dir")

    ex = sub.add_parser("export-mural",
                        help="render a face's mural (art window only) to an RGB PNG")
    ex.add_argument("deck")
    ex.add_argument("--face", required=True, metavar="KEY",
                    help="face key, e.g. major_00 / court_cups_queen / wizard")
    ex.add_argument("--scale", type=int, default=1,
                    help="NEAREST zoom for the exported PNG (default: 1)")
    ex.add_argument("--out", type=Path, metavar="FILE",
                    help="output path (default: artifacts murals dir)")

    rd = sub.add_parser("rd", help="generate candidate art for a face (Retro Diffusion)")
    rd.add_argument("deck")
    rd.add_argument("--face", required=True, metavar="KEY",
                    help="face key, e.g. major_00")
    rd.add_argument("--init", type=Path, metavar="PATH",
                    help="seed image: an RWS scan (cropped and fitted automatically) "
                         "or an art-window-sized PNG used as-is. "
                         "Default: the face's current art or placeholder")
    rd.add_argument("--prompt", metavar="TEXT",
                    help="one-off prompt, overriding generation.yaml")
    rd.add_argument("--seed", type=int, metavar="N",
                    help="generation seed, for reproducibility")
    rd.add_argument("--check-cost", action="store_true",
                    help="dry run: build and send the request without generating")

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
                   args.medallion_scale, args.no_murals,
                   args.configs_root, args.artifacts_root,
                   face=args.face, strict=args.strict)
    elif args.command == "import-mural":
        cmd_import_mural(args.deck, args.png, args.face,
                         args.tolerance, args.force, args.out,
                         args.configs_root, args.artifacts_root)
    elif args.command == "export-mural":
        cmd_export_mural(args.deck, args.face, args.scale,
                         args.out, args.configs_root, args.artifacts_root)
    elif args.command == "rd":
        cmd_rd(args.deck, args.face, args.seed, args.prompt, args.init,
               args.check_cost, args.configs_root, args.artifacts_root)
    elif args.command == "seed":
        cmd_seed(args.deck, args.configs_root, args.artifacts_root)
    else:
        ap.print_help()
        return 1
    return 0
