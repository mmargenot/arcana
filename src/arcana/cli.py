"""
Thin CLI for arcana. The engine lives in the package; this just wires a deck's
config to the render pipeline and writes the results.

    arcana generate <deck> [--scale N]   seed placeholder tiles if missing,
                                         render per-suit borders
    arcana cards <deck> [--layout NAME]  render minor-arcana pip cards (ranks
        [--suit S | --all-suits]         1-10) with numeral/title labels;
        [--no-labels]                    --layout forces one everywhere,
                                         --no-labels renders bare
    arcana majors <deck> [--face KEY]    render the deck's face cards: mural +
        [--no-labels] [--no-murals]      frame + label. Committed art wins, a
        [--strict]                       placeholder fills in, --strict is the
                                         print gate
    arcana rd <deck> --face KEY          generate candidate art: fetch the RWS
        [--init PATH] [--prompt TEXT]    scan, seed from it, write candidates
        [--seed N]                       for a human to choose between
    arcana import-mural <deck> <png>     quantize external pixel art (Retro
        --face KEY [--force] [--bleed]   Diffusion output, edited exports...)
                                         into a face's committed mural layers,
                                         then render the card
    arcana export-mural <deck> --face K  render a face's mural (art window
                                         only) to RGB PNG, for external tools;
                                         the exact inverse of import-mural
    arcana seed <deck>                   (re)write placeholder tiles + faces

`--scale` is a NEAREST-neighbour zoom applied ONLY to the preview contact sheet,
so the tiny pixel-art cards are legible on screen. The individual PNGs are
always written at native resolution; nothing in the deck depends on it. For
print, set `print_scale` in deck.yaml — that is deck identity, not a flag.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from arcana.deck import load_deck, assets_dir, config_dir, render_dir, CONFIGS, ARTIFACTS
from arcana.elements import AssetError, Element, load_all, audit
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


CARD_INCHES = (2.75, 4.75)      # a true tarot card; the deck's 1:1.725 matches


def _write_print(rgb: np.ndarray, cfg: dict, geo, dest: Path) -> Path | None:
    """Write a print-resolution copy of a card, when the deck asks for one.

    `print_scale` in deck.yaml, not a CLI flag: a deck prints at one resolution,
    so it is deck identity, and typing it per card 22 times over would be the
    same tax the rest of this pipeline just shed.

    INTEGER NEAREST only. At 160x276 a card is 58 DPI at true size, so it must
    be scaled for print — and any fractional ratio or smooth filter destroys the
    pixel grid, which is the entire aesthetic. 6x is 960x1656, about 349 DPI;
    5x is about 291, just under the 300 most printers want. The physical size
    goes into the PNG's pDPI metadata so a shop sees 2.75x4.75in rather than
    guessing from the pixel count."""
    scale = cfg.get("print_scale")
    if not scale:
        return None
    scale = int(scale)
    img = Image.fromarray(rgb)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, dpi=(img.width / CARD_INCHES[0], img.height / CARD_INCHES[1]))
    return dest


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
        _write_print(rgb, cfg, geo, out / "print" / f"{key}.png")
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
                     configs_root: Path, artifacts_root: Path,
                     bleed: bool = False) -> Path:
    """Externally generated pixel art -> committed mural layers: quantize the
    RGB to the deck's 14 drawable colors, factor global indices into per-bank
    local layers (mural.split_global), and write the same dotted-stem ASCII a
    hand would author. The exact inverse of `export-mural`, so external art
    and authored art are indistinguishable on disk."""
    deck = load_deck(name, configs_root)
    pal, geo = deck.palette, deck.geometry
    from arcana.tileio import coherence, fidelity, quantize_rgb_global, write_ascii
    try:
        g = quantize_rgb_global(png, pal.for_suit("majors"), tolerance, force)
    except AssetError as e:
        raise SystemExit(str(e))
    fid = fidelity(png, pal.for_suit("majors"))
    sw, sh = mural.safe_size(geo)
    if g.shape not in ((geo.art_h, geo.art_w), (sh, sw)):
        raise SystemExit(
            f"{png.name} is {g.shape[1]}x{g.shape[0]}; expected the art window "
            f"{geo.art_w}x{geo.art_h} or the visible safe area {sw}x{sh} "
            f"(what `arcana rd` generates)")
    from arcana.retro import Generation
    gen = Generation.load(config_dir(name, configs_root) / "generation.yaml")
    # The mural's ground and the card's field must agree, or they meet at a
    # visible rectangle. `reground` moves the ground into the field bank at its
    # existing value rung — same shape, same values, right hue family, and it
    # recolours with the palette from then on. `knockout_ground` is the stronger
    # form for art that should carry no sky at all.
    g = mural.knockout_ground(g) if gen.knockout_ground else mural.reground(g)
    if not bleed:
        g = mural.fit_safe(g, geo)
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
    # Was this a TRANSLATION or an approximation? Both make a plausible card, so
    # the difference has to be measured: art already in the deck's colours lands
    # on its slots (exact ~1.0, few source colours), where anti-aliased or
    # off-palette art gets dragged there one small lie at a time.
    print(f"  translation: {fid['exact']:.0%} of pixels already on a deck colour, "
          f"{fid['colours']} source colours, snap mean {fid['mean']:.0f} / "
          f"p95 {fid['p95']:.0f} / max {fid['max']:.0f}")
    if fid["exact"] < 0.5 or fid["colours"] > 64:
        print("  ! this is an approximation, not a translation — the source is "
              "not pixel art in this palette. Check `input_palette` reached the "
              "generator, and prefer a smaller generation grid over upscaling")
    # Glyphable? Storing indices only buys portability if each bank's three
    # slots carry a VALUE ramp. Where they do, any palette on the same rungs
    # re-renders this art; where they do not, it looks right in this palette
    # only and inverts in the next one.
    coh = coherence(png, pal.for_suit("majors"))
    bad = [b for b, v in coh["banks"].items() if v["used"] > 1 and not v["ordered"]]
    # `ordered` is None when no bank spans two rungs -- flat art, by design. That
    # is not 0%: there is no ramp to get wrong.
    rungs = ("n/a, one rung per bank" if coh["ordered"] is None
             else f"{coh['ordered']:.0%} of banks hold dark<mid<light")
    print(f"  glyphable:   {coh['fragments_per_1k']:.1f} bank fragments/1000px, "
          f"{rungs}" + (f" — rungs broken in {', '.join(bad)}" if bad else ""))
    # 40, from measurement rather than taste. The score divides by bank area, so
    # a legitimately SMALL bank cannot approach zero however clean it is -- one
    # unbroken 88px ring scores 11.4. Real failures are an order up: an RWS scan
    # runs 205, uniform confetti 499, and even 90 loose 3x3 blobs 108. 40 clears
    # the ceiling for small tidy banks with room to spare and still sits well
    # under the cheapest genuine failure. Art with big regions runs under 1.
    if coh["fragments_per_1k"] > 40:
        print("  ! banks are scattered rather than regions, so this will not "
              "survive a palette swap — recolouring confetti, not shapes. Art "
              "built from the palette runs under 1")
    # The charter is about HUE FAMILIES, not slots: art must touch line, paper and
    # all four banks. An unused RUNG is not a fault — on deliberately flat art it
    # is the point, since the generator is handed two tones per bank precisely so
    # it cannot shade. Warning per slot would fire on every card we now want.
    #
    # Checked on the mural COMPOSED ON ITS FIELD, which is how the charter is
    # written (tests/test_mural.py: build_mural, then assert every bank). It
    # matters: the mat supplies the field bank, so art with a transparent ground
    # inherits teal without painting any, and demanding teal of the emblem itself
    # would mean writing water into twenty cards that have none. Border, motif
    # and figure have no such donor and must come from the art.
    from arcana.palette import BANKS as _BANKS
    from arcana.compose import build_mural as _bm
    el = Element(name=face, role="mural", size=g.shape, layers=layers)
    composed = _bm(pal, geo, {}, field.field_for_suit(deck.config, "majors", None),
                   image=el)
    ox, oy = geo.art_origin
    seen = set(np.unique(composed[oy:oy + geo.art_h, ox:ox + geo.art_w]).tolist())
    bare = [b for i, b in enumerate(_BANKS)
            if not seen & {3 + 3 * i, 4 + 3 * i, 5 + 3 * i}]
    if bare:
        print(f"  ! unused BANK: {', '.join(bare)} — a hue family missing from "
              f"this card entirely")
    idle = [lab for lab, n in zip(labels[1:], counts[1:]) if not n]
    if idle:
        print(f"    (rungs unused: {', '.join(idle)} — expected on flat art)")
    ink = mural.margin_ink(g, geo)
    if ink > 0.12:
        print(f"  ! {ink:.0%} of the covered margin is line work — the generator "
              f"may have drawn its own frame, which the deck's frame will "
              f"collide with rather than cover")
    # Render the card immediately. There is no workflow where you import art and
    # deliberately do not look at the result, so making it a second command just
    # charges the same tax 22 times. `--out` means a probe into a scratch dir,
    # not the deck, so there is nothing to render then.
    if out is None:
        cmd_majors(name, 2, False, None, None, False,
                   configs_root, artifacts_root, face=face)
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
           init: Path | None, configs_root: Path, artifacts_root: Path,
           no_init: bool = False) -> Path:
    """Generate candidate art for one face and write it for review.

    The seed image is the point. With no `--init` the deck's source scan is
    FETCHED — the Commons path is computable from the face key, so there is no
    reason to make you run curl first — then cropped and fitted. `--init` takes
    a scan you supply at any size, or an already-fitted PNG used as-is.

    Nothing is imported automatically: choosing a candidate is the one step in
    this pipeline that needs a human, so `import-mural` stays deliberate."""
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

    out = render_dir(name, artifacts_root) / "rd"
    out.mkdir(parents=True, exist_ok=True)
    init_bytes = None
    if no_init:
        print(f"no seed: generating from the prompt alone with {gen.seedless_style}")
    elif init is None:
        try:
            init = retro.fetch_scan(
                face, render_dir(name, artifacts_root) / "scans" /
                (retro.RWS_FILES.get(face) or f"{face}.jpg"))
        except retro.GenerationError as e:
            raise SystemExit(str(e))
        print(f"source scan: {init}")
    elif not init.exists():
        raise SystemExit(f"no seed image at {init}")

    if not no_init:
        from arcana.pixelate import pixelate
        sw, sh = mural.safe_size(geo)
        img = Image.open(init)
        if img.size != (sw, sh):
            print(f"preparing seed: {init.name} {img.width}x{img.height} -> {sw}x{sh}")
            img = Image.fromarray(pixelate(init, sw, sh))
        seed_png = out / f"{face}.seed.png"
        img.convert("RGB").save(seed_png)
        init_bytes = seed_png.read_bytes()
        print(f"seed -> {seed_png}")

    payload = retro.build_payload(gen, geo, pal.for_suit("majors"), prompt=text,
                                  seed=seed, init=init_bytes)
    style = gen.style if init_bytes else gen.seedless_style
    sw, sh = mural.safe_size(geo)
    print(f"style {style}  {sw}x{sh}  x{gen.candidates}"
          + (f"  strength {gen.strength}" if init_bytes else ""))
    try:
        response = retro.request(payload)
    except retro.GenerationError as e:
        raise SystemExit(str(e))
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
    im.add_argument("--bleed", action="store_true",
                    help="fill the whole art window instead of fitting the art to the "
                         "visible safe rectangle (see field.insets)")
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
    seedgrp = rd.add_mutually_exclusive_group()
    seedgrp.add_argument("--init", type=Path, metavar="PATH",
                         help="seed image: a scan at any size (cropped and fitted) or an "
                              "already-fitted PNG. Default: fetch the face's source scan")
    seedgrp.add_argument("--no-init", action="store_true",
                         help="generate from the prompt alone, with no seed image "
                              "(uses generation.yaml's seedless_style)")
    rd.add_argument("--prompt", metavar="TEXT",
                    help="one-off prompt, overriding generation.yaml")
    rd.add_argument("--seed", type=int, metavar="N",
                    help="generation seed, for reproducibility")

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
                         args.configs_root, args.artifacts_root, args.bleed)
    elif args.command == "export-mural":
        cmd_export_mural(args.deck, args.face, args.scale,
                         args.out, args.configs_root, args.artifacts_root)
    elif args.command == "rd":
        cmd_rd(args.deck, args.face, args.seed, args.prompt, args.init,
               args.configs_root, args.artifacts_root, args.no_init)
    elif args.command == "seed":
        cmd_seed(args.deck, args.configs_root, args.artifacts_root)
    else:
        ap.print_help()
        return 1
    return 0
