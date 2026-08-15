"""Mural tooling regression tests. Every one guards a bug that is invisible at 1x.

No deck currently commits mural art, so these tests exercise the PIPELINE —
loader, compose seam, split/bind factoring, and the bidirectional pixel<->ASCII
path — against a synthetic fixture mural written to a tmp dir in the exact
committed format (`murals/major_NN.<bank>.txt`). The fixture is built
to the majors' charter (all four banks plus LINE/PAPER in play), so it also
documents the contract future art must meet. Elements and the font come from
seeded placeholders, as in test_deck.
"""
from pathlib import Path

import numpy as np
import pytest

from arcana import compose, field, mural, seed, tileio
from arcana.elements import AssetError, Element, load_all
from arcana.geometry import Geometry, load_config
from arcana.palette import BANKS, LINE, MAX_LOCAL, PAPER, Palette

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "decks" / "configs" / "vaporwave-rws"

FIXTURE_N = 7          # arbitrary major number the fixture mural is filed under


@pytest.fixture(scope="session")
def ctx(tmp_path_factory):
    assets = tmp_path_factory.mktemp("assets")
    seed.seed_deck(assets)
    pal = Palette.load(CONFIG / "palette.yaml")
    geo = Geometry.load(CONFIG / "deck.yaml")
    cfg = load_config(CONFIG / "deck.yaml")
    return pal, geo, cfg, load_all(assets, cfg["elements"])


@pytest.fixture(scope="session")
def fixture_image(ctx) -> Element:
    """A deterministic image-box mural meeting the majors' charter: a LINE
    frame, a PAPER strip, and a dark/mid/light band from every bank — all kept
    a `margin` inside the box (the uniform foreground inset). Built in GLOBAL
    index space and factored through split_global, so the fixture exercises
    the same layer format a hand (or import-mural) would produce."""
    _, geo, _, _ = ctx
    _, _, bw, bh = mural.image_box(geo)
    g = np.zeros((bh, bw), np.uint8)
    ix = iy = geo.margin
    top, bot = iy + 12, bh - iy - 12
    left, right = ix + 8, bw - ix - 8
    g[top:bot, left:right] = LINE                 # frame, then carve interior
    inner = g[top + 2:bot - 2, left + 2:right - 2]
    inner[:4, :] = PAPER
    body = inner[4:]
    h = body.shape[0] // len(BANKS)
    for i in range(len(BANKS)):                   # one band per bank
        base = 3 + 3 * i
        band = body[i * h:(i + 1) * h]
        w3 = band.shape[1] // 3
        band[:, :w3] = base                       # dark
        band[:, w3:2 * w3] = base + 1             # mid
        band[:, 2 * w3:] = base + 2               # light
    return Element(name="fixture.mural", role="mural", size=g.shape,
                   layers=mural.split_global(g))


@pytest.fixture(scope="session")
def murals_dir(tmp_path_factory, fixture_image) -> Path:
    """The fixture written to disk in the committed layout, so loader tests go
    through the real file path (dotted stems, per-bank probing)."""
    d = tmp_path_factory.mktemp("murals")
    s = mural.stem(FIXTURE_N)
    for bank, layer in fixture_image.layers.items():
        tileio.write_ascii(layer, d / f"{s}.{bank}.txt", name=f"{s}.{bank}")
    return d


# --- loader + authored-format contract ----------------------------------
def test_loader_round_trips_written_layers(ctx, murals_dir, fixture_image):
    """write_ascii -> load_mural reproduces the layers exactly: the on-disk
    ASCII format carries a mural without loss."""
    _, geo, _, _ = ctx
    el = mural.load_mural(murals_dir, FIXTURE_N, geo, required=True)
    assert el is not None and el.role == "mural"
    assert set(el.layers) == set(fixture_image.layers)
    _, _, bw, bh = mural.image_box(geo)
    for bank, a in el.layers.items():
        assert bank in BANKS
        assert a.shape == (bh, bw)
        assert a.dtype == np.uint8 and int(a.max(initial=0)) <= MAX_LOCAL
        assert np.array_equal(a, fixture_image.layers[bank])


def test_fixture_meets_the_majors_charter(ctx, fixture_image):
    """The majors' charter, encoded: a mural composed on the field must touch
    LINE, PAPER, and every bank — all 14 drawable colours in play, not a
    3-colour subset. The fixture passes; future committed art must too."""
    pal, geo, cfg, els = ctx
    m = compose.build_mural(pal, geo, els,
                            field.field_for_suit(cfg, "majors", None),
                            image=fixture_image)
    bx, by, bw, bh = mural.image_box(geo)
    window = m[by:by + bh, bx:bx + bw]
    present = set(np.unique(window).tolist())
    assert LINE in present and PAPER in present
    for i, bank in enumerate(BANKS):
        base = 3 + 3 * i
        assert present & {base, base + 1, base + 2}, f"{bank} bank unused"


def test_mural_indices_are_suit_invariant(ctx, fixture_image):
    """The wrong-LUT bug, extended to murals: colour comes from the LUT at
    render, never from the index matrix."""
    pal, geo, cfg, els = ctx
    fname = field.field_for_suit(cfg, "majors", None)
    base = compose.build_mural(pal, geo, els, fname, image=fixture_image)
    for suit in cfg["suit_pips"]:
        other = compose.build_mural(pal.for_suit(suit), geo, els, fname,
                                    image=fixture_image)
        assert np.array_equal(base, other)


def test_image_respects_foreground_inset(ctx, fixture_image):
    """Foreground discipline: the image keeps a uniform `margin` clear inside
    the box, where the frame band (and a full-scale top medallion) can never
    touch it. The box already sits clear of the side bands, so the inset is
    even on all four sides."""
    pal, geo, _, _ = ctx
    bound = fixture_image.bind(pal)
    _, _, bw, bh = mural.image_box(geo)
    g = geo.margin
    assert (bound[:g, :] == 0).all() and (bound[bh - g:, :] == 0).all()
    assert (bound[:, :g] == 0).all() and (bound[:, bw - g:] == 0).all()


# --- compose contract ---------------------------------------------------
def test_no_image_is_the_bare_full_bleed_field(ctx):
    """`image=None` (and `--no-murals`) is exactly the field and nothing
    else — this is the default today: no deck commits mural art. The field
    runs FULL-BLEED at the SAME depth on all four sides: from `geo.margin` in,
    under the frame band's repeating ornament top and bottom just like the
    sides always were, so the title floats on the field, never on bare paper.
    The asymmetry this guards: a field starting at the art top ends before the
    top band while still running under the side bands."""
    pal, geo, cfg, els = ctx
    fname = field.field_for_suit(cfg, "majors", None)
    m = compose.build_mural(pal, geo, els, fname)
    ox, oy = geo.art_origin
    g = geo.margin
    expected = np.zeros((geo.card_h, geo.card_w), np.uint8)
    compose.paste(expected,
                  pal.bind(compose.build_field(fname, geo, full_bleed=True),
                           "field"), g, g)
    assert np.array_equal(m, expected)
    assert (m[:g, :] == 0).all() and (m[geo.card_h - g:, :] == 0).all()
    assert (m[:, :g] == 0).all() and (m[:, geo.card_w - g:] == 0).all()
    # the same field depth under the band on every side...
    assert (m[g, ox:ox + geo.art_w] != 0).all()          # under the top band
    assert (m[geo.card_h - g - 1, ox:ox + geo.art_w] != 0).all()  # bottom band
    title_row = oy + geo.art_h + 1                       # inside the title band
    assert (m[title_row, ox:ox + geo.art_w] != 0).all()  # field runs under it


def test_mural_only_touches_image_box(ctx, fixture_image):
    """A mural changes nothing outside the image box — in particular the side
    strips under the frame band, where the wider art window would have let it
    leak: frame, bands, and label are identical with and without the image."""
    pal, geo, cfg, els = ctx
    from arcana.text import Font
    font = seed.placeholder_font()
    kw = dict(top="VII", bottom="THE CHARIOT", field_design="plain",
              pip_key=cfg["suit_pips"]["majors"], med_style="suit", med_scale=0.5)
    bare = compose.build_major_card(pal, geo, els, font, **kw)
    with_ = compose.build_major_card(pal, geo, els, font, image=fixture_image, **kw)
    bx, by, bw, bh = mural.image_box(geo)
    mask = np.ones_like(bare, bool)
    mask[by:by + bh, bx:bx + bw] = False
    assert np.array_equal(bare[mask], with_[mask])
    assert isinstance(font, Font)


def test_major_card_keeps_outer_rule(ctx, fixture_image):
    """The outer LINE ring survives a mural on all four sides."""
    pal, geo, cfg, els = ctx
    font = seed.placeholder_font()
    m = compose.build_major_card(pal, geo, els, font, top="VII",
                                 bottom="THE CHARIOT", image=fixture_image,
                                 pip_key=cfg["suit_pips"]["majors"])
    W, H = geo.card_w, geo.card_h
    assert (m[0, :] == LINE).all() and (m[H - 1, :] == LINE).all()
    assert (m[:, 0] == LINE).all() and (m[:, W - 1] == LINE).all()


# --- opt-in semantics ---------------------------------------------------
def test_missing_mural_errors_when_required(ctx, tmp_path):
    """A required mural must fail loudly, naming the expected files — never a
    silently bare card."""
    _, geo, _, _ = ctx
    assert mural.load_mural(tmp_path, 7, geo) is None
    with pytest.raises(AssetError, match="major_07"):
        mural.load_mural(tmp_path, 7, geo, required=True)


def test_has_murals_is_presence(murals_dir, tmp_path):
    """Presence of any layer file is the opt-in switch — no config knob."""
    assert mural.has_murals(murals_dir)
    assert not mural.has_murals(tmp_path)            # empty dir
    assert not mural.has_murals(tmp_path / "absent")  # no dir at all


def test_partial_mural_set_fails_loudly(cli_configs_root, tmp_path):
    """All-or-nothing: once a deck ships ANY mural layers, `arcana majors`
    requires every major — a partial set must name the first missing stem, not
    print a deck with silently bare cards mixed in."""
    from arcana.cli import main
    with pytest.raises(SystemExit, match="major_00"):
        main(["--configs-root", str(cli_configs_root),
              "--artifacts-root", str(tmp_path / "artifacts"),
              "majors", "vaporwave-rws"])


# --- split_global (the import path's writer) ----------------------------
def test_split_global_round_trips_bind(ctx, fixture_image):
    """split_global must be the exact inverse of Element.bind: factor a
    composed mural into layers, bind them again, and get the same matrix —
    the guarantee that imported art is indistinguishable from authored art."""
    pal, _, _, _ = ctx
    g = fixture_image.bind(pal)
    again = Element(name="x", role="mural", size=fixture_image.size,
                    layers=mural.split_global(g)).bind(pal)
    assert np.array_equal(g, again)


def test_split_global_puts_universals_in_figure():
    """LINE/PAPER pixels land in the `figure` layer by convention — a
    deterministic home keeps the ASCII files stable across re-imports."""
    g = np.zeros((4, 4), np.uint8)
    g[0, 0], g[1, 1], g[2, 2] = LINE, PAPER, 7      # field mid
    layers = mural.split_global(g)
    assert set(layers) == {"field", "figure"}
    assert layers["figure"][0, 0] == LINE and layers["figure"][1, 1] == PAPER
    assert layers["field"][2, 2] == 4                # local MID


def test_split_global_rejects_out_of_range():
    with pytest.raises(AssetError, match="15"):
        mural.split_global(np.full((2, 2), 15, np.uint8))


# --- the bidirectional pixel <-> ASCII path -----------------------------
def _window(ctx, image):
    """The image composed onto the field, cropped to the image box — exactly
    what export-mural renders."""
    pal, geo, cfg, els = ctx
    m = compose.build_mural(pal, geo, els,
                            field.field_for_suit(cfg, "majors", None),
                            image=image)
    bx, by, bw, bh = mural.image_box(geo)
    return m[by:by + bh, bx:bx + bw]


def test_pixel_ascii_round_trip_is_lossless(ctx, fixture_image, tmp_path):
    """The whole point of the pair: ASCII map -> pixels (export) -> ASCII map
    (import) must be index-identical under a matching palette, so external
    tools can edit a mural without degrading it. Offline by construction."""
    from PIL import Image
    from arcana.tileio import quantize_rgb_global
    pal, _, _, _ = ctx
    window = _window(ctx, fixture_image)
    png = tmp_path / "window.png"
    Image.fromarray(pal.for_suit("majors").render(window)).save(png)
    back = quantize_rgb_global(png, pal.for_suit("majors"))
    assert np.array_equal(window, back)
    # and the import path's layer factoring recomposes to the same window
    el = Element(name="x", role="mural", size=back.shape,
                 layers=mural.split_global(back))
    assert np.array_equal(el.bind(pal), back)


def test_import_rejects_off_palette_pixels(ctx, fixture_image, tmp_path):
    """An unquantized source (anti-aliasing, a raw generation) must fail
    loudly, naming the count and the first bad pixel — never silently become
    the wrong slot. --force opts into deterministic snapping instead."""
    from PIL import Image
    from arcana.tileio import quantize_rgb_global
    pal, geo, _, _ = ctx
    rgb = pal.for_suit("majors").render(_window(ctx, fixture_image))
    rgb[10, 10] = (17, 250, 30)                  # one alien green pixel
    png = tmp_path / "poked.png"
    Image.fromarray(rgb).save(png)
    with pytest.raises(AssetError, match=r"1 pixel\(s\).*\(10,10\)"):
        quantize_rgb_global(png, pal.for_suit("majors"))
    snapped = quantize_rgb_global(png, pal.for_suit("majors"), force=True)
    _, _, bw, bh = mural.image_box(geo)
    assert snapped.shape == (bh, bw)
    assert 0 < int(snapped[10, 10]) <= 14        # snapped to a real slot


@pytest.fixture()
def cli_configs_root(tmp_path, fixture_image):
    """A throwaway configs root holding a copy of the real deck config plus
    the fixture mural — so the CLI commands run end-to-end without any
    committed mural art existing in the repo."""
    import shutil
    root = tmp_path / "configs"
    deck = root / "vaporwave-rws"
    deck.mkdir(parents=True)
    shutil.copy(CONFIG / "palette.yaml", deck / "palette.yaml")
    shutil.copy(CONFIG / "deck.yaml", deck / "deck.yaml")
    d = deck / "murals"
    s = mural.stem(FIXTURE_N)
    for bank, layer in fixture_image.layers.items():
        tileio.write_ascii(layer, d / f"{s}.{bank}.txt")
    return root


def test_import_export_cli_round_trip(ctx, fixture_image, cli_configs_root, tmp_path):
    """The commands themselves: export-mural a card, import-mural the PNG into
    a fresh dir, and the re-loaded mural composes to the identical window."""
    from arcana.cli import main
    png = tmp_path / "out.png"
    roots = ["--configs-root", str(cli_configs_root),
             "--artifacts-root", str(tmp_path / "artifacts")]
    rc = main(roots + ["export-mural", "vaporwave-rws",
                       "--major", str(FIXTURE_N), "--out", str(png)])
    assert rc == 0 and png.exists()
    dest = tmp_path / "imported"
    rc = main(roots + ["import-mural", "vaporwave-rws", str(png),
                       "--major", str(FIXTURE_N), "--out", str(dest)])
    assert rc == 0
    pal, geo, _, _ = ctx
    el = mural.load_mural(dest, FIXTURE_N, geo)
    assert el is not None
    assert np.array_equal(el.bind(pal), _window(ctx, fixture_image))


def test_import_clears_stale_layers(ctx, cli_configs_root, tmp_path):
    """The stale-layer bug: re-importing a stem must delete its old layer
    files, not just overwrite the ones it rewrites — a bank dropped between
    imports would otherwise linger and silently cover the new art. A pure
    field-colour PNG imports to a field layer only, so a pre-existing motif
    layer must be GONE afterwards."""
    from PIL import Image
    from arcana.cli import main
    pal, geo, cfg, _ = ctx
    fname = field.field_for_suit(cfg, "majors", None)
    window = np.zeros((geo.art_h, geo.art_w), np.uint8)
    compose.paste(window, pal.bind(compose.build_field(fname, geo), "field"), 0, 0)
    png = tmp_path / "field-only.png"
    Image.fromarray(pal.for_suit("majors").render(window)).save(png)
    dest = tmp_path / "imported"
    s = mural.stem(FIXTURE_N)
    dest.mkdir()
    (dest / f"{s}.motif.txt").write_text("# stale\n.\n")   # must be cleared
    rc = main(["--configs-root", str(cli_configs_root),
               "--artifacts-root", str(tmp_path / "artifacts"),
               "import-mural", "vaporwave-rws", str(png),
               "--major", str(FIXTURE_N), "--out", str(dest)])
    assert rc == 0
    assert not (dest / f"{s}.motif.txt").exists()
    assert (dest / f"{s}.field.txt").exists()
