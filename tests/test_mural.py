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
FIXTURE_KEY = mural.major_key(FIXTURE_N)


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
    """A deterministic art-window mural meeting the majors' charter: a LINE
    frame, a PAPER strip, and a dark/mid/light band from every bank — all kept
    inside `field.insets`.

    This is `seed.placeholder_face`, imported rather than duplicated: the image
    a deck seeds for an un-drawn face and the image these tests assert against
    are the same contract, so there must be exactly one generator. If the
    charter changes, both move together or neither does."""
    _, geo, _, _ = ctx
    g = seed.placeholder_face(geo)
    return Element(name="fixture.mural", role="mural", size=g.shape,
                   layers=mural.split_global(g))


@pytest.fixture(scope="session")
def murals_dir(tmp_path_factory, fixture_image) -> Path:
    """The fixture written to disk in the committed layout, so loader tests go
    through the real file path (dotted stems, per-bank probing)."""
    d = tmp_path_factory.mktemp("murals")
    s = FIXTURE_KEY
    for bank, layer in fixture_image.layers.items():
        tileio.write_ascii(layer, d / f"{s}.{bank}.txt", name=f"{s}.{bank}")
    return d


# --- loader + authored-format contract ----------------------------------
def test_loader_round_trips_written_layers(ctx, murals_dir, fixture_image):
    """write_ascii -> load_mural reproduces the layers exactly: the on-disk
    ASCII format carries a mural without loss."""
    _, geo, _, _ = ctx
    el = mural.load_mural(murals_dir, FIXTURE_KEY, geo, required=True)
    assert el is not None and el.role == "mural"
    assert set(el.layers) == set(fixture_image.layers)
    for bank, a in el.layers.items():
        assert bank in BANKS
        assert a.shape == (geo.art_h, geo.art_w)
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
    ox, oy = geo.art_origin
    window = m[oy:oy + geo.art_h, ox:ox + geo.art_w]
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


def test_image_respects_field_insets(ctx, fixture_image):
    """Foreground discipline: the image stays inside `field.insets`, where the
    frame band (and a full-scale top medallion) can never touch it. Content in
    the overlap ring is invisible on screen and clipped in print."""
    pal, geo, _, _ = ctx
    bound = fixture_image.bind(pal)
    ix, iy = field.insets(geo)
    assert (bound[:iy, :] == 0).all() and (bound[geo.art_h - iy:, :] == 0).all()
    assert (bound[:, :ix] == 0).all() and (bound[:, geo.art_w - ix:] == 0).all()


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


def test_mural_only_touches_art_window(ctx, fixture_image):
    """A mural changes nothing outside the art window: frame, bands, and label
    are identical with and without the image."""
    pal, geo, cfg, els = ctx
    from arcana.text import Font
    font = seed.placeholder_font()
    kw = dict(top="VII", bottom="THE CHARIOT", field_design="plain",
              pip_key=cfg["suit_pips"]["majors"], med_style="suit", med_scale=0.5)
    bare = compose.build_major_card(pal, geo, els, font, **kw)
    with_ = compose.build_major_card(pal, geo, els, font, image=fixture_image, **kw)
    ox, oy = geo.art_origin
    mask = np.ones_like(bare, bool)
    mask[oy:oy + geo.art_h, ox:ox + geo.art_w] = False
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
    assert mural.load_mural(tmp_path, FIXTURE_KEY, geo) is None
    with pytest.raises(AssetError, match="major_07"):
        mural.load_mural(tmp_path, FIXTURE_KEY, geo, required=True)


def test_has_murals_is_presence(murals_dir, tmp_path):
    """Presence of any layer file for a known key — no config knob."""
    assert mural.has_murals(murals_dir, (FIXTURE_KEY,))
    assert not mural.has_murals(tmp_path, (FIXTURE_KEY,))            # empty dir
    assert not mural.has_murals(tmp_path / "absent", (FIXTURE_KEY,))  # no dir


def test_committed_art_beats_placeholder(ctx, murals_dir, fixture_image, tmp_path):
    """Committed art always wins over a seeded placeholder, and a face with no
    committed art still loads. This is what replaced all-or-nothing: the deck
    renders while art lands one card at a time."""
    pal, geo, _, _ = ctx
    fallback = tmp_path / "assets"
    seed.seed_faces(fallback, geo, [FIXTURE_KEY, "major_00"])
    el = mural.load_mural(murals_dir, FIXTURE_KEY, geo,
                          fallback_dir=fallback / "murals")
    assert np.array_equal(el.bind(pal), fixture_image.bind(pal))   # committed
    assert mural.load_mural(murals_dir, "major_00", geo,
                            fallback_dir=fallback / "murals") is not None
    assert mural.is_committed(murals_dir, FIXTURE_KEY)
    assert not mural.is_committed(murals_dir, "major_00")


def test_strict_is_the_print_gate(cli_configs_root, tmp_path):
    """`--strict` is where all-or-nothing's intent now lives: a deck may render
    with placeholders, but it must not go to print with one. The error names
    every face that is still a placeholder."""
    from arcana.cli import main
    roots = ["--configs-root", str(cli_configs_root),
             "--artifacts-root", str(tmp_path / "artifacts")]
    assert main(roots + ["majors", "vaporwave-rws"]) == 0     # renders happily
    with pytest.raises(SystemExit, match="major_00"):
        main(roots + ["majors", "vaporwave-rws", "--strict"])


def test_face_keys_are_free_form(ctx):
    """The seam is not tarot-specific: a deck declares its own face set, so
    JQKA courts and one-off specials use the identical path."""
    _, geo, _, _ = ctx
    assert mural.face_keys() == mural.MAJOR_KEYS and len(mural.MAJOR_KEYS) == 22
    keys = mural.face_keys({"faces": ["court_cups_queen", "wizard"]})
    assert keys == ("court_cups_queen", "wizard")
    g = seed.placeholder_face(geo, "wizard")
    assert sorted(mural.split_global(g)) == sorted(BANKS)


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
    """The image composed onto the field, cropped to the art window — exactly
    what export-mural renders."""
    pal, geo, cfg, els = ctx
    m = compose.build_mural(pal, geo, els,
                            field.field_for_suit(cfg, "majors", None),
                            image=image)
    ox, oy = geo.art_origin
    return m[oy:oy + geo.art_h, ox:ox + geo.art_w]


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
    assert snapped.shape == (geo.art_h, geo.art_w)
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
    s = FIXTURE_KEY
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
                       "--face", FIXTURE_KEY, "--out", str(png)])
    assert rc == 0 and png.exists()
    dest = tmp_path / "imported"
    rc = main(roots + ["import-mural", "vaporwave-rws", str(png),
                       "--face", FIXTURE_KEY, "--out", str(dest)])
    assert rc == 0
    pal, geo, _, _ = ctx
    el = mural.load_mural(dest, FIXTURE_KEY, geo)
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
    s = FIXTURE_KEY
    dest.mkdir()
    (dest / f"{s}.motif.txt").write_text("# stale\n.\n")   # must be cleared
    rc = main(["--configs-root", str(cli_configs_root),
               "--artifacts-root", str(tmp_path / "artifacts"),
               "import-mural", "vaporwave-rws", str(png),
               "--face", FIXTURE_KEY, "--out", str(dest)])
    assert rc == 0
    assert not (dest / f"{s}.motif.txt").exists()
    assert (dest / f"{s}.field.txt").exists()


def test_margin_ink_flags_a_drawn_frame(ctx):
    """Border cruft cannot be prompted away (the API ignores `negative`), so
    import warns instead. Background in the covered margin is normal; a RING
    of line work there means the generator drew its own frame, which will
    collide with the deck's rather than sit under it."""
    _, geo, _, _ = ctx
    from arcana import field
    ix, iy = field.insets(geo)
    clean = np.zeros((geo.art_h, geo.art_w), np.uint8)
    clean[iy:-iy, ix:-ix] = LINE                      # art fills the safe area
    assert mural.margin_ink(clean, geo) == 0.0

    framed = np.zeros((geo.art_h, geo.art_w), np.uint8)
    framed[:] = LINE
    framed[4:-4, 4:-4] = 0                            # a drawn border ring
    assert mural.margin_ink(framed, geo) > 0.12


def test_fit_safe_seats_overflowing_art_without_cropping(ctx):
    """Generated art fills its whole canvas, so it overflows the rectangle the
    frame leaves visible. Seat it by SCALING, never by clearing the margin:
    on the Fool that ring holds his feet and the sun, and cropping amputates
    the composition."""
    _, geo, _, _ = ctx
    full = np.full((geo.art_h, geo.art_w), LINE, np.uint8)
    fitted = mural.fit_safe(full, geo)
    sw, sh = mural.safe_size(geo)
    assert fitted.shape == (geo.art_h, geo.art_w)
    assert mural.fits_safe(fitted, geo)
    kept = int((fitted != 0).sum())
    assert kept > 0.8 * sw * sh          # scaled in, not cropped away
    assert mural.margin_ink(fitted, geo) == 0.0


def test_fit_safe_leaves_conforming_art_untouched(ctx, fixture_image):
    """The export/import round trip is documented as lossless, so art already
    inside the insets must pass through bit-identical — rescaling it would be
    destructive and would silently break that guarantee."""
    pal, geo, _, _ = ctx
    g = fixture_image.bind(pal)
    assert mural.fits_safe(g, geo)
    assert np.array_equal(mural.fit_safe(g, geo), g)
    # and the same once the field background is baked in, which is what
    # export-mural writes and what a re-import therefore sees
    baked = g.copy()
    baked[baked == 0] = 3 + 3 * BANKS.index("field") + 1
    assert mural.fits_safe(baked, geo)
    assert np.array_equal(mural.fit_safe(baked, geo), baked)


def test_knockout_ground_spares_the_figure(ctx):
    """The 69.9% bug, pinned. A naive flood from the border follows LINE inward
    — the figure's outline is contiguous with the border's line pixels — and
    dissolves the figure. Ground goes; ink stays."""
    _, geo, _, _ = ctx
    sky = 3 + 3 * BANKS.index("field") + 2
    g = np.full((geo.art_h, geo.art_w), sky, np.uint8)
    g[40:180, 30:110] = LINE                      # a figure, touching nothing
    g[0, :] = LINE                                # ink ON the border, joined to it
    g[0:41, 60:62] = LINE                         # ...and contiguous with the figure
    out = mural.knockout_ground(g)
    assert (out == 0).sum() > 0.4 * g.size        # the sky went
    assert (out[40:180, 30:110] == LINE).all()    # the figure did not


def test_knockout_ground_ignores_small_edge_regions(ctx):
    """Only a LARGE edge-touching region is ground. A shadow or a staff that
    happens to reach the border is composition, not sky."""
    _, geo, _, _ = ctx
    sky = 3 + 3 * BANKS.index("field") + 2
    speck = 3 + 3 * BANKS.index("motif") + 1
    g = np.full((geo.art_h, geo.art_w), sky, np.uint8)
    g[0:6, 0:6] = speck                           # tiny, touches the corner
    out = mural.knockout_ground(g)
    assert (out[0:6, 0:6] == speck).all()
    assert (out[100, 70] == 0)                    # the sky still went


def test_reground_moves_the_sky_to_the_field_bank_at_its_rung(ctx):
    """Background is a BANK, not a colour. A quantiser that files the sky under
    `figure` (a warm source sky lands in the flesh ramp) puts the mural's ground
    in a different hue family from the card's field mat, and they meet at a
    visible rectangle. Reground fixes it in bank space — and the value rung
    survives, because every bank is held to the same rungs."""
    _, geo, _, _ = ctx
    fig = 3 + 3 * BANKS.index("figure")
    fld = 3 + 3 * BANKS.index("field")
    g = np.full((geo.art_h, geo.art_w), fig + 2, np.uint8)      # figure.light sky
    g[60:160, 40:100] = LINE                                     # a figure
    out = mural.reground(g)
    assert (out[0, 0], out[-1, -1]) == (fld + 2, fld + 2)        # -> field.light
    assert (out[60:160, 40:100] == LINE).all()                   # figure untouched


def test_reground_is_idempotent_and_keeps_dark_ground_dark(ctx):
    """Art whose ground is already field-bank is left alone, and a dark ground
    stays dark — the rung is preserved, not normalised to one slot."""
    _, geo, _, _ = ctx
    fld = 3 + 3 * BANKS.index("field")
    dark = np.full((geo.art_h, geo.art_w), fld, np.uint8)        # field.dark night
    assert np.array_equal(mural.reground(dark), dark)

    mot = 3 + 3 * BANKS.index("motif")
    g = np.full((geo.art_h, geo.art_w), mot, np.uint8)           # motif.DARK ground
    once = mural.reground(g)
    assert (once == fld).all()                                   # -> field.DARK
    assert np.array_equal(mural.reground(once), once)


def test_reground_tints_rather_than_erases_on_a_leak(ctx):
    """The reason reground is preferred over knockout. If the flood leaks
    through a gap in the outline it recolours the figure — visible at a glance —
    where a knockout would silently delete it."""
    _, geo, _, _ = ctx
    fig = 3 + 3 * BANKS.index("figure")
    g = np.full((geo.art_h, geo.art_w), fig + 2, np.uint8)
    g[60:160, 40:100] = LINE
    g[100:104, 40:100] = fig + 2                  # a gap: sky leaks into the figure
    leaked = (mural.reground(g)[100:104, 40:100] != 0).all()
    erased = (mural.knockout_ground(g)[100:104, 40:100] == 0).all()
    assert leaked and erased
