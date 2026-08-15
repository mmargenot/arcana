"""Generation-seam tests. No test makes a network call.

The payload is where the expensive mistakes live: a wrong dimension wastes a
run and fails on import, and prompt expansion quietly rewrites the subject.
`retro.build_payload` is pure so all of that is assertable without stubbing an
HTTP layer — the only thing a stub would prove is that `urllib` works.
"""
from pathlib import Path

import base64
import io

import numpy as np
import pytest
from PIL import Image

from arcana import retro
from arcana.elements import AssetError
from arcana.geometry import Geometry
from arcana.palette import Palette

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "decks" / "configs" / "vaporwave-rws"


@pytest.fixture(scope="session")
def ctx():
    pal = Palette.load(CONFIG / "palette.yaml").for_suit("majors")
    geo = Geometry.load(CONFIG / "deck.yaml")
    gen = retro.Generation.load(CONFIG / "generation.yaml")
    return pal, geo, gen


def payload(ctx, **kw):
    pal, geo, gen = ctx
    return retro.build_payload(gen, geo, pal, prompt="a figure", **kw)


def test_generates_at_the_visible_safe_size(ctx):
    """Never literals, and never the full art window: the frame band overlaps
    that window, so a full-window generation spends a quarter of its pixels on
    a ring that is clipped in print -- and gives a model-drawn border a place
    to land exactly where the deck's frame is."""
    from arcana.mural import safe_size
    _, geo, _ = ctx
    p = payload(ctx)
    assert (p["width"], p["height"]) == safe_size(geo)
    assert (p["width"], p["height"]) != (geo.art_w, geo.art_h)


def test_prompt_expansion_is_always_bypassed(ctx):
    """The consistency killer. With expansion on, a test generation returned a
    forest scene with a deer instead of the figure; across 22 cards that is
    fatal. There is deliberately no way to turn this off."""
    assert payload(ctx)["bypass_prompt_expansion"] is True
    assert payload(ctx, seed=1, init=b"x")["bypass_prompt_expansion"] is True


def test_output_is_never_upscaled(ctx):
    """`import-mural` requires exactly the art window's size; any upscale
    factor makes the result unimportable."""
    assert payload(ctx)["upscale_output_factor"] == 1


def test_palette_payload_is_the_fourteen_drawable_colours(ctx):
    """Index 0 is transparent and `rgb_lut` substitutes paper for it, so it
    must be dropped — including it would weight paper twice and claim a colour
    the deck does not have."""
    pal, _, _ = ctx
    png = base64.b64decode(payload(ctx)["input_palette"])
    strip = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).reshape(-1, 3)
    assert len(strip) == 14
    assert np.array_equal(strip, pal.rgb_lut()[1:])


def test_init_image_carries_strength_and_omitting_it_does_not(ctx):
    """`strength` is meaningless without an init, and sending it anyway
    invites a service-side default to apply to a text-only run."""
    _, _, gen = ctx
    p = payload(ctx, init=b"\x89PNG-not-really")
    assert p["strength"] == gen.strength and "input_image" in p
    assert "strength" not in payload(ctx) and "input_image" not in payload(ctx)


def test_missing_key_is_a_clear_error_not_a_bare_request(monkeypatch):
    """An unauthenticated call is a wasted round trip and a confusing 401; the
    error must say what to set and where NOT to put it."""
    monkeypatch.delenv(retro.KEY_ENV, raising=False)
    with pytest.raises(retro.GenerationError, match=retro.KEY_ENV):
        retro.api_key()


def test_generation_config_defaults_and_prompt_lookup(tmp_path):
    """A deck with no generation.yaml still has a usable house style, and a
    face with no prompt says so by name instead of generating something
    arbitrary."""
    gen = retro.Generation.load(tmp_path / "absent.yaml")
    assert gen.style == retro.DEFAULTS["style"]
    assert gen.candidates == retro.DEFAULTS["candidates"]
    with pytest.raises(AssetError, match="major_09"):
        gen.prompt_for("major_09")


def test_shipped_prompt_never_says_pixel_art(ctx):
    """The style handles rendering. Saying "pixel art" in the prompt tends to
    produce a picture *of* pixel art, so the shipped prompts must not."""
    _, _, gen = ctx
    assert gen.prompts, "the deck ships at least one prompt"
    for key, text in gen.prompts.items():
        assert "pixel" not in text.lower(), key


def test_decode_images_rejects_an_unexpected_response():
    """A contract change should fail loudly rather than write zero files and
    report success."""
    assert retro.decode_images({"base64_images": [base64.b64encode(b"x").decode()]}) == [b"x"]
    with pytest.raises(retro.GenerationError):
        retro.decode_images({"images": ["..."]})


# --- 1-to-1 mapping, not generation --------------------------------------
def test_style_is_transformative_not_generative(ctx):
    """The deck maps existing RWS cards into pixel space; it does not invent
    new ones. rd_plus__*/rd_fast__* take the prompt as the subject and the
    input only as a hint -- rd_pro__pixelate/edit take the IMAGE as the
    subject. Drifting back to a generative style is how 22 cards stop being
    the same deck."""
    _, _, gen = ctx
    assert gen.style in ("rd_pro__pixelate", "rd_pro__edit"), gen.style


def test_strength_holds_the_composition(ctx):
    """Strength is the only fidelity knob the API offers -- the `negative`
    field is a documented placeholder that current models ignore. Above ~0.5
    the model is redrawing rather than re-rendering."""
    _, _, gen = ctx
    assert 0.0 < gen.strength <= 0.5, gen.strength


def test_prompts_are_anchors_not_descriptions(ctx):
    """With a pixelate style the seed is the subject; a long narrative prompt
    invites the invention this whole path is trying to avoid -- extra props, a
    drawn frame, a vignette."""
    _, _, gen = ctx
    for key, text in gen.prompts.items():
        assert len(text.split()) <= 20, f"{key}: {len(text.split())} words"


# --- transparent ground --------------------------------------------------
def test_remove_bg_is_opt_in_and_reaches_the_payload(ctx):
    """A transparent ground is what makes the mural's sky and the card's field
    the same pixels, so no seam can exist between them. It is opt-in because a
    scene may want an opaque ground on purpose -- a night sky painted in
    field-bank tones."""
    import dataclasses
    _, geo, gen = ctx
    assert "remove_bg" not in retro.build_payload(
        dataclasses.replace(gen, remove_bg=False), geo, ctx[0], prompt="x")
    assert retro.build_payload(
        dataclasses.replace(gen, remove_bg=True), geo, ctx[0],
        prompt="x")["remove_bg"] is True


def test_scan_url_is_computed_not_looked_up():
    """Commons stores a file under the first one and two hex digits of the md5
    of its name, so `arcana rd` can fetch a seed itself rather than failing
    with instructions to go run curl. Pinned against the verified live path."""
    assert retro.scan_url("major_00").endswith("/9/90/RWS_Tarot_00_Fool.jpg")
    assert len(retro.RWS_FILES) == 22
    assert retro.scan_url("court_cups_queen") is None   # unknown face, not a crash
