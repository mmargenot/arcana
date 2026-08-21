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


# --- shown a look, not told one -------------------------------------------
def test_the_style_is_a_tier_that_accepts_reference_images(ctx):
    """RD Pro is the ONLY tier that takes `reference_images`, and references are
    the entire strategy: the service quantises after generating, so
    `input_palette` re-maps colour and can never make the model draw flat. The
    look has to be shown.

    That makes a downgrade to rd_plus__* or rd_fast__* silently catastrophic —
    cheaper, still generates, and quietly ignores the nine exemplars that are the
    reason the output resembles the deck at all. It fails here instead."""
    _, _, gen = ctx
    for style in (gen.style, gen.seedless_style):
        assert style.startswith("rd_pro__"), (
            f"{style} is not an RD Pro style, so reference_images would be dropped")


def test_generated_size_fits_the_styles_range(ctx):
    """RD Pro runs 64-256px where Plus and Fast reach 384. The safe area is
    112x208 and fits, but the margin at the top is 48px — a geometry change that
    pushed height past 256 would be rejected by the service, and the cheapest
    place to learn that is here rather than mid-run."""
    from arcana.mural import safe_size
    _, geo, gen = ctx
    w, h = safe_size(geo)
    lo, hi = (64, 256) if gen.seedless_style.startswith("rd_pro__") else (64, 384)
    assert lo <= w <= hi and lo <= h <= hi, f"{w}x{h} outside {lo}-{hi}"


def test_strength_holds_the_composition(ctx):
    """Strength is the only fidelity knob the API offers -- the `negative`
    field is a documented placeholder that current models ignore. Above ~0.5
    the model is redrawing rather than re-rendering."""
    _, _, gen = ctx
    assert 0.0 < gen.strength <= 0.5, gen.strength


def test_every_major_has_a_prompt(ctx):
    """All 22, or a run stops partway with a face nobody wrote a subject for."""
    from arcana.mural import MAJOR_KEYS
    _, _, gen = ctx
    assert set(MAJOR_KEYS) <= set(gen.prompts), sorted(set(MAJOR_KEYS) - set(gen.prompts))


def test_prompts_are_emblems_and_banks_not_narrative(ctx):
    """A prompt names the card's emblems and the bank each should land in, with
    ONE marked dominant -- the reference sheet is a big mass plus small accents,
    not a list of co-equal objects. It is not a scene description; prose invites
    the invention this path exists to avoid. Style adjectives are not counted
    here at all: they live once in `style_prefix`."""
    _, _, gen = ctx
    for key, text in gen.prompts.items():
        assert len(text.split()) <= 45, f"{key}: {len(text.split())} words"


def test_prompt_colour_words_still_match_the_palette(ctx):
    """The prompts name banks by hue -- "violet stone", "teal sky" -- so they are
    COUPLED to palette.yaml. Recolour a bank and every prompt is quietly wrong,
    describing a deck that no longer exists. Pin the words to the actual hues so
    that edit fails here instead of surfacing 22 bad generations later."""
    import colorsys
    pal, _, gen = ctx
    expected = {"border": ("violet", 240, 290), "field": ("teal", 170, 215),
                "motif": ("magenta", 310, 350), "figure": ("warm tan", 5, 45)}
    corpus = " ".join(gen.prompts.values()).lower()
    for bank, (word, lo, hi) in expected.items():
        hexc = pal.banks[bank].mid
        r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
        hue = colorsys.rgb_to_hls(r, g, b)[0] * 360
        assert lo <= hue <= hi, (
            f"{bank} is {hexc} (hue {hue:.0f}) but the prompts call it {word!r}")
        assert word in corpus, f"no prompt uses {word!r} for the {bank} bank"


def test_the_banks_are_painted_across_the_deck_not_on_every_card(ctx):
    """Only the FIELD bank has a donor — the card's mat paints it, so an emblem on
    a transparent ground inherits teal for free. Border, motif and figure have to
    come from the art, which is how `major_00` imported with the motif bank empty.

    But that is checked PER DECK here, not per card, because a sparse white emblem
    is a legitimate card: the reference sheet's own Fool is a white flower with a
    brown centre and almost no bank at all. Requiring three hues of every prompt
    forbids that, and forbids it on a GUESS — predicting from the words what the
    art will contain. `import-mural` already warns per card on the actual pixels,
    which is the same check made against evidence, so this one stays loose.

    Teal is checked the other way round: majors have plain fields, and writing
    water into twenty cards that have none would spend real shapes buying a bank
    the mat was giving away."""
    _, _, gen = ctx
    corpus = " ".join(gen.prompts.values()).lower()
    n = len(gen.prompts)
    for word in ("violet", "magenta", "warm tan"):
        painted = sum(word in t.lower() for t in gen.prompts.values())
        assert painted >= n // 2, f"only {painted}/{n} prompts paint {word}"
        assert word in corpus
    watery = {k for k, v in gen.prompts.items() if "teal" in v.lower()}
    assert watery and len(watery) <= 8, sorted(watery)


def test_style_asks_the_emblem_to_fill_the_frame(ctx):
    """This once asked for "generous empty margins" and got them: major_00 came
    back 9.4% opaque, an emblem lost in its own window. Generating at the safe
    size already cuts the margin the card needs, so asking again spends the art
    twice."""
    _, _, gen = ctx
    assert "margins" not in gen.style_prefix, gen.style_prefix
    assert "filling the frame" in gen.style_prefix


def test_the_style_notes_are_positive_only(ctx):
    """Negations do not work and we keep re-introducing them. There is no
    negation mechanism in a positive prompt -- naming "horizon" or "drop shadows"
    raises their odds rather than lowering them -- and RD's `negative` field is a
    documented non-functional placeholder, so there is nowhere correct to put
    them. The old suffix carried six and produced shaded emblems anyway."""
    _, _, gen = ctx
    banned = [c.strip() for c in gen.style_prefix.split(",")
              if c.strip().startswith(("no ", "without ", "avoid "))]
    assert not banned, f"negations in style_prefix: {banned}"


def test_the_style_notes_do_not_restate_a_parameter(ctx):
    """`remove_bg` gives the transparent ground as a first-class flag, so asking
    for one in words buys nothing and spends tokens in the position that matters
    most. Same for upscaling and the size, which come from Geometry."""
    _, _, gen = ctx
    low = gen.style_prefix.lower()
    for word in ("transparent", "background", "upscale"):
        assert word not in low, f"{word!r} is a parameter, not a style note"


def test_the_prompt_stays_inside_the_token_budget(ctx):
    """CLIP conditioning truncates around 77 tokens, and the old 70-word prompt
    ran near 90 -- which put the style notes, then at the TAIL, exactly where
    truncation eats them. We cannot verify RD's tokenizer, so this keeps a margin
    rather than trusting one: ~1.3 tokens per word is the usual English rate."""
    _, _, gen = ctx
    for key in gen.prompts:
        words = len(gen.prompt_for(key).split())
        assert words * 1.3 < 77, f"{key}: {words} words, ~{words * 1.3:.0f} tokens"


def test_style_leads_and_is_shared_not_per_card(ctx):
    """22 prompts each carrying their own style adjectives is how 22 cards stop
    matching. The look is written once and PREPENDED to every subject.

    Leading, not trailing, and separated by a full stop. Joined with a comma at
    the tail, "a single isolated emblem" landed in the same grammatical slot as
    "a magenta ibis" and read as one more thing to draw."""
    _, _, gen = ctx
    assert gen.style_prefix, "the deck states a look"
    for key in gen.prompts:
        full = gen.prompt_for(key)
        assert full.startswith(gen.style_prefix), key
        assert gen.prompts[key] in full, key
        assert f"{gen.style_prefix}. " in full, f"{key}: style must end in a stop"
    assert "pixel" not in gen.style_prefix.lower()


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
    """`arcana rd` fetches its own seed rather than failing with instructions to
    go run curl. Special:FilePath resolves a name to bytes, and `width=` asks
    for a thumbnail — requesting multi-megabyte originals for 22 cards is what
    earns a 429."""
    assert "RWS_Tarot_00_Fool.jpg" in retro.scan_url("major_00")
    assert "width=" in retro.scan_url("major_00")     # a thumbnail, not the original
    assert len(retro.RWS_FILES) == 22
    assert retro.scan_url("court_cups_queen") is None   # unknown face, not a crash


def test_a_seedless_run_carries_no_seed_only_fields(ctx):
    """Seedless is now the DEFAULT, so this is the common path rather than the
    exception. `strength` is meaningless without an init and sending it anyway
    invites a service-side default to apply to a text-only run."""
    pal, geo, gen = ctx
    seeded = retro.build_payload(gen, geo, pal, prompt="x", init=b"png")
    bare = retro.build_payload(gen, geo, pal, prompt="x")
    assert seeded["prompt_style"] == gen.style
    assert bare["prompt_style"] == gen.seedless_style
    assert "input_image" not in bare and "strength" not in bare


def test_the_style_returns_one_object_not_a_sheet(ctx):
    """`item_sheet` and `character_turnaround` mean what they say and return a
    grid — we shipped `item_sheet` once and got exactly that. A face card is one
    emblem, so no style whose name promises several."""
    _, _, gen = ctx
    for style in (gen.style, gen.seedless_style):
        assert "sheet" not in style and "turnaround" not in style, style


def test_references_reach_the_payload_and_respect_the_ceiling(ctx):
    """Nine is RD Pro's documented maximum, and sending a tenth is a rejected
    request rather than a silently dropped image. Sending NONE is the failure
    this whole change exists to prevent, so an empty list must not put an empty
    key in the body either."""
    pal, geo, gen = ctx
    p = retro.build_payload(gen, geo, pal, prompt="x",
                            refs=[b"a", b"b", b"c"])
    assert len(p["reference_images"]) == 3
    over = retro.build_payload(gen, geo, pal, prompt="x",
                               refs=[bytes([i]) for i in range(20)])
    assert len(over["reference_images"]) == retro.MAX_REFS == 9
    assert "reference_images" not in retro.build_payload(gen, geo, pal, prompt="x")


def test_the_deck_ships_its_own_exemplars(ctx):
    """The references are deck identity — committed under configs, not artifacts,
    because they are hand-supplied input rather than something `arcana`
    regenerates. Losing them degrades every future generation silently, so their
    presence is pinned like any other config."""
    refs = retro.load_references(CONFIG)
    assert len(refs) == retro.MAX_REFS, f"{len(refs)} reference images"
    assert all(r.startswith(b"\x89PNG") for r in refs), "references must be PNG"


def test_the_un_quantised_twin_is_always_requested(ctx):
    """A shaded candidate has two possible authors and they need opposite fixes:
    the model drew it shaded, or our palette flattened a detailed image into
    stripes. `return_pre_palette` separates those in one comparison, so it is not
    worth a flag — it is always on."""
    assert payload(ctx)["return_pre_palette"] is True
    assert retro.decode_pre_palette({}) == []          # absent is not an error
    assert retro.decode_pre_palette(
        {"base64_images_pre_palette": [base64.b64encode(b"x").decode()]}) == [b"x"]


def test_cost_check_is_opt_in(ctx):
    """`check_cost` is a free dry run, which matters at RD Pro prices — but it
    returns a price INSTEAD of images, so it must never be on by default."""
    assert "check_cost" not in payload(ctx)
    assert payload(ctx, cost_only=True)["check_cost"] is True


def test_style_override_beats_config_without_editing_it(ctx):
    """The comparison matrix is several styles over the same card. Without an
    override that means editing config between runs, which is how a run gets
    attributed to the wrong style."""
    pal, geo, gen = ctx
    p = retro.build_payload(gen, geo, pal, prompt="x", style="rd_pro__simple")
    assert p["prompt_style"] == "rd_pro__simple" != gen.seedless_style


def test_generation_palette_offers_no_room_to_shade(ctx):
    """Emblems came back SHADED because the palette let them: a dark/mid/light
    ramp per bank is the material for a gradient, and the prompt asking for
    flatness was competing with it. Restricting the rungs makes shading within a
    bank impossible rather than merely discouraged.

    Index 0 is transparent and `rgb_lut` substitutes paper for it, so it is
    dropped whatever the rungs are — sending it would list paper twice and
    weight the model toward it."""
    pal, _, _ = ctx
    png = base64.b64decode(payload(ctx)["input_palette"])
    strip = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).reshape(-1, 3)
    lut = pal.rgb_lut()
    assert len(strip) == 2 + 4 * len(retro.GENERATION_RUNGS) < 14
    assert sum(np.array_equal(s, lut[2]) for s in strip) == 1, "paper listed once"
    # line and paper lead, universal and unshaded
    assert np.array_equal(strip[0], lut[1]) and np.array_equal(strip[1], lut[2])
    # every bank still represented, so the art can use all four hue families
    for i in range(4):
        band = lut[3 + 3 * i:6 + 3 * i]
        assert any(any(np.array_equal(s, c) for c in band) for s in strip[2:])
