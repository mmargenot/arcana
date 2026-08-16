"""
Generation: an RWS scan in, palette-constrained pixel art out.

This is the seam the whole face-card path is built around. `arcana` composes
frames, fields and labels algorithmically; the one thing it cannot compute is a
scene. So a generative model draws the art window, seeded by the public-domain
RWS card so composition carries across, and the result comes back through
`import-mural` like any hand-authored art.

WHAT THE MODEL IS FOR. Not pixelation -- a downscale can pixelate. The model
supplies SEMANTIC COLOUR: it knows the sky is a sky and should take the field
bank's teal, where a nearest-colour mapping only knows the sky is warm and puts
it in the flesh ramp. Measured on the Fool, direct quantisation of the scan put
19,509 px in `figure` and 3 px in `border`. That is the problem being solved.

THREE NON-NEGOTIABLES, each a bug that has already happened or is one call
away:

  * `bypass_prompt_expansion` is always on. With expansion, a test generation
    returned a forest scene with a deer instead of the intended figure. Across
    22 cards that is the consistency killer.
  * `upscale_output_factor` is always 1. Import requires an exact pixel size
    and rejects anything else.
  * width/height come from `Geometry`, never literals, so a deck that changes
    its art window does not silently generate at the old size.

GENERATE AT THE SAFE SIZE, not the art window's. The frame band overlaps the
window, so a full-window generation spends a quarter of its pixels on a ring
that is half-covered on screen and clipped in print -- and puts whatever border
the model decides to draw right where it will collide with the deck's frame.
Asking for `mural.safe_size` instead means every generated pixel is visible,
nothing needs rescaling, and edge decoration has nowhere to land.

`input_palette` constrains colour but does not guarantee the deck's exact
hexes, and it cannot guarantee that every bank gets used. Import therefore
still snaps (`--force`) and still prints the histogram -- that histogram is
the charter check, not a formality.

THE KEY NEVER TOUCHES CONFIG. `RD_API_KEY` is read from the environment only.
Cloud environments have no secrets store, so it belongs on the machine that
runs the call, not in a deck file or an environment variable box.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from arcana.elements import AssetError
from arcana.geometry import Geometry
from arcana.palette import BANKS, DARK, LIGHT, LINE, MID, PAPER, Palette

ENDPOINT = "https://api.retrodiffusion.ai/v1/inferences"
KEY_ENV = "RD_API_KEY"

DEFAULTS = {
    "provider": "retro-diffusion",
    "style": "rd_plus__default",
    "strength": 0.6,
    "candidates": 4,
    "remove_bg": False,
    "knockout_ground": False,
    "style_suffix": "",
    "seedless_style": "rd_plus__default",
}


class GenerationError(RuntimeError):
    """A generation could not be requested or the service refused it."""


@dataclass(frozen=True, slots=True)
class Generation:
    """A deck's generation config: house style plus per-face prompts.

    Style, strength and candidate count are the DECK's identity -- you settle
    them once and every card inherits them, which is what makes 22 cards look
    like one deck. They live in config for the same reason `pip_layouts` and
    `field_designs` do, and are deliberately not CLI flags."""
    style: str
    strength: float
    candidates: int
    remove_bg: bool
    knockout_ground: bool
    style_suffix: str
    seedless_style: str
    prompts: dict[str, str]

    @classmethod
    def load(cls, path: str | Path) -> "Generation":
        p = Path(path)
        d = dict(DEFAULTS)
        if p.exists():
            loaded = yaml.safe_load(p.read_text()) or {}
            if not isinstance(loaded, dict):
                raise AssetError(f"{p}: expected a mapping")
            d.update(loaded)
        return cls(style=str(d["style"]), strength=float(d["strength"]),
                   candidates=int(d["candidates"]),
                   remove_bg=bool(d["remove_bg"]),
                   knockout_ground=bool(d["knockout_ground"]),
                   style_suffix=str(d["style_suffix"]).strip(),
                   seedless_style=str(d["seedless_style"]),
                   prompts=dict(d.get("prompts") or {}))

    def prompt_for(self, key: str) -> str:
        """A face's subject, plus the deck's shared style suffix.

        The split is the point. The SUBJECT is per card and stays terse -- with
        a pixelate style the seed is the subject, so a long description only
        invites invention. The STYLE is per deck and written once, which is what
        keeps 22 cards looking like one deck instead of 22 prompts drifting
        apart."""
        try:
            subject = self.prompts[key]
        except KeyError:
            raise AssetError(
                f"no prompt for face {key!r}. Add it under `prompts:` in the "
                f"deck's generation.yaml, or pass --prompt for a one-off.")
        return f"{subject}, {self.style_suffix}" if self.style_suffix else subject


# Which rung of each bank the generator may use.
#
# Offering all three is what produced SHADED emblems: a dark/mid/light ramp per
# hue family is exactly the material for a gradient, and the model duly built 3D
# form out of it -- a plinth with a lit top, a modelled side, a shadowed base.
# Asking for flatness in the prompt did not stop it, because the prompt was
# competing with the palette. Hand it two violets and a shaded violet is not
# something it can render.
#
# Two rungs is the target look: a lit face and a body, form carried by
# silhouette rather than modelling. Drop to (MID,) for one flat tone per bank if
# anything still reads dimensional. Deliberately a constant rather than config:
# there is one deck and one aesthetic, and this is a one-line change either way.
GENERATION_RUNGS = (MID,)


def palette_png(pal: Palette, rungs: tuple[int, ...] = GENERATION_RUNGS) -> bytes:
    """The colours the generator may use, as a PNG for `input_palette`.

    Always line and paper -- outlines and whites are universal, and every emblem
    needs them -- plus the named rungs of each bank. Index 0 is transparent and
    `rgb_lut` substitutes paper for it, so it MUST be dropped: including it
    would weight paper twice and claim a colour the deck does not have.

    This only narrows what comes BACK. Import still quantises against all 14, so
    art that used ten of them simply stays flat."""
    lut = pal.rgb_lut()
    keep = [LINE, PAPER]
    for i in range(len(BANKS)):
        keep += [3 + 3 * i + (r - DARK) for r in rungs]
    buf = io.BytesIO()
    Image.fromarray(lut[keep].reshape(1, -1, 3).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def build_payload(gen: Generation, geo: Geometry, pal: Palette, *,
                  prompt: str, seed: int | None = None,
                  init: bytes | None = None) -> dict:
    """The request body. Pure -- no network, no key -- so tests can assert the
    invariants without stubbing anything."""
    from arcana.mural import safe_size
    w, h = safe_size(geo)
    # A transformative style takes the IMAGE as its subject, so without one there
    # is nothing to transform and the service rejects the call. Seedless work
    # therefore needs a generative style -- a different job, named separately in
    # config rather than silently reusing `style`.
    payload: dict = {
        "prompt": prompt,
        "prompt_style": gen.style if init is not None else gen.seedless_style,
        "width": w,
        "height": h,
        "num_images": gen.candidates,
        "input_palette": _b64(palette_png(pal)),
        "bypass_prompt_expansion": True,
        "upscale_output_factor": 1,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if init is not None:
        payload["input_image"] = _b64(init)
        payload["strength"] = gen.strength
    if gen.remove_bg:
        # transparent ground straight from the service: the card's field then
        # shows through the sky, and `quantize_rgb_global` already maps alpha
        # below 128 to index 0, so nothing downstream changes
        payload["remove_bg"] = True
    return payload


def estimate_cost(gen: Generation, geo: Geometry) -> float:
    """Local USD estimate from Retro Diffusion's published formulas, so a run's
    price is visible before it is spent. `check_cost` is authoritative; this is
    for the dry print."""
    px = geo.art_w * geo.art_h
    style = gen.style
    if style.startswith("rd_pro__"):
        per = 0.18
    elif "low_res" in style or "mc_" in style or style.endswith(("classic", "skill_icon")):
        per = max(0.02, (px + 13_700) / 600_000)
    elif style.startswith("rd_fast__"):
        per = max(0.015, (px + 100_000) / 6_000_000)
    else:
        per = max(0.025, (px + 50_000) / 2_000_000)
    return per * gen.candidates


# Wikimedia serves the thumbnail endpoint rather than originals, and REQUIRES a
# descriptive User-Agent: a bare urllib request is answered with 403, and a run
# of 22 fetches with 429 pointing here. Both were shipped bugs.
COMMONS_FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
USER_AGENT = "arcana/0.1 (pixel-art card engine; https://github.com/mmargenot/arcana)"

# Width to request. The seed is decimated to ~112px wide, so an original scan is
# many times more data than the pipeline can use, and asking for it is what
# earns the rate limit.
SCAN_WIDTH = 1200

# Wikimedia Commons filenames for the majors, which are what the deck seeds
# from. The 1909 Rider printing is public domain; these are the scans, not a
# modern recolour (checked against the warm-cream paper the notes record).
RWS_FILES: dict[str, str] = {
    f"major_{i:02d}": f"RWS_Tarot_{i:02d}_{n}.jpg" for i, n in enumerate((
        "Fool", "Magician", "High_Priestess", "Empress", "Emperor",
        "Hierophant", "Lovers", "Chariot", "Strength", "Hermit",
        "Wheel_of_Fortune", "Justice", "Hanged_Man", "Death", "Temperance",
        "Devil", "Tower", "Star", "Moon", "Sun", "Judgement", "World"))
}


def scan_url(face: str, width: int = SCAN_WIDTH) -> str | None:
    """The Commons URL for a face's source scan, or None if we do not know one.

    `Special:FilePath` resolves a file NAME to its bytes, so no md5 path
    arithmetic and no API lookup -- and `?width=` gets a thumbnail rather than a
    multi-megabyte original the pipeline would immediately throw away."""
    fname = RWS_FILES.get(face)
    if not fname:
        return None
    return f"{COMMONS_FILEPATH}{urllib.parse.quote(fname)}?width={width}"


def fetch_scan(face: str, dest: Path, *, attempts: int = 4) -> Path:
    """Download a face's source scan if it is not already on disk.

    Wikimedia answers an unidentified client with 403 and a burst of requests
    with 429, so this sends a descriptive User-Agent, asks for a thumbnail, and
    backs off. Fetching 22 majors in a row is exactly the burst they mean.
    """
    if dest.exists():
        return dest
    url = scan_url(face)
    if url is None:
        raise GenerationError(
            f"no known source scan for face {face!r} — pass --init with an "
            f"image to seed from")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise GenerationError(
                f"could not fetch {url}: {e}"
                + (" — Wikimedia is rate-limiting this address; wait a few "
                   "minutes or pass --init with a local scan" if e.code == 429
                   else ""))
        except urllib.error.URLError as e:
            raise GenerationError(f"could not fetch {url}: {e}")
    return dest


def api_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise GenerationError(
            f"{KEY_ENV} is not set. Retro Diffusion keys start with 'rdpk-'; "
            f"export it in the shell that runs this command. Do not put it in "
            f"a deck config or a cloud environment's variables -- those are "
            f"readable by anyone with access to the environment.")
    return key


def request(payload: dict, *, timeout: int = 180) -> dict:
    """POST to the inference endpoint. Stdlib only: this is one JSON call, and
    a dependency for it would not earn its place."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-RD-Token": api_key()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise GenerationError(f"Retro Diffusion returned {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise GenerationError(f"could not reach {ENDPOINT}: {e.reason}")


def decode_images(response: dict) -> list[bytes]:
    """The returned images as raw PNG bytes. The API answers with base64 under
    `base64_images`; a hosted-URL response is not requested, so anything else
    is a contract change worth failing on rather than guessing about."""
    images = response.get("base64_images")
    if not images:
        raise GenerationError(
            f"no images in response (keys: {', '.join(sorted(response)) or 'none'})")
    return [base64.b64decode(b) for b in images]
