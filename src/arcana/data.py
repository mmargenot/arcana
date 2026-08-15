"""
Tarot label data: the plaintext a card's numeral/title bands render.

Engine ships the canonical Rider–Waite–Smith names and rank words as defaults;
a deck overrides them under a `labels:` block in deck.yaml (`suit_titles`,
`major_titles`) — the same "engine ships defaults, deck customises" split as
`pip_layouts` / `field_designs`. Everything here is uppercase, ASCII, and
suit-agnostic string work; pixels happen in `arcana.text`.
"""
from __future__ import annotations

# Major arcana 0..21, canonical RWS order. Index IS the major's number.
MAJOR_NAMES: tuple[str, ...] = (
    "THE FOOL", "THE MAGICIAN", "THE HIGH PRIESTESS", "THE EMPRESS",
    "THE EMPEROR", "THE HIEROPHANT", "THE LOVERS", "THE CHARIOT",
    "STRENGTH", "THE HERMIT", "WHEEL OF FORTUNE", "JUSTICE",
    "THE HANGED MAN", "DEATH", "TEMPERANCE", "THE DEVIL",
    "THE TOWER", "THE STAR", "THE MOON", "THE SUN",
    "JUDGEMENT", "THE WORLD",
)

# Minor ranks Ace..Ten as words.
RANK_WORDS: dict[int, str] = {
    1: "ACE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE",
    6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN",
}

_ROMAN = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


def roman(n: int) -> str:
    """Roman numeral for a major (1..21). `0` is a literal for The Fool."""
    if n == 0:
        return "0"
    out: list[str] = []
    for value, sym in _ROMAN:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def arabic_rank(n: int) -> str:
    """Compact rank token for a minor: `A` for the ace, else the digits."""
    return "A" if n == 1 else str(n)


# ---------------------------------------------------------------- overrides
def _labels_cfg(cfg: dict | None) -> dict:
    return (cfg or {}).get("labels", {}) or {}


def suit_title(suit: str, cfg: dict | None = None) -> str:
    """Display name for a suit — `labels.suit_titles` override else UPPERCASED."""
    return (_labels_cfg(cfg).get("suit_titles") or {}).get(suit, suit.upper())


def major_name(number: int, cfg: dict | None = None) -> str:
    """Display name for a major — `labels.major_titles[number]` override else RWS."""
    over = _labels_cfg(cfg).get("major_titles") or ()
    if number < len(over) and over[number]:
        return str(over[number]).upper()
    return MAJOR_NAMES[number]


# ---------------------------------------------------------------- composers
# Each returns (top, bottom): the string for the numeral band and the title
# band. `None` means that band is empty. Combined (split=False) puts everything
# in the roomy bottom title band — the safe default that never fights the top
# medallion. Split (split=True) is the two-band look, tuned on renders.
def minor_label(rank: int, suit: str, *, style: str = "arabic",
                split: bool = False, cfg: dict | None = None) -> tuple[str | None, str | None]:
    name = suit_title(suit, cfg)
    if split:
        token = arabic_rank(rank) if style == "arabic" else RANK_WORDS[rank]
        return token, name
    return None, f"{RANK_WORDS[rank]} OF {name}"


def major_label(number: int, *, split: bool = False,
                cfg: dict | None = None) -> tuple[str | None, str | None]:
    name = major_name(number, cfg)
    numeral = roman(number)
    if split:
        return numeral, name
    return None, f"{numeral} {name}"


def face_label(key: str, *, split: bool = False,
               cfg: dict | None = None) -> tuple[str | None, str | None]:
    """Label for any face card, by key. A tarot major (`major_07`) keeps its
    roman numeral and canonical name; any other key (`court_cups_queen`,
    `wizard`) reads its title from `labels.face_titles[key]`, else its own key
    prettified. Numbering is a tarot convention, not a face-card one, so a
    non-major gets a title and no numeral."""
    if key.startswith("major_") and key[6:].isdigit() and int(key[6:]) < len(MAJOR_NAMES):
        return major_label(int(key[6:]), split=split, cfg=cfg)
    over = _labels_cfg(cfg).get("face_titles") or {}
    return None, str(over.get(key, key.replace("_", " "))).upper()


def label_options(cfg: dict | None) -> dict:
    """Read the `labels:` block with defaults, for the CLI/builders."""
    lc = _labels_cfg(cfg)
    return {
        "enabled": bool(lc.get("enabled", True)),
        "style": lc.get("numeral_style", "arabic"),
        "split": bool(lc.get("split", False)),
    }
