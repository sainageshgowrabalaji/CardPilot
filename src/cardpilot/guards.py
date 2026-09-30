"""Guardrails that run in plain code, before and after any model.

Input side. Card numbers and ID numbers are removed before anything is logged, traced or sent to a
model. Prompt-injection attempts and topics outside credit cards are refused politely.

Output side. Advice wording ("you should get this card") is removed, because CardPilot explains and
compares, it never recommends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
CVV = re.compile(r"\b(?:cvv|cvc|cvv2|security code)\s*(?:is|:|=)?\s*\d{3,4}\b", re.IGNORECASE)
EXPIRY = re.compile(r"\b(?:exp(?:iry|ires)?|valid thru)\s*(?:is|:)?\s*(?:0[1-9]|1[0-2])\s*/\s*\d{2,4}\b", re.IGNORECASE)


def luhn_ok(digits: str) -> bool:
    """The checksum real card numbers pass. It keeps phone numbers and amounts from being removed."""
    total, double = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if double:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        double = not double
    return total % 10 == 0


@dataclass
class Screened:
    text: str
    removed: list[str] = field(default_factory=list)
    refused: str | None = None
    advice_request: bool = False
    ranking_request: bool = False


def redact(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []

    def card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            removed.append("card number")
            return "[card number removed]"
        return match.group()

    text = CARD_CANDIDATE.sub(card, text)
    for pattern, label in (
        (CVV, "security code"),
        (EXPIRY, "expiry date"),
        (SSN, "Social Security number"),
        (AADHAAR, "Aadhaar number"),
        (PAN, "PAN"),
    ):
        if pattern.search(text):
            removed.append(label)
            text = pattern.sub(f"[{label} removed]", text)
    return text, removed


INJECTION = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)|disregard (?:your|the|all) "
    r"(?:instructions|rules)|system prompt|developer mode|you are now|jailbreak|reveal (?:your|the) (?:prompt|instructions|rules)"
    r"|act as (?:an? )?(?:unrestricted|different)|pretend (?:you are|to be)|\bDAN\b",
    re.IGNORECASE,
)

OFF_TOPIC = re.compile(
    r"\b(stocks?|shares|crypto|bitcoin|ethereum|mutual funds?|sip|invest(?:ing|ment)?s?|trading|forex trading|mortgage|home loan|"
    r"personal loan|car loan|tax (?:return|filing|advice)|income tax|insurance claim|lottery|gambling|betting)\b",
    re.IGNORECASE,
)
CARD_WORDS = re.compile(
    r"\b(card|cards|credit|apr|interest|fee|fees|reward|rewards|cash ?back|points|miles|limit|statement|minimum|payment|lounge|"
    r"forex|foreign|annual|joining|secured|balance|cibil|score|bill|emi)\b",
    re.IGNORECASE,
)
ADVICE_REQUEST = re.compile(
    r"\b(which|what) (?:card|one) should i\b|\bshould i (?:get|apply|take|choose|pick|open)\b|\bbest card for me\b|"
    r"\brecommend (?:a|me|one)\b|\bwhich is better for me\b",
    re.IGNORECASE,
)


RANKING_REQUEST = re.compile(
    r"\b(?:top|best|most popular|popular|top[- ]rated|highest[- ]rated|graded|ranked|leading|number one)\b"
    r"(?:\W+\w+){0,3}?\W+cards?\b|\bcards?\b(?:\W+\w+){0,4}?\W+(?:ranked|graded|rated|ranking|most popular)\b",
    re.IGNORECASE,
)


def screen_input(text: str, max_chars: int = 800) -> Screened:
    clean, removed = redact(text.strip())
    if not clean:
        return Screened(text=clean, removed=removed, refused="Ask me anything about credit cards in your country.")
    if len(clean) > max_chars:
        return Screened(
            text=clean[:max_chars],
            removed=removed,
            refused="That question is too long. Please ask in a sentence or two.",
        )
    if INJECTION.search(clean):
        return Screened(
            text=clean,
            removed=removed,
            refused="I can only help with questions about credit cards, and I keep my own rules. Try asking about a card's fees or rewards.",
        )
    if OFF_TOPIC.search(clean) and not CARD_WORDS.search(clean):
        return Screened(
            text=clean,
            removed=removed,
            refused="I only explain credit cards, so I can't help with that. For money questions beyond cards, a licensed adviser is the right person.",
        )
    return Screened(
        text=clean,
        removed=removed,
        advice_request=bool(ADVICE_REQUEST.search(clean)),
        ranking_request=bool(RANKING_REQUEST.search(clean)),
    )


ADVICE_WORDING = re.compile(
    r"\byou should (?:get|apply|choose|pick|go with|sign up)|\bi (?:would )?recommend\b|\bi suggest\b|\bbest (?:card|choice|option) "
    r"for you\b|\bgo with the\b|\bthe best card\b|\bthe clear winner\b|\bdefinitely get\b|\bmost popular card|"
    r"\btop[- ]rated card|\bbest overall\b",
    re.IGNORECASE,
)


def screen_sentence(sentence: str) -> bool:
    """False when a sentence gives advice instead of facts. Those sentences are removed."""
    return not ADVICE_WORDING.search(sentence)


def notice_for(removed: list[str]) -> str | None:
    if not removed:
        return None
    what = ", ".join(sorted(set(removed)))
    return (
        f"I removed a {what} from your message before doing anything with it. CardPilot never needs card "
        "numbers or ID numbers, and never stores them."
    )
