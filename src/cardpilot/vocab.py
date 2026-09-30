"""Everyday words mapped to the words card documents use, so "cost per year" finds "annual fee".

Query expansion only adds terms, it never removes the user's words. Vector search already handles
many paraphrases. This lifts keyword search and the offline answer picker, which are lexical.
Built from the misses on the tuning golden set only. The held-out set was frozen before these rules
existed and none of its phrasings are here, so its score shows how well the rules generalise.
"""

from __future__ import annotations

import re

# (pattern in the question, terms to add)
_RULES: list[tuple[str, str]] = [
    (r"per year|a year|every year|yearly|annually", "annual fee"),
    (r"\bcosts?\b|\bcharges?\b|\bprice\b|\bpay for\b", "fee"),
    (
        r"abroad|overseas|international|outside (?:the )?(?:us|country|india)|another country|foreign currency",
        "foreign transaction fee forex markup",
    ),
    (r"restaurants?|takeout|takeaway|food delivery", "dining"),
    (r"supermarkets?|grocery stores?", "groceries grocery"),
    (r"petrol|gas stations?", "gas fuel"),
    (r"\batms?\b|withdraw(?:al)? cash|cash withdrawal|cash out|take cash", "cash advance"),
    (r"partner|spouse|family member|add (?:someone|a person)|second card", "authorized user add-on"),
    (r"never (?:applied|asked)|didn't apply|did not apply|without applying", "unsolicited"),
    (r"without (?:my )?permission|someone used my card|unauthori[sz]ed", "unauthorised liability dispute"),
    (r"stolen|thief", "lost stolen liability"),
    (r"wrong charge|dispute", "dispute error"),
    (r"\bmean\b|meaning|definition|what is an?\b", "stands means"),
    (r"late fee|late charge|missed (?:a )?payment", "late payment penalty fee"),
    (r"whole bill|entire bill|total bill", "total outstanding"),
    (r"\binterest\b", "interest apr"),
    (r"sign ?up bonus|joining bonus", "welcome bonus offer"),
    (r"get back|money back", "cashback cash back earn"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), terms) for p, terms in _RULES]


def expansion(text: str) -> str:
    """Only the added terms, deduplicated, in rule order."""
    added: list[str] = []
    for pattern, terms in _COMPILED:
        if pattern.search(text):
            added += [t for t in terms.split() if t not in added]
    return " ".join(added)


def expand(text: str) -> str:
    """The question followed by the added terms."""
    extra = expansion(text)
    return f"{text} {extra}" if extra else text
