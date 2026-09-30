"""The card catalog of one country: structured facts, who says so, and the math around them.

Numbers in answers come from here, never from a model's memory. Every value carries the source it
was read from, so a comparison table can cite each cell.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from .countries import Country, get_country
from .schemas import Chunk, Evidence


class Source(BaseModel):
    id: str
    url: str
    title: str


class Passage(BaseModel):
    text: str
    source: str


class Card(BaseModel):
    id: str
    name: str
    issuer: str
    network: str | None = None
    kind: str
    annual_fee: float | None = None
    joining_fee: float | None = None
    fee_waiver: str | None = None
    foreign_transaction_fee_pct: float | None = None
    intro_apr: str | None = None
    regular_apr: str | None = None
    interest_rate: str | None = None
    credit_level: str | None = None
    earn: dict[str, float | None]
    earn_assumption: str | None = None
    earn_caps: str | None = None
    key_benefits: list[str] = []
    welcome_offer: str | None = None
    sources: list[Source]
    passages: list[Passage]
    verified: str | None = None
    notes: str | None = None

    def source(self, sid: str) -> Source:
        for s in self.sources:
            if s.id == sid:
                return s
        return self.sources[0]

    def source_for(self, *keywords: str) -> Source:
        """The source of the first passage that talks about these words, else the card's main page."""
        for p in self.passages:
            low = p.text.lower()
            if any(k in low for k in keywords):
                return self.source(p.source)
        return self.sources[0]


class GeneralPassage(BaseModel):
    text: str
    source: Source | dict


class Catalog:
    """All cards of one country. There is no way to reach another country's cards from here."""

    def __init__(self, country: Country, data: dict):
        if data.get("country") != country.code:
            raise ValueError(f"Card file is for {data.get('country')!r}, not {country.code!r}")
        self.country = country
        self.as_of: str = data.get("as_of", "")
        self.cards: list[Card] = [Card.model_validate(c) for c in data["cards"]]
        self.general = data.get("general", [])
        self._by_id = {c.id: c for c in self.cards}
        self._aliases = _build_aliases(self.cards)

    @classmethod
    def load(cls, cards_dir: Path, code: str) -> Catalog:
        country = get_country(code)
        data = json.loads((cards_dir / f"{country.code}.json").read_text(encoding="utf-8"))
        return cls(country, data)

    # ------------------------------------------------------------ lookup

    def get(self, card_id_or_name: str) -> Card | None:
        if card_id_or_name in self._by_id:
            return self._by_id[card_id_or_name]
        found = self.mentions(card_id_or_name)
        return found[0] if found else None

    def mentions(self, text: str) -> list[Card]:
        """Cards named in a question, longest alias first, so "Discover it Secured" beats "Discover it"."""
        low = " " + re.sub(r"[^a-z0-9&+]+", " ", text.lower()) + " "
        found: list[Card] = []
        taken: list[tuple[int, int]] = []
        for alias, card_id in self._aliases:
            start = low.find(f" {alias} ")
            while start != -1:
                span = (start, start + len(alias) + 2)
                if not any(a < span[1] and span[0] < b for a, b in taken):
                    taken.append(span)
                    card = self._by_id[card_id]
                    if card not in found:
                        found.append(card)
                    break
                start = low.find(f" {alias} ", start + 1)
        return found

    # ------------------------------------------------------------ chunks for the search index

    def chunks(self) -> list[Chunk]:
        """Every passage, a few fact sentences per card built from the structured fields, and the basics."""
        out: list[Chunk] = []
        c = self.country
        for card in self.cards:
            for i, p in enumerate(card.passages):
                src = card.source(p.source)
                out.append(
                    Chunk(
                        id=f"{c.code}:{card.id}:p{i}",
                        country=c.code,
                        card_id=card.id,
                        card_name=card.name,
                        kind="card",
                        text=f"{card.name}. {p.text}",
                        url=src.url,
                        title=src.title,
                        as_of=self.as_of,
                    )
                )
            for j, (text, src) in enumerate(self.fact_sentences(card)):
                out.append(
                    Chunk(
                        id=f"{c.code}:{card.id}:f{j}",
                        country=c.code,
                        card_id=card.id,
                        card_name=card.name,
                        kind="fact",
                        text=text,
                        url=src.url,
                        title=src.title,
                        as_of=self.as_of,
                    )
                )
        for k, g in enumerate(self.general):
            src = g["source"]
            out.append(
                Chunk(
                    id=f"{c.code}:general:{k}",
                    country=c.code,
                    kind="general",
                    text=g["text"],
                    url=src.get("url"),
                    title=src.get("title", "Consumer guide"),
                    as_of=self.as_of,
                )
            )
        return out

    def fact_sentences(self, card: Card) -> list[tuple[str, Source]]:
        c = self.country
        facts: list[tuple[str, Source]] = []
        if card.annual_fee is not None:
            fee = "no annual fee" if card.annual_fee == 0 else f"an annual fee of {c.money(card.annual_fee)}"
            if c.code == "in":
                fee += " (GST extra)" if card.annual_fee else ""
            facts.append((f"The {card.name} has {fee}.", card.source_for("annual fee", "renewal fee", "fee")))
        if card.joining_fee is not None:
            facts.append(
                (f"The {card.name} has a joining fee of {c.money(card.joining_fee)}.", card.source_for("joining"))
            )
        if card.foreign_transaction_fee_pct is not None:
            word = "foreign transaction fee" if c.code == "us" else "forex markup fee"
            text = (
                f"The {card.name} has no {word}."
                if card.foreign_transaction_fee_pct == 0
                else f"The {card.name} charges a {word} of {card.foreign_transaction_fee_pct:g}%."
            )
            facts.append((text, card.source_for("foreign", "forex", "markup")))
        rates = [(k, v) for k, v in card.earn.items() if v]
        if rates:
            labels = dict(c.categories)
            parts = ", ".join(f"{v:g}% on {labels.get(k, k).lower()}" for k, v in rates)
            facts.append(
                (f"The {card.name} earns an estimated {parts}.", card.source_for("earn", "%", "cashback", "cash back"))
            )
        if card.intro_apr:
            facts.append(
                (f"The {card.name} intro APR is {card.intro_apr.rstrip('. ')}.", card.source_for("intro", "apr"))
            )
        if card.credit_level and c.code == "us":
            facts.append(
                (f"The {card.name} is aimed at {card.credit_level.rstrip('. ')} credit.", card.source_for("credit"))
            )
        elif card.credit_level:
            facts.append(
                (
                    f"Eligibility for the {card.name}. {card.credit_level.rstrip('. ')}.",
                    card.source_for("eligib", "income", "age"),
                )
            )
        return facts

    # ------------------------------------------------------------ tools

    def card_evidence(self, card: Card) -> list[Evidence]:
        """What get_card returns: the structured facts plus the card's own passages."""
        items = [
            Evidence(id="", text=text, title=src.title, url=src.url, card=card.name)
            for text, src in self.fact_sentences(card)
        ]
        if card.earn_caps:
            src = card.source_for("cap", "only", "limit")
            items.append(
                Evidence(
                    id="", text=f"{card.name} limits. {card.earn_caps}", title=src.title, url=src.url, card=card.name
                )
            )
        if card.key_benefits:
            items.append(
                Evidence(
                    id="",
                    text=f"{card.name} key benefits include {', '.join(card.key_benefits[:6])}.",
                    title=card.sources[0].title,
                    url=card.sources[0].url,
                    card=card.name,
                )
            )
        for p in card.passages:
            src = card.source(p.source)
            items.append(Evidence(id="", text=p.text, title=src.title, url=src.url, card=card.name))
        return items

    def compare(self, card_ids: list[str]) -> dict:
        cards = [c for c in (self.get(i) for i in card_ids) if c is not None][:3]
        if len(cards) < 2:
            raise ValueError("Pick two or three cards from this country's list to compare.")
        c = self.country
        sources: list[Source] = []

        def cite(src: Source) -> int:
            for n, s in enumerate(sources, 1):
                if s.url == src.url:
                    return n
            sources.append(src)
            return len(sources)

        def fee(card: Card) -> dict:
            return {
                "value": c.money(card.annual_fee) if card.annual_fee is not None else "not published",
                "source": cite(card.source_for("annual fee", "renewal", "fee")),
            }

        rows = [
            {"label": "Issuer", "cells": [{"value": card.issuer, "source": cite(card.sources[0])} for card in cards]},
            {"label": "Type", "cells": [{"value": card.kind, "source": cite(card.sources[0])} for card in cards]},
            {"label": "Annual fee", "cells": [fee(card) for card in cards]},
        ]
        if c.code == "in":
            rows.append(
                {
                    "label": "Joining fee",
                    "cells": [
                        {
                            "value": c.money(card.joining_fee) if card.joining_fee is not None else "not published",
                            "source": cite(card.source_for("joining")),
                        }
                        for card in cards
                    ],
                }
            )
        rows.append(
            {
                "label": "Foreign transaction fee" if c.code == "us" else "Forex markup",
                "cells": [
                    {
                        "value": "not published"
                        if card.foreign_transaction_fee_pct is None
                        else f"{card.foreign_transaction_fee_pct:g}%",
                        "source": cite(card.source_for("foreign", "forex", "markup")),
                    }
                    for card in cards
                ],
            }
        )
        for key, label in c.categories:
            rows.append(
                {
                    "label": f"Earn on {label.lower()}",
                    "cells": [
                        {
                            "value": "not published" if card.earn.get(key) is None else f"{card.earn[key]:g}%",
                            "source": cite(card.source_for("earn", "%", "cashback", "cash back")),
                        }
                        for card in cards
                    ],
                }
            )
        rows.append(
            {
                "label": "Limits",
                "cells": [
                    {
                        "value": card.earn_caps or "none published",
                        "source": cite(card.source_for("cap", "only", "limit")),
                    }
                    for card in cards
                ],
            }
        )
        rows.append(
            {
                "label": "Key benefits",
                "cells": [
                    {"value": ", ".join(card.key_benefits[:5]) or "none listed", "source": cite(card.sources[0])}
                    for card in cards
                ],
            }
        )
        return {
            "columns": [card.name for card in cards],
            "rows": rows,
            "sources": [{"n": n, "title": s.title, "url": s.url} for n, s in enumerate(sources, 1)],
        }

    def estimate(self, monthly_spend: dict[str, float]) -> dict:
        """Yearly rewards from monthly spend, per card, with the fee taken off. Plain arithmetic."""
        keys = {k for k, _ in self.country.categories}
        spend = {k: max(0.0, float(v)) for k, v in monthly_spend.items() if k in keys and v is not None}
        if not spend or sum(spend.values()) <= 0:
            raise ValueError("Enter at least one monthly amount.")
        rows = []
        for card in self.cards:
            yearly = sum(amount * 12 * (card.earn.get(k) or 0) / 100 for k, amount in spend.items())
            fee = card.annual_fee or 0
            rows.append(
                {
                    "card_id": card.id,
                    "name": card.name,
                    "yearly_rewards": round(yearly, 2),
                    "annual_fee": fee,
                    "net": round(yearly - fee, 2),
                }
            )
        rows.sort(key=lambda r: r["net"], reverse=True)
        note = (
            "A simple estimate from each card's published earn rates, with points valued conservatively. "
            "It ignores spending caps, welcome offers, first-year fee waivers and rotating categories"
            + (", and fees shown exclude GST" if self.country.code == "in" else "")
            + ". Check each card's limits before relying on it."
        )
        return {"rows": rows, "note": note}

    def listing(self) -> dict:
        return {
            "cards": [
                {"id": c.id, "name": c.name, "issuer": c.issuer, "kind": c.kind, "annual_fee": c.annual_fee}
                for c in self.cards
            ],
            "categories": [{"key": k, "label": label} for k, label in self.country.categories],
        }


# Words that name the issuer or the product type rather than the card itself.
_GENERIC = {
    "card",
    "credit",
    "bank",
    "the",
    "from",
    "rewards",
    "cash",
    "of",
}
_MANUAL_ALIASES = {
    "amex-gold": ["amex gold", "gold card", "american express gold"],
    "amex-blue-cash-everyday": ["blue cash everyday", "amex blue cash", "bce"],
    "chase-sapphire-preferred": ["sapphire preferred", "csp", "sapphire"],
    "chase-freedom-unlimited": ["freedom unlimited", "cfu"],
    "citi-double-cash": ["double cash", "citi double cash"],
    "capital-one-savor": ["savor", "savorone"],
    "capital-one-venture": ["venture", "venture rewards"],
    "capital-one-quicksilver": ["quicksilver"],
    "wells-fargo-active-cash": ["active cash"],
    "discover-it-cash-back": ["discover it", "discover it cash back"],
    "discover-it-secured": ["discover it secured", "discover secured"],
    "bofa-customized-cash-rewards": ["customized cash", "bank of america customized cash", "bofa customized cash"],
    "hdfc-millennia": ["hdfc millennia", "millennia hdfc"],
    "hdfc-regalia-gold": ["regalia gold", "regalia"],
    "sbi-simplyclick": ["simplyclick", "simply click", "sbi simplyclick"],
    "sbi-cashback": ["sbi cashback", "cashback sbi", "sbi cash back"],
    "icici-amazon-pay": ["amazon pay icici", "amazon pay", "amazon icici"],
    "axis-flipkart": ["flipkart axis", "flipkart"],
    "axis-ace": ["axis ace", "ace"],
    "idfc-first-millennia": ["idfc millennia", "idfc first millennia"],
    "idfc-first-wow": ["idfc wow", "wow card", "idfc first wow"],
    "hdfc-tata-neu-infinity": ["tata neu", "tata neu infinity", "neu infinity"],
}


def _build_aliases(cards: list[Card]) -> list[tuple[str, str]]:
    aliases: dict[str, str] = {}
    for card in cards:
        name = re.sub(r"[^a-z0-9&+ ]+", " ", card.name.lower())
        aliases[" ".join(name.split())] = card.id
        aliases[card.id.replace("-", " ")] = card.id
        core = [w for w in name.split() if w not in _GENERIC]
        if len(core) >= 2:
            aliases[" ".join(core)] = card.id
        for alias in _MANUAL_ALIASES.get(card.id, []):
            aliases.setdefault(alias, card.id)
    return sorted(aliases.items(), key=lambda kv: -len(kv[0]))


@lru_cache
def load_catalog(cards_dir: str, code: str) -> Catalog:
    return Catalog.load(Path(cards_dir), code)
