"""The agent's tools. All read-only, all locked to one country.

The model chooses which tool to call and with what words. It never chooses the country: each
ToolBox is built for one country's knowledge and cannot see any other.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .retrieval import Knowledge
from .schemas import Evidence


class SearchDocs(BaseModel):
    """Search this country's card documents and consumer guides. Use it for any fact you state."""

    query: str = Field(description="What to look for, in plain words, like 'Amex Gold foreign transaction fee'.")


class GetCard(BaseModel):
    """Get the published facts for one card: fees, earn rates, limits and benefits."""

    card: str = Field(description="The card's name as the user said it, like 'Sapphire Preferred'.")


class CompareCards(BaseModel):
    """Get side-by-side facts for two or three cards."""

    cards: list[str] = Field(description="Two or three card names.")


class EstimateRewards(BaseModel):
    """Estimate yearly rewards for every card from monthly spending per category."""

    monthly_spend: dict[str, float] = Field(
        description="Monthly amount per category key, like {'dining': 300, 'groceries': 500}."
    )


ListAttribute = Literal[
    "overview",
    "annual_fee",
    "apr",
    "intro_apr",
    "foreign_fee",
    "credit_level",
    "welcome_offer",
    "earn_dining",
    "earn_groceries",
    "earn_gas",
    "earn_fuel",
    "earn_travel",
    "earn_online_shopping",
    "earn_streaming",
    "earn_bills_utilities",
    "earn_other",
]


class ListCards(BaseModel):
    """One fact for every card in this country's catalog, each with its source. Use it for questions about many
    cards at once, like the APRs of all cards, which cards have no annual fee, or the top or best cards.
    CardPilot does not rank cards, so for "top" or "best" use overview or the attribute the user named."""

    attribute: ListAttribute = Field(
        description="overview, annual_fee, apr (purchase APR or interest rate), intro_apr, foreign_fee, credit_level, "
        "welcome_offer, or earn_<category> for an earn rate, like earn_dining."
    )


TOOL_SCHEMAS = {
    "search_docs": SearchDocs,
    "get_card": GetCard,
    "compare_cards": CompareCards,
    "estimate_rewards": EstimateRewards,
    "list_cards": ListCards,
}


def tool_specs() -> list[dict]:
    """OpenAI-style function specs, which both Groq and Claude accept through LangChain."""
    specs = []
    for name, schema in TOOL_SCHEMAS.items():
        params = schema.model_json_schema()
        params.pop("title", None)
        specs.append(
            {"type": "function", "function": {"name": name, "description": schema.__doc__, "parameters": params}}
        )
    return specs


class ToolBox:
    def __init__(self, knowledge: Knowledge):
        self.k = knowledge

    def run(self, name: str, args: dict) -> tuple[list[Evidence], str]:
        """Runs one tool call. Returns the evidence found and a one-line summary for the trace."""
        if name not in TOOL_SCHEMAS:
            return [], f"unknown tool {name}"
        parsed = TOOL_SCHEMAS[name].model_validate(args)
        return getattr(self, name)(parsed)

    def search_docs(self, a: SearchDocs) -> tuple[list[Evidence], str]:
        mentioned = [c.id for c in self.k.catalog.mentions(a.query)]
        chunks = self.k.search(a.query, k=5, card_ids=mentioned or None)
        items = [Evidence(id="", text=c.text, title=c.title, url=c.url, card=c.card_name) for c in chunks]
        return items, f"search_docs({a.query!r}) found {len(items)}"

    def get_card(self, a: GetCard) -> tuple[list[Evidence], str]:
        card = self.k.catalog.get(a.card)
        if card is None:
            return [], f"get_card({a.card!r}) is not in the {self.k.catalog.country.name} catalog"
        items = self.k.catalog.card_evidence(card)
        return items, f"get_card({card.name}) returned {len(items)} facts"

    def compare_cards(self, a: CompareCards) -> tuple[list[Evidence], str]:
        items: list[Evidence] = []
        names = []
        for name in a.cards[:3]:
            card = self.k.catalog.get(name)
            if card:
                names.append(card.name)
                items += self.k.catalog.card_evidence(card)[:6]
        return items, f"compare_cards({', '.join(names) or 'none found'})"

    def estimate_rewards(self, a: EstimateRewards) -> tuple[list[Evidence], str]:
        try:
            result = self.k.catalog.estimate(a.monthly_spend)
        except ValueError as exc:
            return [], f"estimate_rewards failed: {exc}"
        money = self.k.catalog.country.money
        items = [
            Evidence(
                id="",
                text=f"Estimated yearly rewards on the {r['name']} are {money(r['yearly_rewards'])}, "
                f"or {money(r['net'])} after its {money(r['annual_fee'])} annual fee.",
                title="CardPilot estimate from the issuers' published earn rates",
                url=None,
                card=r["name"],
            )
            for r in result["rows"][:5]
        ]
        items.append(Evidence(id="", text=result["note"], title="How the estimate works", url=None))
        return items, f"estimate_rewards over {len(a.monthly_spend)} categories"

    def list_cards(self, a: ListCards) -> tuple[list[Evidence], str]:
        catalog = self.k.catalog
        country = catalog.country
        scope = Evidence(
            id="",
            text=f"CardPilot's {country.name} catalog has {len(catalog.cards)} well-known cards. It does not rank or grade them.",
            title=f"CardPilot {country.name} catalog, as of {catalog.as_of}",
            url=None,
            kind="scope",
        )
        items = [scope]
        try:
            for card in catalog.cards:
                text, src = catalog.attribute(card, a.attribute)
                items.append(Evidence(id="", text=text, title=src.title, url=src.url, card=card.name, kind="list"))
        except ValueError as exc:
            return [scope], f"list_cards({a.attribute}) failed: {exc}"
        return items, f"list_cards({a.attribute}) returned {len(items) - 1} cards"
