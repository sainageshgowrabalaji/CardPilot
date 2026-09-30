"""The agent's tools. All read-only, all locked to one country.

The model chooses which tool to call and with what words. It never chooses the country: each
ToolBox is built for one country's knowledge and cannot see any other.
"""

from __future__ import annotations

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


TOOL_SCHEMAS = {
    "search_docs": SearchDocs,
    "get_card": GetCard,
    "compare_cards": CompareCards,
    "estimate_rewards": EstimateRewards,
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
