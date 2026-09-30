"""CardPilot's tools as an MCP server, so any MCP client (Claude Desktop, Claude Code, Cursor, an agent
built on another framework) can use the same country-scoped knowledge.

Every tool is read-only and takes the country as an explicit argument. There is no tool that
searches both countries at once, so the isolation the web app has holds here too.

Run it over stdio:
    uv run cardpilot-mcp
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import __version__
from .config import get_settings
from .countries import COUNTRIES
from .graph import ask, build_graph
from .llm import build_models
from .retrieval import Knowledge, load_knowledge
from .tools import EstimateRewards, GetCard, SearchDocs, ToolBox

Country = Literal["us", "in"]
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

server = MCPServer(
    name="cardpilot",
    title="CardPilot",
    version=__version__,
    instructions=(
        "Facts about credit cards in the United States (us) or India (in), each with its source. "
        "Pick the country first and pass it to every tool. Education only, never financial advice."
    ),
)


@lru_cache
def knowledge(country: str) -> Knowledge:
    if country not in COUNTRIES:
        raise ValueError("country must be 'us' or 'in'")
    return load_knowledge(get_settings(), country)


@lru_cache
def graph(country: str):
    others = [knowledge(c).catalog for c in COUNTRIES if c != country]
    s = get_settings()
    return build_graph(knowledge(country), build_models(s), others, s.max_tool_calls, s.max_question_chars)


def _evidence(found) -> list[dict]:
    return [{"text": e.text, "card": e.card, "source_title": e.title, "source_url": e.url} for e in found]


@server.tool(annotations=READ_ONLY)
def search_card_docs(country: Country, query: str) -> dict:
    """Search one country's card documents and regulator guides. Returns passages with their sources."""
    found, note = ToolBox(knowledge(country)).search_docs(SearchDocs(query=query))
    return {"country": country, "note": note, "results": _evidence(found)}


@server.tool(annotations=READ_ONLY)
def get_card(country: Country, card: str) -> dict:
    """Published facts for one card (fees, earn rates, limits, benefits), each with its source."""
    found, note = ToolBox(knowledge(country)).get_card(GetCard(card=card))
    return {"country": country, "note": note, "facts": _evidence(found)}


@server.tool(annotations=READ_ONLY)
def compare_cards(country: Country, cards: list[str]) -> dict:
    """A side-by-side table for two or three cards from the same country, every cell cited."""
    catalog = knowledge(country).catalog
    ids = [c.id for c in (catalog.get(name) for name in cards) if c]
    if len(ids) < 2:
        return {
            "country": country,
            "error": "Name two or three cards from this country's catalog.",
            "catalog": [c.name for c in catalog.cards],
        }
    return {"country": country, **catalog.compare(ids)}


@server.tool(annotations=READ_ONLY)
def estimate_rewards(country: Country, monthly_spend: dict[str, float]) -> dict:
    """Yearly rewards per card from monthly spending per category. Plain arithmetic on published earn rates."""
    k = knowledge(country)
    keys = [key for key, _ in k.catalog.country.categories]
    try:
        return {
            "country": country,
            "categories": keys,
            **k.catalog.estimate(EstimateRewards(monthly_spend=monthly_spend).monthly_spend),
        }
    except ValueError as exc:
        return {"country": country, "error": str(exc), "categories": keys}


@server.tool(
    annotations=ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=False, open_world_hint=True
    )
)
def ask_cardpilot(country: Country, question: str) -> dict:
    """The full CardPilot agent: guardrails, tool use, a cited answer and its verification trace."""
    return ask(graph(country), question)


@server.resource("cardpilot://{country}/cards", name="cards", mime_type="application/json")
def card_list(country: str) -> dict:
    """Every card in one country's catalog, with its id."""
    return knowledge(country).catalog.listing()


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
