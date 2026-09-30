import asyncio
import json

import pytest

from cardpilot import mcp_server
from cardpilot.config import get_settings


@pytest.fixture(autouse=True)
def offline(monkeypatch, settings):
    monkeypatch.setenv("CARDPILOT_ENGINE", "offline")
    monkeypatch.setenv("CARDPILOT_INDEX_DIR", str(settings.index_dir))
    for cached in (get_settings, mcp_server.knowledge, mcp_server.graph):
        cached.cache_clear()
    yield
    for cached in (get_settings, mcp_server.knowledge, mcp_server.graph):
        cached.cache_clear()


def call(name: str, args: dict):
    result = asyncio.run(mcp_server.server.call_tool(name, args))
    return result


def payload(result) -> dict:
    return json.loads(result.content[0].text)


def test_tools_are_listed_read_only():
    tools = asyncio.run(mcp_server.server.list_tools())
    names = {t.name for t in tools}
    assert names == {"search_card_docs", "get_card", "list_cards", "compare_cards", "estimate_rewards", "ask_cardpilot"}
    for t in tools:
        assert t.annotations.read_only_hint and t.input_schema["properties"]["country"]["enum"] == ["us", "in"]


def test_get_card():
    body = payload(call("get_card", {"country": "in", "card": "HDFC Millennia"}))
    assert body["facts"] and all(f["card"] == "HDFC Bank Millennia Credit Card" for f in body["facts"])


def test_search_stays_in_its_country():
    body = payload(call("search_card_docs", {"country": "us", "query": "HDFC Millennia cashback"}))
    assert all("HDFC" not in (r["card"] or "") for r in body["results"])


def test_compare_and_estimate():
    table = payload(call("compare_cards", {"country": "us", "cards": ["Amex Gold", "Savor"]}))
    assert len(table["columns"]) == 2
    est = payload(call("estimate_rewards", {"country": "in", "monthly_spend": {"dining": 5000}}))
    assert est["rows"] and est["categories"][0] == "dining"


def test_the_full_agent_as_a_tool():
    body = payload(call("ask_cardpilot", {"country": "us", "question": "Tell me about the HDFC Millennia"}))
    assert body["refused"]


def test_an_unknown_country_is_rejected():
    # In process the server raises; over stdio the client receives the same message with is_error set.
    with pytest.raises(Exception, match="Input should be 'us' or 'in'"):
        call("search_card_docs", {"country": "uk", "query": "fees"})


def test_list_cards():
    body = payload(call("list_cards", {"country": "us", "attribute": "apr"}))
    assert len(body["cards"]) == 13  # the catalog line plus 12 cards
    assert "does not rank" in body["cards"][0]["text"]
