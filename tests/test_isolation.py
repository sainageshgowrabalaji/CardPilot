"""A session for one country must never see, search or answer about the other country's cards."""

import pytest

from cardpilot.graph import ask
from cardpilot.store import SQLiteStore

GENERIC = [
    "What is the annual fee?",
    "Which cards have no foreign transaction fee?",
    "cash back on groceries",
    "What happens if I pay late?",
    "lounge access",
    "fuel surcharge waiver",
    "cashback on online shopping",
    "How does a secured card work?",
    "welcome bonus",
    "travel rewards points",
]


def test_store_refuses_a_chunk_from_another_country(knowledge, tmp_path):
    store = SQLiteStore(tmp_path, "us")
    india_chunks = knowledge["in"].catalog.chunks()[:2]
    with pytest.raises(ValueError):
        store.rebuild(india_chunks, knowledge["in"].embedder.embed([c.text for c in india_chunks]), "x")


@pytest.mark.parametrize("code,other", [("us", "in"), ("in", "us")])
def test_search_never_returns_the_other_country(knowledge, code, other):
    other_names = [c.name for c in knowledge[other].catalog.cards]
    own_ids = {c.id for c in knowledge[code].catalog.cards} | {None}
    for q in GENERIC + other_names:
        for mode in ("keyword", "vector", "hybrid"):
            for c in knowledge[code].search(q, k=8, mode=mode):
                assert c.country == code and c.card_id in own_ids, (q, mode, c.id)


@pytest.mark.parametrize("code,other", [("us", "in"), ("in", "us")])
def test_agent_refuses_cards_from_the_other_country(graphs, catalogs, code, other):
    for card in catalogs[other].cards[:5]:
        result = ask(graphs[code], f"Tell me about the {card.name}")
        assert result["refused"], card.name
        assert "not in the" in result["sentences"][0]["text"]
        assert result["sources"] == []


@pytest.mark.parametrize("code,other", [("us", "in"), ("in", "us")])
def test_answers_only_cite_their_own_country(graphs, catalogs, code, other):
    other_names = {c.name for c in catalogs[other].cards}
    for q in GENERIC:
        result = ask(graphs[code], q)
        for s in result["sources"]:
            assert s["card"] not in other_names
        for s in result["sentences"]:
            assert not any(name in s["text"] for name in other_names)
