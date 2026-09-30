"""The production store, on a real Postgres with pgvector started inside the test (pgserver)."""

import pytest

pgserver = pytest.importorskip("pgserver")
pytest.importorskip("pgvector")

from cardpilot.config import Settings
from cardpilot.graph import ask
from cardpilot.retrieval import build_index, load_knowledge
from cardpilot.store import PostgresStore

from .conftest import graph_for


@pytest.fixture(scope="module")
def pg(tmp_path_factory):
    try:
        server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    except Exception as exc:  # no Postgres binary for this platform
        pytest.skip(f"embedded Postgres unavailable: {exc}")
    yield Settings(engine="offline", store="postgres", database_url=server.get_uri(), _env_file=None)
    server.cleanup()


@pytest.fixture(scope="module")
def pg_knowledge(pg):
    return {code: build_index(pg, code) for code in ("us", "in")}


def test_each_country_has_its_own_schema(pg_knowledge):
    assert isinstance(pg_knowledge["us"].store, PostgresStore)
    assert pg_knowledge["us"].store.schema == "cp_us" and pg_knowledge["in"].store.schema == "cp_in"
    assert pg_knowledge["us"].store.count() == len(pg_knowledge["us"].catalog.chunks())


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid"])
def test_search_modes(pg_knowledge, mode):
    results = pg_knowledge["us"].search("Amex Gold foreign transaction fee", k=5, mode=mode)
    assert any(c.card_id == "amex-gold" for c in results)
    assert all(c.country == "us" for c in results)


def test_card_filter(pg_knowledge):
    results = pg_knowledge["in"].search("annual fee", k=6, card_ids=["axis-ace"])
    assert results and {c.card_id for c in results} <= {"axis-ace", None}


def test_hostile_query_text_is_safe(pg_knowledge):
    pg_knowledge["us"].search("'; DROP TABLE cp_us.chunks; --", k=3)
    assert pg_knowledge["us"].store.count() > 0


def test_reopening_reuses_the_index(pg, pg_knowledge):
    k = load_knowledge(pg, "us")
    from cardpilot.retrieval import index_version

    assert k.store.embedder_id == index_version(k.embedder, k.catalog.chunks()) and k.store.count() > 0


def test_the_agent_runs_on_postgres(pg_knowledge):
    r = ask(graph_for(pg_knowledge, "us"), "Does the Amex Gold charge foreign transaction fees?")
    assert r["sentences"][0]["sources"]
