import numpy as np
import pytest

from cardpilot.config import Settings
from cardpilot.embeddings import GloveEmbedder
from cardpilot.retrieval import load_knowledge
from cardpilot.schemas import Chunk
from cardpilot.store import fts_query, rrf


def chunk(i: str) -> Chunk:
    return Chunk(id=i, country="us", text=i, title=i)


def test_rrf_rewards_agreement():
    a = [chunk("x"), chunk("y"), chunk("z")]
    b = [chunk("y"), chunk("w")]
    fused = rrf([a, b], 3)
    assert fused[0].id == "y"
    assert {c.id for c in fused} <= {"x", "y", "z", "w"}


@pytest.mark.parametrize("text", ['fee" OR 1=1 --', "NEAR(annual fee)", "*", "", "a"])
def test_fts_query_cannot_inject(text, knowledge):
    q = fts_query(text)
    assert all(part.startswith('"') and part.endswith('"') for part in q.split(" OR ") if part)
    knowledge["us"].search(text, mode="keyword")  # must not raise


def test_embedder_is_normalised():
    e = GloveEmbedder()
    v = e.embed(["annual fee on the card", "zzzz qqqq"])
    assert v.shape == (2, e.dim)
    assert np.linalg.norm(v[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.isfinite(v).all()


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid"])
def test_each_mode_finds_the_fact(knowledge, mode):
    results = knowledge["us"].search("Amex Gold foreign transaction fee", k=5, mode=mode)
    assert any(c.card_id == "amex-gold" and "foreign" in c.text.lower() for c in results)


def test_card_filter_keeps_general_guides(knowledge):
    results = knowledge["us"].search(
        "what is a grace period on the Citi Double Cash", k=8, card_ids=["citi-double-cash"]
    )
    assert results
    assert {c.card_id for c in results} <= {"citi-double-cash", None}
    assert any(c.card_id is None for c in results)


def test_index_rebuilds_when_the_embedder_changes(settings, tmp_path):
    s = Settings(engine="offline", index_dir=tmp_path, _env_file=None)
    k = load_knowledge(s, "us")
    k.store.rebuild(k.catalog.chunks()[:3], k.embedder.embed(["a", "b", "c"]), "some-other-embedder")
    again = load_knowledge(s, "us")
    assert again.store.embedder_id.startswith(again.embedder.id + "+")
    assert again.store.count() == len(again.catalog.chunks())


def test_sqlite_files_are_separate_per_country(settings, knowledge):
    files = sorted(p.name for p in settings.index_dir.iterdir() if p.suffix in {".sqlite", ".db"})
    assert len(files) == 2 and files[0] != files[1]


def test_expansion_adds_document_words_and_keeps_the_question():
    from cardpilot.vocab import expand, expansion

    assert "annual" in expansion("What does it cost per year?") and "fee" in expansion("What does it cost per year?")
    assert "forex" in expansion("Any charge when I spend abroad?")
    assert expand("fees abroad").startswith("fees abroad")
    assert expansion("Tell me about the Venture") == ""


def test_expansion_can_be_turned_off(knowledge):
    plain = knowledge["us"].search("What does the Amex Gold cost per year?", k=5, expand_query=False)
    expanded = knowledge["us"].search("What does the Amex Gold cost per year?", k=5)
    assert any("annual fee" in c.text.lower() for c in expanded)
    assert [c.id for c in plain] != [c.id for c in expanded]


def test_index_rebuilds_when_the_card_data_changes(tmp_path):
    from cardpilot.retrieval import index_version

    s = Settings(engine="offline", index_dir=tmp_path, _env_file=None)
    k = load_knowledge(s, "us")
    chunks = k.catalog.chunks()
    stale = [c.model_copy(update={"text": c.text + " (old)"}) if i == 0 else c for i, c in enumerate(chunks)]
    k.store.rebuild(stale, k.embedder.embed([c.text for c in stale]), index_version(k.embedder, stale))
    again = load_knowledge(s, "us")
    assert again.store.embedder_id == index_version(again.embedder, chunks)
    assert not any("(old)" in c.text for c in again.search("annual fee", k=50, mode="keyword"))
