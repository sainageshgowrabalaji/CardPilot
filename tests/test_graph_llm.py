"""The model paths, with scripted fake models, so the tests need no key and no network."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from cardpilot.graph import ask
from cardpilot.llm import Models
from cardpilot.schemas import CitedAnswer, CitedSentence

from .conftest import graph_for


class Script:
    """A fake model. Returns queued replies in order and records every prompt it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen: list[list] = []

    def __call__(self, messages):
        self.seen.append(list(messages))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply(messages) if callable(reply) else reply


def call(name: str, i: int = 0, **args) -> dict:
    return {"name": name, "args": args, "id": f"call_{name}_{i}", "type": "tool_call"}


def models(agent: Script, composer: Script) -> Models:
    return Models("fake", RunnableLambda(agent), RunnableLambda(composer))


def evidence_ids(messages) -> list[str]:
    import re

    text = " ".join(m.content for m in messages if hasattr(m, "content") and isinstance(m.content, str))
    return re.findall(r"\[(E\d+)\]", text)


def first_fact(messages, containing: str) -> tuple[str, str]:
    for m in messages:
        for line in str(m.content).splitlines():
            if line.startswith("[E") and containing in line:
                eid, rest = line[1:].split("] ", 1)
                return eid, rest.split(" (source:")[0]
    raise AssertionError(f"no evidence containing {containing!r}")


def test_tool_use_then_a_cited_answer(knowledge):
    agent = Script(AIMessage("", tool_calls=[call("get_card", card="Amex Gold")]), AIMessage("done"))

    def compose(messages):
        eid, fact = first_fact(messages, "annual fee of $325")
        return CitedAnswer(sentences=[CitedSentence(text=fact, evidence_ids=[eid])])

    r = ask(graph_for(knowledge, "us", models(agent, Script(compose))), "What is the Amex Gold annual fee?")
    assert r["engine"] == "fake"
    assert "$325" in r["sentences"][0]["text"] and r["sentences"][0]["sources"] == [1]
    assert any(isinstance(m, ToolMessage) for m in agent.seen[1])  # the model saw the tool result


def test_unsupported_sentences_are_dropped(knowledge):
    agent = Script(AIMessage("", tool_calls=[call("get_card", card="Amex Gold")]), AIMessage("done"))

    def compose(messages):
        eid, fact = first_fact(messages, "annual fee of $325")
        return CitedAnswer(
            sentences=[
                CitedSentence(text=fact, evidence_ids=[eid]),
                CitedSentence(text="The annual fee is $999.", evidence_ids=[eid]),  # number not in the evidence
                CitedSentence(text="It has great lounges.", evidence_ids=["E99"]),  # evidence that does not exist
                CitedSentence(text="Also no cites here.", evidence_ids=[]),
            ]
        )

    r = ask(graph_for(knowledge, "us", models(agent, Script(compose))), "Amex Gold annual fee?")
    texts = [s["text"] for s in r["sentences"]]
    assert len(texts) == 1 and "$325" in texts[0]
    assert "dropped 3" in next(t["detail"] for t in r["trace"] if t["step"] == "verify")


def test_advice_wording_from_the_model_is_removed(knowledge):
    agent = Script(AIMessage("", tool_calls=[call("get_card", card="Amex Gold")]), AIMessage("done"))

    def compose(messages):
        eid, fact = first_fact(messages, "annual fee of $325")
        return CitedAnswer(
            sentences=[
                CitedSentence(text=fact, evidence_ids=[eid]),
                CitedSentence(text="You should get this card.", evidence_ids=[eid]),
            ]
        )

    r = ask(graph_for(knowledge, "us", models(agent, Script(compose))), "Amex Gold annual fee?")
    assert not any("should get" in s["text"] for s in r["sentences"])


def test_a_model_outage_falls_back_to_offline(knowledge):
    agent = Script(TimeoutError("model timed out"))
    r = ask(
        graph_for(knowledge, "us", models(agent, Script(TimeoutError()))),
        "Does the Amex Gold charge foreign transaction fees?",
    )
    assert r["engine"] == "offline"
    assert r["sentences"] and r["sentences"][0]["sources"]
    assert "model unavailable" in r["trace"][1]["detail"]


def test_a_composer_outage_falls_back_to_extracted_facts(knowledge):
    agent = Script(AIMessage("", tool_calls=[call("get_card", card="Amex Gold")]), AIMessage("done"))
    r = ask(
        graph_for(knowledge, "us", models(agent, Script(RuntimeError("503")))), "Amex Gold foreign transaction fee?"
    )
    assert r["engine"] == "offline" and r["sentences"][0]["sources"]


def test_a_model_that_skips_tools_is_made_to_search(knowledge):
    agent = Script(AIMessage("The Amex Gold fee is $250."), AIMessage("done"))  # from memory, and out of date

    def compose(messages):
        assert evidence_ids(messages), "compose must receive evidence"
        eid, fact = first_fact(messages, "annual fee of $325")
        return CitedAnswer(sentences=[CitedSentence(text=fact, evidence_ids=[eid])])

    r = ask(graph_for(knowledge, "us", models(agent, Script(compose))), "What is the Amex Gold annual fee?")
    assert "$325" in r["sentences"][0]["text"]


def test_tool_calls_are_capped(knowledge):
    agent = Script(lambda m: AIMessage("", tool_calls=[call("search_docs", len(m), query="fees")]))
    r = ask(graph_for(knowledge, "us", models(agent, Script(CitedAnswer(sentences=[]))), max_tool_calls=3), "fees?")
    tool_steps = [t for t in r["trace"] if t["step"] == "tools"]
    assert len(tool_steps) == 3


def test_bad_tool_arguments_are_reported_not_raised(knowledge):
    agent = Script(
        AIMessage("", tool_calls=[call("compare_cards", cards="not a list"), call("no_such_tool", 1)]),
        AIMessage("done"),
    )
    r = ask(graph_for(knowledge, "us", models(agent, Script(CitedAnswer(sentences=[])))), "compare?")
    detail = next(t["detail"] for t in r["trace"] if t["step"] == "tools")
    assert "failed" in detail and "unknown tool" in detail


def test_the_model_never_sees_a_card_number(knowledge):
    agent = Script(AIMessage("", tool_calls=[call("get_card", card="Quicksilver")]), AIMessage("done"))
    composer = Script(CitedAnswer(sentences=[]))
    ask(graph_for(knowledge, "us", models(agent, composer)), "card 4111 1111 1111 1111, fee on quicksilver?")
    sent = str(agent.seen) + str(composer.seen)
    assert "4111" not in sent and "[card number removed]" in sent


def test_the_model_is_told_its_country(knowledge):
    agent = Script(AIMessage("done"))
    ask(graph_for(knowledge, "in", models(agent, Script(CitedAnswer(sentences=[])))), "What is a forex markup?")
    assert "India" in agent.seen[0][0].content


@pytest.mark.parametrize("engine", ["groq", "claude"])
def test_real_models_are_built_without_calling_them(engine):
    from pydantic import SecretStr

    from cardpilot.config import Settings
    from cardpilot.llm import build_models

    s = Settings(
        engine=engine, groq_api_key=SecretStr("gsk_test"), anthropic_api_key=SecretStr("sk-ant-test"), _env_file=None
    )
    m = build_models(s)
    assert m.engine == engine and m.agent is not None and m.composer is not None


def test_a_missing_key_fails_fast():
    from cardpilot.config import Settings
    from cardpilot.llm import build_models

    with pytest.raises(ValueError):
        build_models(Settings(engine="groq", groq_api_key=None, _env_file=None))


def test_auto_engine_order():
    from pydantic import SecretStr

    from cardpilot.config import Settings

    assert (
        Settings(engine="auto", groq_api_key=None, anthropic_api_key=None, _env_file=None).chosen_engine() == "offline"
    )
    assert (
        Settings(engine="auto", groq_api_key=None, anthropic_api_key=SecretStr("k"), _env_file=None).chosen_engine()
        == "claude"
    )
    assert (
        Settings(
            engine="auto", groq_api_key=SecretStr("k"), anthropic_api_key=SecretStr("k"), _env_file=None
        ).chosen_engine()
        == "groq"
    )
