"""CardPilot's agent, as a LangGraph state graph.

    START
      |
    guard_in ----(refused, or a card from another country)----------------+
      |                                                                   |
    agent  <------+     the model picks tools, at most max_tool_calls     |
      |           |                                                       |
      +--> tools -+     search_docs, get_card, compare_cards, estimate    |
      |                                                                   |
    compose             the model writes a CitedAnswer from the evidence  |
      |                                                                   |
    verify              drop sentences with unknown ids or unsupported    |
      |                 numbers, fall back to extracted facts             |
    finalize  <-----------------------------------------------------------+
      |                 advice wording removed, sources numbered,
     END                disclaimer and "as of" date added

The flow is fixed and the model only works inside it. That makes every answer auditable: what
was asked, which tools ran, what they returned, and which evidence each sentence rests on.
"""

from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .catalog import Catalog
from .guards import notice_for, screen_input, screen_sentence
from .llm import Models
from .retrieval import Knowledge
from .schemas import CitedAnswer, Evidence
from .tools import ToolBox
from .vocab import expansion


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class State(TypedDict, total=False):
    question: str
    notice: str | None
    refused: str | None
    advice_request: bool
    ranking_request: bool
    history: list[dict]  # earlier turns of this session, already redacted: question, answer, mentioned
    list_attribute: str | None  # set when the question is about many cards at once
    mentioned: list[str]
    other_country: list[str]
    messages: Annotated[list[AnyMessage], add_messages]
    evidence: Annotated[dict[str, Evidence], _merge]
    tool_calls: int
    sentences: list[dict]
    engine: str
    trace: Annotated[list[dict], operator.add]
    result: dict


AGENT_PROMPT = """You are CardPilot, a careful guide to credit cards in {country}.
You only know what your tools return. Use them before stating any fact.
- Use get_card for a named card, compare_cards for two or three cards, list_cards for questions about many cards at once (the APRs of all cards, which cards have no annual fee, top or best cards), search_docs for anything else, and estimate_rewards only when the user gives monthly amounts.
- Category keys for estimate_rewards and list_cards earn_<category> are {categories}.
- CardPilot does not rank or grade cards. For "top", "best" or "popular" cards, use list_cards with the attribute the user asked about, or overview.
- Earlier turns of this conversation come before the question. Use them to understand words like "these" or "it", and look facts up again with the tools.
- Everything is about {country} only. Never use outside knowledge about cards, and never mention cards from other countries.
- Call the tools you need, then stop calling tools. Do not write the final answer here."""

COMPOSE_PROMPT = """You are CardPilot. Write the answer to the user's question about credit cards in {country}, using only the evidence below.
Rules:
- 2 to 5 short, plain sentences. When the question covers many cards, write one short sentence per card, up to 12. Each sentence lists the ids of the evidence that supports it, like ["E2"].
- Every number (fees, percents, amounts) must appear in the cited evidence exactly.
- Explain and compare. Never recommend a card, never say which card the user should get, and never say "best for you".{advice_note}
- Earlier turns of the conversation may come before the question. Use them only to understand what the user means, never as evidence.
- If the evidence does not answer the question, say so in one sentence and cite nothing.

Evidence:
{evidence}"""

ADVICE_NOTE = "\n- The user asked which card to choose. Say kindly that you can't choose for them, then lay out the differences that matter."
RANKING_NOTE = (
    "\n- The user asked for top, best or popular cards. CardPilot does not rank cards, and a note saying so is added "
    "for you. Do not call any card top, best or popular. Give the facts card by card."
)

REFERS_BACK = re.compile(
    r"\b(it|its|it's|they|them|their|these|those|this card|that card|the same|both|either)\b", re.IGNORECASE
)
LIST_WORDS = re.compile(
    r"\b(all|every|each|list|which cards|what cards|any cards|the cards|these cards|those cards|cards (?:with|that|have|"
    r"charge|offer|earn|give)|aprs|annual fees|interest rates|earn rates|forex (?:fees|markups)|foreign transaction fees)\b",
    re.IGNORECASE,
)
ATTRIBUTE_WORDS: list[tuple[str, str]] = [
    ("intro_apr", r"\bintro(?:ductory)?\b|0% apr|zero percent|balance transfer offer"),
    ("foreign_fee", r"foreign|forex|abroad|international|overseas|markup"),
    ("apr", r"\baprs?\b|interest rates?|\binterest\b|finance charges?"),
    ("credit_level", r"credit score|eligib|who can get|qualify|approval|income"),
    ("welcome_offer", r"welcome|sign ?up bonus|joining bonus|\bbonus"),
    ("earn_dining", r"dining|restaurants?|food delivery|swiggy|zomato"),
    ("earn_groceries", r"grocer|supermarket"),
    ("earn_gas", r"\bgas\b|petrol|\bfuel\b"),
    ("earn_fuel", r"\bfuel\b|petrol|\bgas\b"),
    ("earn_travel", r"travel|flights?|hotels?|airlines?"),
    ("earn_online_shopping", r"online shopping|amazon|flipkart|e-?commerce"),
    ("earn_streaming", r"streaming|netflix|spotify"),
    ("earn_bills_utilities", r"bills?|utilit|electricity|recharge"),
    ("annual_fee", r"\bfees?\b|cost|charges?"),
]

COMPARE_WORDS = re.compile(r"\b(compare|comparison|vs\.?|versus|difference|differ|better)\b", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "the a an is are does do what how much on of for to in and or my i it card cards with any this that".split()
)
QUESTION_WORDS = frozenset(
    "which should get choose pick better vs versus between me you need want know can whats tell about work works compare "
    "comparison difference differ one ones two".split()
)
NUMBER = re.compile(r"(?:[$₹]\s?)?\d[\d,]*(?:\.\d+)?%?")


def _content(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1}


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[,\s$₹]", "", n).rstrip(".") for n in NUMBER.findall(text)}


def _tick(state_trace: list[dict], step: str, detail: str, started: float) -> list[dict]:
    return [{"step": step, "detail": detail, "ms": int((time.perf_counter() - started) * 1000)}]


def build_graph(
    knowledge: Knowledge, models: Models, others: list[Catalog], max_tool_calls: int = 6, max_question_chars: int = 800
):
    """One compiled graph per country. The country is fixed when the graph is built."""
    catalog = knowledge.catalog
    country = catalog.country
    box = ToolBox(knowledge)
    categories = ", ".join(k for k, _ in country.categories)

    # -------------------------------------------------------------- guard_in
    def list_attribute_for(text: str) -> str:
        keys = {k for k, _ in country.categories}
        for attribute, pattern in ATTRIBUTE_WORDS:
            if attribute.startswith("earn_") and attribute[len("earn_") :] not in keys:
                continue
            if re.search(pattern, text, re.IGNORECASE):
                return attribute
        return "overview"

    def guard_in(state: State) -> dict:
        t = time.perf_counter()
        screened = screen_input(state["question"], max_question_chars)
        history = [h for h in (state.get("history") or []) if h.get("question")][-3:]
        mentioned = [c.id for c in catalog.mentions(screened.text)]
        other = [] if mentioned else [c.name for cat in others for c in cat.mentions(screened.text)]
        notes = ["clean" if not screened.removed else f"removed {', '.join(screened.removed)}"]
        if screened.refused:
            notes.append("refused")
        if not mentioned and not other and history and REFERS_BACK.search(screened.text):
            mentioned = [m for m in history[-1].get("mentioned", []) if catalog.get(m)]
            if mentioned:
                notes.append("follow-up about the previous cards")
        list_attribute = None
        if not mentioned and (screened.ranking_request or LIST_WORDS.search(screened.text)):
            list_attribute = list_attribute_for(screened.text)
            notes.append(f"about every card, {list_attribute}")
        if mentioned:
            notes.append(f"cards {', '.join(mentioned)}")
        if history:
            notes.append(f"{len(history)} earlier turns")
        earlier: list[AnyMessage] = []
        for turn in history:
            earlier += [HumanMessage(turn["question"]), AIMessage(turn.get("answer") or "")]
        return {
            "question": screened.text,
            "notice": notice_for(screened.removed),
            "refused": screened.refused,
            "advice_request": screened.advice_request,
            "ranking_request": screened.ranking_request,
            "history": history,
            "list_attribute": list_attribute,
            "mentioned": mentioned,
            "other_country": other,
            "tool_calls": 0,
            "evidence": {},
            "messages": [
                SystemMessage(AGENT_PROMPT.format(country=country.name, categories=categories)),
                *earlier,
                HumanMessage(screened.text),
            ],
            "trace": _tick([], "guard", ", ".join(notes), t),
        }

    def after_guard(state: State) -> str:
        if state.get("refused") or state.get("other_country"):
            return "finalize"
        return "agent"

    # -------------------------------------------------------------- agent
    def planned_calls(state: State) -> list[dict]:
        """The same tool choices the model would make, by rules. Used offline and as a safety net."""
        q, mentioned = state["question"], state.get("mentioned", [])
        calls = []
        if state.get("list_attribute"):
            calls.append({"name": "list_cards", "args": {"attribute": state["list_attribute"]}})
            return [{**c, "id": f"plan{i}", "type": "tool_call"} for i, c in enumerate(calls)]
        if len(mentioned) >= 2 and (COMPARE_WORDS.search(q) or state.get("advice_request")):
            calls.append({"name": "compare_cards", "args": {"cards": mentioned[:3]}})
        else:
            calls += [{"name": "get_card", "args": {"card": m}} for m in mentioned[:3]]
        calls.append({"name": "search_docs", "args": {"query": q}})
        return [{**c, "id": f"plan{i}", "type": "tool_call"} for i, c in enumerate(calls)]

    def agent(state: State) -> dict:
        t = time.perf_counter()
        used = state.get("tool_calls", 0)
        if models.agent is None or state.get("engine") == "offline":
            if used:
                return {
                    "messages": [AIMessage("done")],
                    "engine": "offline",
                    "trace": _tick([], "agent", "offline plan done", t),
                }
            calls = planned_calls(state)
            return {
                "messages": [AIMessage("", tool_calls=calls)],
                "engine": "offline",
                "trace": _tick([], "agent", f"offline plan, {len(calls)} tool calls", t),
            }
        try:
            reply: AIMessage = models.agent.invoke(state["messages"])
        except Exception as exc:  # model down, rate limited, or timed out after retries and fallback
            calls = [] if used else planned_calls(state)
            return {
                "messages": [AIMessage("", tool_calls=calls) if calls else AIMessage("done")],
                "engine": "offline",
                "trace": _tick([], "agent", f"model unavailable ({type(exc).__name__}), answering offline", t),
            }
        calls = list(reply.tool_calls or [])
        if not calls and not used:
            # The model tried to answer from memory. Every answer must rest on evidence, so search first.
            calls = planned_calls(state)
            reply = AIMessage(reply.content or "", tool_calls=calls)
        if not used and state.get("list_attribute") and not any(c["name"] == "list_cards" for c in calls):
            # The question is plainly about every card. Search alone returns five passages, so list them all too.
            extra = {
                "name": "list_cards",
                "args": {"attribute": state["list_attribute"]},
                "id": "route0",
                "type": "tool_call",
            }
            calls.append(extra)
            reply = AIMessage(reply.content or "", tool_calls=calls)
        if used + len(calls) > max_tool_calls:
            calls = calls[: max(0, max_tool_calls - used)]
            reply = AIMessage(reply.content or "", tool_calls=calls)
        detail = ", ".join(c["name"] for c in calls) or "enough evidence"
        return {"messages": [reply], "engine": models.engine, "trace": _tick([], "agent", detail, t)}

    def after_agent(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "compose"

    # -------------------------------------------------------------- tools
    def tools(state: State) -> dict:
        t = time.perf_counter()
        last: AIMessage = state["messages"][-1]
        evidence = dict(state.get("evidence", {}))
        messages, notes = [], []
        for call in last.tool_calls:
            try:
                found, note = box.run(call["name"], call.get("args", {}))
            except Exception as exc:  # bad arguments from the model are reported back, not raised
                found, note = [], f"{call['name']} failed: {exc}"
            lines = []
            for item in found:
                duplicate = next((eid for eid, e in evidence.items() if e.text == item.text), None)
                eid = duplicate or f"E{len(evidence) + 1}"
                if not duplicate:
                    evidence[eid] = item.model_copy(update={"id": eid})
                elif item.kind and not evidence[eid].kind:
                    evidence[eid] = evidence[eid].model_copy(update={"kind": item.kind})
                lines.append(f"[{eid}] {item.text} (source: {item.title})")
            messages.append(ToolMessage("\n".join(lines) or note, tool_call_id=call["id"], name=call["name"]))
            notes.append(note)
        return {
            "messages": messages,
            "evidence": evidence,
            "tool_calls": state.get("tool_calls", 0) + len(last.tool_calls),
            "trace": _tick([], "tools", "; ".join(notes), t),
        }

    # -------------------------------------------------------------- compose
    def extractive(state: State, limit: int = 4) -> list[dict]:
        """An answer made only of evidence sentences: relevant to the question, no repeats, balanced across cards."""
        rows = [(eid, e) for eid, e in state.get("evidence", {}).items() if e.kind == "list"]
        if rows and (state.get("list_attribute") or not state.get("mentioned")):  # about every card: one sentence each
            return [{"text": e.text, "evidence_ids": [eid]} for eid, e in rows[:12]]
        mentioned = [c for c in (catalog.get(m) for m in state.get("mentioned", [])) if c]
        names = list(dict.fromkeys(c.name for c in mentioned))  # a list, so the answer order never depends on hashing
        ids = {c.id for c in mentioned}
        alias_text = " ".join(a for a, cid in catalog._aliases if cid in ids)
        name_words = set().union(*(_content(c.name) for c in mentioned)) | _content(alias_text) if mentioned else set()
        canon = _content(expansion(state["question"]))  # the document words for what was asked, like "annual fee"
        focus = (_content(state["question"]) | canon) - name_words - QUESTION_WORDS
        comparing = len(names) >= 2
        scored = []
        for order, (eid, e) in enumerate(state.get("evidence", {}).items()):
            words = _content(e.text) - name_words
            hits = len(focus & words) + len(canon & words)  # document words count double
            # In a comparison every fact about a named card is relevant. The focus words only rank them.
            if focus and not hits and not (comparing and e.card in names):
                continue
            bonus = 1.0 if e.card in names else 0.0
            scored.append((hits + bonus - order * 0.01, eid, e))
        scored.sort(key=lambda s: s[0], reverse=True)

        picked, seen = [], []

        def take(item) -> bool:
            _, eid, e = item
            words = _content(e.text) - name_words
            if any(card == e.card and len(words & w) >= 0.75 * max(1, min(len(words), len(w))) for card, w in seen):
                return False  # says the same thing about the same card as a sentence already picked
            seen.append((e.card, words))
            text = e.text if not e.card or e.card.lower() in e.text.lower() else f"{e.card}. {e.text}"
            picked.append({"text": text, "evidence_ids": [eid]})
            return True

        if comparing:  # take the cards in turn, and a card whose next fact repeats itself offers its following one
            buckets = {n: [s for s in scored if s[2].card == n] for n in names}
            while len(picked) < limit and any(buckets.values()):
                for n in names:
                    while buckets[n] and len(picked) < limit and not take(buckets[n].pop(0)):
                        pass
            scored = [s for s in scored if s[2].card not in names]
        for item in scored:
            if len(picked) >= limit:
                break
            take(item)
        return picked

    def compose(state: State) -> dict:
        t = time.perf_counter()
        evidence = state.get("evidence", {})
        if not evidence:
            return {"sentences": [], "trace": _tick([], "compose", "no evidence", t)}
        if models.composer is None or state.get("engine") == "offline":
            return {"sentences": extractive(state), "trace": _tick([], "compose", "extracted from evidence", t)}
        listing = "\n".join(f"[{eid}] {e.text} (source: {e.title})" for eid, e in evidence.items())
        note = ADVICE_NOTE if state.get("advice_request") else RANKING_NOTE if state.get("ranking_request") else ""
        prompt = COMPOSE_PROMPT.format(country=country.name, evidence=listing, advice_note=note)
        earlier: list[AnyMessage] = []
        for turn in state.get("history") or []:
            earlier += [HumanMessage(turn["question"]), AIMessage(turn.get("answer") or "")]
        try:
            answer: CitedAnswer = models.composer.invoke(
                [SystemMessage(prompt), *earlier, HumanMessage(state["question"])]
            )
            sentences = [s.model_dump() for s in answer.sentences]
            return {
                "sentences": sentences,
                "trace": _tick([], "compose", f"{len(sentences)} sentences by {models.engine}", t),
            }
        except Exception as exc:
            return {
                "sentences": extractive(state),
                "engine": "offline",
                "trace": _tick([], "compose", f"model unavailable ({type(exc).__name__}), extracted instead", t),
            }

    # -------------------------------------------------------------- verify
    def verify(state: State) -> dict:
        t = time.perf_counter()
        evidence = state.get("evidence", {})
        kept, dropped = [], 0
        for s in state.get("sentences", []):
            ids = [i for i in s.get("evidence_ids", []) if i in evidence]
            if not ids:
                dropped += 1
                continue
            supported = set().union(*(_numbers(evidence[i].text) for i in ids))
            if not _numbers(s["text"]) <= supported:
                dropped += 1  # a number the evidence does not contain
                continue
            kept.append({"text": s["text"].strip(), "evidence_ids": ids})
        if not kept and evidence and state.get("sentences") is not None:
            kept = extractive(state)
        return {"sentences": kept, "trace": _tick([], "verify", f"kept {len(kept)}, dropped {dropped}", t)}

    # -------------------------------------------------------------- finalize
    def finalize(state: State) -> dict:
        t = time.perf_counter()
        evidence = state.get("evidence", {})
        sources: list[dict] = []
        numbers: dict[str, int] = {}

        def cite(eid: str) -> int | None:
            e = evidence[eid]
            key = e.url or e.title
            if key not in numbers:
                sources.append({"n": len(sources) + 1, "title": e.title, "url": e.url, "card": e.card})
                numbers[key] = len(sources)
            return numbers[key]

        refused = bool(state.get("refused"))
        if state.get("refused"):
            sentences = [{"text": state["refused"], "sources": []}]
        elif state.get("other_country"):
            names = ", ".join(sorted(set(state["other_country"])))
            sentences = [
                {
                    "text": f"{names} is not in the {country.name} catalog. CardPilot keeps each country's cards separate, so start over and pick the other country to ask about it.",
                    "sources": [],
                }
            ]
            refused = True
        else:
            sentences, removed = [], 0
            ranking_note = state.get("ranking_request") and not state.get("advice_request")
            for s in state.get("sentences", []):
                if not screen_sentence(s["text"]):
                    removed += 1
                    continue
                if ranking_note and all(evidence[i].kind == "scope" for i in s["evidence_ids"]):
                    continue  # the note below already says what the catalog covers
                sentences.append(
                    {"text": s["text"], "sources": sorted({n for n in (cite(i) for i in s["evidence_ids"]) if n})}
                )
            if state.get("advice_request"):
                sentences.insert(
                    0,
                    {
                        "text": "I can't tell you which card to get, but here is how they differ so you can decide.",
                        "sources": [],
                    },
                )
            elif ranking_note:
                sentences.insert(
                    0,
                    {
                        "text": f"CardPilot doesn't rank or grade cards. It covers {len(catalog.cards)} well-known cards "
                        f"in {'the ' if country.code == 'us' else ''}{country.name}, and here is what each one publishes.",
                        "sources": [],
                    },
                )
            if not sentences:
                sentences = [
                    {
                        "text": f"I couldn't find that in the {country.name} sources I have. Try naming the card, or ask about fees, rewards or how cards work.",
                        "sources": [],
                    }
                ]
        result = {
            "sentences": sentences,
            "sources": sources,
            "as_of": catalog.as_of,
            "disclaimer": country.disclaimer,
            "notice": state.get("notice"),
            "refused": refused,
            "engine": state.get("engine", models.engine),
        }
        return {
            "result": result,
            "trace": _tick([], "finalize", f"{len(sentences)} sentences, {len(sources)} sources", t),
        }

    g = StateGraph(State)
    for name, fn in [
        ("guard_in", guard_in),
        ("agent", agent),
        ("tools", tools),
        ("compose", compose),
        ("verify", verify),
        ("finalize", finalize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "guard_in")
    g.add_conditional_edges("guard_in", after_guard, {"agent": "agent", "finalize": "finalize"})
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "compose": "compose"})
    g.add_edge("tools", "agent")
    g.add_edge("compose", "verify")
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def ask(graph, question: str, history: list[dict] | None = None, callbacks: list[Any] | None = None) -> dict:
    """Runs one question through the graph and returns the API response, trace included.

    `history` is the session's earlier turns. The result carries a `turn` entry, the redacted question and a
    short form of the answer, for the caller to keep as history. Refused questions leave no turn."""
    config = {"recursion_limit": 25}
    if callbacks:
        config["callbacks"] = callbacks
    state = graph.invoke({"question": question, "history": history or []}, config=config)
    result = {**state["result"], "trace": state.get("trace", [])}
    if not result["refused"]:
        result["turn"] = {
            "question": state["question"],
            "answer": " ".join(s["text"] for s in result["sentences"])[:600],
            "mentioned": state.get("mentioned", []),
        }
    return result
