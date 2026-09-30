"""Typed shapes shared across CardPilot. Every model output and tool result is validated against these."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One searchable piece of text, always tied to one country and one source page."""

    id: str
    country: str
    card_id: str | None = None
    card_name: str | None = None
    kind: str = "card"  # card, fact or general
    text: str
    url: str | None = None
    title: str
    as_of: str | None = None
    score: float = 0.0


class Evidence(BaseModel):
    """Something a tool found, with an id the answer must cite, like E3."""

    id: str
    text: str
    title: str
    url: str | None = None
    card: str | None = None
    kind: str | None = None  # "list" for one row of a list_cards result, "scope" for the catalog line


class CitedSentence(BaseModel):
    text: str = Field(description="One short factual sentence in plain English.")
    evidence_ids: list[str] = Field(description="Ids of the evidence that supports this sentence, like ['E1', 'E3'].")


class CitedAnswer(BaseModel):
    """The only shape an answer may take. Sentences without valid evidence are removed later."""

    sentences: list[CitedSentence] = Field(description="The answer, one sentence per item, each with its evidence ids.")


class ToolTrace(BaseModel):
    step: str
    detail: str
    ms: int
