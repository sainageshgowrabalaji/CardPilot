"""Shared fixtures. Everything runs offline: no model key and no network are needed."""

from __future__ import annotations

import pytest

from cardpilot.catalog import Catalog
from cardpilot.config import Settings
from cardpilot.graph import build_graph
from cardpilot.llm import Models
from cardpilot.retrieval import load_knowledge

OFFLINE = Models("offline", None, None)


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return Settings(engine="offline", index_dir=tmp_path_factory.mktemp("index"), _env_file=None)


@pytest.fixture(scope="session")
def knowledge(settings):
    return {code: load_knowledge(settings, code) for code in ("us", "in")}


@pytest.fixture(scope="session")
def catalogs(knowledge) -> dict[str, Catalog]:
    return {code: k.catalog for code, k in knowledge.items()}


def graph_for(knowledge, code: str, models: Models = OFFLINE, max_tool_calls: int = 6):
    others = [k.catalog for c, k in knowledge.items() if c != code]
    return build_graph(knowledge[code], models, others, max_tool_calls)


@pytest.fixture(scope="session")
def graphs(knowledge):
    return {code: graph_for(knowledge, code) for code in knowledge}
