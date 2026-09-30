"""The model gateway. One place decides which model answers, with timeouts, retries and a fallback.

- groq: free tier, gpt-oss-120b with gpt-oss-20b as the fallback.
- claude: Anthropic's API, pay per use. Claude Haiku by default.
- offline: no model at all. CardPilot still answers with extracted, cited facts.

If the chosen model fails even after retries and the fallback, the request is answered offline
instead of failing, and the answer says which engine produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from .config import Settings
from .schemas import CitedAnswer
from .tools import tool_specs


@dataclass
class Models:
    engine: str
    agent: Runnable | None  # chat model with the tools bound
    composer: Runnable | None  # chat model that must return a CitedAnswer


def _groq(settings: Settings, model: str) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=0,
        timeout=settings.model_timeout_s,
        max_retries=2,
        api_key=settings.groq_api_key,
    )


def _claude(settings: Settings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.claude_model,
        temperature=0,
        timeout=settings.model_timeout_s,
        max_retries=2,
        max_tokens=1024,
        api_key=settings.anthropic_api_key,
    )


def build_models(settings: Settings) -> Models:
    engine = settings.chosen_engine()
    if engine == "offline":
        return Models("offline", None, None)
    tools = tool_specs()
    if engine == "claude":
        if not settings.anthropic_api_key:
            raise ValueError("CARDPILOT_ENGINE=claude needs ANTHROPIC_API_KEY.")
        m = _claude(settings)
        return Models("claude", m.bind_tools(tools), m.with_structured_output(CitedAnswer))
    if not settings.groq_api_key:
        raise ValueError("CARDPILOT_ENGINE=groq needs GROQ_API_KEY.")
    primary, backup = _groq(settings, settings.groq_model), _groq(settings, settings.groq_fallback_model)
    return Models(
        "groq",
        primary.bind_tools(tools).with_fallbacks([backup.bind_tools(tools)]),
        primary.with_structured_output(CitedAnswer).with_fallbacks([backup.with_structured_output(CitedAnswer)]),
    )
