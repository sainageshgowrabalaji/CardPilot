"""The HTTP API and the web page.

Each browser session picks a country once and is locked to it. Every endpoint looks the country up
from the session, never from the request, so a request cannot reach the other country's data.

    POST /api/session      start a session for "us" or "in"
    GET  /api/suggestions  sample questions for the session's country
    POST /api/ask          one question through the agent graph, with sources and a trace
    GET  /api/cards        the country's card list and spending categories
    POST /api/compare      side-by-side facts for two or three cards
    POST /api/estimate     yearly rewards from monthly spending, plain arithmetic
    GET  /api/lessons      short lessons from the regulator's own pages
    GET  /api/tap          plain text for the iOS Shortcut that runs when a Wallet card is tapped
    GET  /healthz          readiness, engine and index sizes
    GET  /metrics          Prometheus counters
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .catalog import Catalog
from .config import Settings, get_settings
from .countries import COUNTRIES
from .graph import ask, build_graph
from .lessons import build_lessons
from .llm import build_models
from .retrieval import Knowledge, load_knowledge
from .store import describe

log = logging.getLogger("cardpilot.api")


# ------------------------------------------------------------------ sessions and limits


@dataclass
class Session:
    id: str
    country: str
    created: float
    last_seen: float
    # The last few redacted questions and short answers, so follow-ups like "what about these?" make sense.
    # Kept only in memory for the life of the session.
    history: list[dict] = field(default_factory=list)


class Sessions:
    """In-memory sessions with a time limit. Enough for one server; Redis would replace it for several."""

    def __init__(self, ttl_seconds: float, max_sessions: int = 5000):
        self.ttl = ttl_seconds
        self.max = max_sessions
        self._items: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, country: str) -> Session:
        now = time.time()
        with self._lock:
            self._sweep(now)
            if len(self._items) >= self.max:
                oldest = min(self._items.values(), key=lambda s: s.last_seen)
                del self._items[oldest.id]
            s = Session(secrets.token_urlsafe(24), country, now, now)
            self._items[s.id] = s
            return s

    def get(self, session_id: str | None) -> Session:
        now = time.time()
        with self._lock:
            s = self._items.get(session_id or "")
            if s is None or now - s.last_seen > self.ttl:
                self._items.pop(session_id or "", None)
                raise HTTPException(404, "Your session has ended. Pick your country to start again.")
            s.last_seen = now
            return s

    def _sweep(self, now: float) -> None:
        for sid in [sid for sid, s in self._items.items() if now - s.last_seen > self.ttl]:
            del self._items[sid]

    def __len__(self) -> int:
        return len(self._items)


class RateLimiter:
    """A sliding one-minute window per client and bucket."""

    def __init__(self):
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, bucket: str, per_minute: int) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(client, bucket)]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= per_minute:
                retry = int(60 - (now - hits[0])) + 1
                raise HTTPException(
                    429, "Too many requests. Try again in a minute.", headers={"Retry-After": str(retry)}
                )
            hits.append(now)


class Metrics:
    def __init__(self):
        self.counts: dict[str, int] = defaultdict(int)
        self.latency_ms_sum = 0.0
        self.latency_count = 0
        self._lock = threading.Lock()

    def inc(self, name: str, **labels: Any) -> None:
        key = name + ("{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}" if labels else "")
        with self._lock:
            self.counts[key] += 1

    def observe(self, ms: float) -> None:
        with self._lock:
            self.latency_ms_sum += ms
            self.latency_count += 1

    def render(self, sessions: int) -> str:
        lines = [f"cardpilot_{k} {v}" for k, v in sorted(self.counts.items())]
        lines += [
            f"cardpilot_answer_latency_ms_sum {self.latency_ms_sum:.1f}",
            f"cardpilot_answer_latency_ms_count {self.latency_count}",
            f"cardpilot_active_sessions {sessions}",
        ]
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ runtime


@dataclass
class Runtime:
    settings: Settings
    engine: str
    knowledge: dict[str, Knowledge]
    graphs: dict[str, Any]
    lessons: dict[str, list[dict]]
    sessions: Sessions
    limiter: RateLimiter = field(default_factory=RateLimiter)
    metrics: Metrics = field(default_factory=Metrics)

    @classmethod
    def start(cls, settings: Settings) -> Runtime:
        models = build_models(settings)
        knowledge = {code: load_knowledge(settings, code) for code in COUNTRIES}
        graphs = {}
        for code, k in knowledge.items():
            others: list[Catalog] = [other.catalog for c, other in knowledge.items() if c != code]
            graphs[code] = build_graph(k, models, others, settings.max_tool_calls, settings.max_question_chars)
        lessons = {code: build_lessons(k.catalog) for code, k in knowledge.items()}
        log.info("CardPilot ready: engine=%s, countries=%s", models.engine, ",".join(knowledge))
        return cls(settings, models.engine, knowledge, graphs, lessons, Sessions(settings.session_ttl_hours * 3600))


def tracing_callbacks(settings: Settings) -> list[Any]:
    """Langfuse traces every node, tool call and model call when its keys are set. LangSmith works through
    its own environment variables (LANGSMITH_TRACING, LANGSMITH_API_KEY) with no code here."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        from langfuse.langchain import CallbackHandler  # langfuse 3

        return [CallbackHandler()]
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # langfuse 2

            return [
                CallbackHandler(
                    public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key.get_secret_value()
                )
            ]
        except ImportError:
            log.warning("Langfuse keys are set but the langfuse package is not installed. Run uv sync --extra tracing.")
            return []


# ------------------------------------------------------------------ request bodies


class SessionIn(BaseModel):
    country: str = Field(max_length=8)


class AskIn(BaseModel):
    session_id: str = Field(max_length=64)
    question: str = Field(max_length=4000)


class CompareIn(BaseModel):
    session_id: str = Field(max_length=64)
    card_ids: list[str] = Field(min_length=2, max_length=3)


class EstimateIn(BaseModel):
    session_id: str = Field(max_length=64)
    spend: dict[str, float | None]


# ------------------------------------------------------------------ merchants for the Wallet tap

MERCHANT_WORDS = {
    "dining": [
        "starbucks",
        "mcdonald",
        "chipotle",
        "restaurant",
        "cafe",
        "coffee",
        "pizza",
        "doordash",
        "grubhub",
        "uber eats",
        "swiggy",
        "zomato",
        "domino",
        "burger",
        "kfc",
        "subway",
        "dunkin",
        "taco bell",
        "panera",
        "chaayos",
        "haldiram",
        "bar",
        "grill",
        "kitchen",
        "diner",
        "bakery",
    ],
    "groceries": [
        "whole foods",
        "trader joe",
        "kroger",
        "safeway",
        "aldi",
        "publix",
        "wegmans",
        "heb",
        "h-e-b",
        "grocery",
        "supermarket",
        "bigbasket",
        "blinkit",
        "zepto",
        "dmart",
        "reliance fresh",
        "instamart",
        "nature's basket",
    ],
    "gas": [
        "shell",
        "chevron",
        "exxon",
        "mobil",
        "bp",
        "speedway",
        "circle k",
        "valero",
        "sunoco",
        "arco",
        "gas station",
    ],
    "fuel": [
        "indian oil",
        "iocl",
        "bharat petroleum",
        "bpcl",
        "hpcl",
        "hp petrol",
        "shell",
        "nayara",
        "petrol",
        "fuel",
    ],
    "travel": [
        "delta",
        "united airlines",
        "american airlines",
        "southwest",
        "jetblue",
        "marriott",
        "hilton",
        "hyatt",
        "airbnb",
        "expedia",
        "booking.com",
        "uber",
        "lyft",
        "amtrak",
        "indigo",
        "air india",
        "akasa",
        "irctc",
        "makemytrip",
        "goibibo",
        "cleartrip",
        "ola",
        "oyo",
        "hotel",
        "airline",
    ],
    "online_shopping": ["amazon", "flipkart", "myntra", "ebay", "etsy", "ajio", "nykaa", "meesho", "tata cliq"],
    "streaming": [
        "netflix",
        "hulu",
        "disney",
        "spotify",
        "youtube",
        "hbo",
        "peacock",
        "paramount",
        "apple music",
        "apple tv",
        "prime video",
    ],
    "bills_utilities": [
        "electricity",
        "bescom",
        "tata power",
        "adani electricity",
        "airtel",
        "jio",
        "vodafone",
        "bsnl",
        "act fibernet",
        "broadband",
        "mahanagar gas",
        "tata play",
        "recharge",
        "water bill",
    ],
}


def merchant_category(country_code: str, merchant: str) -> str:
    keys = {k for k, _ in COUNTRIES[country_code].categories}
    low = f" {merchant.lower()} "
    matches = [
        (len(word), cat)
        for cat, words in MERCHANT_WORDS.items()
        if cat in keys
        for word in words
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", low)
    ]
    return max(matches)[1] if matches else "other"


# ------------------------------------------------------------------ the app


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rt = Runtime.start(settings)
        yield

    app = FastAPI(title="CardPilot", version=__version__, lifespan=lifespan, docs_url="/docs", redoc_url=None)

    def rt(request: Request) -> Runtime:
        return request.app.state.rt

    def client_of(request: Request) -> str:
        if settings.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        if (
            request.method == "POST"
            and request.url.path.startswith("/api/")
            and not request.headers.get("x-cardpilot-client")
        ):
            # A custom header cannot be sent cross-site without CORS approval, so this blocks forged form posts.
            response: Response = JSONResponse({"detail": "Missing the X-CardPilot-Client header."}, status_code=403)
        else:
            try:
                response = await call_next(request)
            except Exception:
                log.exception("unhandled error, request %s", rid)
                response = JSONResponse(
                    {"detail": "Something went wrong on our side. Please try again."}, status_code=500
                )
        ms = (time.perf_counter() - started) * 1000
        response.headers.update(
            {
                "X-Request-Id": rid,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            }
        )
        # The question text is never logged. Only the route, status and time.
        log.info(
            json.dumps(
                {
                    "rid": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(ms, 1),
                }
            )
        )
        if hasattr(request.app.state, "rt"):
            route = request.scope.get("route")  # the route template, so unknown paths cannot grow the label set
            request.app.state.rt.metrics.inc(
                "http_requests_total", path=getattr(route, "path", "unmatched"), status=response.status_code
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return JSONResponse(
            {"detail": f"Invalid {where or 'request'}: {first.get('msg', 'check the fields')}."}, status_code=422
        )

    # -------------------------------------------------------------- pages and health

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(settings.web_dir / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/healthz")
    def healthz(request: Request):
        r = rt(request)
        return {
            "status": "ok",
            "version": __version__,
            "engine": r.engine,
            "countries": {
                code: {
                    "cards": len(k.catalog.cards),
                    "chunks": k.store.count(),
                    "store": describe(k.store),
                    "embedder": k.embedder.id,
                    "as_of": k.catalog.as_of,
                }
                for code, k in r.knowledge.items()
            },
        }

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request):
        r = rt(request)
        return PlainTextResponse(r.metrics.render(len(r.sessions)), media_type="text/plain; version=0.0.4")

    # -------------------------------------------------------------- sessions

    @app.post("/api/session")
    def new_session(body: SessionIn, request: Request):
        r = rt(request)
        r.limiter.check(client_of(request), "session", 30)
        code = body.country.strip().lower()
        if code not in COUNTRIES:
            raise HTTPException(400, "Pick the United States or India.")
        s = r.sessions.create(code)
        c = COUNTRIES[code]
        r.metrics.inc("sessions_total", country=code)
        return {"session_id": s.id, "country": code, "country_name": c.name, "currency": c.currency, "symbol": c.symbol}

    @app.get("/api/suggestions")
    def suggestions(request: Request, session_id: str = Query(max_length=64)):
        s = rt(request).sessions.get(session_id)
        return {"suggestions": list(COUNTRIES[s.country].suggestions)}

    # -------------------------------------------------------------- the agent

    @app.post("/api/ask")
    def ask_question(body: AskIn, request: Request):
        r = rt(request)
        s = r.sessions.get(body.session_id)
        r.limiter.check(client_of(request), "ask", settings.rate_limit_per_minute)
        started = time.perf_counter()
        result = ask(r.graphs[s.country], body.question, history=list(s.history), callbacks=tracing_callbacks(settings))
        turn = result.pop("turn", None)
        if turn:
            s.history = [*s.history, turn][-3:]
        r.metrics.observe((time.perf_counter() - started) * 1000)
        r.metrics.inc(
            "answers_total", country=s.country, engine=result["engine"], refused=str(result["refused"]).lower()
        )
        return result

    # -------------------------------------------------------------- tools without the model

    @app.get("/api/cards")
    def cards(request: Request, session_id: str = Query(max_length=64)):
        s = rt(request).sessions.get(session_id)
        return {
            **rt(request).knowledge[s.country].catalog.listing(),
            "as_of": rt(request).knowledge[s.country].catalog.as_of,
        }

    @app.post("/api/compare")
    def compare(body: CompareIn, request: Request):
        r = rt(request)
        s = r.sessions.get(body.session_id)
        r.limiter.check(client_of(request), "tools", settings.rate_limit_per_minute * 3)
        try:
            return r.knowledge[s.country].catalog.compare(body.card_ids)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/estimate")
    def estimate(body: EstimateIn, request: Request):
        r = rt(request)
        s = r.sessions.get(body.session_id)
        r.limiter.check(client_of(request), "tools", settings.rate_limit_per_minute * 3)
        spend = {k: v for k, v in body.spend.items() if v is not None}
        if any(v < 0 or v > 10_000_000 for v in spend.values()):
            raise HTTPException(400, "Monthly amounts must be between 0 and 10,000,000.")
        try:
            return r.knowledge[s.country].catalog.estimate(spend)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/lessons")
    def lessons(request: Request, session_id: str = Query(max_length=64)):
        s = rt(request).sessions.get(session_id)
        return {"lessons": rt(request).lessons[s.country]}

    # -------------------------------------------------------------- the Wallet tap

    @app.get("/api/tap", response_class=PlainTextResponse)
    def tap(
        request: Request,
        country: str = Query(max_length=8),
        merchant: str = Query(default="", max_length=120),
        cards: str = Query(default="", max_length=400, description="Your cards, comma separated, names or ids."),
    ):
        """Called by an iOS Shortcut when a Wallet card is tapped. Returns one short notification line.
        It states earn rates for the category. It does not tell anyone which card to use."""
        r = rt(request)
        r.limiter.check(client_of(request), "tap", settings.rate_limit_per_minute)
        code = country.strip().lower()
        if code not in COUNTRIES:
            raise HTTPException(400, "country must be us or in")
        catalog = r.knowledge[code].catalog
        c = COUNTRIES[code]
        category = merchant_category(code, merchant)
        label = dict(c.categories)[category].lower()
        chosen = [catalog.get(x.strip()) for x in cards.split(",") if x.strip()]
        chosen = [card for card in chosen if card]
        if not chosen:
            chosen = sorted(catalog.cards, key=lambda card: card.earn.get(category) or 0, reverse=True)[:3]
        rates = sorted(chosen, key=lambda card: card.earn.get(category) or 0, reverse=True)
        parts = [
            f"{card.name} {card.earn[category]:g}%"
            if card.earn.get(category) is not None
            else f"{card.name} not published"
            for card in rates
        ]
        where = (
            f"{merchant.strip()} looks like {label}"
            if category != "other"
            else f"I could not tell what {merchant.strip() or 'this'} is, so these are everyday rates"
        )
        r.metrics.inc("taps_total", country=code, category=category)
        return f"{where}. Earn on {label}: {', '.join(parts)}. Caps may apply. Rates as of {catalog.as_of}. Education only."

    return app


def __getattr__(name: str):
    # `uvicorn cardpilot.api:app` builds the app on first use, so importing this module stays cheap in tests.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
