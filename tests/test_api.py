import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cardpilot.api import Sessions, create_app, merchant_category

H = {"X-CardPilot-Client": "test"}


@pytest.fixture(scope="module")
def client(settings):
    s = settings.model_copy(update={"rate_limit_per_minute": 1000})
    with TestClient(create_app(s)) as c:
        yield c


def start(client, country="us") -> str:
    r = client.post("/api/session", json={"country": country}, headers=H)
    assert r.status_code == 200
    return r.json()["session_id"]


def test_health(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["engine"] == "offline"
    assert body["countries"]["us"]["chunks"] > 100 and body["countries"]["in"]["chunks"] > 100


def test_page_is_served_with_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "CardPilot" in r.text
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-request-id"]


def test_posts_need_the_client_header(client):
    assert client.post("/api/session", json={"country": "us"}).status_code == 403


@pytest.mark.parametrize("code,symbol", [("us", "$"), ("in", "₹"), ("IN", "₹")])
def test_session(client, code, symbol):
    body = client.post("/api/session", json={"country": code}, headers=H).json()
    assert body["country"] == code.lower() and body["symbol"] == symbol and len(body["session_id"]) >= 32


def test_unknown_country(client):
    r = client.post("/api/session", json={"country": "uk"}, headers=H)
    assert r.status_code == 400 and "India" in r.json()["detail"]


def test_ask(client):
    sid = start(client)
    r = client.post(
        "/api/ask",
        json={"session_id": sid, "question": "Does the Amex Gold charge foreign transaction fees?"},
        headers=H,
    )
    body = r.json()
    assert r.status_code == 200
    assert body["sentences"][0]["sources"] and body["sources"][0]["url"].startswith("https://")
    assert {"as_of", "disclaimer", "notice", "refused", "engine", "trace"} <= set(body)


def test_the_session_decides_the_country(client):
    sid = start(client, "us")
    body = client.post(
        "/api/ask", json={"session_id": sid, "question": "Tell me about the HDFC Millennia"}, headers=H
    ).json()
    assert body["refused"] and "United States" in body["sentences"][0]["text"]
    r = client.post("/api/compare", json={"session_id": sid, "card_ids": ["hdfc-millennia", "axis-ace"]}, headers=H)
    assert r.status_code == 400


def test_unknown_or_expired_session(client):
    r = client.get("/api/suggestions", params={"session_id": "nope"})
    assert r.status_code == 404 and "session has ended" in r.json()["detail"]


def test_sessions_expire():
    s = Sessions(ttl_seconds=0.01)
    sid = s.create("us").id
    time.sleep(0.02)
    with pytest.raises(HTTPException):
        s.get(sid)


def test_sessions_are_capped():
    s = Sessions(ttl_seconds=60, max_sessions=3)
    ids = [s.create("us").id for _ in range(5)]
    assert len(s) == 3
    s.get(ids[-1])


def test_suggestions_cards_and_lessons(client):
    sid = start(client, "in")
    assert len(client.get("/api/suggestions", params={"session_id": sid}).json()["suggestions"]) >= 4
    cards = client.get("/api/cards", params={"session_id": sid}).json()
    assert len(cards["cards"]) == 10 and {"key": "fuel", "label": "Fuel"} in cards["categories"]
    lessons = client.get("/api/lessons", params={"session_id": sid}).json()["lessons"]
    assert len(lessons) >= 5 and all(
        lesson["sources"] and "rbi.org.in" in lesson["sources"][0]["url"] for lesson in lessons
    )


def test_compare(client):
    sid = start(client)
    body = client.post(
        "/api/compare", json={"session_id": sid, "card_ids": ["amex-gold", "capital-one-savor"]}, headers=H
    ).json()
    assert len(body["columns"]) == 2 and body["rows"] and body["sources"]


def test_compare_validation_message_is_readable(client):
    sid = start(client)
    r = client.post("/api/compare", json={"session_id": sid, "card_ids": ["amex-gold"]}, headers=H)
    assert r.status_code == 422 and isinstance(r.json()["detail"], str) and "card_ids" in r.json()["detail"]


def test_estimate(client):
    sid = start(client)
    body = client.post(
        "/api/estimate", json={"session_id": sid, "spend": {"dining": 300, "groceries": None}}, headers=H
    ).json()
    assert body["rows"][0]["net"] >= body["rows"][-1]["net"] and body["note"]
    bad = client.post("/api/estimate", json={"session_id": sid, "spend": {"dining": -5}}, headers=H)
    assert bad.status_code == 400


def test_tap(client):
    text = client.get(
        "/api/tap", params={"country": "us", "merchant": "STARBUCKS 0231", "cards": "Amex Gold,Citi Double Cash"}
    ).text
    assert text.startswith("STARBUCKS 0231 looks like dining") and "American Express Gold Card 4%" in text
    assert "Education only" in text
    assert client.get("/api/tap", params={"country": "xx", "merchant": "a"}).status_code == 400


@pytest.mark.parametrize(
    "code,merchant,category",
    [
        ("us", "Shell Oil 123", "gas"),
        ("in", "Shell Petrol Pump", "fuel"),
        ("us", "UBER EATS", "dining"),
        ("us", "Uber Trip", "travel"),
        ("in", "Airtel Postpaid", "bills_utilities"),
        ("us", "Airtel", "other"),
        ("us", "Hubspot", "other"),
    ],
)
def test_merchant_category(code, merchant, category):
    assert merchant_category(code, merchant) == category


def test_rate_limit(settings):
    s = settings.model_copy(update={"rate_limit_per_minute": 2})
    with TestClient(create_app(s)) as c:
        sid = start(c)
        codes = [
            c.post("/api/ask", json={"session_id": sid, "question": "APR?"}, headers=H).status_code for _ in range(3)
        ]
        assert codes == [200, 200, 429]
        r = c.post("/api/ask", json={"session_id": sid, "question": "APR?"}, headers=H)
        assert r.headers["retry-after"] and "Too many" in r.json()["detail"]


def test_metrics(client):
    start(client)
    text = client.get("/metrics").text
    assert 'cardpilot_sessions_total{country="us"}' in text and "cardpilot_active_sessions" in text
    assert 'path="/api/session"' in text
