import json
from pathlib import Path

import pytest

from cardpilot.catalog import Catalog
from cardpilot.countries import COUNTRIES, get_country, indian_grouping

CARDS_DIR = Path(__file__).resolve().parents[1] / "data" / "cards"


@pytest.mark.parametrize("code", ["us", "in"])
def test_card_files_are_complete(code, catalogs):
    catalog = catalogs[code]
    ids = [c.id for c in catalog.cards]
    assert len(ids) == len(set(ids)) >= 10
    keys = {k for k, _ in COUNTRIES[code].categories}
    for card in catalog.cards:
        assert card.sources and all(s.url.startswith("https://") for s in card.sources), card.id
        for p in card.passages:
            card.source(p.source)  # every passage points at a real source
        assert set(card.earn) <= keys, card.id
    assert catalog.general and catalog.as_of


def test_a_file_for_another_country_is_refused():
    data = json.loads((CARDS_DIR / "in.json").read_text())
    with pytest.raises(ValueError):
        Catalog(get_country("us"), data)


def test_unknown_country_is_refused():
    with pytest.raises((KeyError, ValueError)):
        get_country("uk")


@pytest.mark.parametrize(
    "code,text,expected",
    [
        ("us", "Tell me about Discover it Secured", ["discover-it-secured"]),
        ("us", "Amex Gold vs Savor", ["amex-gold", "capital-one-savor"]),
        ("us", "compare the sapphire preferred and the venture", ["chase-sapphire-preferred", "capital-one-venture"]),
        ("in", "HDFC Millennia fees", ["hdfc-millennia"]),
        ("in", "Flipkart Axis card cashback", ["axis-flipkart"]),
        ("in", "Tata Neu Infinity", ["hdfc-tata-neu-infinity"]),
        ("us", "How does interest work?", []),
    ],
)
def test_mentions(catalogs, code, text, expected):
    assert [c.id for c in catalogs[code].mentions(text)] == expected


def test_compare_cites_every_cell(catalogs):
    table = catalogs["us"].compare(["amex-gold", "capital-one-savor"])
    assert table["columns"] == ["American Express Gold Card", "Capital One Savor Cash Rewards"]
    n = len(table["sources"])
    for row in table["rows"]:
        assert len(row["cells"]) == 2
        assert all(1 <= cell["source"] <= n for cell in row["cells"])


def test_compare_needs_two_cards(catalogs):
    with pytest.raises(ValueError):
        catalogs["us"].compare(["amex-gold"])
    with pytest.raises(ValueError):
        catalogs["us"].compare(["amex-gold", "hdfc-millennia"])  # the Indian card is not in the US catalog


def test_estimate_is_plain_arithmetic(catalogs):
    catalog = catalogs["us"]
    result = catalog.estimate({"dining": 100, "groceries": 200})
    for row in result["rows"]:
        card = catalog.get(row["card_id"])
        expected = 100 * 12 * (card.earn.get("dining") or 0) / 100 + 200 * 12 * (card.earn.get("groceries") or 0) / 100
        assert row["yearly_rewards"] == pytest.approx(expected, abs=0.01)
        assert row["net"] == pytest.approx(expected - (card.annual_fee or 0), abs=0.01)
    nets = [r["net"] for r in result["rows"]]
    assert nets == sorted(nets, reverse=True)


def test_estimate_ignores_unknown_categories_and_needs_an_amount(catalogs):
    with pytest.raises(ValueError):
        catalogs["us"].estimate({"fuel": 100})  # "fuel" is an Indian category; the US one is "gas"
    with pytest.raises(ValueError):
        catalogs["in"].estimate({"dining": 0})


def test_money_formats():
    assert indian_grouping(10000000) == "1,00,00,000"
    assert get_country("in").money(250000) == "₹2,50,000"
    assert get_country("us").money(1250) == "$1,250"


@pytest.mark.parametrize("code", ["us", "in"])
def test_fact_sentences_have_no_double_periods(code, catalogs):
    for card in catalogs[code].cards:
        for text, _ in catalogs[code].fact_sentences(card):
            assert ".." not in text, text
