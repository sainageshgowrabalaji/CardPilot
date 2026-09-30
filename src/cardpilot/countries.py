"""The countries CardPilot knows. Each one is a closed world with its own cards, index and wording.

Adding a country means adding an entry here and a data/cards/<code>.json file. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CountryCode = Literal["us", "in"]


@dataclass(frozen=True)
class Country:
    code: CountryCode
    name: str
    currency: str
    symbol: str
    categories: tuple[tuple[str, str], ...]
    suggestions: tuple[str, ...]
    regulator: str
    disclaimer: str

    def money(self, amount: float | None) -> str:
        if amount is None:
            return "not published"
        if self.code == "in":
            return f"{self.symbol}{indian_grouping(round(amount))}"
        return f"{self.symbol}{amount:,.0f}" if float(amount).is_integer() else f"{self.symbol}{amount:,.2f}"


def indian_grouping(n: int) -> str:
    """1234567 as 12,34,567, the way amounts are written in India."""
    s = str(abs(n))
    if len(s) <= 3:
        out = s
    else:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        out = ",".join(groups) + "," + tail
    return ("-" if n < 0 else "") + out


COUNTRIES: dict[str, Country] = {
    "us": Country(
        code="us",
        name="United States",
        currency="USD",
        symbol="$",
        categories=(
            ("dining", "Dining"),
            ("groceries", "Groceries"),
            ("gas", "Gas"),
            ("travel", "Travel"),
            ("online_shopping", "Online shopping"),
            ("streaming", "Streaming"),
            ("other", "Everything else"),
        ),
        suggestions=(
            "Does the Amex Gold charge foreign transaction fees?",
            "What does the Chase Sapphire Preferred earn on dining?",
            "Compare Citi Double Cash and Wells Fargo Active Cash",
            "What happens if I only pay the minimum?",
            "How does a secured card work?",
        ),
        regulator="the Consumer Financial Protection Bureau (consumerfinance.gov)",
        disclaimer=(
            "Education only, not financial advice. Card terms change often, so check the issuer's site before you apply."
        ),
    ),
    "in": Country(
        code="in",
        name="India",
        currency="INR",
        symbol="₹",
        categories=(
            ("dining", "Dining"),
            ("groceries", "Groceries"),
            ("fuel", "Fuel"),
            ("travel", "Travel"),
            ("online_shopping", "Online shopping"),
            ("bills_utilities", "Bills and utilities"),
            ("other", "Everything else"),
        ),
        suggestions=(
            "What is the annual fee on the HDFC Millennia?",
            "Does the Axis ACE give cashback on bill payments?",
            "Compare SBI Cashback and Amazon Pay ICICI",
            "What happens if I pay only the minimum amount due?",
            "How does a card against a fixed deposit work?",
        ),
        regulator="the Reserve Bank of India (rbi.org.in)",
        disclaimer=(
            "Education only, not financial advice. Card terms change often, so check the bank's site before you apply."
        ),
    ),
}


def get_country(code: str) -> Country:
    """The one place a country code is checked. Anything else is refused."""
    country = COUNTRIES.get(code)
    if country is None:
        raise ValueError(f"Unknown country {code!r}. CardPilot supports {', '.join(COUNTRIES)}.")
    return country
