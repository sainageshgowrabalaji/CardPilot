"""Short lessons built from each country's regulator passages (CFPB and FTC for the US, RBI for India).

Every sentence in a lesson is a passage from the card file, shown with its source. Nothing is written
here that the regulator did not publish.
"""

from __future__ import annotations

from .catalog import Catalog

# (id, title, words that place a passage in the lesson). The first lesson that matches a passage wins.
LESSONS = {
    "us": [
        ("billing", "Due dates and the grace period", ["grace period", "21 days", "bills at least"]),
        ("interest", "APR and minimum payments", ["apr stands", "more than the minimum"]),
        ("late", "Late payments and penalty fees", ["counts as late", "penalty fee"]),
        ("credit", "Building credit and secured cards", ["credit score", "secured card", "authorized user"]),
        ("protection", "Lost cards, fraud and disputes", ["lost or stolen", "dispute"]),
        (
            "costs",
            "Balance transfers, cash advances and other costs",
            ["balance transfer", "introductory rate", "cash advance", "transaction fees"],
        ),
    ],
    "in": [
        ("basics", "How a credit card works", ["revolving credit", "interest-free period", "interest can be charged"]),
        (
            "statements",
            "Statements and the minimum amount due",
            ["statements must be sent", "minimum amount due", "paying only the minimum"],
        ),
        ("late", "Late payments and your credit report", ["past due", "late payment charges", "credit information"]),
        ("rights", "Closing a card and cards you never asked for", ["close a credit card", "unsolicited", "consent"]),
        ("fees", "Fees the bank must tell you about", ["most important terms", "key fact statement"]),
        ("protection", "Fraud and disputed transactions", ["zero liability", "shadow reversal"]),
    ],
}


def build_lessons(catalog: Catalog) -> list[dict]:
    plan = LESSONS[catalog.country.code]
    buckets: dict[str, list[dict]] = {lid: [] for lid, _, _ in plan}
    for passage in catalog.general:
        low = passage["text"].lower()
        for lid, _, words in plan:
            if any(w in low for w in words):
                buckets[lid].append(passage)
                break
    lessons = []
    for lid, title, _ in plan:
        passages = buckets[lid]
        if not passages:
            continue
        sources, seen = [], set()
        for p in passages:
            src = p["source"]
            if src["url"] not in seen:
                seen.add(src["url"])
                sources.append({"title": src["title"], "url": src["url"]})
        lessons.append({"id": lid, "title": title, "body": " ".join(p["text"] for p in passages), "sources": sources})
    return lessons
