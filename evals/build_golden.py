"""Writes the golden question sets once, so every eval run scores against the same frozen questions.

Generated questions paraphrase a fact about each card (fee, foreign fee, an earn rate). Handwritten
questions ask about the regulator's rules in everyday words. A question's relevant chunks are the ones
for the right card whose text contains any of its `must` phrases. Re-run only when the card data changes.

    uv run python evals/build_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cardpilot.catalog import _MANUAL_ALIASES, Catalog
from cardpilot.config import get_settings

HERE = Path(__file__).parent

TEMPLATES = {
    "us": [
        ("What does the {alias} cost per year?", ["annual fee"], "fee"),
        (
            "Is there an extra charge for using the {alias} abroad?",
            ["foreign transaction fee", "outside the us"],
            "foreign",
        ),
        ("How much do I get back at restaurants with the {alias}?", ["dining"], "dining"),
        ("What do I earn at the supermarket with the {alias}?", ["groceries", "grocery"], "groceries"),
    ],
    "in": [
        ("What is the yearly charge for the {alias}?", ["annual fee", "renewal fee"], "fee"),
        ("What markup does the {alias} add on international spends?", ["forex", "markup"], "foreign"),
        ("How much cashback does the {alias} give on online shopping?", ["online shopping"], "online_shopping"),
        ("Does the {alias} give anything back on dining?", ["dining"], "dining"),
    ],
}

HANDWRITTEN = {
    "us": [
        ("How long do I have to pay my bill before interest kicks in?", ["grace period"]),
        ("What does APR actually mean?", ["apr stands"]),
        ("What happens if I only pay the minimum each month?", ["more than the minimum"]),
        ("How big can a late fee legally be?", ["penalty fee"]),
        ("How can I build credit when I have none?", ["secured card", "credit score", "credit history"]),
        ("My wallet was stolen. Do I owe for charges the thief made?", ["lost or stolen"]),
        ("How do I dispute a wrong charge on my card?", ["dispute"]),
        ("Does a 0% balance transfer cover my new purchases too?", ["balance transfer"]),
        ("Can I take cash out of an ATM with a credit card?", ["cash advance"]),
        ("Can I add my partner to my card?", ["authorized user"]),
        ("How many days before the due date must the bill arrive?", ["21 days"]),
        ("How long must an intro rate last?", ["introductory rate"]),
    ],
    "in": [
        ("When do I get an interest free period on my card?", ["interest-free period"]),
        ("What is the minimum amount due and how is it set?", ["minimum amount due"]),
        ("When can the bank report me to CIBIL as a defaulter?", ["credit information", "past due"]),
        ("How quickly must the bank close my card when I ask?", ["close a credit card"]),
        ("I got a credit card I never applied for. What now?", ["unsolicited"]),
        ("Someone used my card online without my permission. Am I liable?", ["zero liability", "shadow reversal"]),
        ("What is a key fact statement?", ["key fact statement"]),
        ("Is the late fee charged on my whole bill?", ["late payment charges"]),
        ("What charges apply if I withdraw cash with my card?", ["cash advance"]),
        ("Can the bank raise my limit without asking me?", ["consent"]),
        ("How much time do I get to pay after the statement?", ["fortnight"]),
        ("Can interest be charged on amounts I already paid?", ["interest can be charged"]),
    ],
}


def alias_for(card) -> str:
    aliases = _MANUAL_ALIASES.get(card.id)
    return (
        aliases[0].title().replace("Hdfc", "HDFC").replace("Sbi", "SBI").replace("Idfc", "IDFC")
        if aliases
        else card.name
    )


def build(code: str) -> list[dict]:
    catalog = Catalog.load(get_settings().cards_dir, code)
    rows = []
    for card in catalog.cards:
        for template, must, field in TEMPLATES[code]:
            if field in ("dining", "groceries", "online_shopping") and not card.earn.get(field):
                continue
            if field == "foreign" and card.foreign_transaction_fee_pct is None:
                continue
            if field == "fee" and card.annual_fee is None:
                continue
            q = template.format(alias=alias_for(card))
            rows.append({"source": "generated", "question": q, "card_id": card.id, "must": must, "field": field})
    for q, must in HANDWRITTEN[code]:
        rows.append({"source": "handwritten", "question": q, "card_id": None, "must": must, "field": "general"})
    return rows


def main() -> None:
    for code in ("us", "in"):
        rows = build(code)
        path = HERE / f"golden_{code}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{path.name}: {len(rows)} questions")


if __name__ == "__main__":
    main()
