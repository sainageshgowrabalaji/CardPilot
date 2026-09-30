# CardPilot eval results

Engine **offline**, embedder `glove-64-v1`, run 2026-09-30. Golden sets: 59 US and 51 India questions, frozen in `evals/golden_*.jsonl`.

## Retrieval

| Country | Mode | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|---|
| US | keyword | 56% | 83% | 0.68 |
| US | vector | 49% | 88% | 0.64 |
| US | hybrid | 64% | 97% | 0.77 |
| IN | keyword | 84% | 100% | 0.92 |
| IN | vector | 29% | 90% | 0.56 |
| IN | hybrid | 73% | 100% | 0.85 |

Held-out questions, written before any tuning and never tuned against:

| Country | Mode | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|---|
| US | keyword | 60% | 70% | 0.68 |
| US | vector | 40% | 50% | 0.49 |
| US | hybrid | 60% | 80% | 0.68 |
| IN | keyword | 30% | 70% | 0.47 |
| IN | vector | 30% | 60% | 0.44 |
| IN | hybrid | 40% | 70% | 0.51 |

## Agent and safety

| Metric | US | India |
|---|---|---|
| Fee and foreign-fee answers state the right value | 100% | 100% |
| Answers where every sentence cites a source | 100% | 100% |
| Questions answered (not "couldn't find") | 100% | 100% |
| Other country's cards refused | 100% | 100% |
| Answers mentioning the other country's cards | 0% | 0% |
| Card and ID numbers removed and never echoed | 100% | 100% |
| Prompt injections refused | 100% | 100% |
| Off-topic questions refused | 100% | 100% |
| "Which card should I get" answered without a pick | 100% | 100% |
| Questions about every card that name every card | 100% | 100% |
| "Top" or "best" cards answered without a ranking | 100% | 100% |

Latency per question: p50 4 ms, p95 7 ms over 222 questions.

## Where it misses

- Retrieval, not in the hybrid top 5: How much do I get back at restaurants with the Amex Gold?
- Retrieval, not in the hybrid top 5: Can I take cash out of an ATM with a credit card?
