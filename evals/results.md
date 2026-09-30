# CardPilot eval results

Engine **offline**, embedder `glove-64-v1`, run 2026-09-29. Golden sets: 59 US and 51 India questions, frozen in `evals/golden_*.jsonl`.

## Retrieval

| Country | Mode | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|---|
| US | keyword | 56% | 81% | 0.68 |
| US | vector | 49% | 88% | 0.64 |
| US | hybrid | 64% | 97% | 0.77 |
| IN | keyword | 88% | 100% | 0.94 |
| IN | vector | 29% | 92% | 0.57 |
| IN | hybrid | 75% | 100% | 0.86 |

Held-out questions, written before any tuning and never tuned against:

| Country | Mode | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|---|
| US | keyword | 60% | 70% | 0.68 |
| US | vector | 50% | 50% | 0.54 |
| US | hybrid | 60% | 80% | 0.68 |
| IN | keyword | 30% | 70% | 0.50 |
| IN | vector | 30% | 60% | 0.44 |
| IN | hybrid | 40% | 70% | 0.53 |

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

Latency per question: p50 4 ms, p95 5 ms over 210 questions.

## Where it misses

- Retrieval, not in the hybrid top 5: How much do I get back at restaurants with the Amex Gold?
- Retrieval, not in the hybrid top 5: Can I take cash out of an ATM with a credit card?
