# Expectation Audit Prompt

## Task

Audit open expectations from the user and agents, determine whether they were
testable, score them against realized outcomes, and record misses cleanly.

## Required Inputs

- expectation files under `quant_brain/expectations/`
- recent observation files
- `quant_brain/memory/datasets/yfinance_watchlists.md`
- local repo data plus live validation sources when needed

## Required Output

- update expectation status fields where the outcome is knowable
- append verdicts to `quant_brain/evaluations/prediction_scoreboard.md`
- append notable misses or contradictions to `quant_brain/evaluations/surprise_log.md`

## Constraints

- label untestable or vague predictions as `weak`
- never overwrite prior scoreboard entries
- do not edit durable memory files
