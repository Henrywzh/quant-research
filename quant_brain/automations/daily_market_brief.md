# Daily Market Brief Prompt

## Task

Summarize yesterday's market behavior and compare it against explicit open
expectations from the user and prior agents.

## Required Inputs

- `quant_brain/workspace/current_focus.md`
- `quant_brain/workspace/session_handoff.md`
- open expectation files under `quant_brain/expectations/`
- `quant_brain/memory/datasets/yfinance_watchlists.md`
- local repo data when relevant
- live market context from `yfinance`, macro context from `FRED`, and recent
  web/news validation when necessary

## Required Output

- create or update `quant_brain/observations/daily/YYYY-MM-DD.md`
- append a scored note to `quant_brain/evaluations/daily_reality_checks.md`
- append any important mismatch to `quant_brain/evaluations/surprise_log.md`

## Constraints

- separate `what happened` from `why we think it happened`
- mark vague expectations as `weak`
- if evidence conflicts, state the conflict clearly
- do not modify durable memory files
