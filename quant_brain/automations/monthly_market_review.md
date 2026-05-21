# Monthly Market Review Prompt

## Task

Review the completed month, score the major expectations, identify repeated
forecasting errors, and propose which findings deserve promotion into curated
memory.

## Required Inputs

- monthly and weekly observation history
- `quant_brain/evaluations/prediction_scoreboard.md`
- current strategy folders and active project files
- `quant_brain/memory/datasets/yfinance_watchlists.md`
- local repo data plus live price, macro, and recent event context as needed

## Required Output

- create or update `quant_brain/observations/monthly/YYYY-MM.md`
- append an entry to `quant_brain/evaluations/monthly_review.md`
- append durable-memory promotion suggestions to `quant_brain/workspace/inbox.md`

## Constraints

- suggest promotions, do not finalize them silently
- identify repeated failure modes rather than isolated misses only
- distinguish durable findings from temporary commentary
