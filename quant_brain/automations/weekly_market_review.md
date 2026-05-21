# Weekly Market Review Prompt

## Task

Evaluate the completed week, compare active strategy and market expectations with
realized behavior, and summarize whether the current regime assumptions still
look aligned.

## Required Inputs

- daily observation files for the week
- open weekly expectations and strategy expectations
- current active projects and handoff files
- `quant_brain/memory/datasets/yfinance_watchlists.md`
- local repo data plus live price, macro, and recent event context as needed

## Required Output

- create or update `quant_brain/observations/weekly/YYYY-Www.md`
- append an entry to `quant_brain/evaluations/weekly_review.md`
- update `quant_brain/workspace/current_focus.md` only if the regime or
  priority changed materially

## Constraints

- do not rewrite durable findings
- call out contradictions between the week's action and prior theses
- keep trade-idea evaluation tied to explicit expectations
