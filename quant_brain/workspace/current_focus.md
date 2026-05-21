# Current Focus

## Summary

The current initiative is to establish `quant_brain` as the canonical research
overlay and use the seeded `smh_regime_filter` example to validate the memory,
evaluation, and automation workflow while starting four concrete research lanes
based on Yahoo Finance market data. The repo is also gaining a discretionary
trade journal and systematization bridge so manual ideas can be reviewed and,
when appropriate, promoted into quantitative research.

## Active Priority

- Confirm the markdown schema is stable for multi-agent use.
- Keep SMH as the first end-to-end reference strategy.
- Add the discretionary trade layer without letting it blur into validated
  systematic strategy memory.
- Start the first four cross-asset and relative-performance research lanes:
  - large vs small caps
  - sector and industry ETF relative performance
  - gold, silver, copper, oil, and bitcoin tracking
  - US, UK, and Japan sovereign duration tracking
- Keep daily, weekly, monthly, and expectation-audit automation coverage aligned
  with those themes.

## Success Condition

A new agent should be able to read this file, the handoff file, and the SMH
strategy folder, then understand:

- what the repo is currently trying to learn
- which files are canonical
- what the next research iterations should test

## Current Risks

- Legacy notes under `研究/...` still hold useful context that is not yet indexed.
- Daily automation summaries can become noisy if expectations are not explicit.
- Durable findings need human or curated review, not blind promotion.
- Mixed prompts can cause agents to jump into code too early unless the
  discuss-first rule is explicit.
- Japan duration-specific Yahoo proxies are still in transition because the new
  1-3 year and 20+ year JGB ETFs are only being listed in late May 2026.
