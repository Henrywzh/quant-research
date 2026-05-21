# Automation Manifest

## Write Policy

Automations may write only to observation and evaluation targets plus
`workspace/inbox.md` for promotion suggestions.

Automations must not directly rewrite:

- `memory/strategies/*/findings.md`
- `memory/knowledge/*.md`
- `memory/decisions/*.md`

Automations should use:

- `memory/datasets/yfinance_watchlists.md` for reusable ticker sets
- topic notes under `memory/knowledge/` for interpretation and context

## Jobs

### daily-market-brief

- Schedule: every weekday at 9:00 AM in the user's locale
- Writes:
  - `observations/daily/YYYY-MM-DD.md`
  - `evaluations/daily_reality_checks.md`
  - `evaluations/surprise_log.md`

### weekly-market-review

- Schedule: every Saturday morning
- Writes:
  - `observations/weekly/YYYY-Www.md`
  - `evaluations/weekly_review.md`
  - `workspace/current_focus.md` only for material priority shifts

### monthly-market-review

- Schedule: first calendar day of the month in the morning
- Writes:
  - `observations/monthly/YYYY-MM.md`
  - `evaluations/monthly_review.md`
  - `workspace/inbox.md`

### expectation-audit

- Schedule: every weekday after the daily brief
- Writes:
  - expectation status updates in `expectations/*`
  - `evaluations/prediction_scoreboard.md`
  - `evaluations/surprise_log.md`
