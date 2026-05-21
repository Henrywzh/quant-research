# Discretionary Manifest

## Purpose

The discretionary layer captures manual trade ideas, executed trades,
post-trade review, and the bridge into systematized quantitative research.
It exists so manual thinking has a canonical home without being confused with
validated systematic strategy memory.

## File Roles

- `ideas/open_ideas.md`: active discretionary trade ideas before or during
  execution.
- `ideas/archived_ideas.md`: stale, invalidated, or superseded ideas.
- `trades/open_trades.md`: live discretionary positions and their current plan.
- `trades/closed_trades.md`: completed discretionary positions with exit and
  outcome detail.
- `trades/postmortems.md`: process-quality reviews that separate decision
  quality from outcome quality.
- `playbooks/*.md`: recurring discretionary setups, risk practices, and
  invalidation rules.
- `systematization/*.md`: the funnel for moving discretionary patterns into
  testable quant hypotheses.

## Required Schemas

### Idea Schema

- `Idea ID`
- `Status`
- `Source`
- `Asset`
- `Direction`
- `Time Horizon`
- `Theme`
- `Thesis`
- `Why Now`
- `Entry Trigger`
- `Invalidation`
- `Risk`
- `What Would Prove This Wrong`
- `Can This Be Systematized?`
- `Candidate Quant Proxies`

### Trade Schema

- `Trade ID`
- `Linked Idea`
- `Status`
- `Asset`
- `Direction`
- `Entry Date`
- `Entry Price`
- `Size`
- `Stop`
- `Target`
- `Max Risk Budget`
- `Original Plan`
- `Updates`
- `Exit`
- `PnL`
- `Decision Quality Review`
- `Outcome Review`
- `Systematization Note`

### Postmortem Schema

- `Trade Ref`
- `Was The Thesis Good?`
- `Was The Risk Good?`
- `Was The Execution Good?`
- `Was The Discipline Good?`
- `Outcome Quality`
- `Repeatable?`
- `Should This Be Systematized?`
- `Next Adjustment`

## Promotion Funnel

1. Log the idea in `ideas/open_ideas.md`.
2. Record any executed position in `trades/open_trades.md`.
3. Move completed trades to `trades/closed_trades.md`.
4. Review them in `trades/postmortems.md`.
5. Promote repeatable patterns to `systematization/candidates.md`.
6. Only after a testable rule definition exists should the pattern move into
   `memory/strategies/...` or research code.
