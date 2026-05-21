# Quant Brain Manifest

## Purpose

`quant_brain` is the canonical overlay for research state in this repository.
It gives Codex, Claude, and other agents one predictable place to read current
context, inspect durable strategy memory, compare expectations with outcomes,
and append new observations without rewriting curated knowledge.

Existing files under `研究/...`, notebooks under `experiments/...`, and datasets
under `data/...` remain valid source material. They are not the canonical place
for active state or durable findings unless explicitly linked here.

## Required Read Order

Agents should orient in this order before proposing changes or new research:

1. [workspace/current_focus.md](workspace/current_focus.md)
2. [workspace/session_handoff.md](workspace/session_handoff.md)
3. relevant `memory/strategies/<strategy>/strategy.md`
4. relevant expectation and evaluation files
5. supporting source notes or notebooks only after the canonical files above
6. relevant discretionary files when the prompt is about manual trades, trade
   review, or systematization

## File Classes And Write Rules

`Mutable`
- `workspace/*`
- `observations/*`

`Append-only`
- `memory/research_log.md`
- `evaluations/prediction_scoreboard.md`
- `evaluations/surprise_log.md`
- `evaluations/daily_reality_checks.md`
- `evaluations/weekly_review.md`
- `evaluations/monthly_review.md`

`Curated`
- `memory/strategies/*/strategy.md`
- `memory/strategies/*/findings.md`
- `memory/knowledge/*.md`
- `memory/heuristics/*.md`
- `memory/decisions/*.md`
- `discretionary/playbooks/*.md`
- `discretionary/systematization/*.md`

Rules:
- Agents may append to append-only files, but must not rewrite prior entries.
- Automations may write only to designated observation and evaluation files.
- Automations must queue durable-memory promotions in `workspace/inbox.md`.
- Discretionary trade files and systematic strategy files must remain distinct
  until an idea passes through explicit systematization.
- No agent may change canonical headings or schemas without updating this file.

## Canonical Schemas

### Strategy Files

Every strategy file must include:

- `Strategy`
- `Status`
- `Scope`
- `Linked Experiments`
- `Known Failure Modes`
- `Open Questions`

### Expectation Entries

Every expectation entry must include:

- `Source`
- `Timestamp`
- `Horizon`
- `Claim`
- `Measurable Check`
- `Confidence`
- `Status`

### Observation Files

Every daily, weekly, and monthly observation file must include:

- `Market Summary`
- `Leadership`
- `Breadth`
- `Rates/Vol/Macro`
- `Surprises`
- `Links`

### Evaluation Entries

Every evaluation entry must include:

- `Expectation Ref`
- `Outcome`
- `Verdict`
- `What Was Wrong`
- `Next Adjustment`

### Discretionary Idea Entries

Every discretionary idea entry must include:

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

### Discretionary Trade Entries

Every discretionary trade entry must include:

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

## Standard Operating Procedure

### Manual Research Logging

1. Run or inspect the relevant strategy test.
2. Record the run in `memory/research_log.md`.
3. Store structured metrics as a JSON block.
4. Write a critic evaluation with failure points and next iteration steps.
5. Promote only durable conclusions into curated files.

### Automation Behavior

1. Read the current focus and relevant open expectations.
2. Use local repo data first when relevant.
3. Supplement with live APIs and web/news context when needed.
4. Write only to designated observation and evaluation files.
5. Add durable-memory suggestions to `workspace/inbox.md`, never silently
   rewrite findings or decisions.

### Discretionary Workflow

1. Log manual ideas in `discretionary/ideas/open_ideas.md`.
2. Record executed discretionary trades in `discretionary/trades/open_trades.md`.
3. Move closed trades to `discretionary/trades/closed_trades.md`.
4. Review process quality in `discretionary/trades/postmortems.md`.
5. Promote repeatable discretionary patterns into
   `discretionary/systematization/candidates.md`.
6. Only promote a discretionary setup into `memory/strategies/...` after it has
   a testable rule definition and explicit systematization record.

## Optional Mode Overrides

- Agents should infer the working mode from the user prompt by default.
- Users may optionally prefix prompts with:
  - `Mode: manual idea`
  - `Mode: trade review`
  - `Mode: systematize`
  - `Mode: systematic`
- [workspace/mode.md](workspace/mode.md) is an optional override and strong
  hint when it has been updated recently.
- If a prompt mixes discretionary trade discussion and coding, the default is
  `discuss first`, then propose a short systematization path only if useful.

## Source Policy

- Local repo artifacts are the first source for strategy-specific context.
- `yfinance` or similar may be used for recent price context when local data is
  stale or absent.
- `FRED` or similar may be used for macro and rates context.
- Web search may be used for recent events, validation, and contradiction checks.
- If sources conflict, the automation must note the conflict explicitly.

## Canonical Links

- [workspace/current_focus.md](workspace/current_focus.md)
- [workspace/session_handoff.md](workspace/session_handoff.md)
- [workspace/mode.md](workspace/mode.md)
- [memory/research_log.md](memory/research_log.md)
- [memory/strategies/smh_regime_filter/strategy.md](memory/strategies/smh_regime_filter/strategy.md)
- [agent_response_contract.md](agent_response_contract.md)
- [discretionary/manifest.md](discretionary/manifest.md)
- [automations/automation_manifest.md](automations/automation_manifest.md)
