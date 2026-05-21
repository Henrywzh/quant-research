# Agent Response Contract

## Purpose

This file tells agents how to behave when the repo is being used for
systematic research, manual trade discussion, trade review, or
systematization. The goal is to keep mode changes clear without forcing the
user to restate the same instructions every time.

## Behavior Modes

### `systematic_research`

- Prioritize code, datasets, backtests, signal design, robustness, and tests.
- Default to implementation when the user is clearly asking for research code or
  infrastructure changes.
- Keep discussion tied to measurable rules and verifiable outputs.

### `manual_trade_discussion`

- Prioritize thesis quality, timing, scenario analysis, invalidation, sizing,
  and risk.
- Do not force a coding solution when the user is exploring a discretionary
  idea.
- Surface alternative scenarios and missing information before systematization.

### `trade_review`

- Prioritize decision quality, risk quality, execution discipline, and process
  over raw PnL.
- Separate a good losing trade from a bad winning trade.
- Focus on whether the trade matched the stated plan and whether the risk was
  appropriately framed.

### `systematization`

- Translate discretionary logic into observable proxies, rules, datasets, and a
  minimal test design.
- Identify what is measurable, what is subjective, and what would need a
  research approximation.
- Propose the smallest useful backtest or monitoring framework first.

## Auto-Inference Rules

- Prompts about trades entered, stops, targets, timing, invalidation, risk,
  or "was this a good trade?" should map to a discretionary mode.
- Prompts asking to convert an idea into rules, signals, backtests, or code
  should map to `systematization` or `systematic_research`.
- If a prompt mixes discretionary critique and coding, default to
  `discuss first`, then include a short systematization path only if it is
  clearly useful.
- When ambiguity is high and the choice would materially change the response,
  ask one short clarification question instead of forcing code or critique.

## Working Agreement

- Agents should read `workspace/current_focus.md` and
  `workspace/session_handoff.md` first.
- Agents should use `workspace/mode.md` as an override if it exists and has
  been updated recently.
- Discretionary journals should stay separate from systematic strategy memory
  until an explicit systematization step occurs.
