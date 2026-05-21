# Strategy: smh_regime_filter

## Strategy

- Name: `smh_regime_filter`
- Status: `exploratory`
- Scope: sector ETF regime overlay for semiconductor exposure

## Thesis

Semiconductor leadership is cyclical and may benefit from a regime filter that
steps out of sustained drawdown phases while preserving much of the upside
during healthy risk-on environments.

## Canonical Definition

- Universe: `SMH`
- Baseline Rule: long only when prior close is above the 200-day moving average
- Benchmark: `SMH` buy-and-hold
- Execution Proxy: daily close-to-close with one-day lag

## Linked Experiments

- `EXP-001`

## Known Failure Modes

- single-factor trend gating can lag fast regime reversals
- pure price trend may ignore breadth deterioration or broad macro stress
- concentrated ETF behavior may not generalize to wider market regime filters

## Open Questions

- Does a breadth confirmation improve timing?
- Does large-vs-small cap leadership help identify better entry windows?
- Is benchmark-relative strength more informative than the standalone moving average?
