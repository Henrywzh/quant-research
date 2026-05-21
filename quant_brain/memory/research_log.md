# Research Log

This file is append-only. Each entry records what was tested, the structured
result, the critic view, and the next exact iteration.

## Entry: 2026-05-20 / SMH Regime Filter / EXP-001

### Metadata

- Strategy: `smh_regime_filter`
- Experiment ID: `EXP-001`
- Source: `manual_seed`
- Status: `completed`
- Data Window: `2015-01-02 to 2026-05-19`
- Benchmark: `SMH buy-and-hold`

### Configuration

- Rule: invest in `SMH` only when the prior close is above the 200-day moving average
- Execution Convention: daily close-to-close proxy with a one-day signal lag
- Days In Market: `0.768263`

### Result

```json
{
  "strategy": {
    "ann_return_geo": 0.261197,
    "ann_vol": 0.243152,
    "sharpe": 1.076721,
    "max_dd": -0.324537,
    "total_return": 12.93822,
    "n_obs": 2861
  },
  "benchmark": {
    "ann_return_geo": 0.314975,
    "ann_vol": 0.314625,
    "sharpe": 1.028209,
    "max_dd": -0.453026,
    "total_return": 21.39189,
    "n_obs": 2861
  }
}
```

### Critic Evaluation

- The filter improved risk-adjusted performance and reduced max drawdown.
- The rule gave up a meaningful amount of absolute return during persistent
  semiconductor bull phases.
- The setup is useful as a seed memory drill, but too simple to treat as a
  durable edge without breadth, leadership, or macro context.

### Failure Points

- Single-input regime logic is likely too blunt for a concentrated sector ETF.
- A pure 200-day trend gate may exit late during sharp selloffs and re-enter late
  during strong rebounds.
- No transaction cost, tax, or slippage adjustments were included.

### Next Iteration

- Add breadth and large-vs-small leadership conditions around the trend gate.
- Compare 200-day, 150-day, and dual-threshold variants.
- Test whether a benchmark-relative filter works better than a standalone one.
