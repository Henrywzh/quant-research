# quant-research

A personal quantitative research repository for systematic investing across multiple asset classes. This repo consolidates my experiments and notes on portfolio construction, alpha research, and robust evaluation workflows, with a focus on practical, reproducible research.

## Scope

This project covers:

- **Multi-asset research:** equities, rates, FX, commodities, and major ETFs/indices
- **Market timing & overlays:** regime-aware exposure management, volatility targeting, risk controls
- **Relative value / long–short:** cross-sectional signals, ranking frameworks, spread trades, hedged portfolios
- **Macro regime research:** slow-moving macro proxies used as context for positioning and risk management
- **ML in quant finance:** feature engineering, model selection, validation discipline, and backtest integration

## Repository Highlights

- **Research notebooks / experiments:** iterative exploration of signals, regimes, and portfolio rules
- **Backtesting & evaluation:** performance attribution, drawdowns, turnover, and robustness checks
- **Reusable outputs and notes:** distilled findings and write-ups for future reference

## Key Findings and Write-ups

Useful research notes, summaries, and findings are stored in the **`pdf/`** directory.  
If you want the quickest overview of what has been tested and what worked/didn’t, start there.

## Typical Research Workflow

1. Define hypothesis (signal, regime overlay, or relative-value construct)
2. Validate data quality and tradability assumptions
3. Implement baseline strategy + sensible benchmark
4. Evaluate (returns, risk, drawdowns, turnover, stability)
5. Run robustness checks (subperiods, parameter sensitivity, alternative universes)
6. Document conclusions and next steps in `pdf/`

## Conventions

- Focus on **no-lookahead** logic and realistic rebalancing assumptions
- Prefer **simple baselines** before introducing complexity
- Separate research exploration from conclusions (write-ups live in `pdf/`)

## Disclaimer

This repository is for research and educational purposes only and does not constitute investment advice. Past performance is not indicative of future results.
