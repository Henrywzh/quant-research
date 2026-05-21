# Source Registry

## Canonical Sources

- Local repo data under `data/processed` for stable research artifacts
- `yfinance` for recent market context or missing local ETF price history
- `FRED` for macro and rates context
- web/news sources for recent events and contradiction checks
- [yfinance_watchlists.md](yfinance_watchlists.md) is the canonical reusable ticker universe file for automations and future research.

## Equity Proxy Policy

- Prefer ETF proxies instead of raw index tickers when building reusable market watchlists.
- For China equity size work, use ETF proxies for `上证50`, `CSI 300`, `CSI 500`, and `CSI 1000`.
