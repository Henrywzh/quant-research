# Large Vs Small Caps

## Purpose

Track the relative performance of large-cap US equity proxies versus small caps
to support regime work, risk-appetite monitoring, and cross-checks against
strategy timing ideas.

## Current Focus

- Compare large-cap leadership across:
  - Dow proxy
  - S&P 500 proxy
  - Nasdaq 100 proxy
- Compare those against the Russell 2000 small-cap proxy.
- Monitor both absolute returns and pairwise relative-strength behavior.
- Extend the regional equity map with major Asia index ETF proxies and explicit
  China size buckets.

## Yahoo Finance Universe

### US

- `DIA` - Dow proxy
- `SPY` - S&P 500 proxy
- `QQQ` - Nasdaq 100 proxy
- `IWM` - Russell 2000 proxy

### Japan

- `1321.T` - Nikkei 225 ETF proxy

### Korea

- `069500.KS` - KOSPI 200 ETF proxy

### Hong Kong

- `2800.HK` - Hang Seng Index ETF proxy

### China Size Buckets

- `510050.SS` - SSE 50 ETF proxy
- `510300.SS` - CSI 300 ETF proxy
- `510500.SS` - CSI 500 ETF proxy
- `159845.SZ` - CSI 1000 ETF proxy

## Research Questions

- When does large-cap growth leadership diverge from broad large-cap leadership?
- Does small-cap underperformance align with tightening financial conditions?
- Can large-vs-small relative moves improve regime filters for equity strategies?
- How do Japan, Korea, Hong Kong, and China size buckets line up with US risk appetite?
- Do China size buckets behave like a usable large-mid-small ladder for regime work?

## Notes

- Use dividend-adjusted close series from Yahoo Finance for comparative total-return-like tracking.
- Pairwise ratios such as `QQQ / IWM` and `SPY / IWM` are likely useful first diagnostics.
- For Asia, start with relative-return panels before introducing FX overlays.
- Treat `KOSPI 200` as the practical Yahoo ETF proxy for Korea large-cap equity tracking.
