# Yahoo Finance Watchlists

This file is the canonical reusable list of Yahoo Finance tickers for recurring
automations and future research in this repo.

## Rules

- Prefer these watchlists over retyping ticker sets in new prompts or notes.
- If a ticker is replaced, update this file and the linked topic note together.
- Topic notes may explain why a watchlist matters, but this file is the
  canonical place for the actual reusable universe.

## Large Vs Small Caps

- `DIA` - Dow proxy
- `SPY` - S&P 500 proxy
- `QQQ` - Nasdaq 100 proxy
- `IWM` - Russell 2000 proxy

## Asia Equity Index ETF Proxies

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

## Sector And Industry Relative Performance

- `XLC` - communication services
- `XLY` - consumer discretionary
- `XLP` - consumer staples
- `XLE` - energy
- `XLF` - financials
- `XLV` - health care
- `XLI` - industrials
- `XLB` - materials
- `XLRE` - real estate
- `XLK` - technology
- `XLU` - utilities
- `SMH` - semiconductors
- `IGV` - software

## Hard Assets And Bitcoin

- `GC=F` - gold futures
- `SI=F` - silver futures
- `HG=F` - copper futures
- `CL=F` - WTI crude oil futures
- `BTC-USD` - bitcoin spot

## Sovereign Duration Tracks

### US

- `SHY` - short US Treasury proxy
- `IEF` - medium US Treasury proxy
- `TLT` - long US Treasury proxy

### UK

- `IGLS.L` - short UK gilt proxy
- `VGOV.L` - medium or broad UK gilt proxy
- `GLTL.L` - long UK gilt proxy

### Japan

- `2510.T` - broad Japan bond proxy
- `236A.T` - 7-10 year Japan government bond proxy
- `2561.T` - core Japan government bond proxy

## Known Caveats

- Commodity futures symbols are directional tracking tools, not ETF total-return proxies.
- Japan short and long duration ladders are provisional until the newly approved
  `571A` and `573A` JGB ETFs become available in Yahoo Finance data.
- Korea uses `KOSPI 200` as the practical large-cap equity benchmark proxy rather
  than a direct `KOSPI` index ticker.
- China size buckets are defined with ETF proxies rather than raw index symbols.
