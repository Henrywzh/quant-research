# Sovereign Duration Tracks

## Purpose

Track sovereign duration behavior across the US, UK, and Japan using Yahoo
Finance instruments that are practical to query and compare.

## US Yahoo Finance Universe

- `SHY` - short US Treasury proxy
- `IEF` - medium US Treasury proxy
- `TLT` - long US Treasury proxy

## UK Yahoo Finance Universe

- `IGLS.L` - short UK gilt proxy
- `VGOV.L` - medium or broad UK gilt proxy
- `GLTL.L` - long UK gilt proxy

## Japan Yahoo Finance Universe

- `2510.T` - broad Japan bond proxy
- `236A.T` - 7-10 year Japan government bond proxy
- `2561.T` - core Japan government bond proxy

## Research Questions

- How do US, UK, and Japan duration tracks behave around equity leadership shifts?
- Does long-duration strength or weakness line up with large-vs-small cap behavior?
- Do bond tracks provide cleaner macro confirmation than single yield points?

## Notes

- US and UK duration ladders are straightforward with current Yahoo symbols.
- Japan is still a provisional ladder today:
  - `236A.T` is a clean intermediate-duration JGB proxy.
  - `2510.T` and `2561.T` are broader JGB proxies rather than perfect short or
    long buckets.
  - Newly approved `571A` (1-3Y JGB) and `573A` (20+Y JGB) products are expected
    to begin trading on May 27, 2026 and should be revisited once Yahoo data is available.
