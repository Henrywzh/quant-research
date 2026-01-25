# Context for ChatGPT: Current Quant Research + Backtest Modules + Coding Workflow

You are assisting me (Henry) on an ongoing quantitative equity research project. Please **follow my existing backtest modules and workflow** when writing code, and **do not invent a new framework** unless explicitly requested.

---

## 1) What I am doing right now (Current Research)

### Project: HSCI (Hong Kong) Cross-Sectional Single-Factor Research
I have completed single-factor testing on an HSCI universe with point-in-time eligibility filters and monthly rebalancing (every 20 trading days). The goal is **alpha research** (no stop-loss / no risk management overlay), and to understand **where performance comes from** (stock-selection vs structural exposures).

I tested three factor directions:
- **Trend / price momentum** (strongest)
- **Volatility** using high-low std over 3 months (low-vol has higher score) (strong)
- **Liquidity (Amihud)** (weaker; not a focus)

I also ran both:
- **Raw (un-neutralised)** factor scores
- **Neutralised** factor scores (sector + size using approximated market cap)

Key additional work:
- I computed and analyzed:
  - **Sector exposures** (Long/Short/Net weights by sector over time)
  - **Size exposures** (ln(mcap) mean for Long/Short/Universe over time)
  - **Explained variance** (per-date **R²** of score ~ sector, and score ~ sector + ln(mcap))
- For the Trend factor, **R²_sector** shows evidence of a **2–3 year quasi-cycle**:
  - ACF plateau at **22–32 months** is significant
  - FFT strongest period ≈ **27.14 months**
- For Vol factor, **R²_sector** does **not** show a statistically significant 22–32 plateau.

I also built a condition test:
- **2×2 conditional comparison**: R²_sector HIGH vs LOW (top 20% vs bottom 20%) comparing **Raw L/S vs Neutralised L/S** performance.

---

## 2) Universe & Data Assumptions

### Universe: HSCI constituents (point-in-time)
Eligibility at each rebalance date:
- Market cap > **HKD 1bn** (approximated; from yfinance)
- 1-month average price > **HKD 1**
- Stock is in **HSCI** at that time
- Suspended / missing price data → treated as **not tradable**

### Data sources
- OHLCV, market cap (approx), sector from **yfinance**
- HSCI membership history from **Hang Seng Index website**
- Benchmark: **^HSI** index (price series)

---

## 3) Portfolio / Backtest Setup (Standard)
- Rebalance / holding: **H = 20 trading days** (monthly)
- Construction: **Long best 10%** / **Short worst 10%**
- Entry mode: **next_close**
- Trading fees included (per my backtest cost model)
- No stop-loss / no additional risk management (pure alpha research)

---

## 4) My Backtest Modules / Objects (Use These)

### Market data container
- `md` contains:
  - `md.close` (date × ticker close prices)
  - (and other OHLCV fields as available)

### Signal
- I compute signals using:
  - `sig = compute_signal(md, name=signal_name, **params)`
  - then align: `sig = sig.reindex(index=md.close.index, columns=md.close.columns)`

### Eligibility mask
- `universe_eligible`: DataFrame (date × ticker bool)

### Neutralisation (sector + size, FWL)
- `neu = neutralize_sector_and_mcap_fwl(sig, hsci_sector_map['sector'], hsci_market_cap, universe_eligible)`

### Evaluation / report (tearsheet)
- `rep = make_tearsheet(md=md, signal=sig, H=20, n_buckets=10, entry_mode="next_close", min_assets_ic=50, plot=True, benchmark_price=hsi, benchmark_name="^HSI", universe_eligible=universe_eligible)`
- `rep_neu = make_tearsheet(md=md, signal=neu, ...)`

Each `rep` contains keys:
`['meta','bucket_ret','bucket_labels','ret_fwd','coverage','n_valid','ic','ic_stats','ic_roll','bucket_summary',
 'monotonic_spearman_bucket_vs_mean','best_bucket','worst_bucket','turnover_best','turnover_worst',
 'ls_ret','ls_perf','ls_eq','ls_dd','ls_yearly',
 'benchmark_ret','benchmark_perf','benchmark_eq','benchmark_dd','benchmark_yearly']`

### Exposure diagnostics (already implemented / expected)
- Sector weights: Long/Short/Universe per date
- Net exposure: Long − Short
- Size exposure: ln(mcap) stats for Long/Short/Universe per date
- Explained variance:
  - per-date **R²_sector**: score ~ sector
  - per-date **R²_sector+size**: score ~ sector + ln(mcap)
  - per-date **R²_market_risk**: score ~ beta_to_^HSI (rolling beta)

---

## 5) Coding Requirements (Very Important)
When writing code for me:
1. **Integrate with my existing workflow** (use `compute_signal`, `neutralize_sector_and_mcap_fwl`, `make_tearsheet`, `universe_eligible`, `md`, `hsi`).
2. Prefer **vectorized pandas** solutions; avoid unnecessary loops unless required.
3. Keep outputs consistent:
   - return DataFrames/Series aligned on dates and tickers
   - store results into dicts like `results[...] = rep`
4. When adding new diagnostics:
   - provide both the computation + a plot function
   - avoid seaborn; use matplotlib only
5. Do not introduce new external dependencies unless I ask.

---

## 6) What I want from you (How you should help next)
- Write code that plugs into my backtest objects to:
  - generate additional diagnostics or plots
  - test hypotheses (e.g., whether sector leadership drives trend factor effectiveness)
  - implement conditional regime logic using my computed R² series
- When suggesting new analyses, propose:
  - the minimal experiment
  - the expected output artifacts (plots/tables)
  - how to interpret results in an investor-friendly way

---

## 7) Current Priority Tasks (Immediate)
1) Improve interpretability of Trend factor:
   - sector net exposure time series
   - sector concentration (top-k sector share) vs time
   - relate these to R²_sector high/low regimes
2) Compare trend vs vol on the above, focusing on why only trend shows 2–3y R²_sector cycle.

Please proceed accordingly and always write code consistent with my modules and data structures.
