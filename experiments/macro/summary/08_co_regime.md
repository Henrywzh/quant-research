# Copper/Oil Ratio Macro Regime Overlay for Equity Indices (Public Summary)

> **Abstract**  
This paper presents a rule-based **macro regime overlay** driven by the **Copper/Oil ratio** to manage long-only equity index exposure. The method does not target short-horizon return prediction; instead, it classifies market conditions into **Risk-On (ON)** vs **Risk-Off (OFF)** regimes and uses the regime state to **gate or scale** equity exposure. Across multiple major indices, the overlay exhibits strong **regime state separation**—ON regimes have materially better risk-adjusted outcomes than OFF regimes. The dominant contribution comes from **tail-risk mitigation**, evidenced by meaningfully reduced losses during worst-market-day quantiles. Failures are concentrated in **event-driven/policy shock** episodes, where discontinuous repricing can occur faster than a macro proxy can adapt. This is a public summary; exact implementation parameterization is intentionally omitted.

---

## 1. Motivation

Macro proxies can be useful not as short-term predictors, but as **slow-moving risk context** for exposure management. The **Copper/Oil ratio** is a compact representation of the balance between:
- **industrial demand / growth impulse** (copper), and
- **energy cost / inflation pressure** (oil).

The central hypothesis is that changes in this balance can coincide with persistent **risk appetite** shifts. Rather than forecasting next-day returns, the Copper/Oil ratio is used as a **macro-frequency overlay** to:
- remain invested during favorable conditions (**ON**), and
- reduce exposure during unfavorable conditions (**OFF**).

---

## 2. Strategy Overview (High Level)

### 2.1 Overlay objective
The overlay is designed as a **top-level risk switch** applied to a baseline long-only equity index exposure:
- **ON (Risk-On):** allow long exposure  
- **OFF (Risk-Off):** reduce/neutralize exposure

### 2.2 Stability and turnover control
To avoid excessive switching caused by boundary noise, the regime classifier is constructed to be robust via:
- signal standardization, and
- a two-threshold switching rule (hysteresis-style behavior) to reduce regime churn.

### 2.3 Realistic timing convention
The regime state is applied with a realistic convention (signal at *t* affects exposure from *t+1* onward) to avoid look-ahead effects.

---

## 3. Data, Markets, and Evaluation Design

### 3.1 Markets covered
This research evaluates the overlay across several major equity indices. For clarity, the presentation is split into:

**Primary figures shown in the report (one per region)**
- **Hong Kong:** `^HSI`
- **China (A-share proxy):** `SS001`
- **Japan:** `^N225`

**Additional validation indices (tested, referenced for robustness)**
- **Hong Kong:** `^HSCE`
- **China:** `CSI300`

### 3.2 Backtest framing (public)
- Frequency: daily data
- Overlay exposure: long-only when ON; reduced/neutral when OFF
- Core outputs:
  - annualized return/volatility, Sharpe, max drawdown
  - regime share (time in ON) and flip frequency
  - tail-loss behavior (worst-day quantile loss reduction)

---

## 4. Backtest Evidence

The figures below provide the primary empirical evidence.

### 4.1 Equity curve: Overlay vs Buy & Hold
- **Figure 1:** `^HSI` equity curve (Overlay vs Buy & Hold)  
![^HSI_long_only_equity_curve](../img/^HSI_long_only_equity_curve.png)

- **Figure 2:** `SS001` equity curve (Overlay vs Buy & Hold)  
![SS001_long_only_equity_curve](../img/SS001_long_only_equity_curve.png)

- **Figure 3:** `^N225` equity curve (Overlay vs Buy & Hold)  
![^N225_long_only_equity_curve](../img/^N225_long_only_equity_curve.png)

### 4.2 Drawdown: tail-risk behavior
- **Figure 4:** drawdown comparison for `^HSI` (Overlay vs Buy & Hold)  
![^HSI_drawdown](../img/^HSI_drawdown.png)

### 4.3 Year-by-year performance (attribution-friendly view)
- **Figure 5:** `^HSI` yearly returns (Overlay vs Buy & Hold)  
![^HSI_yearly_returns](../img/^HSI_yearly_returns.png)

### 4.4 Regime decision timeline (position state)

This figure visualizes the **final regime decision** used for exposure control: **ON (1)** vs **OFF (0)**.
- **Figure 6:** `^HSI` position state over time (ON=1, OFF=0)  
![^HSI_long_only_position](../img/^HSI_long_only_position.png)

---

## 5. Table A — Regime State Separation (Primary Indices)
Table A summarizes how return/risk characteristics differ between OFF and ON regimes.  
This table focuses on the primary indices shown in the figures: `^HSI`, `SS001`, and `^N225`.

**Table A. ON vs OFF regime statistics**

| Market | State | Days | Ann. Ret | Ann. Vol | Sharpe | Max DD (in state) | ON Share | Flips/Yr |
|---|---:|---:|---:|---:|---:|------------------:|---:|---:|
| ^HSI  | OFF | 2721 | -10.9% | 25.3% | -0.43 |            -81.6% | 0.54 | 4.8 |
| ^HSI  | ON  | 3161 | 19.7%  | 21.8% | 0.90  |            -31.0% | 0.54 | 4.8 |
| SS001 | OFF | 2598 | -13.4% | 25.3% | -0.53 |            -86.4% | 0.54 | 4.8 |
| SS001 | ON  | 3007 | 21.8%  | 23.1% | 0.94  |            -38.0% | 0.54 | 4.8 |
| ^N225 | OFF | 2562 | -5.2%  | 24.9% | -0.21 |            -79.9% | 0.54 | 4.8 |
| ^N225 | ON  | 2972 | 21.4%  | 23.6% | 0.91  |            -31.1% | 0.54 | 4.8 |

**Interpretation**
- Across Hong Kong, China, and Japan, ON regimes are associated with **positive annualized returns and materially higher Sharpe**, while OFF regimes are associated with **negative or substantially weaker outcomes**.
- ON share is approximately **0.53–0.54**, and flip frequency is approximately **4–5 per year**, consistent with a **macro-frequency overlay** rather than high-turnover timing.

---

## 6. Tail-Risk Evidence: Worst-Day Loss Reduction

This section evaluates whether the overlay meaningfully reduces losses during the market’s worst days—an explicit test of the strategy’s risk-control mechanism.

### 6.1 Table B — Worst-day loss reduction (q = 1%, 5%)

**Table B. Worst-day loss reduction**

**Definition (worst-day quantiles):** for each market, “worst q% days” are identified using the **buy-and-hold daily returns** over the full sample. The overlay strategy return is then evaluated **on those same dates**. The overlay follows a realistic timing convention: the regime decision observed at time *t* affects exposure from *t+1* onward.


| Market | q | Avg BH (worst) | Avg Overlay (worst) | Capture Ratio | Avg Exposure (worst) |
|---|---:|---:|---:|---:|---:|
| ^HSI  | 1% | -5.53% | -1.83% | 0.33 | 0.34 |
| ^HSI  | 5% | -3.35% | -1.36% | 0.41 | 0.43 |
| ^HSCE | 1% | -6.78% | -1.81% | 0.27 | 0.27 |
| ^HSCE | 5% | -4.14% | -1.62% | 0.39 | 0.41 |
| SS001 | 1% | -6.26% | -2.95% | 0.47 | 0.47 |
| SS001 | 5% | -3.74% | -1.66% | 0.44 | 0.44 |
| CSI300 | 1% | -7.59% | -4.76% | 0.63 | 0.58 |
| CSI300 | 5% | -3.99% | -2.02% | 0.50 | 0.46 |

### 6.2 How to read Table B
- **Avg BH (worst):** the average buy-and-hold return on the **worst q% days** (worst days are selected by ranking BH daily returns from lowest to highest).
- **Avg Overlay (worst):** the overlay strategy’s average return on that **same set of worst days**.
- **Capture Ratio:** `Avg Overlay (worst) / Avg BH (worst)`; **smaller is better** (values **< 1** indicate the overlay reduces exposure to tail-loss days).
- **Avg Exposure (worst):** the overlay’s average absolute exposure on those worst days (lower values indicate more effective de-risking when tail events hit).

**Key takeaway**
The overlay substantially reduces losses on the worst 1%–5% days across the tested markets, supporting the view that the performance improvement is driven primarily by **tail-risk mitigation** rather than return chasing.

---

## 7. Mechanism Evidence and Risk Management Interpretation

### 7.1 Primary mechanism: tail-risk mitigation
The strongest evidence for the overlay’s edge is the combination of:
1) clear state separation (Table A), and  
2) reduced exposure and smaller losses in the worst-tail windows (Table B).

This is consistent with the overlay acting as a **macro-frequency drawdown control layer**.

### 7.2 How to position the overlay in a portfolio
The overlay is best treated as a **top-level risk control module**, rather than a standalone alpha engine. Practical integrations include:
- applying it to a baseline long-only exposure, or
- using it as a cap on risk allocation in broader systematic portfolios.

---

## 8. Boundary Conditions and Failure Modes

The overlay is not designed to be optimal in all regimes. Its core limitation is that **macro proxies are slow-moving** relative to certain shock types.

### 8.1 When it tends to work
- multi-month macro trend environments where risk appetite evolves gradually,
- drawdowns driven by macro deterioration rather than single-day discontinuities.

### 8.2 When it can fail (and why)
Failures concentrate in **event-driven and policy-driven shocks**, where large discontinuous moves can occur while the regime remains ON. Common categories include:
- **trade and tariff escalation** episodes (rapid repricing of growth and risk premia),
- **domestic policy pivots** that reset expectations abruptly,
- **pandemic-style shocks** (gap moves, liquidity stress),
- fast geopolitical escalations.

In these scenarios, the market may reprice faster than a macro regime proxy can update, causing **late de-risking**.

### 8.3 How failures should be communicated (interpretability)
A clean interpretability approach is to document a small set of widely-known macro episodes and show:
- the regime state (ON/OFF) during the episode window,
- whether worst-tail loss reduction was achieved,
- failure examples where large down days occurred while ON, tied to the episode’s shock narrative.

This explains “why it failed here” without introducing a second signal.

---

## 9. Conclusion

Across multiple major equity indices, a Copper/Oil-driven regime overlay provides a robust macro-frequency exposure control layer, improving risk-adjusted outcomes primarily through tail-loss reduction, while remaining vulnerable to fast, event-driven policy shocks.

**One-line summary:**  
A Copper/Oil macro regime overlay improves long-only index exposure outcomes mainly by mitigating tail losses, but can lag in abrupt event-driven repricing regimes.
