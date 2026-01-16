# 📘 Notebook 11 — Findings, Weaknesses, and Improvement Plan  
*(Eq/Bond/Cash Regime Strategy based on EWMA `p_recession`)*

---

## 🧭 Summary of Findings

- The strategy dynamically adjusts equity and bond exposure using the **EWMA-smoothed recession probability (`p_recession`)**.  
- It performs **exceptionally well during inflationary risk-off regimes** (e.g. 2022) where both equities and bonds declined.  
- **Drawdowns are controlled** relative to a static 60/40 portfolio, and the **Sharpe ratio** remains stable over long periods (>1.0).  
- The macro signal successfully aligns with most **NBER recessions**, confirming its **economic interpretability**.

### Key Observations
| Period | Macro Context | Model Behaviour |
|:-------|:--------------|:----------------|
| **2000–2002** | Dot-com bubble burst | Large drawdown (~–30%), late risk-off response |
| **2008–2009** | Global Financial Crisis | Large drawdown (~–30%), signal lagged liquidity shock |
| **2011–2019** | Stable expansion | Strategy stayed risk-on, minimal turnover |
| **2022** | Inflation shock (Eq–Bd correlation ↑) | Excellent performance; quickly de-risked, avoided duration losses |

✅ The EWMA smoothing helped prevent whipsaws, providing gradual transitions between risk-on and risk-off phases.  
✅ The model’s structure is **simple, interpretable, and robust** for most macroeconomic regimes.

---

## ⚠️ Struggles & Weaknesses

1. **Lagging Response to Crashes**
   - The model relies on macro variables (yield curve, spreads, PMI), which react **after** markets move.
   - In 2000 and 2008, equities fell before recession probabilities spiked.

2. **Liquidity Crisis Blind Spot**
   - 2008 was driven by systemic liquidity stress.
   - Both equities and bonds fell simultaneously, breaking the “bonds hedge equities” assumption.

3. **No Volatility or Market-Stress Filter**
   - The strategy only reacts to macro deterioration, not to sudden market stress.
   - Misses “fast crashes” that occur before economic data confirms a slowdown.

4. **Drawdown Size**
   - Maximum drawdown ≈ **–30%** — acceptable for unlevered macro models but **too large for a defensive portfolio**.
   - Lacks volatility targeting or capital preservation overlay.

5. **Single Signal Dependency**
   - Entire portfolio allocation depends on one variable (`p_recession`).
   - No diversification across other risk factors (e.g., credit spreads, vol spikes, cross-asset trends).

---

## 💡 How to Improve

| Issue | Potential Fix | Implementation Idea |
|:------|:---------------|:--------------------|
| **Signal Lag** | Combine macro `p_recession` with faster *market-based* indicators | Add a “market stress” score using realized volatility, VIX proxy, or credit spreads |
| **Liquidity Crisis Sensitivity** | Include liquidity or funding metrics | Use TED spread, OIS–Libor spread, or credit default swap indices |
| **Large Drawdown** | Add volatility targeting / dynamic scaling | Reduce exposure when realized vol or drawdown > threshold |
| **Single Factor Risk** | Blend multiple macro & market factors | Weighted composite: `0.7*p_recession + 0.3*vol_zscore` |
| **Defensive Layer** | Introduce secondary cash/hedge sleeve | Shift to cash when both equity & bond momentum < 0 |

### Example Extended Risk Score
$\mathrm{RiskScore}_t = 0.7\, p_{\mathrm{rec,EWMA}} + 0.3\, z^{(\mathrm{vol})}_t$

Use this combined score to smooth and accelerate de-risking in future versions.

---

## 🧩 Next Steps (Notebook 12)

1. **Develop a Volatility Shock Detector**
   - Compute realized volatility or a simple GARCH estimate.
   - Trigger partial de-risking when vol spikes beyond 2σ of its rolling mean.

2. **Create a Combined Regime Score**
   - Integrate macro (`p_recession`) + market (`vol_zscore`) → composite risk probability.

3. **Add Conditional Allocation Layer**
   - When both macro and market risk are high → hold 50% cash.
   - When macro risk high but market calm → hold bonds.
   - When both low → full equity exposure.

4. **Re-run robustness tests (Notebook 11 style)**
   - Evaluate new composite strategy across 2000, 2008, and 2022 to verify smoother drawdown.

---

## ✅ Conclusion

- The current EWMA-based regime model is **robust and interpretable**,  
  demonstrating clear value over traditional 60/40 portfolios, especially in inflationary and post-COVID periods.  
- However, its **lagged macro response** leads to large drawdowns during fast market crashes (2000, 2008).  
- The next improvement phase should focus on **faster risk detection and volatility-aware scaling**  
  to preserve capital in abrupt downturns while maintaining the macro intuition of the framework.
