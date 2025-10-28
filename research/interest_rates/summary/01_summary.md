# Research Summary – Interest Rate Model

**Date:** 2025-10-26  
**Researcher:** Henry Wu  
**Objective:**  
Test whether term-spread (10Y-3M) and real rate level predict next-12-month equity returns and recessions.

---

### 1. Data & Features
- Source: FRED, daily → monthly resample  
- Sample: 1980–2025  
- Features: [Term spread, Inflation, Real Fed Funds, ISM PMI]  
- Target: 12M recession indicator (NBER)

---

### 2. Models
- Logistic regression (baseline)
- Random Forest (non-linear test)
- Expanding-window out-of-sample test (1980–2010 train, 2010–2025 test)

---

### 3. Key Results
| Metric | Train | Test |
|:-------|:------|:-----|
| AUC | 0.83 | 0.78 |
| Accuracy | 0.75 | 0.72 |

Backtest Sharpe (probability-weighted allocation): **1.02**

---

### 4. Interpretation
- Yield curve inversion remains the dominant predictor.
- Real rate level adds marginal explanatory power post-2008.
- Model confidence spikes before every major recession except 2001.

---

### 5. Limitations & Next Steps
- Potential look-ahead bias in PMI variable.
- Test robustness using LASSO or macro regime clustering.
- Integrate with asset allocation (risk-on / risk-off weighting).
