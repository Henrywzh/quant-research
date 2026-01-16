# 🧠 Quantitative Macro Research by Henry Wu

Welcome to my **Quantitative Macro & Investment Research Repository** — a collection of reproducible projects combining **economic intuition**, **statistical modeling**, and **systematic portfolio design**.

This repo documents my full research journey — from exploring macroeconomic indicators and yield curves, to building predictive recession models and dynamic asset allocation strategies.

---

Research/
  data/                      # keep
  pdf/                       # keep

  src/
    qresearch/               # reusable library package (new)
      __init__.py
      backtest/
        __init__.py
        portfolio.py         # weights backtest engine
        buckets.py           # bucket backtest + tearsheet
        metrics.py           # sharpe/maxdd/etc (single source of truth)
      portfolio/
        __init__.py
        weights.py           # build_strategy_weights/topk/etc
      signals/
        __init__.py
        momentum.py          # ROC, MA strength, etc
      data/
        __init__.py
        yfinance.py          # download_close/download_ohlc
      utils/
        __init__.py
        artifacts.py         # config/returns/equity/weights export
        checks.py            # alignment + sanity gates

  experiments/               # your current topic folders (moved/renamed)
    factor_model/
    index_timing/
    interest_rates/
    macro/
    mean_reversion/
    momentum/
    options/
    technical_analysis/

  artifacts/                 # standardized outputs per run (new)
  tests/                     # smoke tests (new)

  pyproject.toml             # packaging so imports work
  README.md
  prompt.md
  .env
  .gitignore

---

## 🎯 Research Purpose

The goal of this research is to understand **how macroeconomic variables translate into investment signals**, and to build models that:
1. Quantify the probability of economic recessions or regime shifts.  
2. Convert macro signals into **systematic, testable trading or allocation strategies**.  
3. Evaluate performance through **Sharpe ratio, drawdown, turnover, and robustness metrics**.  
4. Bridge the gap between **economic reasoning** and **data-driven portfolio management**.

---

## 📈 Current Research Roadmap

| Notebook | Title | Description | Core Output |
|:--|:--|:--|:--|
| `01_data_exploration.ipynb` | 数据获取与探索 | 清洗并对齐 FRED 宏观数据，计算利差、通胀、流动性指标 | 完整宏观数据集 |
| `02_yield_curve_dynamics.ipynb` | 收益率曲线因子分析 | 通过 PCA 提取 Level / Slope / Curvature 三因子 | 收益率曲线主成分 |
| `03_macro_relationships.ipynb` | 宏观变量关系 | 用回归方法研究利率因子与通胀、失业率、GDP 的动态关系 | 经济周期因果线索 |
| `04_recession_prediction.ipynb` | 衰退概率建模 | 基于多变量逻辑回归预测未来 12 个月衰退 | AUC≈0.81 的稳定模型 |
| `05_multivariate_oos.ipynb` | 样本外验证 | 扩展窗口 OOS 回测，验证预测稳健性 | 样本外概率序列 |
| `06_backtest_regimes.ipynb` | 固定切换策略 | 衰退概率≥0.6 转债券，否则持股 | Sharpe=0.62，风险防御有效 |
| `07_enhanced_regime_strategy.ipynb` | 动态权重配置 | 平滑权重 + 滞后确认 + 现金过滤层 | Sharpe=0.86，换手率<2% |

---

## 🧩 Core Research Question

> Can the **yield curve and macro variables** be used to predict economic regimes  
> — and can those signals be **translated into dynamic portfolio decisions** that outperform static allocations?

---

## 🧠 Key Insights

- **Yield curve inversion** remains the most consistent early warning of recessions.  
- **Combining macro variables** (inflation, unemployment, liquidity) improves predictive stability (AUC ↑ to 0.81).  
- **Dynamic allocation** based on recession probability reduces volatility and drawdown, improving Sharpe from 0.62 → 0.86.  
- Engineering discipline (data lagging, expanding-window OOS, modular backtesting) is crucial for robustness.  

---

## 💼 Practical Relevance

This research reflects how a **quantitative macro analyst or systematic PM** would approach investment design:
- Build interpretable models grounded in economics.  
- Enforce strict out-of-sample validation and avoid look-ahead bias.  
- Evaluate results statistically and economically, not just by fit.  
- Implement strategies in clean, modular Python workflows.

---
