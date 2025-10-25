# Studying interest rates and macro

## Step 1: Data Exploration
| Step                | Description                                                                           | Output            |
| ------------------- | ------------------------------------------------------------------------------------- | ----------------- |
| ✅ Fetch data        | Downloaded FRED series: `DFF`, `DGS2`, `DGS10`, `CPIAUCSL`, `M2SL`, `GDPC1`, `UNRATE` | 7 key indicators  |
| ✅ Resample & align  | Converted daily/quarterly series to **monthly frequency** (1959–present)              | ~780 observations |
| ✅ Derived variables | Added yield spreads (10Y–2Y), inflation YoY, M2 growth YoY                            | New columns       |
| ✅ Visual checks     | Plotted yield curve, spreads, real rates, M2 vs DFF                                   | Validated trends  |

## Step 2: Exploratory Analysis
| Objective                | Techniques                                             | Deliverable                           |
| ------------------------ | ------------------------------------------------------ | ------------------------------------- |
| 📈 Yield curve structure | Compare DFF, DGS2, DGS10 across cycles                 | Notebook: *01_data_exploration.ipynb* |
| 📉 Spread dynamics       | Plot 10Y–2Y spread and overlay recessions              | —                                     |
| 🔍 Correlation analysis  | Compute relationships with inflation, unemployment, M2 | —                                     |
| 🧠 Real policy rate      | Compute `real_dff = DFF – inflation_yoy`               | —                                     |
| 🧩 Macro co-movement     | Visualize M2 growth, GDP, UNRATE vs DFF                | —                                     |

## Step 3: Yield Curve & Macro Dynamics
| Objective            | Method                                                                              | Notebook                        |
| -------------------- | ----------------------------------------------------------------------------------- | ------------------------------- |
| Yield curve factors  | **PCA** on yields (3M, 2Y, 5Y, 10Y, 30Y) → Level, Slope, Curvature                  | `02_yield_curve_dynamics.ipynb` |
| Macro regression     | OLS or VAR between yield factors and macro variables (inflation, GDP, unemployment) | `03_macro_relationships.ipynb`  |
| Recession prediction | Logistic regression using 2s10s spread or slope factor                              | `04_recession_prediction.ipynb` |

## Step 4: Forecasting Models
| Model                              | Purpose                                                       | Notebook                        |
| ---------------------------------- | ------------------------------------------------------------- | ------------------------------- |
| **VAR (Vector AutoRegression)**    | Jointly model short & long yields with inflation/unemployment | `05_forecasting_VAR.ipynb`      |
| **ARIMA / Kalman / Nelson–Siegel** | Forecast yield curve shape                                    | `06_term_structure_model.ipynb` |
| **Machine learning baseline**      | Random Forest or XGBoost for macro-yield mapping              | `07_ml_baseline.ipynb`          |

## Step 5: Strategy
| Use Case                  | Example                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------- |
| **Macro signals**         | Use yield curve inversion or PCA factors to signal equity/fixed income risk regimes |
| **Bond portfolio tilt**   | Backtest long-short strategies along the yield curve                                |
| **Cross-market analysis** | Compare U.S. vs Eurozone or China rates once your infra supports multiple regions   |
