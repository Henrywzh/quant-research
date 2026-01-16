"""
Refactored runner: analyze/monitor a specific historical period with configurable dates.

What you can control:
- analysis_start / analysis_end: slice the ratio series to a period of interest (e.g., 2010–2011)
- monitor_start / monitor_end: run walk-forward monitoring only inside [monitor_start, monitor_end]
- piecewise_start: run piecewise scan on log(R) using data from this start date
- oos_start + oos_train_end_1/2: run OOS OLS checks inside a chosen regime window
- target_ratios: the targets you want days-to-target for
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from dataclasses import dataclass
from scipy.stats import norm


# =========================
# Core utilities (same as before)
# =========================

def download_close(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")


def compute_ratio(px: pd.DataFrame, gold_col: str, silver_col: str) -> pd.Series:
    df = px[[gold_col, silver_col]].dropna()
    return (df[gold_col] / df[silver_col]).rename("gs_ratio")


def plot_ratio_and_log(ratio: pd.Series, title_prefix: str = "Gold/Silver Ratio"):
    r = ratio.dropna()
    lr = np.log(r)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(r.index, r.values, linewidth=1)
    ax[0].set_title(title_prefix)
    ax[0].set_ylabel("Ratio")
    ax[1].plot(lr.index, lr.values, linewidth=1)
    ax[1].set_title("log(Ratio)")
    ax[1].set_ylabel("log Ratio")
    plt.tight_layout()
    plt.show()


def rolling_slope_log_ratio(ratio: pd.Series, start: str | None = None, end: str | None = None, roll: int = 30) -> pd.Series:
    r = ratio.dropna()
    if start: r = r.loc[start:]
    if end:   r = r.loc[:end]
    lr = np.log(r)
    y = lr.values
    idx = lr.index
    if len(y) < roll:
        raise ValueError(f"Not enough points for roll={roll}. Have {len(y)}.")
    slopes = []
    for i in range(roll - 1, len(y)):
        yy = y[i - roll + 1:i + 1]
        tt = np.arange(roll, dtype=float)
        b = np.polyfit(tt, yy, deg=1)[0]
        slopes.append(b)
    s = pd.Series(slopes, index=idx[roll - 1:], name=f"roll_slope_{roll}")
    plt.figure(figsize=(12, 3.5))
    plt.plot(s.index, s.values, linewidth=1)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"Rolling slope of log(R) (window={roll})")
    plt.ylabel("slope per day")
    plt.tight_layout()
    plt.show()
    return s


def fit_piecewise_linear_scan(ratio: pd.Series, start: str | None = None, end: str | None = None, min_seg: int = 30):
    r = ratio.dropna()
    if start: r = r.loc[start:]
    if end:   r = r.loc[:end]
    y = np.log(r).values
    n = len(y)
    t = np.arange(n, dtype=float)
    if n < 2 * min_seg + 1:
        raise ValueError(f"Not enough points for min_seg={min_seg}. n={n}")

    best = None
    for tau in range(min_seg, n - min_seg):
        X1 = sm.add_constant(t[:tau].reshape(-1, 1), has_constant="add")
        m1 = sm.OLS(y[:tau], X1).fit()
        sse1 = np.sum(m1.resid ** 2)

        X2 = sm.add_constant(t[tau:].reshape(-1, 1), has_constant="add")
        m2 = sm.OLS(y[tau:], X2).fit()
        sse2 = np.sum(m2.resid ** 2)

        sse = sse1 + sse2
        if (best is None) or (sse < best["sse"]):
            best = {"tau": tau, "sse": sse, "m1": m1, "m2": m2}

    break_date = r.index[best["tau"]]
    return best["m1"], best["m2"], break_date, float(best["sse"]), int(best["tau"])


def oos_ols_logR(ratio: pd.Series, oos_start: str, train_end: str) -> dict:
    """
    Fit log(R) ~ const + t on [oos_start .. train_end], test on (train_end .. end of ratio slice].
    """
    r = ratio.dropna().loc[oos_start:]
    y = np.log(r)

    y_train = y.loc[:train_end]
    y_test = y.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1):]

    # Guard: ensure we have a test set
    if len(y_test) < 2:
        return {
            "rmse_model": np.nan, "rmse_naive": np.nan,
            "params": [np.nan, np.nan], "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "note": "Not enough test observations after train_end."
        }

    t_all = np.arange(len(y), dtype=float)
    t = pd.Series(t_all, index=y.index)

    X_train = sm.add_constant(t.loc[y_train.index].values.reshape(-1, 1), has_constant="add")
    m = sm.OLS(y_train.values, X_train).fit()

    X_test = sm.add_constant(t.loc[y_test.index].values.reshape(-1, 1), has_constant="add")
    yhat = m.predict(X_test)

    rmse = float(np.sqrt(np.mean((y_test.values - yhat) ** 2)))

    y_naive = y_test.shift(1).dropna()
    rmse_naive = float(np.sqrt(np.mean((y_naive.values - y_test.loc[y_naive.index].values) ** 2)))

    return {
        "rmse_model": rmse,
        "rmse_naive": rmse_naive,
        "params": m.params.tolist(),  # [intercept, slope]
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def daily_oos_monitor(ratio: pd.Series, start: str, end: str | None = None, min_train: int = 20) -> pd.DataFrame:
    """
    Walk-forward 1-step monitoring: fit log(R)~t each day, predict next day, track errors & slope.
    Runs on ratio[start:end].
    """
    r = ratio.dropna().loc[start:]
    if end:
        r = r.loc[:end]
    y = np.log(r)

    t_all = np.arange(len(y), dtype=float)
    t = pd.Series(t_all, index=y.index)

    rows = []
    for i in range(min_train, len(y)):
        train_idx = y.index[:i]
        test_idx = y.index[i]

        y_train = y.loc[train_idx].values
        t_train = t.loc[train_idx].values.reshape(-1, 1)

        X_train = sm.add_constant(t_train, has_constant="add")
        m = sm.OLS(y_train, X_train).fit()

        t_test = np.array([[t.loc[test_idx]]], dtype=float)
        X_test = sm.add_constant(t_test, has_constant="add")

        yhat = float(m.predict(X_test)[0])
        ytrue = float(y.loc[test_idx])

        # RW baseline in log space
        yhat_rw = float(y.loc[y.index[i - 1]])

        rows.append({
            "date": test_idx,
            "ytrue": ytrue,
            "yhat_model": yhat,
            "err_model": ytrue - yhat,
            "yhat_rw": yhat_rw,
            "err_rw": ytrue - yhat_rw,
            "slope_b": float(m.params[1]),
            "slope_t": float(m.tvalues[1]),
            "n_train": int(len(train_idx)),
        })

    df = pd.DataFrame(rows).set_index("date")

    # rolling RMSE
    for w in [10, 20]:
        df[f"rmse_model_{w}"] = df["err_model"].rolling(w).apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)
        df[f"rmse_rw_{w}"] = df["err_rw"].rolling(w).apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)

    df["trend_alive"] = (df["slope_b"] < 0) & (df["slope_t"] < -2)
    df["model_beats_rw_10"] = df["rmse_model_10"] < df["rmse_rw_10"]
    df["model_beats_rw_20"] = df["rmse_model_20"] < df["rmse_rw_20"]

    # healthy: use 20-day when available, else 10-day
    df["healthy"] = np.where(
        df["rmse_model_20"].notna() & df["rmse_rw_20"].notna(),
        df["trend_alive"] & df["model_beats_rw_20"],
        df["trend_alive"] & df["model_beats_rw_10"]
    )

    return df


def days_to_target_quantiles(R0: float, R_star: float, b: float, sigma: float, ps=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    """
    Quantile time-to-target under log-linear trend with normal forecast noise:
      t_p = (ln(R*/R0) - z_p*sigma) / b, b<0
    """
    dy = float(np.log(R_star / R0))
    out = {}
    for p in ps:
        z = float(norm.ppf(p))
        out[p] = float((dy - z * sigma) / b)
    return out


# =========================
# Refactored config + runner
# =========================

@dataclass
class RunConfig:
    # data
    download_start: str = "2000-01-01"
    download_end: str | None = None
    gold_ticker: str = "GC=F"
    silver_ticker: str = "SI=F"

    # analysis slice (what period you want to study)
    analysis_start: str | None = None
    analysis_end: str | None = None

    # piecewise scan range
    piecewise_start: str | None = None
    piecewise_end: str | None = None
    min_seg: int = 30

    # monitoring range
    monitor_start: str | None = None
    monitor_end: str | None = None
    min_train: int = 20

    # OOS checks (fit from oos_start to train_end, test after train_end within analysis slice)
    oos_start: str | None = None
    oos_train_end_1: str | None = None
    oos_train_end_2: str | None = None

    # targets
    target_ratios: tuple[float, ...] = (40.0,)

    # plotting
    do_plots: bool = True
    roll_slope_start: str | None = None
    roll_slope_end: str | None = None
    roll: int = 30


def run_period_study(cfg: RunConfig):
    # ---- download ----
    px = download_close([cfg.gold_ticker, cfg.silver_ticker], start=cfg.download_start, end=cfg.download_end)
    px.columns = [cfg.gold_ticker, cfg.silver_ticker]
    ratio_full = compute_ratio(px, gold_col=cfg.gold_ticker, silver_col=cfg.silver_ticker)

    # ---- slice analysis window ----
    ratio = ratio_full.copy()
    if cfg.analysis_start:
        ratio = ratio.loc[cfg.analysis_start:]
    if cfg.analysis_end:
        ratio = ratio.loc[:cfg.analysis_end]

    print("Ratio sample:", ratio.index.min().date(), "->", ratio.index.max().date())
    print("Ratio last:", float(ratio.iloc[-1]))

    if cfg.do_plots:
        plot_ratio_and_log(ratio, title_prefix="Gold/Silver Ratio (analysis slice)")
        if cfg.roll_slope_start or cfg.roll_slope_end:
            rolling_slope_log_ratio(ratio, start=cfg.roll_slope_start, end=cfg.roll_slope_end, roll=cfg.roll)
        else:
            rolling_slope_log_ratio(ratio, start=ratio.index.min().strftime("%Y-%m-%d"), end=ratio.index.max().strftime("%Y-%m-%d"), roll=cfg.roll)

    # ---- piecewise scan ----
    if cfg.piecewise_start:
        m1, m2, break_date, sse_total, tau = fit_piecewise_linear_scan(
            ratio, start=cfg.piecewise_start, end=cfg.piecewise_end, min_seg=cfg.min_seg
        )

        def seg_summary(m):
            return {
                "params": m.params.tolist(),  # [const, slope]
                "rsq": float(m.rsquared),
                "sse": float(np.sum(m.resid ** 2)),
                "resid_std": float(np.std(m.resid, ddof=1)),
                "n": int(m.nobs),
            }

        print("\n=== Piecewise log(R) fit ===")
        print("piecewise_start:", cfg.piecewise_start, "piecewise_end:", cfg.piecewise_end or ratio.index.max().date())
        print("break_date:", break_date, "tau:", tau, "sse_total:", sse_total)
        print("seg1:", seg_summary(m1))
        print("seg2:", seg_summary(m2))

    # ---- OOS checks inside the analysis slice ----
    oos_runs = []
    if cfg.oos_start and cfg.oos_train_end_1:
        print("\n=== OOS OLS checks ===")
        r1 = oos_ols_logR(ratio, oos_start=cfg.oos_start, train_end=cfg.oos_train_end_1)
        print("OOS1:", r1)
        oos_runs.append(r1)
    if cfg.oos_start and cfg.oos_train_end_2:
        r2 = oos_ols_logR(ratio, oos_start=cfg.oos_start, train_end=cfg.oos_train_end_2)
        print("OOS2:", r2)
        oos_runs.append(r2)

    # ---- monitoring only inside [monitor_start, monitor_end] ----
    monitor = None
    if cfg.monitor_start:
        monitor = daily_oos_monitor(ratio, start=cfg.monitor_start, end=cfg.monitor_end, min_train=cfg.min_train)
        print("\n=== Monitor tail (within monitor period) ===")
        cols = ["slope_b", "slope_t", "rmse_model_20", "rmse_rw_20", "healthy"]
        print(monitor.tail(10)[cols])

    # ---- target ratio ranges (use last point within analysis slice) ----
    R0 = float(ratio.iloc[-1])
    for tgt in cfg.target_ratios:
        print(f"\n=== Target ratio: {tgt} ===")
        if monitor is not None and len(monitor) > 0:
            last = monitor.iloc[-1]
            # prefer rmse_20, else rmse_10
            sigma = float(last["rmse_model_20"]) if pd.notna(last.get("rmse_model_20")) else float(last["rmse_model_10"])
            b = float(last["slope_b"])
            q = days_to_target_quantiles(R0=R0, R_star=tgt, b=b, sigma=sigma)
            print("Live monitor quantiles:", q)
        else:
            print("No monitor computed; skipping live quantiles.")

    # ---- quick monitor plots ----
    if cfg.do_plots and monitor is not None and len(monitor) > 0:
        plt.figure(figsize=(12, 3.5))
        plt.plot(monitor.index, monitor["slope_b"], linewidth=1)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title("Daily OOS fitted slope_b (log-ratio trend) — monitor period")
        plt.ylabel("slope_b")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 2.5))
        plt.plot(monitor.index, monitor["healthy"].astype(int), linewidth=1)
        plt.ylim(-0.1, 1.1)
        plt.title("Model health (1=healthy, 0=not) — monitor period")
        plt.tight_layout()
        plt.show()

    return {"ratio": ratio, "monitor": monitor, "oos_runs": oos_runs}


# =========================
# Example usage (your requested setup)
# =========================

def main():
    cfg = RunConfig(
        analysis_start="2010-12-01",
        analysis_end="2011-06-15",          # <-- set the monitor end date / analysis end date

        piecewise_start="2010-12-01",
        piecewise_end="2011-04-01",
        min_seg=20,

        monitor_start="2011-02-01",
        monitor_end="2011-06-15",
        min_train=20,

        oos_start="2010-12-01",
        oos_train_end_1="2011-02-05",       # <-- customize these two
        oos_train_end_2="2011-04-01",

        target_ratios=(30.0, 40.0),              # <-- customize target ratio(s)
        do_plots=True,
        roll_slope_start="2010-12-01",
        roll_slope_end="2011-06-15",
        roll=30,
    )

    out = run_period_study(cfg)
    return out


if __name__ == "__main__":
    main()
