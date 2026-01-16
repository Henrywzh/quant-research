from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qresearch.backtest.metrics import (
    TRADING_DAYS,
    perf_summary,
    equity_curve,
    drawdown_series_from_equity,
    yearly_returns,
)

EntryMode = Literal["next_close", "next_open", "open_to_close"]


def bucket_backtest(
    price_df: pd.DataFrame,
    signal: pd.DataFrame,
    H: int = 1,
    n_buckets: int = 10,
    entry_mode: EntryMode = "next_close",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    price_df : yfinance-style MultiIndex columns with ['Open','Close',...] on level 0
    signal   : DataFrame (dates x tickers), computed at close[t]
    H        : holding horizon (trading days)
    Rebalance: every H days, non-overlapping

    Returns
    -------
    bucket_ret : DataFrame indexed by rebalance dates
    bucket_lbl : DataFrame indexed by ALL dates (ffill for convenience)
    ret_fwd    : DataFrame indexed by ALL dates, formation-aligned forward return
    """
    close = price_df["Close"].sort_index()

    if entry_mode == "next_close":
        entry_px = close.shift(-1)
        exit_px = close.shift(-(H + 1))
    elif entry_mode == "next_open":
        open_ = price_df["Open"].sort_index()
        entry_px = open_.shift(-1)
        exit_px = open_.shift(-(H + 1))
    elif entry_mode == "open_to_close":
        open_ = price_df["Open"].sort_index()
        entry_px = open_.shift(-1)
        exit_px = close.shift(-(H + 1))
    else:
        raise ValueError("entry_mode must be one of: next_close, next_open, open_to_close")

    # formation-aligned forward return: ret_fwd[t] = exit(t+H+1) / entry(t+1) - 1
    ret_fwd = exit_px / entry_px - 1

    dates = close.index
    n_dates = len(dates)

    # Rebalance dates: every H days (non-overlapping positions)
    rebalance_mask = np.zeros(n_dates, dtype=bool)
    rebalance_mask[::H] = True
    rebalance_dates = dates[rebalance_mask]

    bucket_ret = {f"bucket_{k}": [] for k in range(1, n_buckets + 1)}
    out_dates = []

    bucket_lbl = pd.DataFrame(index=dates, columns=signal.columns, dtype=float)

    for t in rebalance_dates:
        if t not in signal.index or t not in ret_fwd.index:
            continue

        sig_t = signal.loc[t]

        # valid: have a signal AND have a realized forward return
        valid_mask = sig_t.notna() & ret_fwd.loc[t].notna()
        valid_tickers = sig_t.index[valid_mask]

        if len(valid_tickers) == 0 or len(valid_tickers) < n_buckets:
            continue

        sig_valid = sig_t[valid_tickers]

        # tie-breaker for discrete signals
        eps = pd.Series(np.arange(len(sig_valid), dtype=float), index=sig_valid.index) * 1e-12
        sig_valid = sig_valid + eps

        # high signal => rank 1
        ranks = sig_valid.rank(ascending=False, method="average")

        n_valid = len(valid_tickers)
        bucket_size = n_valid / n_buckets

        bucket_ids = ((ranks - 1) / bucket_size).astype(int) + 1
        bucket_ids = bucket_ids.clip(1, n_buckets)

        bucket_lbl.loc[t, valid_tickers] = bucket_ids

        r_fwd_t = ret_fwd.loc[t, valid_tickers]

        for k in range(1, n_buckets + 1):
            members = bucket_ids.index[bucket_ids == k]
            bucket_ret[f"bucket_{k}"].append(r_fwd_t[members].mean() if len(members) else np.nan)

        out_dates.append(t)

    bucket_ret = pd.DataFrame(bucket_ret, index=out_dates)
    bucket_lbl = bucket_lbl.ffill()

    return bucket_ret, bucket_lbl, ret_fwd


def _compute_ic_spearman(
    signal: pd.DataFrame,
    ret_fwd: pd.DataFrame,
    dates: pd.Index,
    min_assets: int,
) -> pd.Series:
    signal = signal.reindex(index=ret_fwd.index)
    dates = pd.Index(dates).intersection(signal.index).intersection(ret_fwd.index)

    ic_vals = []
    for dt in dates:
        x = signal.loc[dt]
        y = ret_fwd.loc[dt]
        m = x.notna() & y.notna()
        ic_vals.append(x[m].corr(y[m], method="spearman") if m.sum() >= min_assets else np.nan)

    return pd.Series(ic_vals, index=dates, name="IC_spearman")


def _compute_coverage_and_nvalid(
    signal: pd.DataFrame,
    ret_fwd: pd.DataFrame,
    dates: pd.Index,
) -> Tuple[pd.Series, pd.Series]:
    dates = pd.Index(dates).intersection(signal.index).intersection(ret_fwd.index)
    universe_size = signal.shape[1]

    cov, nvalid = [], []
    for dt in dates:
        m = signal.loc[dt].notna() & ret_fwd.loc[dt].notna()
        n = int(m.sum())
        nvalid.append(n)
        cov.append(n / universe_size if universe_size > 0 else np.nan)

    return (
        pd.Series(cov, index=dates, name="coverage"),
        pd.Series(nvalid, index=dates, name="n_valid"),
    )


def _turnover_proxy_from_labels(
    bucket_lbl: pd.DataFrame,
    rebalance_dates: pd.Index,
    bucket_k: int,
) -> pd.Series:
    dts = pd.Index(rebalance_dates).intersection(bucket_lbl.index)

    prev_members = None
    vals = []
    for dt in dts:
        lbl = bucket_lbl.loc[dt]
        members = set(lbl.index[lbl == bucket_k])

        if prev_members is None or len(members) == 0:
            vals.append(np.nan)
        else:
            overlap = len(members.intersection(prev_members))
            vals.append(1.0 - overlap / len(members))

        prev_members = members

    return pd.Series(vals, index=dts, name=f"turnover_bucket_{bucket_k}")


def _compute_benchmark_ret_fwd(
    bench_price: pd.DataFrame | pd.Series,
    H: int,
    entry_mode: EntryMode,
) -> pd.Series:
    if isinstance(bench_price, pd.Series):
        px = bench_price.sort_index()
        bench_df = None
    else:
        bench_df = bench_price.sort_index()
        if entry_mode == "next_open":
            px = bench_df["Open"]
        else:
            px = bench_df["Close"]

    if entry_mode == "open_to_close":
        if bench_df is None:
            raise ValueError("For entry_mode='open_to_close', benchmark must be a DataFrame with Open/Close.")
        entry_px = bench_df["Open"].shift(-1)
        exit_px = bench_df["Close"].shift(-(H + 1))
        return (exit_px / entry_px - 1).rename("benchmark_ret_fwd")

    entry_px = px.shift(-1)
    exit_px = px.shift(-(H + 1))
    return (exit_px / entry_px - 1).rename("benchmark_ret_fwd")


def make_tearsheet(
    price_df: pd.DataFrame,
    signal: pd.DataFrame,
    H: int = 5,
    n_buckets: int = 20,
    entry_mode: EntryMode = "next_close",
    min_assets_ic: int = 50,
    plot: bool = True,
    rolling_window_obs: Optional[int] = None,
    benchmark_price: Optional[pd.DataFrame | pd.Series] = None,
    benchmark_name: str = "Benchmark",
) -> Dict[str, Any]:
    if rolling_window_obs is None:
        rolling_window_obs = max(10, int(round(TRADING_DAYS / H)))

    bucket_ret, bucket_lbl, ret_fwd = bucket_backtest(
        price_df=price_df,
        signal=signal,
        H=H,
        n_buckets=n_buckets,
        entry_mode=entry_mode,
    )

    rebalance_dates = bucket_ret.index
    freq = TRADING_DAYS / H

    signal_aligned = signal.reindex(index=ret_fwd.index)
    coverage, n_valid = _compute_coverage_and_nvalid(signal_aligned, ret_fwd, rebalance_dates)

    ic = _compute_ic_spearman(signal_aligned, ret_fwd, rebalance_dates, min_assets=min_assets_ic)
    ic_roll = ic.rolling(rolling_window_obs).mean()

    ic_std = ic.std(ddof=0)
    ic_stats = {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic_std),
        "icir": float(ic.mean() / ic_std) if ic_std and np.isfinite(ic_std) else np.nan,
        "hit_rate": float((ic > 0).mean()),
        "n_obs": int(ic.dropna().shape[0]),
    }

    bucket_summary_rows = []
    for col in bucket_ret.columns:
        r = bucket_ret[col].dropna()
        mu = float(r.mean()) if len(r) else np.nan
        sd = float(r.std(ddof=0)) if len(r) >= 2 else np.nan
        tstat = (mu / (sd / np.sqrt(len(r)))) if (len(r) >= 2 and sd > 0) else np.nan
        ps = perf_summary(bucket_ret[col], freq=int(freq))
        bucket_summary_rows.append({
            "bucket": col,
            "mean_ret_per_period": mu,
            "std_ret_per_period": sd,
            "t_stat_mean": tstat,
            **ps,
        })
    bucket_summary = pd.DataFrame(bucket_summary_rows).set_index("bucket")

    mean_by_bucket = bucket_summary["mean_ret_per_period"].copy()
    tmp = mean_by_bucket.copy()
    tmp.index = tmp.index.str.replace("bucket_", "", regex=False).astype(int)
    tmp = tmp.sort_index()
    monotonic_spearman = float(pd.Series(tmp.index, index=tmp.index).corr(tmp, method="spearman"))

    turnover_bottom = _turnover_proxy_from_labels(bucket_lbl, rebalance_dates, bucket_k=1)
    turnover_top = _turnover_proxy_from_labels(bucket_lbl, rebalance_dates, bucket_k=n_buckets)

    bench_ret = bench_eq = bench_dd = bench_yearly = None
    if benchmark_price is not None:
        bench_ret_full = _compute_benchmark_ret_fwd(benchmark_price, H=H, entry_mode=entry_mode)
        bench_ret = bench_ret_full.reindex(rebalance_dates)
        bench_eq = equity_curve(bench_ret)
        bench_dd = drawdown_series_from_equity(bench_eq)
        bench_yearly = yearly_returns(bench_ret).rename(benchmark_name)

    if plot:
        plt.figure()
        ic.cumsum().plot()
        plt.title("Cumulative IC (Spearman)")
        plt.ylabel("Cumulative IC")
        plt.xlabel("Date")
        plt.show()

        plt.figure()
        ic_roll.plot()
        plt.title(f"Rolling Mean IC (window={rolling_window_obs} obs)")
        plt.ylabel("Rolling IC mean")
        plt.xlabel("Date")
        plt.show()

        plt.figure(figsize=(10, 6))
        mean_by_bucket.sort_index(
            key=lambda idx: idx.str.replace("bucket_", "", regex=False).astype(int)
        ).plot(kind="bar")
        plt.title("Mean Return per Bucket (per holding period)")
        plt.ylabel("Mean H-day return")
        plt.xlabel("Bucket")
        plt.tight_layout()
        plt.show()

        plt.figure()
        eq_all = (1.0 + bucket_ret.fillna(0.0)).cumprod()
        eq_all.plot(legend=True)
        plt.title("Bucket Cumulative Curves")
        plt.ylabel("Cumulative growth")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

        # Drawdown plot (top/bottom/benchmark)
        top_col = "bucket_1"
        bot_col = f"bucket_{n_buckets}"

        plt.figure()
        dd_top = drawdown_series_from_equity(equity_curve(bucket_ret[top_col]))
        dd_bot = drawdown_series_from_equity(equity_curve(bucket_ret[bot_col]))

        if len(dd_top): dd_top.plot(label=f"{top_col} (max_dd={dd_top.min():.2%})")
        if len(dd_bot): dd_bot.plot(label=f"{bot_col} (max_dd={dd_bot.min():.2%})")
        if bench_dd is not None and len(bench_dd):
            bench_dd.plot(label=f"{benchmark_name} (max_dd={bench_dd.min():.2%})")

        plt.title("Drawdown Curves (Top / Bottom / Benchmark)")
        plt.ylabel("Drawdown")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Yearly returns bar
        yr_top = yearly_returns(bucket_ret[top_col]).rename(top_col)
        yr_bot = yearly_returns(bucket_ret[bot_col]).rename(bot_col)
        parts = [yr_top, yr_bot]
        if bench_yearly is not None and len(bench_yearly):
            parts.append(bench_yearly)
        yr_tbl = pd.concat(parts, axis=1).sort_index()
        if len(yr_tbl):
            plt.figure(figsize=(10, 5))
            yr_tbl.plot(kind="bar")
            plt.title("Calendar-Year Returns (Top / Bottom / Benchmark)")
            plt.ylabel("Return")
            plt.xlabel("Year")
            plt.tight_layout()
            plt.show()

    return {
        "meta": {
            "H": H,
            "n_buckets": n_buckets,
            "entry_mode": entry_mode,
            "freq": freq,
            "rolling_window_obs": rolling_window_obs,
            "universe_size": int(signal.shape[1]),
            "n_rebalance_obs": int(len(rebalance_dates)),
        },
        "bucket_ret": bucket_ret,
        "benchmark_ret": bench_ret,
        "bucket_labels": bucket_lbl,
        "ret_fwd": ret_fwd,
        "coverage": coverage,
        "n_valid": n_valid,
        "ic": ic,
        "ic_stats": ic_stats,
        "bucket_summary": bucket_summary,
        "monotonic_spearman_bucket_vs_mean": monotonic_spearman,
        "turnover_bottom": turnover_bottom,
        "turnover_top": turnover_top,
    }
