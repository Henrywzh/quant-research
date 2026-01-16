import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS: int = 252


def bucket_backtest(price_df, signal, H=1, n_buckets=10, entry_mode="next_close"):
    """
    price_df  : MultiIndex ('Close', ticker) as from yfinance
    signal    : DataFrame (dates x tickers) computed at date t
    H         : holding horizon in trading days (e.g. 1=1d, 5=5d)
    n_buckets : number of quantile groups
    entry_mode:
        - "next_close": enter at close[t+1], exit at close[t+H+1]
        - "next_open":  enter at open[t+1],  exit at open[t+H+1]
        - "open_to_close": enter open[t+1], exit close[t+H+1]

    Trading assumption (consistent with factor testing):
      - Form buckets at date t using signal[t]
      - Enter at t+1; exit at t+H+1
      - Rebalance every H days (non-overlapping holds)
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

    # fwd_ret[t]
    ret_fwd = exit_px / entry_px - 1

    dates = close.index
    n_dates = len(dates)

    # Rebalance dates: every H days (non-overlapping positions)
    rebalance_mask = np.zeros(n_dates, dtype=bool)
    rebalance_mask[::H] = True
    rebalance_dates = dates[rebalance_mask]

    # Output containers
    bucket_ret = {f"bucket_{k}": [] for k in range(1, n_buckets + 1)}
    out_dates = []

    # Bucket labels for analysis (NaN for excluded tickers)
    bucket_lbl = pd.DataFrame(index=dates, columns=signal.columns, dtype=float)

    for t in rebalance_dates:
        # If t is too close to the end, ret_fwd[t] may be all NaN
        if t not in signal.index or t not in ret_fwd.index:
            continue

        sig_t = signal.loc[t]

        # -----------------------------------------
        # IMPORTANT: exclude tickers with:
        # - NaN signal
        # - NaN close at t (can't form portfolio)
        # - NaN forward return at t (no realized return for horizon H)
        # -----------------------------------------
        # valid_mask = (~sig_t.isna()) & (~close.loc[t].isna()) & (~ret_fwd.loc[t].isna())
        valid_mask = sig_t.notna() & ret_fwd.loc[t].notna()

        valid_tickers = sig_t.index[valid_mask]

        if len(valid_tickers) == 0:
            continue

        sig_valid = sig_t[valid_tickers]
        # In case of a tie for discrete signal
        eps = pd.Series(np.arange(len(sig_valid), dtype=float), index=sig_valid.index) * 1e-12
        sig_valid = sig_valid + eps

        # Rank: high signal = high rank
        ranks = sig_valid.rank(ascending=False, method="average")

        # Assign buckets (approximately equal count)
        n_valid = len(valid_tickers)
        bucket_size = n_valid / n_buckets

        if n_valid < n_buckets:
            continue

        bucket_ids = ((ranks - 1) / bucket_size).astype(int) + 1
        bucket_ids = bucket_ids.clip(1, n_buckets)

        # Save labels (only at rebalance dates)
        bucket_lbl.loc[t, valid_tickers] = bucket_ids

        # Earn H-day forward return
        r_fwd_t = ret_fwd.loc[t, valid_tickers]

        # Compute bucket-level equal-weight return
        for k in range(1, n_buckets + 1):
            mask_k = (bucket_ids == k)
            if isinstance(mask_k.dtype, pd.BooleanDtype):
                mask_k = mask_k.fillna(False)

            members = bucket_ids.index[mask_k]
            if len(members) == 0:
                bucket_ret[f"bucket_{k}"].append(np.nan)
            else:
                bucket_ret[f"bucket_{k}"].append(r_fwd_t[members].mean())

        out_dates.append(t)

    # Convert bucket returns to DataFrame
    bucket_ret = pd.DataFrame(bucket_ret, index=out_dates)

    # Forward-fill bucket labels for entire holding period (analysis convenience)
    # NOTE: this is NOT used to compute returns here; it's just to know membership between rebalances.
    bucket_lbl = bucket_lbl.ffill()

    return bucket_ret, bucket_lbl, ret_fwd


# =======================================================
# Your metrics (unchanged)
# =======================================================

def icir(ic_series):
    ic_series = ic_series.dropna()
    if len(ic_series) < 2:
        return np.nan
    return ic_series.mean() / ic_series.std(ddof=0)


def annualised_return_geo(r, freq=TRADING_DAYS):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    equity = (1 + r).prod()
    return equity ** (freq / len(r)) - 1


def annualised_vol(r, freq=TRADING_DAYS):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return r.std(ddof=0) * np.sqrt(freq)


def sharpe_ratio(r, freq=TRADING_DAYS, rf=0.0):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    ex = r - rf / freq
    vol = ex.std(ddof=1) * np.sqrt(freq)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (ex.mean() * freq) / vol


def rolling_sharpe(r, window=TRADING_DAYS, freq=TRADING_DAYS, rf=0.0):
    ex = r - rf / freq
    mu = ex.rolling(window).mean() * freq
    vol = ex.rolling(window).std(ddof=0) * np.sqrt(freq)
    return mu / vol


def max_drawdown(r: pd.Series) -> float:
    r = r.fillna(0.0)
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def vol_target_returns(
    ret: pd.Series,
    target_vol: float = 0.10,
    vol_lookback: int = 20,
    ann_factor: int = 252,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    rv = ret.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(ann_factor)
    scale = (target_vol / rv).clip(0.0, max_leverage)
    scale = scale.shift(1).fillna(0.0)  # no lookahead
    ret_vt = scale * ret
    eq_vt = (1 + ret_vt).cumprod()
    return pd.DataFrame({"scale": scale, "ret_vt": ret_vt, "eq_vt": eq_vt})



from typing import Dict, Optional, Tuple, Any

def perf_summary(r: pd.Series, freq: int) -> Dict[str, float]:
    return {
        "ann_return_geo": annualised_return_geo(r, freq=freq),
        "ann_vol": annualised_vol(r, freq=freq),
        "sharpe": sharpe_ratio(r, freq=freq),
        "max_dd": max_drawdown(r),
        "n_obs": int(r.dropna().shape[0]),
    }


# =======================================================
# Tear sheet
# =======================================================

def _compute_ic_spearman(
        signal: pd.DataFrame,
        ret_fwd: pd.DataFrame,
        dates: pd.Index,
        min_assets: int,
) -> pd.Series:
    """Daily (rebalance-date) Spearman IC, cross-sectional."""
    # Align on dates first
    signal = signal.reindex(index=ret_fwd.index)
    dates = pd.Index(dates).intersection(signal.index).intersection(ret_fwd.index)

    ic_vals = []
    for dt in dates:
        x = signal.loc[dt]
        y = ret_fwd.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() < min_assets:
            ic_vals.append(np.nan)
        else:
            ic_vals.append(x[m].corr(y[m], method="spearman"))
    return pd.Series(ic_vals, index=dates, name="IC_spearman")


def _compute_coverage_and_nvalid(
        signal: pd.DataFrame,
        ret_fwd: pd.DataFrame,
        dates: pd.Index,
) -> Tuple[pd.Series, pd.Series]:
    """Coverage (%) and valid count on rebalance dates."""
    dates = pd.Index(dates).intersection(signal.index).intersection(ret_fwd.index)
    universe_size = signal.shape[1]

    cov = []
    nvalid = []
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
    """
    Turnover proxy for a given bucket: 1 - overlap_ratio of bucket membership.
    Computed on rebalance dates using labels at those dates.
    """
    dts = pd.Index(rebalance_dates).intersection(bucket_lbl.index)
    prev_members = None
    vals = []

    for dt in dts:
        lbl = bucket_lbl.loc[dt]
        members = set(lbl.index[lbl == bucket_k])
        if prev_members is None:
            vals.append(np.nan)
        else:
            if len(members) == 0:
                vals.append(np.nan)
            else:
                overlap = len(members.intersection(prev_members))
                vals.append(1.0 - overlap / len(members))
        prev_members = members

    return pd.Series(vals, index=dts, name=f"turnover_bucket_{bucket_k}")


def _equity_curve_from_returns(r: pd.Series) -> pd.Series:
    """Equity curve on the same index as r (no resampling)."""
    r = r.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    return (1.0 + r).cumprod()


def _drawdown_from_equity(eq: pd.Series) -> pd.Series:
    """Drawdown series from equity curve."""
    if eq is None or len(eq) == 0:
        return pd.Series(dtype=float)
    peak = eq.cummax()
    return eq / peak - 1.0


def _yearly_returns_from_returns(r: pd.Series) -> pd.Series:
    """
    Calendar-year compounded returns from periodic returns r.
    Works for H-day returns too (compounds all observations within each year).
    """
    r = r.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    # group by calendar year and compound
    return (1.0 + r).groupby(r.index.year).prod() - 1.0


def _compute_benchmark_ret_fwd(
        bench_price: pd.DataFrame | pd.Series,
        H: int,
        entry_mode: str,
) -> pd.Series:
    """
    Return series aligned to formation date t:
      ret_fwd[t] = Px[t+H+1] / Px[t+1] - 1
    bench_price can be:
      - Series of Close (or Open) already chosen, OR
      - DataFrame with columns including 'Close' and/or 'Open'
    """
    if isinstance(bench_price, pd.Series):
        px = bench_price.sort_index()
    else:
        bench_price = bench_price.sort_index()
        if entry_mode in {"next_open"}:
            if "Open" not in bench_price.columns:
                raise ValueError("benchmark DataFrame must contain 'Open' for entry_mode='next_open'")
            px = bench_price["Open"]
        else:
            if "Close" not in bench_price.columns:
                raise ValueError(
                    "benchmark DataFrame must contain 'Close' for entry_mode='next_close' or 'open_to_close'")
            px = bench_price["Close"]

    # open_to_close benchmark: enter open[t+1], exit close[t+H+1]
    if entry_mode == "open_to_close":
        if isinstance(bench_price, pd.Series):
            raise ValueError("For entry_mode='open_to_close', benchmark must be a DataFrame with Open/Close.")
        entry_px = bench_price["Open"].shift(-1)
        exit_px = bench_price["Close"].shift(-(H + 1))
        return (exit_px / entry_px - 1).rename("benchmark_ret_fwd")

    # next_close / next_open
    entry_px = px.shift(-1)
    exit_px = px.shift(-(H + 1))
    return (exit_px / entry_px - 1).rename("benchmark_ret_fwd")


def make_tearsheet(
        price_df: pd.DataFrame,
        signal: pd.DataFrame,
        H: int = 5,
        n_buckets: int = 20,
        entry_mode: str = "next_close",
        min_assets_ic: int = 50,
        plot: bool = True,
        rolling_window_obs: Optional[int] = None,
        benchmark_price: Optional[pd.DataFrame | pd.Series] = None,
        benchmark_name: str = "Benchmark",
) -> Dict[str, Any]:
    """
    Generate a standardized tear sheet for a signal under your bucket backtest framework.

    Parameters
    ----------
    price_df : DataFrame
        yfinance multiindex columns, must contain 'Close' and (if next_open/open_to_close) 'Open'
    signal : DataFrame
        dates x tickers, computed at close[t]
    H : int
        holding horizon (in trading days after entry), and rebalance step
    n_buckets : int
        number of buckets
    entry_mode : str
        'next_close' | 'next_open' | 'open_to_close'
    min_assets_ic : int
        minimum valid assets required to compute IC on a rebalance date
    plot : bool
        whether to display plots
    rolling_window_obs : int | None
        rolling window in number of rebalance observations; if None, defaults to ~1 year = 252/H

    Returns
    -------
    report : dict
        Contains bucket_ret, ic series, summaries, coverage, turnover proxies, and summary tables.
    """
    if rolling_window_obs is None:
        rolling_window_obs = max(10, int(round(TRADING_DAYS / H)))

    # Run backtest (uses strict NaN logic, next-day entry alignment)
    bucket_ret, bucket_lbl, ret_fwd = bucket_backtest(
        price_df=price_df,
        signal=signal,
        H=H,
        n_buckets=n_buckets,
        entry_mode=entry_mode,
    )

    rebalance_dates = bucket_ret.index
    freq = TRADING_DAYS / H  # annualisation frequency for H-day non-overlapping returns

    # Coverage & valid count
    signal_aligned = signal.reindex(index=ret_fwd.index)
    coverage, n_valid = _compute_coverage_and_nvalid(signal_aligned, ret_fwd, rebalance_dates)

    # IC series + stats
    ic = _compute_ic_spearman(signal_aligned, ret_fwd, rebalance_dates, min_assets=min_assets_ic)
    ic_stats = {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)),
        "icir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) and np.isfinite(ic.std(ddof=1)) else np.nan,
        "hit_rate": float((ic > 0).mean()),
        "n_obs": int(ic.dropna().shape[0]),
    }

    ic_roll = ic.rolling(rolling_window_obs).mean()

    # Bucket summary table
    bucket_summary_rows = []
    for col in bucket_ret.columns:
        r = bucket_ret[col].dropna()
        mu = float(r.mean()) if len(r) else np.nan
        sd = float(r.std(ddof=1)) if len(r) >= 2 else np.nan
        tstat = (mu / (sd / np.sqrt(len(r)))) if (len(r) >= 2 and sd > 0) else np.nan
        ps = perf_summary(bucket_ret[col], freq=freq)
        bucket_summary_rows.append({
            "bucket": col,
            "mean_ret_per_period": mu,
            "std_ret_per_period": sd,
            "t_stat_mean": tstat,
            **ps,
        })
    bucket_summary = pd.DataFrame(bucket_summary_rows).set_index("bucket")

    # Monotonicity diagnostic: correlation between bucket number and mean return
    # (Simple, fast sanity check)
    mean_by_bucket = bucket_summary["mean_ret_per_period"].copy()
    tmp = mean_by_bucket.copy()
    tmp.index = tmp.index.str.replace("bucket_", "", regex=False).astype(int)
    tmp = tmp.sort_index()  # 1..n_buckets
    monotonic_spearman = float(
        pd.Series(tmp.index, index=tmp.index).corr(tmp, method="spearman")
    )

    # Turnover proxies for bottom and top buckets
    turnover_bottom = _turnover_proxy_from_labels(bucket_lbl, rebalance_dates, bucket_k=1)
    turnover_top = _turnover_proxy_from_labels(bucket_lbl, rebalance_dates, bucket_k=n_buckets)

    bench_ret = None
    bench_eq = None
    bench_dd = None
    bench_yearly = None

    if benchmark_price is not None:
        bench_ret_full = _compute_benchmark_ret_fwd(benchmark_price, H=H, entry_mode=entry_mode)

        # Align to rebalance dates (because bucket_ret is on rebalance dates)
        bench_ret = bench_ret_full.reindex(rebalance_dates)

        # Equity / DD / yearly on the same rebalance grid
        bench_eq = _equity_curve_from_returns(bench_ret)
        bench_dd = _drawdown_from_equity(bench_eq)
        bench_yearly = _yearly_returns_from_returns(bench_ret).rename(benchmark_name)

    # Optional plots
    if plot:
        # # 1) Coverage
        # plt.figure()
        # coverage.plot()
        # plt.title(f"Coverage on Rebalance Dates (H={H}, entry={entry_mode})")
        # plt.ylabel("Coverage (valid fraction)")
        # plt.xlabel("Date")
        # plt.show()

        # 2) Cumulative IC + rolling IC
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

        # 3) Bucket mean returns bar chart
        plt.figure(figsize=(10, 6))
        mean_by_bucket.sort_index(key=lambda idx: idx.str.replace("bucket_", "", regex=False).astype(int)).plot(
            kind="bar")
        plt.title("Mean Return per Bucket (per holding period)")
        plt.ylabel("Mean H-day return")
        plt.xlabel("Bucket")
        plt.tight_layout()
        plt.show()

        # 4) Bucket cumulative curves (all buckets; if too busy, you can slice)
        plt.figure()
        # Build equity curves from bucket returns
        eq = (1.0 + bucket_ret.fillna(0.0)).cumprod()
        eq.plot(legend=True)
        plt.legend()
        plt.title("Bucket Cumulative Curves")
        plt.ylabel("Cumulative growth")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

        # # 5) Turnover proxy (top/bottom)
        # plt.figure()
        # pd.DataFrame({"turnover_bottom": turnover_bottom, "turnover_top": turnover_top}).plot()
        # plt.title("Turnover Proxy (Bucket Membership Change)")
        # plt.ylabel("1 - overlap ratio")
        # plt.xlabel("Date")
        # plt.tight_layout()
        # plt.show()

        # 6) Drawdown
        # Select series to plot drawdown/yearly returns for
        top_col = "bucket_1"
        bot_col = f"bucket_{n_buckets}"

        plt.figure()

        # top bucket
        eq_top = _equity_curve_from_returns(bucket_ret[top_col])
        dd_top = _drawdown_from_equity(eq_top)
        if len(dd_top):
            dd_top.plot(label=f"{top_col} (max_dd={dd_top.min():.2%})")

        # bottom bucket
        eq_bot = _equity_curve_from_returns(bucket_ret[bot_col])
        dd_bot = _drawdown_from_equity(eq_bot)
        if len(dd_bot):
            dd_bot.plot(label=f"{bot_col} (max_dd={dd_bot.min():.2%})")

        # benchmark
        if bench_dd is not None and len(bench_dd):
            bench_dd.plot(label=f"{benchmark_name} (max_dd={bench_dd.min():.2%})")

        plt.title("Drawdown Curves (Top / Bottom / Benchmark)")
        plt.ylabel("Drawdown")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 7) Yearly Returns
        yr_top = _yearly_returns_from_returns(bucket_ret[top_col]).rename(top_col)
        yr_bot = _yearly_returns_from_returns(bucket_ret[bot_col]).rename(bot_col)

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

    report = {
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
    return report