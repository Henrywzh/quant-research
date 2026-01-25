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
from qresearch.backtest.visualise import visualize_factor_tearsheet, FactorVizConfig
from qresearch.data.types import MarketData


EntryMode = Literal["next_close", "next_open", "open_to_close"]


def bucket_backtest(
    md: MarketData,
    signal: pd.DataFrame,
    H: int = 1,
    n_buckets: int = 10,
    entry_mode: EntryMode = "next_close",
    universe_eligible: Optional[pd.DataFrame] = None,
    *,
    ffill_labels: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cross-sectional bucket backtest (non-overlapping holding periods).

    Contract / conventions (non-negotiable):
    - signal is computed at close[t]
    - positions are formed on date t using signal[t] and eligibility[t]
    - entry occurs on t+1 (depending on entry_mode)
    - exit occurs on t+H+1
    - forward return used for evaluation is:
        ret_fwd[t] = exit_px(t+H+1) / entry_px(t+1) - 1

    Bucket convention (IMPORTANT):
    - bucket_1 = WORST  (lowest signal)
    - bucket_n = BEST   (highest signal)

    Universe eligibility (optional):
    - universe_eligible is a boolean DataFrame (date x ticker)
    - universe_eligible[t, i] == True means ticker i is allowed to be ranked / traded at formation date t
    - eligibility is applied BEFORE ranking and bucketing (formation-date gate)

    Non-overlapping rebalances:
    - Rebalance dates are every H trading days on the close index grid:
        rebalance_dates = dates[::H]
      This means each rebalance forms a portfolio held for H days, then liquidated, then reformed.

    Outputs:
    - bucket_ret: DataFrame indexed by rebalance dates with columns bucket_1..bucket_n,
                 values are mean forward returns (H-day, non-overlapping) of each bucket at that rebalance.
    - bucket_lbl: DataFrame indexed by all dates and columns tickers,
                  values are bucket id on rebalance dates (NaN elsewhere unless ffill_labels=True).
    - ret_fwd: DataFrame (date x ticker) of forward returns consistent with entry/exit convention above.

    Notes:
    - Uses a small epsilon tie-breaker to stabilize bucketing for discrete or tied signals.
    - Uses pd.qcut on ranks to produce near-equal-count buckets; falls back to arithmetic binning if qcut fails.
    - If ffill_labels=True, bucket_lbl is forward-filled for up to H-1 days to represent "held membership".
      Keep this OFF unless you explicitly need held-period labels; it can be misinterpreted as formation labels.
    """

    close = md.close.sort_index()
    signal = signal.reindex(index=close.index, columns=close.columns)

    # forward returns
    if entry_mode == "next_close":
        entry_px = close.shift(-1)
        exit_px = close.shift(-(H + 1))
    elif entry_mode == "next_open":
        if md.open is None:
            raise ValueError("md.open is required for entry_mode='next_open'")
        open_ = md.open.sort_index()
        entry_px = open_.shift(-1)
        exit_px = open_.shift(-(H + 1))
    elif entry_mode == "open_to_close":
        if md.open is None:
            raise ValueError("md.open is required for entry_mode='open_to_close'")
        open_ = md.open.sort_index()
        entry_px = open_.shift(-1)
        exit_px = close.shift(-(H + 1))
    else:
        raise ValueError("entry_mode must be one of: next_close, next_open, open_to_close")

    ret_fwd = exit_px / entry_px - 1

    dates = close.index
    rebalance_dates = dates[::H]  # non-overlapping grid

    bucket_ret = {f"bucket_{k}": [] for k in range(1, n_buckets + 1)}
    out_dates = []
    bucket_lbl = pd.DataFrame(index=dates, columns=close.columns, dtype=float)

    if universe_eligible is not None:
        universe_eligible = (
            universe_eligible.reindex(index=dates, columns=close.columns)
            .fillna(False)
            .astype(bool)
        )

    for t in rebalance_dates:
        sig_t = signal.loc[t]
        ret_t = ret_fwd.loc[t]

        valid_mask = sig_t.notna() & ret_t.notna()
        if universe_eligible is not None:
            valid_mask &= universe_eligible.loc[t].fillna(False).astype(bool)

        if valid_mask.sum() < n_buckets:
            continue

        sig_valid = sig_t[valid_mask].astype(float)

        # tie-breaker for discrete signals
        eps = pd.Series(np.arange(len(sig_valid), dtype=float), index=sig_valid.index) * 1e-12
        sig_valid = sig_valid + eps

        # stable equal-count buckets (1..n, where 1=worst)
        ranks = sig_valid.rank(ascending=True, method="first")
        try:
            bucket_ids = pd.qcut(ranks, q=n_buckets, labels=range(1, n_buckets + 1))
            bucket_ids = bucket_ids.astype(int)
        except ValueError:
            # fallback to your original sizing if qcut fails
            n_valid = len(sig_valid)
            bucket_size = n_valid / n_buckets
            bucket_ids = ((ranks - 1) / bucket_size).astype(int) + 1
            bucket_ids = bucket_ids.clip(1, n_buckets).astype(int)

        bucket_lbl.loc[t, bucket_ids.index] = bucket_ids

        r_fwd_t = ret_t.loc[bucket_ids.index]
        for k in range(1, n_buckets + 1):
            members = bucket_ids.index[bucket_ids == k]
            bucket_ret[f"bucket_{k}"].append(r_fwd_t[members].mean() if len(members) else np.nan)

        out_dates.append(t)

    bucket_ret = pd.DataFrame(bucket_ret, index=out_dates)

    if ffill_labels:
        bucket_lbl = bucket_lbl.ffill(limit=H-1)

    return bucket_ret, bucket_lbl, ret_fwd


def make_tearsheet(
    md: MarketData,
    signal: pd.DataFrame,
    H: int = 5,
    n_buckets: int = 20,
    entry_mode: EntryMode = "next_close",
    min_assets_ic: int = 50,
    plot: bool = True,
    rolling_window_obs: Optional[int] = None,
    benchmark_price: Optional[pd.DataFrame | pd.Series] = None,
    benchmark_name: str = "Benchmark",
    universe_eligible: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Standard tear sheet for your bucket backtest framework.

    Bucket convention (IMPORTANT):
    ------------------------------
    bucket_1 = WORST
    bucket_n = BEST

    So:
    - "best bucket" = bucket_{n_buckets}
    - "worst bucket" = bucket_1
    - "LS" (factor) = best - worst
    """
    visualise_cfg =  FactorVizConfig(
        benchmark_name=benchmark_name,
        best_col=f'bucket_{n_buckets}',
        worst_col=f'bucket_1'
    )

    if rolling_window_obs is None:
        # number of rebalance observations ~ 1 year
        rolling_window_obs = max(10, int(round(TRADING_DAYS / H)))

    # --- run backtest ---
    bucket_ret, bucket_lbl, ret_fwd = bucket_backtest(
        md=md,
        signal=signal,  # date x ticker, computed at close[t]
        H=H,
        n_buckets=n_buckets,
        entry_mode=entry_mode,
        universe_eligible=universe_eligible,
    )

    rebalance_dates = bucket_ret.index

    # Annualisation factor for H-day non-overlapping returns.
    # Keep float on purpose (avoids accidental rounding).
    freq_per_year = TRADING_DAYS / H

    # --- diagnostics: coverage & IC ---
    signal_aligned = signal.reindex(index=ret_fwd.index)
    coverage, n_valid = _compute_coverage_and_nvalid(signal_aligned, ret_fwd, rebalance_dates)

    ic = _compute_ic_spearman(signal_aligned, ret_fwd, rebalance_dates, min_assets=min_assets_ic)
    ic_roll = ic.rolling(rolling_window_obs).mean()

    # Use the same ddof convention as your metrics module (ddof=0) unless you explicitly want ddof=1.
    ic_std = ic.std(ddof=0)
    ic_stats = {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic_std),
        "icir": float(ic.mean() / ic_std) if ic_std and np.isfinite(ic_std) else np.nan,
        "hit_rate": float((ic > 0).mean()),
        "n_obs": int(ic.dropna().shape[0]),
    }

    # --- bucket summary table ---
    bucket_summary_rows = []
    for col in bucket_ret.columns:
        r = bucket_ret[col].dropna()
        mu = float(r.mean()) if len(r) else np.nan
        sd = float(r.std(ddof=0)) if len(r) >= 2 else np.nan
        tstat = (mu / (sd / np.sqrt(len(r)))) if (len(r) >= 2 and sd > 0) else np.nan

        # perf_summary expects "freq" ~ number of periods per year
        ps = perf_summary(bucket_ret[col], freq=freq_per_year)
        bucket_summary_rows.append({
            "bucket": col,
            "mean_ret_per_period": mu,
            "std_ret_per_period": sd,
            "t_stat_mean": tstat,
            **ps,
        })
    bucket_summary = pd.DataFrame(bucket_summary_rows).set_index("bucket")

    # --- monotonicity sanity check ---
    # Correlation between bucket number (1..n) and mean return.
    # Since bucket_1 is worst, a "good" signal tends to have positive monotonicity.
    mean_by_bucket = bucket_summary["mean_ret_per_period"].copy()
    tmp = mean_by_bucket.copy()
    tmp.index = tmp.index.str.replace("bucket_", "", regex=False).astype(int)
    tmp = tmp.sort_index()  # 1..n_buckets
    monotonic_spearman = float(pd.Series(tmp.index, index=tmp.index).corr(tmp, method="spearman"))

    # --- bucket identities under YOUR convention ---
    WORST_BUCKET = 1
    BEST_BUCKET = n_buckets
    worst_col = f"bucket_{WORST_BUCKET}"
    best_col = f"bucket_{BEST_BUCKET}"

    # Turnover proxies for best/worst buckets (naming matches semantics)
    turnover_worst = _turnover_proxy_from_labels(bucket_lbl, rebalance_dates, bucket_k=WORST_BUCKET)
    turnover_best = _turnover_proxy_from_labels(bucket_lbl, rebalance_dates, bucket_k=BEST_BUCKET)

    # --- long-short (best - worst): crucial for factor research ---
    ls_ret = (bucket_ret[best_col] - bucket_ret[worst_col]).rename("LS_best_minus_worst")
    ls_eq = equity_curve(ls_ret)
    ls_dd = drawdown_series_from_equity(ls_eq)
    ls_yearly = yearly_returns(ls_ret).rename(ls_ret.name)
    ls_perf = perf_summary(ls_ret, freq=freq_per_year)

    # --- benchmark aligned to rebalance grid (optional) ---
    bench_ret = bench_eq = bench_dd = bench_yearly = bench_perf = None
    if benchmark_price is not None:
        bench_ret_full = _compute_benchmark_ret_fwd(benchmark_price, H=H, entry_mode=entry_mode)
        bench_ret = bench_ret_full.reindex(rebalance_dates).rename(benchmark_name)

        bench_eq = equity_curve(bench_ret)
        bench_dd = drawdown_series_from_equity(bench_eq)
        bench_yearly = yearly_returns(bench_ret).rename(benchmark_name)
        bench_perf = perf_summary(bench_ret, freq=freq_per_year)

    # --- plotting ---
    if plot:
        visualize_factor_tearsheet(
            ic=ic,
            ic_roll=ic_roll,
            bucket_ret=bucket_ret,
            mean_by_bucket=mean_by_bucket,
            ls_eq=ls_eq,
            ls_dd=ls_dd,
            ls_yearly=ls_yearly,
            bench_eq=bench_eq if benchmark_price is not None else None,
            bench_dd=bench_dd,
            bench_yearly=bench_yearly,
            cfg=visualise_cfg,
        )

    # --- report dict ---
    return {
        "meta": {
            "H": H,
            "n_buckets": n_buckets,
            "entry_mode": entry_mode,
            "freq_per_year": float(freq_per_year),
            "rolling_window_obs": int(rolling_window_obs),
            "universe_size": int(signal.shape[1]),
            "n_rebalance_obs": int(len(rebalance_dates)),
            "bucket_convention": "bucket_1 = worst, bucket_n = best",
        },
        "bucket_ret": bucket_ret,
        "bucket_labels": bucket_lbl,
        "ret_fwd": ret_fwd,
        "coverage": coverage,
        "n_valid": n_valid,
        "ic": ic,
        "ic_stats": ic_stats,
        "ic_roll": ic_roll,
        "bucket_summary": bucket_summary,
        "monotonic_spearman_bucket_vs_mean": monotonic_spearman,

        # Semantically correct naming under your convention
        "best_bucket": best_col,
        "worst_bucket": worst_col,
        "turnover_best": turnover_best,
        "turnover_worst": turnover_worst,

        # Long-short factor view (best - worst)
        "ls_ret": ls_ret,
        "ls_perf": ls_perf,
        "ls_eq": ls_eq,
        "ls_dd": ls_dd,
        "ls_yearly": ls_yearly,

        # Benchmark (optional)
        "benchmark_ret": bench_ret,
        "benchmark_perf": bench_perf,
        "benchmark_eq": bench_eq,
        "benchmark_dd": bench_dd,
        "benchmark_yearly": bench_yearly,
    }


def _compute_ic_spearman(
    signal: pd.DataFrame,
    ret_fwd: pd.DataFrame,
    dates: pd.Index,
    min_assets: int,
) -> pd.Series:
    """
    Spearman IC on each rebalance date.
    signal[t, :] vs ret_fwd[t, :]
    """
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
    """
    Coverage diagnostics on rebalance dates:
    - fraction of universe that is valid (signal and ret_fwd both available)
    - count of valid assets
    """
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
    """
    Simple turnover proxy for a given bucket:
    turnover(t) = 1 - overlap(members_t, members_{t-1}) / len(members_t)
    """
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


def _as_series(x: pd.Series | pd.DataFrame, *, name: str) -> pd.Series:
    """
    Normalize a Series-or-1col-DataFrame into a Series.
    If x is a multi-column DataFrame, raise with a clear error.
    """
    if isinstance(x, pd.Series):
        return x.rename(name)

    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0].rename(name)

        # Multi-column benchmark is ambiguous: you must choose one
        raise ValueError(
            f"Benchmark resolved to a DataFrame with {x.shape[1]} columns. "
            f"Please pass a single benchmark Series, or a 1-column DataFrame. "
            f"Columns preview: {list(x.columns)[:10]}"
        )

    raise TypeError(f"Expected Series or DataFrame, got {type(x)}")


def _compute_benchmark_ret_fwd(
    bench_price: pd.DataFrame | pd.Series,
    H: int,
    entry_mode: EntryMode,
) -> pd.Series:
    # --- pick the price series(s) ---
    if isinstance(bench_price, pd.Series):
        px_close = bench_price.sort_index()
        px_open = None
    else:
        bench_df = bench_price.sort_index()

        # yfinance-style MultiIndex columns: ["Open","Close",...] at level 0
        if isinstance(bench_df.columns, pd.MultiIndex):
            if "Close" not in bench_df.columns.get_level_values(0):
                raise KeyError("Benchmark MultiIndex DataFrame must contain level-0 'Close'.")
            px_close = bench_df["Close"]
            px_open = bench_df["Open"] if "Open" in bench_df.columns.get_level_values(0) else None
        else:
            cols = list(bench_df.columns)
            if "Close" in cols:
                px_close = bench_df["Close"]
                px_open = bench_df["Open"] if "Open" in cols else None
            elif len(cols) == 1:
                px_close = bench_df.iloc[:, 0]
                px_open = None
            else:
                raise KeyError(
                    f"Benchmark DataFrame must have 'Close' or be 1-column. Got: {cols[:10]}..."
                )

    # --- normalize: ensure close/open are Series (not DataFrame) ---
    px_close = _as_series(px_close, name="bench_close")
    if px_open is not None:
        px_open = _as_series(px_open, name="bench_open")

    # --- compute forward return consistent with your backtest convention ---
    if entry_mode == "open_to_close":
        if px_open is None:
            raise ValueError("entry_mode='open_to_close' requires benchmark Open & Close.")
        entry_px = px_open.shift(-1)
        exit_px = px_close.shift(-(H + 1))
        out = exit_px / entry_px - 1.0

    elif entry_mode == "next_open":
        if px_open is None:
            raise ValueError("entry_mode='next_open' requires benchmark Open.")
        entry_px = px_open.shift(-1)
        exit_px = px_open.shift(-(H + 1))
        out = exit_px / entry_px - 1.0

    else:  # "next_close"
        entry_px = px_close.shift(-1)
        exit_px = px_close.shift(-(H + 1))
        out = exit_px / entry_px - 1.0

    return out.rename("benchmark_ret_fwd")
