from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class FactorVizConfig:
    rolling_window_obs: int = 63
    best_col: Optional[str] = None     # e.g. "bucket_10"
    worst_col: Optional[str] = None    # e.g. "bucket_1"
    benchmark_name: str = "Benchmark"
    figsize: tuple[int, int] = (10, 5)


def visualize_factor_tearsheet(
    *,
    ic: pd.Series,
    ic_roll: Optional[pd.Series] = None,
    bucket_ret: Optional[pd.DataFrame] = None,
    mean_by_bucket: Optional[pd.Series] = None,
    ls_eq: Optional[pd.Series] = None,
    ls_dd: Optional[pd.Series] = None,
    ls_yearly: Optional[pd.Series] = None,
    bench_eq: Optional[pd.Series] = None,
    bench_dd: Optional[pd.Series] = None,
    bench_yearly: Optional[pd.Series] = None,
    cfg: FactorVizConfig = FactorVizConfig(),
) -> None:
    """
    Visual diagnostics for factor / signal analysis.

    Required minimal input:
      - ic: daily IC series (Spearman IC typically)

    Optional:
      - ic_roll: rolling mean IC series (if None, computed from cfg.rolling_window_obs)
      - bucket_ret: DataFrame of bucket returns (index=date, columns like bucket_1..bucket_N)
      - mean_by_bucket: mean return per bucket (if None and bucket_ret provided, computed)
      - ls_eq / ls_dd / ls_yearly: long-short equity/drawdown/yearly returns
      - bench_dd / bench_yearly: benchmark drawdown/yearly returns
    """
    if ic is None or len(ic) == 0:
        raise ValueError("ic must be a non-empty pd.Series indexed by date.")

    ic = ic.dropna().copy()
    ic.index = pd.to_datetime(ic.index)
    ic = ic.sort_index()

    rw = int(cfg.rolling_window_obs)
    if ic_roll is None:
        ic_roll = ic.rolling(rw).mean()

    # -------- 1) Cumulative IC + rolling IC --------
    plt.figure(figsize=cfg.figsize)
    ic.cumsum().plot()
    plt.title("Cumulative IC (Spearman)")
    plt.ylabel("Cumulative IC")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=cfg.figsize)
    ic_roll.plot()
    plt.title(f"Rolling Mean IC (window={rw} obs)")
    plt.ylabel("Rolling IC mean")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

    # If no bucket info, stop here.
    if bucket_ret is None or len(bucket_ret) == 0:
        return

    bucket_ret = bucket_ret.copy()
    bucket_ret.index = pd.to_datetime(bucket_ret.index)
    bucket_ret = bucket_ret.sort_index()

    # infer best/worst if not provided
    if cfg.best_col is None or cfg.worst_col is None:
        cols = [c for c in bucket_ret.columns if str(c).startswith("bucket_")]
        if not cols:
            raise ValueError("bucket_ret columns must include bucket_*.")
        # sort bucket_1..bucket_N
        cols_sorted = sorted(cols, key=lambda s: int(str(s).replace("bucket_", "")))
        worst_col = cols_sorted[0]
        best_col = cols_sorted[-1]
    else:
        best_col = cfg.best_col
        worst_col = cfg.worst_col

    # compute mean_by_bucket if needed
    if mean_by_bucket is None:
        mean_by_bucket = bucket_ret.mean().rename("mean_return")

    # -------- 2) Bucket mean returns bar chart --------
    plt.figure(figsize=cfg.figsize)
    mean_by_bucket.sort_index(
        key=lambda idx: idx.astype(str).str.replace("bucket_", "", regex=False).astype(int)
    ).plot(kind="bar")
    plt.title("Mean Return per Bucket (per holding period)")
    plt.ylabel("Mean H-day return")
    plt.xlabel("Bucket (1=worst, N=best)")
    plt.tight_layout()
    plt.show()

    # -------- 3) Bucket cumulative curves --------
    eq_all = (1.0 + bucket_ret.fillna(0.0)).cumprod()
    eq_all.plot(legend=True, figsize=cfg.figsize)

    plt.title("Bucket Cumulative Curves")
    plt.ylabel("Cumulative growth")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

    # -------- 4) Drawdown plot: best vs worst vs benchmark --------
    def equity_curve(ret: pd.Series) -> pd.Series:
        ret = ret.fillna(0.0)
        return (1.0 + ret).cumprod()

    def drawdown_series_from_equity(eq: pd.Series) -> pd.Series:
        if eq is None or len(eq) == 0:
            return eq
        peak = eq.cummax()
        return eq / peak - 1.0

    plt.figure(figsize=cfg.figsize)

    dd_best = drawdown_series_from_equity(equity_curve(bucket_ret[best_col]))
    dd_worst = drawdown_series_from_equity(equity_curve(bucket_ret[worst_col]))

    if dd_best is not None and len(dd_best):
        dd_best.plot(label=f"{best_col} (BEST, max_dd={dd_best.min():.2%})")
    if dd_worst is not None and len(dd_worst):
        dd_worst.plot(label=f"{worst_col} (WORST, max_dd={dd_worst.min():.2%})")
    if bench_dd is not None and len(bench_dd):
        bench_dd = bench_dd.dropna().copy()
        bench_dd.index = pd.to_datetime(bench_dd.index)
        bench_dd = bench_dd.sort_index()
        bench_dd.plot(label=f"{cfg.benchmark_name} (max_dd={bench_dd.min():.2%})")

    plt.title("Drawdown Curves (Best / Worst / Benchmark)")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- 5) Long-short equity + drawdown --------
    if ls_eq is not None and len(ls_eq):
        ls_eq = ls_eq.dropna().copy()
        ls_eq.index = pd.to_datetime(ls_eq.index)
        ls_eq = ls_eq.sort_index()

        plt.figure(figsize=cfg.figsize)
        ls_eq.plot(label='Long-Short')
        if bench_eq is not None:
            bench_eq.plot(label='Benchmark')
        plt.title("Long-Short Equity (Best - Worst)")
        plt.legend()
        plt.ylabel("Equity")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    if ls_dd is not None and len(ls_dd):
        ls_dd = ls_dd.dropna().copy()
        ls_dd.index = pd.to_datetime(ls_dd.index)
        ls_dd = ls_dd.sort_index()

        plt.figure(figsize=cfg.figsize)
        ls_dd.plot(label=f"LS (max_dd={ls_dd.min():.2%})")
        plt.title("Long-Short Drawdown (Best - Worst)")
        plt.ylabel("Drawdown")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------- 6) Yearly returns bar --------
    def yearly_returns(ret: pd.Series) -> pd.Series:
        ret = ret.dropna()
        if len(ret) == 0:
            return pd.Series(dtype=float)
        ret.index = pd.to_datetime(ret.index)
        # calendar-year compounded return
        return (1.0 + ret).groupby(ret.index.year).prod() - 1.0

    yr_best = yearly_returns(bucket_ret[best_col]).rename(best_col)
    yr_worst = yearly_returns(bucket_ret[worst_col]).rename(worst_col)

    parts = [yr_best, yr_worst]
    if ls_yearly is not None and len(ls_yearly):
        parts.append(ls_yearly.rename("LS"))
    if bench_yearly is not None and len(bench_yearly):
        parts.append(bench_yearly.rename(cfg.benchmark_name))

    yr_tbl = pd.concat(parts, axis=1).sort_index()
    if len(yr_tbl):
        yr_tbl.plot(kind="bar", figsize=cfg.figsize)
        plt.title("Calendar-Year Returns (Best / Worst / LS / Benchmark)")
        plt.ylabel("Return")
        plt.xlabel("Year")
        plt.tight_layout()
        plt.show()


def visualize_from_rep(
    rep: Dict[str, Any],
    *,
    cfg: FactorVizConfig = FactorVizConfig(),
) -> None:
    """
    Convenience wrapper: adapt your make_tearsheet output dict to visualize_factor_tearsheet inputs.

    Assumes common keys; adjust mapping if your rep schema differs.
    """
    ic = rep.get("ic_series") or rep.get("ic")
    ic_roll = rep.get("ic_roll")  # optional
    bucket_ret = rep.get("bucket_ret")
    mean_by_bucket = rep.get("mean_by_bucket") or (rep.get("bucket_ret").mean() if rep.get("bucket_ret") is not None else None)

    # long-short artifacts (optional)
    ls_eq = rep.get("ls_eq")
    ls_dd = rep.get("ls_dd")
    ls_yearly = rep.get("ls_yearly")

    bench_dd = rep.get("bench_dd")
    bench_yearly = rep.get("bench_yearly")

    visualize_factor_tearsheet(
        ic=ic,
        ic_roll=ic_roll,
        bucket_ret=bucket_ret,
        mean_by_bucket=mean_by_bucket,
        ls_eq=ls_eq,
        ls_dd=ls_dd,
        ls_yearly=ls_yearly,
        bench_dd=bench_dd,
        bench_yearly=bench_yearly,
        cfg=cfg,
    )
