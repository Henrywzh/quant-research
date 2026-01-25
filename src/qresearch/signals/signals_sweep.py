from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from qresearch.data.types import MarketData
from qresearch.signals.registry import compute_signal
from qresearch.backtest.buckets import make_tearsheet, EntryMode
from qresearch.backtest.metrics import perf_summary


@dataclass(frozen=True)
class SignalTestConfig:
    # Backtest mechanics
    H: int = 5
    n_buckets: int = 20
    entry_mode: EntryMode = "next_close"
    min_assets_ic: int = 50

    # --- Universe gating ---
    # If True: apply survivorship-safe membership mask (requires events + seed_members)
    use_hsci_membership: bool = True

    # --- Investable filters (applied at formation date t) ---
    # Use None to disable each rule cleanly.

    # Price floor
    ma_window: int = 21
    min_ma_price: Optional[float] = 1.0

    # Market cap floor (rolling mean)
    mcap_window: int = 21
    min_mean_mcap: Optional[float] = 1e9  # e.g. 1e9 / 1e10 enables

    # IPO seasoning (trading days since first valid close)
    min_ipo_trading_days: Optional[int] = 63  # e.g. 63 enables


def _make_long_short(bucket_ret: pd.DataFrame, worst: str, best: str) -> pd.Series:
    """
    Long best bucket, short worst bucket (rebalance-grid series).
    Assumes bucket_ret is indexed by rebalance dates.
    """
    if worst not in bucket_ret.columns or best not in bucket_ret.columns:
        return pd.Series(dtype=float, name="LS_best_minus_worst")
    return (bucket_ret[best] - bucket_ret[worst]).dropna().rename("LS_best_minus_worst")


def _align_universe_eligible(universe_eligible: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive alignment for eligibility masks.
    Ensures:
      - same index/columns as close
      - bool dtype
      - missing -> False
    """
    return (
        universe_eligible.reindex(index=close.index, columns=close.columns)
        .fillna(False)
        .astype(bool)
    )


def run_signal_test(
    md: MarketData,
    signal_name: str,
    signal_params: Optional[Dict[str, Any]] = None,
    cfg: SignalTestConfig = SignalTestConfig(),
    benchmark_price: Optional[pd.DataFrame | pd.Series] = None,
    benchmark_name: str = "Benchmark",
    plot: bool = False,
    *,
    # NEW: prebuilt universe eligibility (formation-date gate)
    universe_eligible: Optional[pd.DataFrame] = None,
    # Optional: whether to mask the signal before testing.
    # Default False (recommended) so coverage diagnostics remain meaningful.
    mask_signal: bool = False,
) -> Dict[str, Any]:
    """
    Signal -> bucket tearsheet -> compact diagnostics.

    NEW contract:
      - This function does NOT build the universe.
      - If universe_eligible is provided, it is passed into make_tearsheet
        (and must be applied at formation dates inside bucket_backtest).
      - ML is paused: pure signal testing.

    Notes on masking:
      - Passing universe_eligible into bucket_backtest is sufficient.
      - Masking signal can hide coverage issues (signal NaN vs ineligible).
      - Keep mask_signal=False unless you have a specific reason.
    """
    signal_params = signal_params or {}
    close = md.close.sort_index()

    # 1) compute signal (date x ticker), aligned to close grid
    sig = compute_signal(md, name=signal_name, **signal_params)
    sig = sig.reindex(index=close.index, columns=close.columns)

    # 2) align universe_eligible once (if provided)
    ue = None
    if universe_eligible is not None:
        ue = _align_universe_eligible(universe_eligible, close)
        if mask_signal:
            sig = sig.where(ue)

    # 3) tearsheet (must pass ue through)
    rep = make_tearsheet(
        md=md,
        signal=sig,
        H=cfg.H,
        n_buckets=cfg.n_buckets,
        entry_mode=cfg.entry_mode,
        min_assets_ic=cfg.min_assets_ic,
        plot=plot,
        benchmark_price=benchmark_price,
        benchmark_name=benchmark_name,
        universe_eligible=ue,  # critical
    )

    # 4) compact summary
    worst = "bucket_1"
    best = f"bucket_{cfg.n_buckets}"

    ls = _make_long_short(rep["bucket_ret"], worst=worst, best=best)

    # IMPORTANT: your metrics module has standardized on freq_per_year, not freq.
    freq_per_year = float(rep["meta"]["freq_per_year"])
    ls_perf = perf_summary(ls, freq=freq_per_year) if len(ls) else {}

    out = {
        "signal_name": signal_name,
        "signal_params": signal_params,

        "ic_mean": rep["ic_stats"]["ic_mean"],
        "icir": rep["ic_stats"]["icir"],
        "hit_rate": rep["ic_stats"]["hit_rate"],
        "monotonic_spearman": rep["monotonic_spearman_bucket_vs_mean"],

        "top_bucket_annret": rep["bucket_summary"].loc[best, "ann_return_geo"],
        "top_bucket_sharpe": rep["bucket_summary"].loc[best, "sharpe"],
        "top_bucket_maxdd": rep["bucket_summary"].loc[best, "max_dd"],

        "ls_annret": ls_perf.get("ann_return_geo", np.nan),
        "ls_sharpe": ls_perf.get("sharpe", np.nan),
        "ls_maxdd": ls_perf.get("max_dd", np.nan),

        # Coverage diagnostics (useful in sweeps)
        "coverage_mean": float(rep["coverage"].mean()) if rep.get("coverage") is not None else np.nan,
        "n_valid_mean": float(rep["n_valid"].mean()) if rep.get("n_valid") is not None else np.nan,

        "rep": rep,
        # Keep ue out of the returned dict by default (avoid huge objects).
        # Add it only if you explicitly want it for debugging.
    }
    return out


def sweep_signals(
    md: MarketData,
    tests: List[Tuple[str, Dict[str, Any]]],
    cfg: SignalTestConfig,
    *,
    universe_eligible: Optional[pd.DataFrame] = None,
    benchmark_price: Optional[pd.DataFrame | pd.Series] = None,
    benchmark_name: str = "Benchmark",
    keep_rep: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    rep_map: Dict[str, Any] = {}

    for name, params in tests:
        out = run_signal_test(
            md=md,
            signal_name=name,
            signal_params=params,
            cfg=cfg,
            benchmark_price=benchmark_price,
            benchmark_name=benchmark_name,
            plot=False,
            universe_eligible=universe_eligible,
            mask_signal=False,
        )

        key = _signal_key(out["signal_name"], out["signal_params"])

        row = {
            "key": key,  # store key in df for easy joins
            "signal": out["signal_name"],
            "params": str(out["signal_params"]),

            "ic_mean": out["ic_mean"],
            "icir": out["icir"],
            "hit_rate": out["hit_rate"],
            "mono": out["monotonic_spearman"],

            "top_annret": out["top_bucket_annret"],
            "top_sharpe": out["top_bucket_sharpe"],
            "top_maxdd": out["top_bucket_maxdd"],

            "ls_annret": out["ls_annret"],
            "ls_sharpe": out["ls_sharpe"],
            "ls_maxdd": out["ls_maxdd"],

            "coverage_mean": out["coverage_mean"],
            "n_valid_mean": out["n_valid_mean"],
        }
        rows.append(row)

        if keep_rep:
            rep_map[key] = out["rep"]

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["ic_mean", "icir", "mono", "coverage_mean"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    return df, rep_map


def _signal_key(name: str, params: dict) -> str:
    items = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{name}({items})"


