from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from qresearch.data.types import MarketData
from qresearch.signals.registry import compute_signal
from qresearch.backtest.buckets import make_tearsheet, UniverseFilterConfig, compute_ma_price_eligibility
from qresearch.backtest.metrics import perf_summary, equity_curve


@dataclass(frozen=True)
class SignalTestConfig:
    H: int = 5
    n_buckets: int = 20
    entry_mode: str = "next_close"
    min_assets_ic: int = 50
    # penny filter
    use_price_floor: bool = True
    ma_window: int = 21
    min_ma_price: float = 1.0


def _make_long_short(bucket_ret: pd.DataFrame, worst: str, best: str) -> pd.Series:
    """Long best bucket, short worst bucket."""
    return (bucket_ret[best] - bucket_ret[worst]).dropna()


def run_signal_test(
    md: MarketData,
    signal_name: str,
    signal_params: Optional[Dict[str, Any]] = None,
    cfg: SignalTestConfig = SignalTestConfig(),
    benchmark_price: Optional[pd.DataFrame | pd.Series] = None,
    benchmark_name: str = "Benchmark",
    plot: bool = False,
) -> Dict[str, Any]:
    """
    Compute a signal -> tearsheet -> compact diagnostics.

    Assumes:
      - price_df has ["Close"] (and maybe ["Open"])
      - bucket convention: bucket_1 = worst, bucket_n = best
    """
    signal_params = signal_params or {}

    close = md.close.sort_index()

    # 1) compute signal (date x ticker)
    sig = compute_signal(md, name=signal_name, **signal_params)

    # 2) universe filter (optional): exclude penny-ish via MA floor
    universe_eligible = None
    if cfg.use_price_floor:
        uf_cfg = UniverseFilterConfig(ma_window=cfg.ma_window, min_ma_price=cfg.min_ma_price)
        universe_eligible = compute_ma_price_eligibility(close, uf_cfg)

        # apply to signal by masking (keeps backtest logic simple)
        sig = sig.where(universe_eligible)

    # 3) tearsheet
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
    )

    # 4) compact summary
    worst = "bucket_1"
    best = f"bucket_{cfg.n_buckets}"

    ls = _make_long_short(rep["bucket_ret"], worst=worst, best=best)
    ls_perf = perf_summary(ls, freq=int(rep["meta"]["freq_per_year"])) if len(ls) else {}

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
        "rep": rep,  # keep full object
    }
    return out


def sweep_signals(
    md: MarketData,
    tests: List[Tuple[str, Dict[str, Any]]],
    cfg: SignalTestConfig,
) -> pd.DataFrame:
    rows = []
    for name, params in tests:
        out = run_signal_test(md, name, params, cfg=cfg, plot=False)
        rows.append({
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
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["ic_mean", "icir", "mono"], ascending=[False, False, False])
