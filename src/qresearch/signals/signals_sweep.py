from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from qresearch.data.types import MarketData
from qresearch.signals.registry import compute_signal
from qresearch.backtest.buckets import make_tearsheet, EntryMode
from qresearch.backtest.metrics import perf_summary
from qresearch.universe.builder import build_final_universe_eligible
from qresearch.universe.filters import UniverseFilterConfig


@dataclass(frozen=True)
class SignalTestConfig:
    # Backtest mechanics
    H: int = 5
    n_buckets: int = 20
    entry_mode: EntryMode = "next_close"
    min_assets_ic: int = 50

    # --- Universe gating (survivorship-safe membership) ---
    # If True: require (events, seed_members) and apply HSCI membership at time t.
    use_hsci_membership: bool = False

    # --- Investable filters (optional; applied on formation date t) ---

    # Price floor (penny filter)
    use_price_floor: bool = True
    ma_window: int = 21
    min_ma_price: float = 1.0

    # Market cap floor (rolling mean of mcap)
    use_mcap_floor: bool = False
    mcap_window: int = 21
    min_mean_mcap: Optional[float] = 1e9  # e.g. 1e9 or 1e10; None disables

    # IPO seasoning (trading-day age)
    use_ipo_seasoning: bool = False
    min_ipo_trading_days: int = 63  # e.g. 63; 0 disables



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
    *,
    # NEW: membership + optional size panel
    events: Optional[pd.DataFrame] = None,
    seed_members: Optional[set[str]] = None,
    mcap: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute a signal -> tearsheet -> compact diagnostics.

    Assumes:
      - signal computed at close[t]
      - bucket convention: bucket_1 = worst, bucket_n = best
      - ML is paused: this is pure signal testing
    """
    signal_params = signal_params or {}
    close = md.close.sort_index()

    # 1) compute signal (date x ticker)
    sig = compute_signal(md, name=signal_name, **signal_params)
    sig = sig.reindex(index=close.index, columns=close.columns)

    # 2) build universe eligibility mask (optional, but recommended)
    universe_eligible = None

    # If you want to support both: (a) old MA-only mode, and (b) new HSCI-membership mode:
    if cfg.use_price_floor or cfg.use_mcap_floor or cfg.use_ipo_seasoning or cfg.use_hsci_membership:
        uf_cfg = UniverseFilterConfig(
            ma_window=cfg.ma_window,
            min_ma_price=cfg.min_ma_price if cfg.use_price_floor else None,
            mcap_window=getattr(cfg, "mcap_window", 21),
            min_mean_mcap=getattr(cfg, "min_mean_mcap", None) if getattr(cfg, "use_mcap_floor", False) else None,
            min_ipo_trading_days=getattr(cfg, "min_ipo_trading_days", 0) if getattr(cfg, "use_ipo_seasoning", False) else 0,
        )

        # If you enable membership gating, events+seed_members must be provided
        if getattr(cfg, "use_hsci_membership", False):
            if events is None or seed_members is None:
                raise ValueError("events and seed_members must be provided when use_hsci_membership=True")

            universe_eligible = build_final_universe_eligible(
                close=close,
                events=events,
                seed_members=seed_members,
                cfg=uf_cfg,
                mcap=mcap,
            )
        else:
            # If no membership gating, universe_eligible can be just investable filters.
            # You can call builder with a dummy 'all True member mask' or expose build_investable_eligible.
            from qresearch.universe.builder import build_investable_eligible
            universe_eligible = build_investable_eligible(close=close, cfg=uf_cfg, mcap=mcap)

    # Optional: mask the signal (not required, but fine)
    if universe_eligible is not None:
        universe_eligible = universe_eligible.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        sig = sig.where(universe_eligible)

    # 3) tearsheet (pass universe_eligible through!)
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
        universe_eligible=universe_eligible,   # NEW
    )

    # 4) compact summary
    worst = "bucket_1"
    best = f"bucket_{cfg.n_buckets}"

    ls = _make_long_short(rep["bucket_ret"], worst=worst, best=best)
    ls_perf = perf_summary(ls, freq=float(rep["meta"]["freq_per_year"])) if len(ls) else {}

    return {
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
        "rep": rep,
        "universe_eligible_used": universe_eligible,  # optional: helps debugging
    }


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
