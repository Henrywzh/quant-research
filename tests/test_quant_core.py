from __future__ import annotations

import importlib

import pandas as pd
import pytest

from qresearch.backtest.buckets import bucket_backtest
from qresearch.backtest.config import (
    AssetVolTargetConfig,
    AssetVolTargetOverlay,
    ExperimentConfig,
    StrategySpec,
    TopKBookConfig,
    TradeDrawdownStopConfig,
    TradeDrawdownStopOverlayWrapper,
)
from qresearch.backtest.metrics import equity_curve
from qresearch.backtest.portfolio import run_one
from qresearch.data.io import load_market_data_from_parquet, save_market_data_to_parquet
from qresearch.data.types import MarketData
from qresearch.data.utils import marketdata_to_yfinance, yfinance_to_marketdata
from qresearch.signals import compute_signal, list_signals
from qresearch.signals.signals_sweep import SignalTestConfig, run_signal_test


def _sample_market_data() -> MarketData:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        [
            [10.0, 11.0, 12.0, 13.0],
            [11.0, 10.5, 12.5, 12.0],
            [12.0, 10.0, 13.0, 11.0],
            [13.0, 9.5, 13.5, 10.0],
            [14.0, 9.0, 14.0, 9.0],
            [15.0, 8.5, 14.5, 8.0],
        ],
        index=dates,
        columns=cols,
    )
    open_ = close - 0.25
    volume = pd.DataFrame(1_000.0, index=dates, columns=cols)
    turnover = close * volume
    mkt_cap = close * 100_000
    return MarketData(
        close=close,
        open=open_,
        high=close + 0.5,
        low=close - 0.5,
        volume=volume,
        turnover=turnover,
        mkt_cap=mkt_cap,
    )


def test_quant_core_import_surface_smoke():
    modules = [
        "qresearch.data",
        "qresearch.signals",
        "qresearch.backtest",
        "qresearch.portfolio",
        "qresearch.universe",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None


def test_equity_curve_keeps_shape_and_treats_nans_as_flat_returns():
    returns = pd.Series([0.10, None, -0.05], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    result = equity_curve(returns)

    expected = pd.Series([1.10, 1.10, 1.045], index=returns.index)
    pd.testing.assert_series_equal(result, expected)


def test_run_one_does_not_mutate_input_market_data():
    md = _sample_market_data()
    original_close = md.close.copy(deep=True)
    original_open = md.open.copy(deep=True)
    scores = md.close.rank(axis=1, pct=True)
    cfg = ExperimentConfig(
        start="2024-01-02",
        end="2024-01-05",
        benchmark_ticker="A",
        strategy=StrategySpec(selector=TopKBookConfig(long_k=1)),
    )

    out = run_one(md, scores, cfg)

    pd.testing.assert_frame_equal(md.close, original_close)
    pd.testing.assert_frame_equal(md.open, original_open)
    assert out["strat"].weights_used.index.min() == pd.Timestamp("2024-01-02")
    assert out["strat"].weights_used.index.max() == pd.Timestamp("2024-01-05")


def test_overlay_pipeline_accepts_supported_overlay_wrappers():
    md = _sample_market_data()
    scores = md.close.rank(axis=1, pct=True)
    cfg = ExperimentConfig(
        benchmark_ticker="A",
        strategy=StrategySpec(
            selector=TopKBookConfig(long_k=1),
            overlays=(
                AssetVolTargetOverlay(AssetVolTargetConfig(enabled=True, vol_window=2)),
                TradeDrawdownStopOverlayWrapper(TradeDrawdownStopConfig(enabled=True)),
            ),
        ),
    )

    out = run_one(md, scores, cfg)

    assert "asset_vol_target" in out["weights_diag"]
    assert "trade_dd_stop" in out["weights_diag"]


def test_bucket_backtest_respects_universe_eligibility_at_formation():
    md = _sample_market_data()
    signal = pd.DataFrame(
        [
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        index=md.close.index,
        columns=md.close.columns,
    )
    eligible = pd.DataFrame(True, index=md.close.index, columns=md.close.columns)
    eligible.loc[:, "A"] = False

    bucket_ret, bucket_lbl, _ = bucket_backtest(md, signal, H=1, n_buckets=2, universe_eligible=eligible)

    assert not bucket_ret.empty
    assert pd.isna(bucket_lbl.loc[md.close.index[0], "A"])
    assert bucket_lbl.loc[md.close.index[0], "B"] in {1.0, 2.0}


def test_market_data_roundtrip_uses_consistent_optional_field_representation(tmp_path):
    md = _sample_market_data()
    file_path = tmp_path / "market.parquet"

    save_market_data_to_parquet(md, file_path)
    loaded = load_market_data_from_parquet(file_path)
    restored = yfinance_to_marketdata(marketdata_to_yfinance(loaded))

    pd.testing.assert_frame_equal(loaded.close, md.close)
    pd.testing.assert_frame_equal(loaded.turnover, md.turnover)
    pd.testing.assert_frame_equal(restored.turnover, md.turnover)
    assert restored.open is not None
    assert restored.volume is not None

    close_only = pd.concat({"Close": md.close}, axis=1)
    close_only_md = yfinance_to_marketdata(close_only)
    assert close_only_md.open is None
    assert close_only_md.volume is None


def test_signal_registry_and_run_signal_test_support_exec_masks():
    md = _sample_market_data()
    cfg = SignalTestConfig(H=1, n_buckets=2, min_assets_ic=2)
    exec_masks = {
        "tradeable": pd.DataFrame(True, index=md.close.index, columns=md.close.columns),
        "limit_up_locked": pd.DataFrame(False, index=md.close.index, columns=md.close.columns),
        "limit_down_locked": pd.DataFrame(False, index=md.close.index, columns=md.close.columns),
        "can_buy_next_open": pd.DataFrame(True, index=md.close.index, columns=md.close.columns),
        "can_sell_next_open": pd.DataFrame(True, index=md.close.index, columns=md.close.columns),
    }

    assert "mom_ret" in list_signals()
    signal = compute_signal(md, "mom_ret", lookback=1, skip=0)
    assert signal.index.equals(md.close.index)
    assert signal.columns.equals(md.close.columns)

    out = run_signal_test(md, "mom_ret", {"lookback": 1, "skip": 0}, cfg=cfg, exec_masks=exec_masks)

    assert out["signal_name"] == "mom_ret"
    assert "rep" in out


def test_run_signal_test_rejects_incomplete_exec_masks():
    md = _sample_market_data()
    cfg = SignalTestConfig(H=1, n_buckets=2, min_assets_ic=2)

    with pytest.raises(ValueError, match="exec_masks missing keys"):
        run_signal_test(
            md,
            "mom_ret",
            {"lookback": 1, "skip": 0},
            cfg=cfg,
            exec_masks={"tradeable": pd.DataFrame(True, index=md.close.index, columns=md.close.columns)},
        )
