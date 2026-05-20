from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qresearch.data.types import MarketData


def _sample_market_data() -> MarketData:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame(
        [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
        index=dates,
        columns=["A", "B"],
    )
    open_ = close - 0.5
    volume = pd.DataFrame([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]], index=dates, columns=close.columns)
    return MarketData(close=close, open=open_, volume=volume)


def test_dataset_root_resolution_is_explicit_and_not_cwd_driven(tmp_path, monkeypatch):
    from qresearch.data.catalog import dataset_root, resolve_dataset_path

    root = tmp_path / "custom-data-root"
    (root / "processed").mkdir(parents=True)
    (root / "raw").mkdir(parents=True)
    manifest = root / "processed" / "dataset_registry.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": {
                    "demo_panel": {
                        "path": "demo/panel.parquet",
                        "source_system": "unit_test",
                        "schema_version": 1,
                        "format": "parquet",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    workdir = tmp_path / "nested" / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workdir)

    assert dataset_root(root) == root
    assert resolve_dataset_path("demo_panel", root=root) == root / "processed" / "demo" / "panel.parquet"


def test_market_data_roundtrip_can_load_provenance_metadata(tmp_path):
    from qresearch.data.catalog import DatasetArtifactMetadata, load_dataset_metadata
    from qresearch.data.io import load_market_data_with_metadata, save_market_data_to_parquet

    root = tmp_path / "data"
    md = _sample_market_data()
    artifact = root / "processed" / "demo" / "market.parquet"
    save_market_data_to_parquet(md, artifact)

    metadata = DatasetArtifactMetadata(
        dataset_name="demo_market",
        path="demo/market.parquet",
        source_system="unit_test",
        schema_version=1,
        format="parquet",
        build_timestamp="2026-05-20T00:00:00Z",
        upstream_raw_inputs=["raw/demo.csv"],
    )
    metadata.write(root=root)

    loaded, loaded_meta = load_market_data_with_metadata(artifact, root=root)

    pd.testing.assert_frame_equal(loaded.close, md.close)
    assert loaded_meta is not None
    assert loaded_meta.dataset_name == "demo_market"
    assert load_dataset_metadata(artifact, root=root).source_system == "unit_test"


def test_yfinance_download_market_data_respects_adjustment_and_fill_policy(monkeypatch):
    from qresearch.data.yfinance import download_close, download_market_data

    calls: list[dict[str, object]] = []

    def fake_download(*args, **kwargs):
        calls.append(kwargs)
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        return pd.DataFrame(
            {
                ("Open", "AAA"): [10.0, None, 12.0],
                ("High", "AAA"): [11.0, None, 13.0],
                ("Low", "AAA"): [9.0, None, 11.0],
                ("Close", "AAA"): [10.0, None, 12.0],
                ("Volume", "AAA"): [100.0, None, 120.0],
            },
            index=idx,
        )

    monkeypatch.setattr("qresearch.data.yfinance.yf.download", fake_download)

    md = download_market_data(
        "AAA",
        start="2024-01-01",
        auto_adjust_close=False,
        fill_method="none",
        drop_all_nan_rows=False,
    )
    px = download_close("AAA", start="2024-01-01", auto_adjust=False, fill_method="none", drop_all_nan_rows=False)

    assert calls[0]["auto_adjust"] is False
    assert calls[1]["auto_adjust"] is False
    assert pd.isna(md.close.iloc[1, 0])
    assert pd.isna(px.iloc[1, 0])


def test_hsci_ticker_master_can_include_current_components_and_emit_audit(tmp_path):
    from qresearch.universe.hsci import build_hsci_ticker_master

    all_df = pd.DataFrame({"Stock Code 股份代號": ["1", "2"]})
    snapshot_2008 = pd.DataFrame({"Stock Code": ["3"]})
    current_components = pd.DataFrame({"Stock Code": ["0004", "bad-code"]})

    master, audit = build_hsci_ticker_master(
        all_df,
        snapshot_2008,
        current_components=current_components,
        save=False,
        return_audit=True,
    )

    assert set(master["ticker"]) == {"0001.HK", "0002.HK", "0003.HK", "0004.HK"}
    assert audit["current_components_rows"] == 2
    assert audit["current_components_unresolved"] == ["bad-code"]


def test_ashare_connect_builders_accept_config_objects_and_return_audit():
    from qresearch.universe.gating.ashare_connect import (
        AShareConnectExecutionConfig,
        AShareConnectMembershipConfig,
        build_ashare_connect_member_mask,
        compute_execution_masks,
    )

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    tickers = pd.Index(["000001.XSHE"])
    events = pd.DataFrame({"code": ["000001.XSHE"], "change_date": ["2024-01-02"], "direction": ["in"]})

    mask, audit = build_ashare_connect_member_mask(
        dates=dates,
        tickers=tickers,
        events=events,
        membership_config=AShareConnectMembershipConfig(effective_on_close=False),
        return_audit=True,
    )

    assert bool(mask.iloc[1, 0]) is False
    assert bool(mask.iloc[2, 0]) is True
    assert audit["events_applied"] == 1

    grid = pd.DataFrame([10.0, 10.0, 10.0], index=dates, columns=tickers)
    masks = compute_execution_masks(
        open_=grid,
        close=grid,
        volume=grid,
        high_limit=grid,
        low_limit=grid,
        config=AShareConnectExecutionConfig(lock_abs_tol=0.0, nan_limit_policy="block"),
    )

    assert masks["limit_up_locked"].iloc[0, 0]


def test_price_cache_can_return_metadata_and_write_sidecar(tmp_path, monkeypatch):
    from qresearch.hkex_nav.pricing import fetch_or_load_price_cache

    idx = pd.DataFrame(
        {
            "stock_code": ["00016"],
            "publish_datetime_hk": ["2024-01-31T12:00:00"],
        }
    )
    px = pd.DataFrame({"0016.HK": [80.0]}, index=pd.date_range("2024-01-31", periods=1, freq="D"))
    cache = tmp_path / "prices_daily.parquet"

    monkeypatch.setattr("qresearch.hkex_nav.pricing.price_cache_path", lambda: cache)
    monkeypatch.setattr("qresearch.hkex_nav.pricing.download_close", lambda **kwargs: px)

    loaded, metadata = fetch_or_load_price_cache(idx, force=True, return_metadata=True)

    pd.testing.assert_frame_equal(loaded, px)
    assert metadata["source_system"] == "yahoo_finance"
    assert (cache.with_suffix(".meta.json")).exists()
