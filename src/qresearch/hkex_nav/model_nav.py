from __future__ import annotations

import json

import pandas as pd

from .config import DEFAULT_ISSUERS
from .pricing import next_trading_close_after_publish


def _pivot_statement_facts(statement_line_items: pd.DataFrame) -> pd.DataFrame:
    if statement_line_items.empty:
        return pd.DataFrame()
    df = statement_line_items.copy()
    df = df[df["line_item_code"].notna()].copy()
    pv = df.pivot_table(
        index=["filing_id", "stock_code", "period_end"],
        columns="line_item_code",
        values="value",
        aggfunc="first",
    ).reset_index()
    pv.columns.name = None
    return pv


def build_rnav_assumptions(index_df: pd.DataFrame, overrides_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for _, r in index_df.iterrows():
        for scenario, cap_rate_shift_bps, ip_value_uplift_pct, dev_hidden_reserve_pct in [
            ("rnav_base", 0, 0.00, 0.05),
            ("rnav_bull", -25, 0.05, 0.10),
            ("rnav_bear", 25, -0.05, 0.00),
        ]:
            rows.append(
                {
                    "assumption_set_id": f"{r['stock_code']}_{r.get('period_end')}_{scenario}",
                    "stock_code": r["stock_code"],
                    "period_end": r.get("period_end"),
                    "scenario": scenario,
                    "ip_adjustment_method": "cap_rate_sensitivity",
                    "cap_rate_shift_bps": cap_rate_shift_bps,
                    "ip_value_uplift_pct": ip_value_uplift_pct,
                    "dev_hidden_reserve_pct": dev_hidden_reserve_pct,
                    "associate_jv_uplift_pct": 0.0,
                    "deferred_tax_haircut_pct": 0.25,
                    "minority_adjustment_mode": "subtract_nci",
                    "notes": "",
                    "source_kind": "default_rule",
                }
            )
    return pd.DataFrame(rows)


def compute_nav_outputs(
    filings_index: pd.DataFrame,
    statement_line_items: pd.DataFrame,
    capital_structure: pd.DataFrame,
    investment_property_notes: pd.DataFrame,
    price_df: pd.DataFrame | None = None,
    assumptions_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stmt = _pivot_statement_facts(statement_line_items)
    caps = capital_structure.copy() if capital_structure is not None else pd.DataFrame()
    if caps.empty:
        caps = pd.DataFrame(columns=["filing_id","shares_outstanding_period_end","weighted_avg_shares_basic"])
    base = filings_index.merge(stmt, on=["filing_id", "stock_code", "period_end"], how="left").merge(
        caps[["filing_id", "shares_outstanding_period_end", "weighted_avg_shares_basic"]],
        on="filing_id",
        how="left",
    )
    if assumptions_df is None:
        assumptions_df = build_rnav_assumptions(filings_index)

    book_rows = []
    rnav_rows = []

    ip_notes = investment_property_notes.copy() if investment_property_notes is not None else pd.DataFrame()
    if ip_notes.empty:
        ip_notes = pd.DataFrame(columns=["filing_id","metric_code","value_num"])
    ip_pv = ip_notes.pivot_table(index="filing_id", columns="metric_code", values="value_num", aggfunc="first").reset_index()
    base = base.merge(ip_pv, on="filing_id", how="left")

    for _, r in base.iterrows():
        equity = _as_num(r.get("equity_attributable_to_owners"))
        nci = _as_num(r.get("non_controlling_interests")) or 0.0
        shares = _as_num(r.get("shares_outstanding_period_end"))
        shares_source = "period_end"
        if not shares or shares == 0:
            shares = _as_num(r.get("weighted_avg_shares_basic"))
            shares_source = "weighted_avg_fallback"
        if not equity or not shares:
            continue
        gross_debt = (_as_num(r.get("total_borrowings_current")) or 0.0) + (_as_num(r.get("total_borrowings_noncurrent")) or 0.0)
        cash = _as_num(r.get("cash_and_bank_balances")) or 0.0
        net_debt = gross_debt - cash
        book_nav = equity / shares
        ticker = DEFAULT_ISSUERS.get(str(r["stock_code"]).zfill(5)).ticker_for_price if str(r["stock_code"]).zfill(5) in DEFAULT_ISSUERS else None
        price_date, price_close = (None, None)
        if ticker and price_df is not None and not price_df.empty:
            price_date, price_close = next_trading_close_after_publish(str(r["publish_datetime_hk"]), ticker, price_df)
        discount = (price_close / book_nav - 1.0) if (price_close is not None and book_nav) else None
        book_rows.append(
            {
                "nav_output_id": f"nav_{r['filing_id']}_book",
                "stock_code": r["stock_code"],
                "period_end": r["period_end"],
                "filing_id": r["filing_id"],
                "publish_datetime_hk": r["publish_datetime_hk"],
                "scenario": "book",
                "nav_per_share": book_nav,
                "shares_used": shares,
                "equity_base_used": equity,
                "net_debt_used": net_debt,
                "price_date_hk": price_date,
                "price_close": price_close,
                "discount_premium_pct": discount,
                "currency": "HKD",
                "calc_version": "v1",
                "assumption_set_id": None,
                "lineage_json": json.dumps({"shares_source": shares_source, "source_filing_id": r["filing_id"]}),
            }
        )
        ip_fv = _as_num(r.get("investment_properties_fair_value_total")) or 0.0
        fair_value_gain = _as_num(r.get("fair_value_gain_loss")) or 0.0
        deferred_tax_related = _as_num(r.get("deferred_tax_revaluation_related")) or 0.0
        period_assump = assumptions_df[
            (assumptions_df["stock_code"] == r["stock_code"]) &
            (assumptions_df["period_end"] == r["period_end"])
        ]
        for _, a in period_assump.iterrows():
            ip_adj = ip_fv * _as_num(a.get("ip_value_uplift_pct"), 0.0) + fair_value_gain * 0.0
            dev_adj = equity * _as_num(a.get("dev_hidden_reserve_pct"), 0.0)
            assoc_adj = equity * _as_num(a.get("associate_jv_uplift_pct"), 0.0)
            deferred_tax_adj = max(deferred_tax_related, ip_adj + dev_adj) * _as_num(a.get("deferred_tax_haircut_pct"), 0.0)
            minority_adj = nci if str(a.get("minority_adjustment_mode")) == "subtract_nci" else 0.0
            rnav_equity = equity + ip_adj + dev_adj + assoc_adj - deferred_tax_adj - minority_adj
            rnav_ps = rnav_equity / shares
            rnav_discount = (price_close / rnav_ps - 1.0) if (price_close is not None and rnav_ps) else None
            rnav_rows.append(
                {
                    "nav_output_id": f"nav_{r['filing_id']}_{a['scenario']}",
                    "stock_code": r["stock_code"],
                    "period_end": r["period_end"],
                    "filing_id": r["filing_id"],
                    "publish_datetime_hk": r["publish_datetime_hk"],
                    "scenario": a["scenario"],
                    "nav_per_share": rnav_ps,
                    "shares_used": shares,
                    "equity_base_used": equity,
                    "net_debt_used": net_debt,
                    "price_date_hk": price_date,
                    "price_close": price_close,
                    "discount_premium_pct": rnav_discount,
                    "currency": "HKD",
                    "calc_version": "v1",
                    "assumption_set_id": a["assumption_set_id"],
                    "lineage_json": json.dumps(
                        {
                            "shares_source": shares_source,
                            "components": {
                                "ip_adj": ip_adj,
                                "dev_adj": dev_adj,
                                "assoc_adj": assoc_adj,
                                "deferred_tax_adj": deferred_tax_adj,
                                "minority_adj": minority_adj,
                            },
                        }
                    ),
                }
            )

    nav_df = pd.DataFrame(book_rows + rnav_rows)
    return nav_df, assumptions_df


def _as_num(v, default=None):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return default

