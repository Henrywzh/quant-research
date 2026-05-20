from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import pandas as pd

from .config import ensure_directories, default_paths


def export_review_sheets(index_df: pd.DataFrame, curated_tables: dict[str, pd.DataFrame]) -> list[Path]:
    paths = ensure_directories(default_paths())
    out_dir = paths.exports_root / "qa_review_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    sli = curated_tables.get("statement_line_items", pd.DataFrame())
    cap = curated_tables.get("capital_structure", pd.DataFrame())
    debt = curated_tables.get("debt_schedule", pd.DataFrame())
    ipn = curated_tables.get("investment_property_notes", pd.DataFrame())

    sheets = {
        "book_nav_core_fields.csv": _review_book_nav_core(sli),
        "capital_structure_fields.csv": _review_generic(cap, "capital_structure"),
        "debt_fields.csv": _review_generic(debt, "debt_schedule"),
        "ip_note_fields.csv": _review_generic(ipn, "investment_property_notes"),
        "rnav_assumptions.csv": _default_rnav_assumptions_sheet(index_df),
    }
    for name, df in sheets.items():
        fp = out_dir / name
        df.to_csv(fp, index=False)
        outputs.append(fp)
    tpl = paths.exports_root / "qa_overrides_import_template.csv"
    if not tpl.exists():
        pd.DataFrame(columns=[
            "target_table","target_key_json","field_name","override_value_num","override_value_text",
            "override_reason","reviewer","status"
        ]).to_csv(tpl, index=False)
        outputs.append(tpl)
    return outputs


def _review_book_nav_core(statement_line_items: pd.DataFrame) -> pd.DataFrame:
    target_codes = [
        "total_equity", "equity_attributable_to_owners", "non_controlling_interests",
        "total_borrowings_current", "total_borrowings_noncurrent", "cash_and_bank_balances",
        "dividends_declared", "dividends_paid", "equity_per_share_disclosed",
    ]
    df = statement_line_items[statement_line_items["line_item_code"].isin(target_codes)].copy() if not statement_line_items.empty else pd.DataFrame()
    if df.empty:
        df = pd.DataFrame(columns=["filing_id","stock_code","period_end","line_item_code","line_item_label_raw","value","page_no","extract_confidence"])
    keep = [c for c in ["filing_id","stock_code","period_end","line_item_code","line_item_label_raw","value","page_no","extract_confidence"] if c in df.columns]
    df = df[keep].copy()
    df["suggested_value"] = df.get("value")
    df["override_value_num"] = None
    df["override_value_text"] = None
    df["override_reason"] = None
    df["reviewer"] = None
    df["status"] = "pending"
    df["target_table"] = "statement_line_items"
    df["target_key_json"] = df.apply(lambda r: json.dumps({"filing_id": r.get("filing_id"), "line_item_code": r.get("line_item_code")}), axis=1)
    df["field_name"] = "value"
    return df


def _review_generic(df: pd.DataFrame, target_table: str) -> pd.DataFrame:
    if df is None or df.empty:
        out = pd.DataFrame(columns=["target_table","target_key_json","field_name","suggested_value","override_value_num","override_value_text","override_reason","reviewer","status"])
        return out
    out = df.copy()
    id_cols = [c for c in out.columns if c.endswith("_id")]
    value_col = "value_num" if "value_num" in out.columns else ("amount" if "amount" in out.columns else None)
    if value_col is None and "shares_outstanding_period_end" in out.columns:
        value_col = "shares_outstanding_period_end"
    out["target_table"] = target_table
    out["target_key_json"] = out.apply(lambda r: json.dumps({c: r[c] for c in id_cols[:1]}), axis=1)
    out["field_name"] = value_col or "value_text"
    out["suggested_value"] = out[value_col] if value_col else None
    out["override_value_num"] = None
    out["override_value_text"] = None
    out["override_reason"] = None
    out["reviewer"] = None
    out["status"] = "pending"
    return out


def _default_rnav_assumptions_sheet(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in index_df.iterrows():
        for scenario, cap_bps, ip_uplift, dev_uplift in [
            ("rnav_base", 0, 0.0, 0.05),
            ("rnav_bull", -25, 0.05, 0.10),
            ("rnav_bear", 25, -0.05, 0.00),
        ]:
            rows.append(
                {
                    "stock_code": r["stock_code"],
                    "period_end": r.get("period_end"),
                    "scenario": scenario,
                    "ip_adjustment_method": "cap_rate_sensitivity",
                    "cap_rate_shift_bps": cap_bps,
                    "ip_value_uplift_pct": ip_uplift,
                    "dev_hidden_reserve_pct": dev_uplift,
                    "associate_jv_uplift_pct": 0.0,
                    "deferred_tax_haircut_pct": 0.25,
                    "minority_adjustment_mode": "subtract_nci",
                    "notes": "",
                    "source_kind": "default_rule",
                    "target_table": "rnav_assumptions",
                    "target_key_json": json.dumps({"stock_code": r["stock_code"], "period_end": r.get("period_end"), "scenario": scenario}),
                    "field_name": "row",
                    "override_value_num": None,
                    "override_value_text": None,
                    "override_reason": None,
                    "reviewer": None,
                    "status": "pending",
                }
            )
    return pd.DataFrame(rows)


def import_overrides_from_csv(csv_paths: list[str | Path]) -> pd.DataFrame:
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for p in csv_paths:
        fp = Path(p)
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        required = {"target_table", "target_key_json", "field_name"}
        if not required.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            status = str(r.get("status", "pending"))
            if status not in {"approved", "rejected", "pending"}:
                status = "pending"
            rows.append(
                {
                    "override_id": f"ovr_{uuid.uuid4().hex[:12]}",
                    "target_table": r.get("target_table"),
                    "target_key_json": r.get("target_key_json"),
                    "field_name": r.get("field_name"),
                    "override_value_num": r.get("override_value_num"),
                    "override_value_text": r.get("override_value_text"),
                    "override_reason": r.get("override_reason"),
                    "reviewer": r.get("reviewer"),
                    "reviewed_at_utc": now,
                    "status": status,
                    "source_csv_path": str(fp),
                }
            )
    return pd.DataFrame(rows)

