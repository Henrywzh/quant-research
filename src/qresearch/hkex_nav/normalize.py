from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import ensure_directories, default_paths


LINE_ITEM_MAP_DEFAULT = {
    "equity attributable to owners": "equity_attributable_to_owners",
    "equity attributable to shareholders": "equity_attributable_to_owners",
    "non-controlling interests": "non_controlling_interests",
    "total equity": "total_equity",
    "cash and bank balances": "cash_and_bank_balances",
    "bank loans": "total_borrowings_noncurrent",
    "borrowings - current": "total_borrowings_current",
    "borrowings - non-current": "total_borrowings_noncurrent",
    "dividend": "dividends_declared",
}


def parse_unit_scale(text: str | None) -> str | None:
    t = (text or "").lower()
    if "million" in t or "$m" in t:
        return "millions"
    if "'000" in t or "thousand" in t:
        return "thousands"
    if t:
        return "ones"
    return None


def parse_currency(text: str | None) -> str | None:
    t = (text or "").upper()
    if "HK$" in t or "HKD" in t:
        return "HKD"
    if "RMB" in t or "CNY" in t:
        return "RMB"
    if "USD" in t or "US$" in t:
        return "USD"
    return None


def canonical_line_item(raw_label: str, candidate: str | None = None) -> str | None:
    if candidate:
        return candidate
    s = raw_label.lower()
    for key, code in LINE_ITEM_MAP_DEFAULT.items():
        if key in s:
            return code
    return None


def _read_jsonl(fp: Path) -> list[dict]:
    out = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_extracted_facts(index_df: pd.DataFrame) -> pd.DataFrame:
    paths = ensure_directories(default_paths())
    rows: list[dict] = []
    for row in index_df.to_dict(orient="records"):
        period_key = str(row.get("period_end") or "unknown_period")
        stock = str(row["stock_code"])
        doc_id = str(row["doc_id"])
        table_dir = paths.extracted_root / "tables" / stock / period_key
        for fp in [table_dir / f"{doc_id}.parquet", table_dir / f"{doc_id}.csv"]:
            if not fp.exists():
                continue
            if fp.suffix == ".parquet":
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp)
            for rec in df.to_dict(orient="records"):
                rows.append(
                    {
                        "filing_id": row["filing_id"],
                        "table_or_note": "table",
                        "raw_label": rec.get("line_text") or rec.get("raw_label"),
                        "candidate_canonical_code": canonical_line_item(str(rec.get("line_text") or ""), rec.get("candidate_canonical_code")),
                        "value": rec.get("value"),
                        "page_no": rec.get("page_no"),
                        "confidence": rec.get("extract_confidence"),
                        "extractor_name": "persisted_table_extract",
                        "source_kind": rec.get("source_kind", "table_extract"),
                        "statement_type": rec.get("statement_type", "balance_sheet"),
                        "currency": rec.get("currency"),
                        "unit_scale": rec.get("unit_scale"),
                        "row_idx": rec.get("row_idx"),
                        "col_idx": rec.get("col_idx"),
                        "raw_table_id": rec.get("raw_table_id"),
                    }
                )
            break
        notes_fp = paths.extracted_root / "text_chunks" / stock / period_key / f"{doc_id}.jsonl"
        if notes_fp.exists():
            note_records = _read_jsonl(notes_fp)
            for rec in note_records:
                txt = str(rec.get("text") or "")
                for line in txt.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    if "investment propert" in s.lower():
                        rows.append(
                            {
                                "filing_id": row["filing_id"],
                                "table_or_note": "note",
                                "raw_label": s,
                                "candidate_canonical_code": "investment_properties_fair_value_total",
                                "value": None,
                                "page_no": rec.get("page_no"),
                                "confidence": 0.2,
                                "extractor_name": "text_chunk_scan",
                                "source_kind": "text_extract",
                                "statement_type": None,
                                "currency": parse_currency(s),
                                "unit_scale": parse_unit_scale(s),
                                "row_idx": None,
                                "col_idx": None,
                                "raw_table_id": None,
                            }
                        )
                        break
    return pd.DataFrame(rows)


def build_curated_tables(index_df: pd.DataFrame, extracted_facts: pd.DataFrame, overrides: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    idx = index_df.copy()
    facts = extracted_facts.copy()
    if facts.empty:
        facts = pd.DataFrame(columns=[
            "filing_id", "table_or_note", "raw_label", "candidate_canonical_code", "value", "page_no",
            "confidence", "extractor_name", "source_kind", "statement_type", "currency", "unit_scale",
            "row_idx", "col_idx", "raw_table_id"
        ])
    if overrides is None:
        overrides = pd.DataFrame(columns=[
            "target_table","target_key_json","field_name","override_value_num","override_value_text","status"
        ])
    facts = facts.merge(
        idx[["filing_id", "stock_code", "period_end"]],
        on="filing_id",
        how="left",
    )
    facts["line_item_code"] = facts["candidate_canonical_code"]
    facts["extract_confidence"] = facts["confidence"]
    facts["line_item_label_raw"] = facts["raw_label"]
    facts["as_reported_sign"] = 1
    facts["scope"] = "group"

    statement_line_items = facts[facts["table_or_note"] == "table"].copy()
    if not statement_line_items.empty:
        statement_line_items = statement_line_items[[
            "filing_id","stock_code","period_end","statement_type","scope","line_item_code",
            "line_item_label_raw","value","currency","unit_scale","as_reported_sign","page_no",
            "source_kind","extract_confidence","raw_table_id","row_idx","col_idx"
        ]].reset_index(drop=True)
        statement_line_items.insert(0, "fact_id", [f"fact_{i+1}" for i in range(len(statement_line_items))])
    else:
        statement_line_items = pd.DataFrame(columns=[
            "fact_id","filing_id","stock_code","period_end","statement_type","scope","line_item_code",
            "line_item_label_raw","value","currency","unit_scale","as_reported_sign","page_no",
            "source_kind","extract_confidence","raw_table_id","row_idx","col_idx"
        ])

    capital_structure = _build_capital_structure(idx, facts)
    debt_schedule = _build_debt_schedule(idx, facts)
    investment_property_notes = _build_investment_property_notes(idx, facts)
    qa_overrides = overrides.copy()
    if qa_overrides.empty:
        qa_overrides = pd.DataFrame(columns=[
            "override_id","target_table","target_key_json","field_name","override_value_num","override_value_text",
            "override_reason","reviewer","reviewed_at_utc","status","source_csv_path"
        ])
    return {
        "filings_index": idx.copy(),
        "statement_line_items": statement_line_items,
        "capital_structure": capital_structure,
        "debt_schedule": debt_schedule,
        "investment_property_notes": investment_property_notes,
        "qa_overrides": qa_overrides,
    }


def _build_capital_structure(index_df: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in index_df.iterrows():
        subset = facts[facts["filing_id"] == r["filing_id"]]
        shares = _first_numeric_from_text(subset, [r"shares in issue", r"issued shares"])
        wavg = _first_numeric_from_text(subset, [r"weighted average"])
        rows.append(
            {
                "capital_fact_id": f"cap_{r['filing_id']}",
                "filing_id": r["filing_id"],
                "stock_code": r["stock_code"],
                "period_end": r.get("period_end"),
                "shares_outstanding_period_end": shares,
                "weighted_avg_shares_basic": wavg,
                "weighted_avg_shares_diluted": None,
                "treasury_shares": None,
                "page_no": None,
                "source_kind": "text_extract",
                "extract_confidence": 0.2,
                "notes": None,
            }
        )
    return pd.DataFrame(rows)


def _build_debt_schedule(index_df: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    out = []
    i = 0
    for _, r in index_df.iterrows():
        subset = facts[(facts["filing_id"] == r["filing_id"]) & facts["raw_label"].str.contains("borrow", case=False, na=False)]
        for _, fr in subset.iterrows():
            i += 1
            out.append(
                {
                    "debt_fact_id": f"debt_{i}",
                    "filing_id": r["filing_id"],
                    "stock_code": r["stock_code"],
                    "period_end": r.get("period_end"),
                    "debt_type": "other",
                    "current_noncurrent": None,
                    "maturity_bucket": None,
                    "secured_flag": None,
                    "currency": fr.get("currency"),
                    "amount": fr.get("value"),
                    "page_no": fr.get("page_no"),
                    "source_kind": fr.get("source_kind", "table_extract"),
                }
            )
    return pd.DataFrame(out)


def _build_investment_property_notes(index_df: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    note_facts = facts[(facts["table_or_note"] == "note") | (facts["line_item_code"].astype(str).str.contains("investment_properties", na=False))]
    rows = []
    for i, (_, fr) in enumerate(note_facts.iterrows(), start=1):
        rows.append(
            {
                "ip_note_fact_id": f"ip_{i}",
                "filing_id": fr["filing_id"],
                "stock_code": fr["stock_code"],
                "period_end": fr.get("period_end"),
                "metric_code": fr.get("line_item_code") or fr.get("candidate_canonical_code"),
                "metric_label_raw": fr.get("raw_label"),
                "value_num": fr.get("value") if isinstance(fr.get("value"), (int, float)) else None,
                "value_text": None if isinstance(fr.get("value"), (int, float)) else fr.get("value"),
                "currency": fr.get("currency"),
                "property_segment": None,
                "geography": None,
                "page_no": fr.get("page_no"),
                "source_kind": fr.get("source_kind"),
                "extract_confidence": fr.get("extract_confidence"),
            }
        )
    return pd.DataFrame(rows)


def _first_numeric_from_text(facts: pd.DataFrame, patterns: list[str]) -> float | None:
    if facts.empty:
        return None
    for p in patterns:
        subset = facts[facts["raw_label"].str.contains(p, case=False, na=False)]
        for _, row in subset.iterrows():
            val = row.get("value")
            if isinstance(val, (int, float)) and pd.notna(val):
                return float(val)
            nums = re.findall(r"\d[\d,]*", str(row.get("raw_label") or ""))
            if nums:
                return float(nums[-1].replace(",", ""))
    return None

