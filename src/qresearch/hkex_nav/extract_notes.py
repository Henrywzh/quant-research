from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd

from .config import ensure_directories, default_paths
from .extract_tables import _try_extract_pdf_text


NOTE_HEADINGS = {
    "investment_properties": re.compile(r"\binvestment properties\b", re.I),
    "fair_value": re.compile(r"\bfair value (measurement|gain|loss)?\b", re.I),
    "valuation": re.compile(r"\bvaluation\b", re.I),
    "deferred_tax": re.compile(r"\bdeferred tax\b", re.I),
    "borrowings": re.compile(r"\b(borrowings|interest-bearing liabilities)\b", re.I),
    "share_capital": re.compile(r"\bshare capital\b", re.I),
    "eps": re.compile(r"\bearn(?:ings)? per share\b", re.I),
}


METRIC_RULES = [
    (re.compile(r"fair value gain", re.I), "fair_value_gain_loss"),
    (re.compile(r"investment properties", re.I), "investment_properties_fair_value_total"),
    (re.compile(r"cap rate", re.I), "valuation_cap_rate_low"),
    (re.compile(r"discount rate", re.I), "discount_rate_low"),
    (re.compile(r"rental income", re.I), "rental_income"),
    (re.compile(r"deferred tax", re.I), "deferred_tax_revaluation_related"),
    (re.compile(r"valuer|valued by", re.I), "valuer_name"),
]


def _extract_scalar_number(line: str) -> float | None:
    vals = re.findall(r"\(?-?\d[\d,]*\.?\d*\)?", line)
    if not vals:
        return None
    tok = vals[-1].replace(",", "")
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()")
    try:
        x = float(tok)
    except ValueError:
        return None
    return -x if neg else x


def extract_notes_for_filing(row: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    pdf_path = row.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return pd.DataFrame(), pd.DataFrame(), []
    pages = _try_extract_pdf_text(Path(pdf_path))
    text_chunks: list[dict] = []
    anchors: list[dict] = []
    facts: list[dict] = []
    active_groups: set[str] = set()
    for page_no, txt in pages:
        text_chunks.append(
            {
                "filing_id": row["filing_id"],
                "page_no": page_no,
                "text": txt,
            }
        )
        for group, pat in NOTE_HEADINGS.items():
            if pat.search(txt):
                anchors.append(
                    {
                        "filing_id": row["filing_id"],
                        "page_no": page_no,
                        "note_group": group,
                        "matched_text": pat.pattern,
                        "confidence": 0.8,
                    }
                )
                active_groups.add(group)
        for line in txt.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            for pat, metric_code in METRIC_RULES:
                if not pat.search(line_s):
                    continue
                num = _extract_scalar_number(line_s)
                facts.append(
                    {
                        "filing_id": row["filing_id"],
                        "table_or_note": "note",
                        "raw_label": line_s,
                        "candidate_canonical_code": metric_code,
                        "value": num if num is not None else line_s[:250],
                        "page_no": page_no,
                        "confidence": 0.35 if num is not None else 0.25,
                        "extractor_name": "pdf_text_note_regex",
                        "source_kind": "text_extract",
                        "statement_type": None,
                        "currency": None,
                        "unit_scale": None,
                        "row_idx": None,
                        "col_idx": None,
                        "raw_table_id": None,
                    }
                )
    return pd.DataFrame(text_chunks), pd.DataFrame(anchors), facts


def persist_note_artifacts(row: dict, text_chunks: pd.DataFrame, anchors: pd.DataFrame) -> tuple[Path | None, Path | None]:
    paths = ensure_directories(default_paths())
    period_key = str(row.get("period_end") or "unknown_period")
    txt_out = anc_out = None
    if not text_chunks.empty:
        txt_out = paths.extracted_root / "text_chunks" / str(row["stock_code"]) / period_key / f"{row['doc_id']}.jsonl"
        txt_out.parent.mkdir(parents=True, exist_ok=True)
        with txt_out.open("w", encoding="utf-8") as f:
            for rec in text_chunks.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if not anchors.empty:
        anc_out = paths.extracted_root / "anchors" / str(row["stock_code"]) / period_key / f"{row['doc_id']}.json"
        anc_out.parent.mkdir(parents=True, exist_ok=True)
        anc_out.write_text(json.dumps(anchors.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    return txt_out, anc_out

