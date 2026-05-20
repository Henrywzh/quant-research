from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd

from .config import ensure_directories, default_paths


TABLE_LABEL_RULES: list[tuple[str, str, str]] = [
    (r"equity attributable to", "equity_attributable_to_owners", "balance_sheet"),
    (r"non-controlling interests", "non_controlling_interests", "balance_sheet"),
    (r"\btotal equity\b", "total_equity", "balance_sheet"),
    (r"cash and bank balances", "cash_and_bank_balances", "balance_sheet"),
    (r"\bborrowings\b", None, "balance_sheet"),
]


def _try_extract_pdf_text(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return pages
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                pages.append((i, txt))
    except Exception:
        return []
    return pages


def _guess_numeric(line: str) -> float | None:
    nums = re.findall(r"\(?-?\d[\d,]*\.?\d*\)?", line)
    if not nums:
        return None
    token = nums[-1].replace(",", "")
    neg = token.startswith("(") and token.endswith(")")
    token = token.strip("()")
    try:
        v = float(token)
    except ValueError:
        return None
    return -v if neg else v


def extract_tables_for_filing(row: dict) -> tuple[pd.DataFrame, list[dict]]:
    pdf_path = row.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return pd.DataFrame(), []
    pages = _try_extract_pdf_text(Path(pdf_path))
    facts = []
    raw_rows = []
    for page_no, txt in pages:
        for ln_idx, line in enumerate(txt.splitlines()):
            line_l = line.lower().strip()
            if not line_l:
                continue
            for pat, code, statement_type in TABLE_LABEL_RULES:
                if re.search(pat, line_l):
                    val = _guess_numeric(line)
                    raw_rows.append(
                        {
                            "filing_id": row["filing_id"],
                            "page_no": page_no,
                            "row_idx": ln_idx,
                            "raw_table_id": f"{row['filing_id']}_p{page_no}",
                            "line_text": line.strip(),
                            "statement_type": statement_type,
                            "candidate_canonical_code": code,
                            "value": val,
                            "extract_confidence": 0.45 if val is not None else 0.2,
                            "currency": None,
                            "unit_scale": None,
                            "source_kind": "table_extract",
                        }
                    )
                    if code and val is not None:
                        facts.append(
                            {
                                "filing_id": row["filing_id"],
                                "table_or_note": "table",
                                "raw_label": line.strip(),
                                "candidate_canonical_code": code,
                                "value": val,
                                "page_no": page_no,
                                "confidence": 0.45,
                                "extractor_name": "pdf_text_table_heuristic",
                                "source_kind": "table_extract",
                                "statement_type": statement_type,
                                "currency": None,
                                "unit_scale": None,
                                "row_idx": ln_idx,
                                "col_idx": None,
                                "raw_table_id": f"{row['filing_id']}_p{page_no}",
                            }
                        )
    return pd.DataFrame(raw_rows), facts


def persist_table_artifacts(row: dict, table_rows: pd.DataFrame) -> Path | None:
    if table_rows.empty:
        return None
    paths = ensure_directories(default_paths())
    period_key = str(row.get("period_end") or "unknown_period")
    out = paths.extracted_root / "tables" / str(row["stock_code"]) / period_key / f"{row['doc_id']}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        table_rows.to_parquet(out, index=False)
    except Exception:
        out = out.with_suffix(".csv")
        table_rows.to_csv(out, index=False)
    return out

