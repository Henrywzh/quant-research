from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from .config import ensure_directories, default_paths


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _pdf_output_path(stock_code: str, doc_id: str, publish_date: str, doc_type: str, language: str, raw_root: Path) -> Path:
    safe_doc_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in doc_id)
    fn = f"{safe_doc_id}_{publish_date}_{doc_type}_{language}.pdf"
    return raw_root / "pdfs" / stock_code / fn


def _meta_output_path(stock_code: str, doc_id: str, raw_root: Path) -> Path:
    safe_doc_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in doc_id)
    return raw_root / "metadata" / stock_code / f"{safe_doc_id}.json"


def _download_bytes(url: str, timeout_sec: int = 60) -> tuple[bytes, dict]:
    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        p = Path(parsed.path if parsed.scheme == "file" else url)
        return p.read_bytes(), {"source": "local_file"}
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (qresearch hkex_nav pipeline)"})
    with urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read()
        headers = {k.lower(): v for k, v in resp.headers.items()}
    return data, headers


def download_pdfs(index_df: pd.DataFrame, force: bool = False, dry_run: bool = False) -> pd.DataFrame:
    paths = ensure_directories(default_paths())
    df = index_df.copy()
    if df.empty:
        return df
    for idx, row in df.iterrows():
        pdf_url = row.get("pdf_url")
        if not pdf_url:
            continue
        publish_date = str(row.get("publish_datetime_hk", ""))[:10] or "unknown-date"
        out_pdf = _pdf_output_path(str(row["stock_code"]), str(row["doc_id"]), publish_date, str(row["doc_type"]), str(row.get("language", "EN")), paths.raw_hkex_root)
        out_meta = _meta_output_path(str(row["stock_code"]), str(row["doc_id"]), paths.raw_hkex_root)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        if out_pdf.exists() and not force:
            pdf_bytes = out_pdf.read_bytes()
            digest = sha256_bytes(pdf_bytes)
            df.at[idx, "pdf_path"] = str(out_pdf)
            df.at[idx, "pdf_sha256"] = digest
            df.at[idx, "file_size_bytes"] = len(pdf_bytes)
            df.at[idx, "downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
            continue
        if dry_run:
            df.at[idx, "pdf_path"] = str(out_pdf)
            continue
        data, headers = _download_bytes(str(pdf_url))
        out_pdf.write_bytes(data)
        digest = sha256_bytes(data)
        (out_pdf.with_suffix(out_pdf.suffix + ".sha256")).write_text(digest + "\n", encoding="utf-8")
        meta = {
            "stock_code": row.get("stock_code"),
            "doc_id": row.get("doc_id"),
            "pdf_url": pdf_url,
            "hkex_url": row.get("hkex_url"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "pdf_sha256": digest,
            "file_size_bytes": len(data),
            "response_headers": headers,
        }
        out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        df.at[idx, "pdf_path"] = str(out_pdf)
        df.at[idx, "pdf_sha256"] = digest
        df.at[idx, "file_size_bytes"] = len(data)
        df.at[idx, "downloaded_at_utc"] = meta["downloaded_at_utc"]
    return df

