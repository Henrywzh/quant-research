from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DOC_TYPE_PATTERNS, NON_TARGET_TITLE_PATTERNS, DEFAULT_ISSUERS, ensure_directories, default_paths
from .types import FilingRecord, RunContext


PERIOD_PATTERNS = [
    re.compile(r"(?:for|ended)\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", re.I),
    re.compile(r"(\d{4})[/-](\d{2})[/-](\d{2})"),
]


class HKEXIndexer(ABC):
    @abstractmethod
    def discover(self, stock_code: str, start_year: int, end_year: int, language: str = "EN") -> list[dict]:
        raise NotImplementedError


class LocalSeedIndexer(HKEXIndexer):
    """
    Reads manually prepared seed files.
    Supported locations:
      data/raw/hkex_nav/filing_index/seeds/{stock_code}.csv|jsonl
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        p = ensure_directories(default_paths())
        self.seed_dir = (base_dir or p.raw_hkex_root / "filing_index" / "seeds")
        self.seed_dir.mkdir(parents=True, exist_ok=True)

    def discover(self, stock_code: str, start_year: int, end_year: int, language: str = "EN") -> list[dict]:
        out: list[dict] = []
        for fp in [self.seed_dir / f"{stock_code}.csv", self.seed_dir / f"{stock_code}.jsonl"]:
            if not fp.exists():
                continue
            if fp.suffix == ".csv":
                df = pd.read_csv(fp)
                out.extend(df.to_dict(orient="records"))
            else:
                with fp.open("r", encoding="utf-8") as f:
                    out.extend(json.loads(line) for line in f if line.strip())
        for r in out:
            r.setdefault("stock_code", stock_code)
            r.setdefault("language", language)
        return out


def classify_doc_type(title: str) -> str | None:
    t = (title or "").strip()
    if not t:
        return None
    for pat in NON_TARGET_TITLE_PATTERNS:
        if pat.search(t):
            return None
    for doc_type, pat in DOC_TYPE_PATTERNS:
        if pat.search(t):
            return doc_type
    return None


def parse_period_end_from_title(title: str) -> str | None:
    if not title:
        return None
    for pat in PERIOD_PATTERNS:
        m = pat.search(title)
        if not m:
            continue
        if len(m.groups()) == 1:
            txt = m.group(1)
            for fmt in ("%d %B %Y", "%d %b %Y"):
                try:
                    return datetime.strptime(txt, fmt).date().isoformat()
                except ValueError:
                    pass
        elif len(m.groups()) == 3:
            y, mo, d = m.groups()
            try:
                return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
            except ValueError:
                return None
    return None


def _make_filing_id(stock_code: str, publish_datetime_hk: str, title: str) -> str:
    h = hashlib.sha1(f"{stock_code}|{publish_datetime_hk}|{title}".encode("utf-8")).hexdigest()[:16]
    return f"{stock_code}_{h}"


def _normalize_seed_row(row: dict, run_id: str) -> FilingRecord | None:
    title = str(row.get("title") or "").strip()
    doc_type = row.get("doc_type") or classify_doc_type(title)
    if not doc_type:
        return None
    publish_dt = str(row.get("publish_datetime_hk") or row.get("publish_datetime") or "").strip()
    if not publish_dt:
        # fallback for seeds with publish_date
        publish_date = str(row.get("publish_date") or "").strip()
        if publish_date:
            publish_dt = f"{publish_date}T00:00:00"
        else:
            return None
    period_end = row.get("period_end") or parse_period_end_from_title(title)
    fy = None
    fh = None
    if period_end:
        fy = int(str(period_end)[:4])
        month = int(str(period_end)[5:7])
        fh = "H1" if month in (6,) else "FY"
    stock_code = str(row.get("stock_code") or "").zfill(5)
    issuer_name = row.get("issuer_name") or DEFAULT_ISSUERS.get(stock_code, DEFAULT_ISSUERS["00016"]).issuer_name
    filing_id = row.get("filing_id") or _make_filing_id(stock_code, publish_dt, title)
    doc_id = str(row.get("doc_id") or filing_id)
    return FilingRecord(
        filing_id=filing_id,
        stock_code=stock_code,
        issuer_name=str(issuer_name),
        doc_id=doc_id,
        doc_type=str(doc_type),
        period_end=str(period_end) if period_end else None,
        fiscal_year=fy,
        fiscal_half=fh,
        publish_datetime_hk=publish_dt,
        title=title,
        language=str(row.get("language") or "EN"),
        hkex_url=row.get("hkex_url"),
        pdf_url=row.get("pdf_url"),
        discovery_run_id=run_id,
        parse_status="indexed",
        notes=row.get("notes"),
    )


def build_filing_index(indexer: HKEXIndexer, ctx: RunContext, out_path: Path | None = None) -> pd.DataFrame:
    paths = ensure_directories(default_paths())
    out_path = out_path or paths.index_jsonl
    rows: list[dict] = []
    for sc in ctx.issuers:
        discovered = indexer.discover(sc, ctx.start_year, ctx.end_year, language=ctx.lang)
        for raw in discovered:
            fr = _normalize_seed_row(raw, ctx.run_id)
            if not fr:
                continue
            rows.append(fr.to_dict())
    if not rows:
        df = pd.DataFrame(columns=[f.name for f in FilingRecord.__dataclass_fields__.values()])
    else:
        df = pd.DataFrame(rows).drop_duplicates(
            subset=["stock_code", "title", "publish_datetime_hk", "language"], keep="last"
        )
        df = df.sort_values(["stock_code", "publish_datetime_hk", "title"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return df


def load_filing_index(index_path: Path | None = None) -> pd.DataFrame:
    paths = ensure_directories(default_paths())
    fp = index_path or paths.index_jsonl
    if not fp.exists():
        return pd.DataFrame()
    recs = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return pd.DataFrame(recs)

