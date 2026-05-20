from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import re

from qresearch.data.utils import get_raw_dir, get_processed_dir
from .types import IssuerConfig


DEFAULT_LANGUAGE = "EN"
HK_TIMEZONE_NAME = "Asia/Hong_Kong"

DEFAULT_ISSUERS: dict[str, IssuerConfig] = {
    "00016": IssuerConfig("00016", "Sun Hung Kai Properties Limited", "0016.HK", fiscal_year_end_month=6),
    "01113": IssuerConfig("01113", "CK Asset Holdings Limited", "1113.HK", fiscal_year_end_month=12),
    "00012": IssuerConfig("00012", "Henderson Land Development Company Limited", "0012.HK", fiscal_year_end_month=12),
}

MANUAL_INPUT_MANIFEST_NAME = "manual_inputs_manifest.json"


DOC_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("annual_report", re.compile(r"\bannual report\b", re.I)),
    ("interim_report", re.compile(r"\binterim report\b", re.I)),
    ("annual_results", re.compile(r"\b(annual|final)\s+results\b", re.I)),
    ("interim_results", re.compile(r"\binterim results\b", re.I)),
]

NON_TARGET_TITLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bnotice of\b", re.I),
    re.compile(r"\bproxy\b", re.I),
]


@dataclass(frozen=True)
class Paths:
    raw_root: Path
    processed_root: Path
    raw_hkex_root: Path
    extracted_root: Path
    curated_root: Path
    model_root: Path
    exports_root: Path
    config_root: Path
    index_jsonl: Path
    duckdb_path: Path


def default_paths() -> Paths:
    raw_root = get_raw_dir()
    processed_root = get_processed_dir()
    raw_hkex_root = raw_root / "hkex_nav"
    proc_hkex_root = processed_root / "hkex_nav"
    extracted_root = proc_hkex_root / "extracted"
    curated_root = proc_hkex_root / "curated"
    model_root = proc_hkex_root / "model"
    exports_root = proc_hkex_root / "exports"
    config_root = proc_hkex_root / "config"
    return Paths(
        raw_root=raw_root,
        processed_root=processed_root,
        raw_hkex_root=raw_hkex_root,
        extracted_root=extracted_root,
        curated_root=curated_root,
        model_root=model_root,
        exports_root=exports_root,
        config_root=config_root,
        index_jsonl=raw_hkex_root / "filing_index" / "filings_index.jsonl",
        duckdb_path=curated_root / "hkex_nav.duckdb",
    )


def ensure_directories(paths: Paths | None = None) -> Paths:
    p = paths or default_paths()
    for d in [
        p.raw_hkex_root,
        p.raw_hkex_root / "filing_index",
        p.raw_hkex_root / "pdfs",
        p.raw_hkex_root / "metadata",
        p.extracted_root / "tables",
        p.extracted_root / "text_chunks",
        p.extracted_root / "anchors",
        p.extracted_root / "extraction_logs",
        p.curated_root,
        p.model_root,
        p.exports_root,
        p.config_root,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    _ensure_mapping_templates(p.config_root)
    _ensure_manual_input_manifest(p.config_root)
    _validate_manual_input_manifest(p.config_root)
    return p


def _ensure_mapping_templates(config_root: Path) -> None:
    templates = {
        "issuers.csv": [
            ["issuer_code", "issuer_name", "ticker_for_price", "fiscal_year_end_month", "lang_policy"],
            ["00016", "Sun Hung Kai Properties Limited", "0016.HK", "6", "EN"],
            ["01113", "CK Asset Holdings Limited", "1113.HK", "12", "EN"],
            ["00012", "Henderson Land Development Company Limited", "0012.HK", "12", "EN"],
        ],
        "line_item_aliases.csv": [
            ["issuer_code", "pattern", "canonical_code", "statement_type", "scope_hint", "priority"],
            ["*", "equity attributable to owners", "equity_attributable_to_owners", "balance_sheet", "attributable_owners", "100"],
            ["*", "cash and bank balances", "cash_and_bank_balances", "balance_sheet", "group", "100"],
        ],
        "note_heading_aliases.csv": [
            ["issuer_code", "pattern", "note_group", "priority"],
            ["*", "investment properties", "investment_properties", "100"],
            ["*", "deferred tax", "deferred_tax", "100"],
        ],
        "issuer_overrides.csv": [
            ["issuer_code", "field", "value", "notes"],
            ["00016", "fiscal_year_end_month", "6", ""],
            ["01113", "fiscal_year_end_month", "12", ""],
            ["00012", "fiscal_year_end_month", "12", ""],
        ],
    }
    for name, rows in templates.items():
        fp = config_root / name
        if fp.exists():
            continue
        with fp.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)


def _ensure_manual_input_manifest(config_root: Path) -> None:
    manifest_path = config_root / MANUAL_INPUT_MANIFEST_NAME
    if manifest_path.exists():
        return
    payload = {
        "manual_inputs": [
            {
                "name": "issuers.csv",
                "owner": "research",
                "source_note": "Maintained issuer universe for HKEX NAV workflow.",
                "last_updated": datetime.utcnow().date().isoformat(),
                "validation_status": "seeded",
                "downstream_datasets": ["hkex_nav_prices_daily", "filings_index"],
            },
            {
                "name": "line_item_aliases.csv",
                "owner": "research",
                "source_note": "Manual canonical mapping rules for extraction normalization.",
                "last_updated": datetime.utcnow().date().isoformat(),
                "validation_status": "seeded",
                "downstream_datasets": ["statement_line_items"],
            },
            {
                "name": "note_heading_aliases.csv",
                "owner": "research",
                "source_note": "Manual note-group alias mapping.",
                "last_updated": datetime.utcnow().date().isoformat(),
                "validation_status": "seeded",
                "downstream_datasets": ["investment_property_notes"],
            },
            {
                "name": "issuer_overrides.csv",
                "owner": "research",
                "source_note": "Explicit issuer-level overrides with audit trail.",
                "last_updated": datetime.utcnow().date().isoformat(),
                "validation_status": "seeded",
                "downstream_datasets": ["rnav_assumptions"],
            },
        ]
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validate_manual_input_manifest(config_root: Path) -> None:
    manifest_path = config_root / MANUAL_INPUT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manual input manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in payload.get("manual_inputs", []):
        required = {"name", "owner", "source_note", "last_updated", "validation_status", "downstream_datasets"}
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"Manual input manifest entry missing keys {missing}: {item}")


def default_year_window(years: int = 10) -> tuple[int, int]:
    now = datetime.now()
    return now.year - years + 1, now.year
