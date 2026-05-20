from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Any, Literal


DocType = Literal["annual_report", "annual_results", "interim_report", "interim_results"]
SourceKind = Literal["table_extract", "text_extract", "manual_override", "default_rule"]


@dataclass(frozen=True)
class IssuerConfig:
    stock_code: str
    issuer_name: str
    ticker_for_price: str
    fiscal_year_end_month: int | None = 6
    enabled_doc_types: tuple[DocType, ...] = (
        "annual_report",
        "annual_results",
        "interim_report",
        "interim_results",
    )
    lang_policy: str = "EN"


@dataclass
class FilingRecord:
    filing_id: str
    stock_code: str
    issuer_name: str
    doc_id: str
    doc_type: str
    period_end: str | None
    fiscal_year: int | None
    fiscal_half: str | None
    publish_datetime_hk: str
    title: str
    language: str
    hkex_url: str | None = None
    pdf_url: str | None = None
    pdf_path: str | None = None
    pdf_sha256: str | None = None
    file_size_bytes: int | None = None
    discovery_run_id: str | None = None
    downloaded_at_utc: str | None = None
    parse_status: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedFact:
    filing_id: str
    table_or_note: str
    raw_label: str
    candidate_canonical_code: str | None
    value: float | str | None
    page_no: int | None
    bbox_json: str | None = None
    confidence: float | None = None
    extractor_name: str | None = None
    source_kind: str = "text_extract"
    statement_type: str | None = None
    currency: str | None = None
    unit_scale: str | None = None
    row_idx: int | None = None
    col_idx: int | None = None
    raw_table_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedFact:
    resolved_value: float | str | None
    resolved_source: Literal["extract", "override"]
    lineage_json: str


@dataclass
class NavScenarioResult:
    stock_code: str
    period_end: str
    scenario: str
    nav_per_share: float
    price_date_hk: str | None
    discount_premium_pct: float | None
    components_json: str = "{}"
    filing_id: str | None = None


@dataclass
class RunContext:
    issuers: list[str]
    start_year: int
    end_year: int
    lang: str = "EN"
    force: bool = False
    dry_run: bool = False
    run_id: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))


def parse_date_str(x: str | None) -> date | None:
    if not x:
        return None
    return date.fromisoformat(x[:10])

