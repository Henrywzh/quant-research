from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .config import ensure_directories, default_paths


DDL = {
    "filings_index": """
        create table if not exists filings_index (
            filing_id varchar primary key,
            stock_code varchar,
            issuer_name varchar,
            doc_id varchar,
            doc_type varchar,
            period_end date,
            fiscal_year integer,
            fiscal_half varchar,
            publish_datetime_hk timestamp,
            title varchar,
            language varchar,
            hkex_url varchar,
            pdf_url varchar,
            pdf_path varchar,
            pdf_sha256 varchar unique,
            file_size_bytes bigint,
            discovery_run_id varchar,
            downloaded_at_utc timestamp,
            parse_status varchar,
            notes varchar
        )
    """,
    "statement_line_items": """
        create table if not exists statement_line_items (
            fact_id varchar primary key,
            filing_id varchar,
            stock_code varchar,
            period_end date,
            statement_type varchar,
            scope varchar,
            line_item_code varchar,
            line_item_label_raw varchar,
            value double,
            currency varchar,
            unit_scale varchar,
            as_reported_sign integer,
            page_no integer,
            source_kind varchar,
            extract_confidence double,
            raw_table_id varchar,
            row_idx integer,
            col_idx integer
        )
    """,
    "capital_structure": """
        create table if not exists capital_structure (
            capital_fact_id varchar primary key,
            filing_id varchar,
            stock_code varchar,
            period_end date,
            shares_outstanding_period_end double,
            weighted_avg_shares_basic double,
            weighted_avg_shares_diluted double,
            treasury_shares double,
            page_no integer,
            source_kind varchar,
            extract_confidence double,
            notes varchar
        )
    """,
    "debt_schedule": """
        create table if not exists debt_schedule (
            debt_fact_id varchar primary key,
            filing_id varchar,
            stock_code varchar,
            period_end date,
            debt_type varchar,
            current_noncurrent varchar,
            maturity_bucket varchar,
            secured_flag boolean,
            currency varchar,
            amount double,
            page_no integer,
            source_kind varchar
        )
    """,
    "investment_property_notes": """
        create table if not exists investment_property_notes (
            ip_note_fact_id varchar primary key,
            filing_id varchar,
            stock_code varchar,
            period_end date,
            metric_code varchar,
            metric_label_raw varchar,
            value_num double,
            value_text varchar,
            currency varchar,
            property_segment varchar,
            geography varchar,
            page_no integer,
            source_kind varchar,
            extract_confidence double
        )
    """,
    "qa_overrides": """
        create table if not exists qa_overrides (
            override_id varchar primary key,
            target_table varchar,
            target_key_json varchar,
            field_name varchar,
            override_value_num double,
            override_value_text varchar,
            override_reason varchar,
            reviewer varchar,
            reviewed_at_utc timestamp,
            status varchar,
            source_csv_path varchar
        )
    """,
    "rnav_assumptions": """
        create table if not exists rnav_assumptions (
            assumption_set_id varchar primary key,
            stock_code varchar,
            period_end date,
            scenario varchar,
            ip_adjustment_method varchar,
            cap_rate_shift_bps double,
            ip_value_uplift_pct double,
            dev_hidden_reserve_pct double,
            associate_jv_uplift_pct double,
            deferred_tax_haircut_pct double,
            minority_adjustment_mode varchar,
            notes varchar,
            source_kind varchar
        )
    """,
    "nav_outputs": """
        create table if not exists nav_outputs (
            nav_output_id varchar primary key,
            stock_code varchar,
            period_end date,
            filing_id varchar,
            publish_datetime_hk timestamp,
            scenario varchar,
            nav_per_share double,
            shares_used double,
            equity_base_used double,
            net_debt_used double,
            price_date_hk date,
            price_close double,
            discount_premium_pct double,
            currency varchar,
            calc_version varchar,
            assumption_set_id varchar,
            lineage_json varchar
        )
    """,
}


def _connect(db_path: Path | None = None):
    try:
        import duckdb  # type: ignore
    except Exception as e:
        raise RuntimeError("duckdb is required for curated/model storage. Install `duckdb`.") from e
    paths = ensure_directories(default_paths())
    return duckdb.connect(str(db_path or paths.duckdb_path))


def initialize_db(db_path: Path | None = None) -> Path:
    paths = ensure_directories(default_paths())
    db = db_path or paths.duckdb_path
    con = _connect(db)
    try:
        for sql in DDL.values():
            con.execute(sql)
    finally:
        con.close()
    return db


def load_tables_to_duckdb(tables: dict[str, pd.DataFrame], db_path: Path | None = None, replace: bool = True) -> Path:
    db = initialize_db(db_path)
    con = _connect(db)
    try:
        for name, df in tables.items():
            if name not in DDL:
                continue
            if df is None:
                continue
            if replace:
                con.execute(f"delete from {name}")
            if df.empty:
                continue
            tmp = df.copy()
            con.register("tmp_df", tmp)
            cols = ", ".join(tmp.columns)
            con.execute(f"insert into {name} ({cols}) select {cols} from tmp_df")
            con.unregister("tmp_df")
    finally:
        con.close()
    return db


def read_table(name: str, db_path: Path | None = None) -> pd.DataFrame:
    con = _connect(db_path)
    try:
        return con.sql(f"select * from {name}").df()
    finally:
        con.close()

