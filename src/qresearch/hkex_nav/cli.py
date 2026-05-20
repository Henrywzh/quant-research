from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import DEFAULT_ISSUERS, DEFAULT_LANGUAGE, ensure_directories, default_paths, default_year_window
from .types import RunContext
from .index_hkex import LocalSeedIndexer, build_filing_index, load_filing_index
from .download import download_pdfs
from .extract_tables import extract_tables_for_filing, persist_table_artifacts
from .extract_notes import extract_notes_for_filing, persist_note_artifacts
from .normalize import load_extracted_facts, build_curated_tables
from .qa_review import export_review_sheets, import_overrides_from_csv
from .duckdb_store import initialize_db, load_tables_to_duckdb
from .model_nav import compute_nav_outputs, build_rnav_assumptions
from .pricing import fetch_or_load_price_cache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ys, ye = default_year_window(10)
    p = argparse.ArgumentParser(prog="python -m qresearch.hkex_nav.cli")
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--issuers", default=",".join(DEFAULT_ISSUERS.keys()))
    common.add_argument("--start-year", type=int, default=ys)
    common.add_argument("--end-year", type=int, default=ye)
    common.add_argument("--lang", default=DEFAULT_LANGUAGE)
    common.add_argument("--force", action="store_true")
    common.add_argument("--dry-run", action="store_true")

    sub.add_parser("index", parents=[common])
    sub.add_parser("download", parents=[common])
    sub.add_parser("extract", parents=[common])
    qx = sub.add_parser("qa-export", parents=[common])
    qx.add_argument("--from-db", action="store_true")
    qi = sub.add_parser("qa-import", parents=[common])
    qi.add_argument("--csv", action="append", default=[])
    sub.add_parser("curate", parents=[common])
    sub.add_parser("model", parents=[common])
    sub.add_parser("run-all", parents=[common])
    return p.parse_args(argv)


def _ctx_from_args(args: argparse.Namespace) -> RunContext:
    issuers = [x.strip().zfill(5) for x in str(args.issuers).split(",") if x.strip()]
    return RunContext(
        issuers=issuers,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        lang=str(args.lang),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )


def cmd_index(ctx: RunContext) -> pd.DataFrame:
    ensure_directories(default_paths())
    indexer = LocalSeedIndexer()
    df = build_filing_index(indexer, ctx)
    print(f"[index] wrote {len(df)} records to filing index")
    return df


def _save_index_back(df: pd.DataFrame) -> None:
    paths = ensure_directories(default_paths())
    with paths.index_jsonl.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_download(ctx: RunContext) -> pd.DataFrame:
    idx = load_filing_index()
    if idx.empty:
        idx = cmd_index(ctx)
    idx = idx[idx["stock_code"].astype(str).isin(ctx.issuers)].copy()
    out = download_pdfs(idx, force=ctx.force, dry_run=ctx.dry_run)
    _save_index_back(out)
    print(f"[download] processed {len(out)} indexed records")
    return out


def cmd_extract(ctx: RunContext) -> pd.DataFrame:
    idx = load_filing_index()
    if idx.empty:
        print("[extract] no filing index found; run index/download first")
        return idx
    idx = idx[idx["stock_code"].astype(str).isin(ctx.issuers)].copy()
    all_facts = []
    log_rows = []
    for row in idx.to_dict(orient="records"):
        tr, table_facts = extract_tables_for_filing(row)
        persist_table_artifacts(row, tr)
        txt, anc, note_facts = extract_notes_for_filing(row)
        persist_note_artifacts(row, txt, anc)
        all_facts.extend(table_facts)
        all_facts.extend(note_facts)
        log_rows.append(
            {
                "filing_id": row["filing_id"],
                "doc_id": row["doc_id"],
                "table_rows": len(tr),
                "note_chunks": len(txt),
                "anchors": len(anc),
                "fact_candidates": len(table_facts) + len(note_facts),
            }
        )
    paths = ensure_directories(default_paths())
    log_fp = paths.extracted_root / "extraction_logs" / f"{ctx.run_id}.jsonl"
    with log_fp.open("w", encoding="utf-8") as f:
        for rec in log_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[extract] logged {len(log_rows)} filings -> {log_fp}")
    return idx


def _load_overrides_from_exports() -> pd.DataFrame:
    paths = ensure_directories(default_paths())
    qa_dir = paths.exports_root / "qa_review_sheets"
    if not qa_dir.exists():
        return pd.DataFrame()
    csvs = sorted(str(p) for p in qa_dir.glob("*.csv"))
    return import_overrides_from_csv(csvs)


def cmd_curate(ctx: RunContext) -> dict[str, pd.DataFrame]:
    idx = load_filing_index()
    if idx.empty:
        raise RuntimeError("No filing index found. Run `index` first.")
    idx = idx[idx["stock_code"].astype(str).isin(ctx.issuers)].copy()
    extracted = load_extracted_facts(idx)
    overrides = _load_overrides_from_exports()
    tables = build_curated_tables(idx, extracted, overrides=overrides)
    db = load_tables_to_duckdb(tables, replace=True)
    print(f"[curate] loaded curated tables into {db}")
    return tables


def cmd_qa_export(ctx: RunContext) -> list[Path]:
    tables = cmd_curate(ctx)
    idx = tables["filings_index"]
    outputs = export_review_sheets(idx, tables)
    print(f"[qa-export] wrote {len(outputs)} files")
    return outputs


def cmd_qa_import(ctx: RunContext, csv_paths: list[str]) -> pd.DataFrame:
    if not csv_paths:
        raise RuntimeError("Pass at least one --csv path, or use qa-export then curate/model.")
    ov = import_overrides_from_csv(csv_paths)
    if ov.empty:
        print("[qa-import] no overrides imported")
        return ov
    initialize_db()
    load_tables_to_duckdb({"qa_overrides": ov}, replace=False)
    print(f"[qa-import] imported {len(ov)} overrides")
    return ov


def _write_model_exports(nav_outputs: pd.DataFrame, assumptions: pd.DataFrame) -> None:
    paths = ensure_directories(default_paths())
    model_dir = paths.model_root
    exports_dir = paths.exports_root
    model_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    book = nav_outputs[nav_outputs["scenario"] == "book"].copy() if not nav_outputs.empty else pd.DataFrame()
    rnav = nav_outputs[nav_outputs["scenario"].str.startswith("rnav_", na=False)].copy() if not nav_outputs.empty else pd.DataFrame()
    disc = nav_outputs[["stock_code","period_end","scenario","price_date_hk","price_close","nav_per_share","discount_premium_pct"]].copy() if not nav_outputs.empty else pd.DataFrame()
    for fp, df in [
        (exports_dir / "book_nav_timeseries.parquet", book),
        (exports_dir / "rnav_timeseries.parquet", rnav),
        (exports_dir / "discount_to_nav_timeseries.parquet", disc),
    ]:
        try:
            df.to_parquet(fp, index=False)
        except Exception:
            df.to_csv(fp.with_suffix(".csv"), index=False)


def cmd_model(ctx: RunContext) -> pd.DataFrame:
    tables = cmd_curate(ctx)
    price_df = fetch_or_load_price_cache(tables["filings_index"], force=ctx.force)
    assumptions = build_rnav_assumptions(tables["filings_index"], tables.get("qa_overrides"))
    nav_outputs, assumptions = compute_nav_outputs(
        filings_index=tables["filings_index"],
        statement_line_items=tables["statement_line_items"],
        capital_structure=tables["capital_structure"],
        investment_property_notes=tables["investment_property_notes"],
        price_df=price_df,
        assumptions_df=assumptions,
    )
    load_tables_to_duckdb({"rnav_assumptions": assumptions, "nav_outputs": nav_outputs}, replace=True)
    _write_model_exports(nav_outputs, assumptions)
    print(f"[model] computed {len(nav_outputs)} nav rows")
    return nav_outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = _ctx_from_args(args)
    ensure_directories(default_paths())
    if args.command == "index":
        cmd_index(ctx)
    elif args.command == "download":
        cmd_download(ctx)
    elif args.command == "extract":
        cmd_extract(ctx)
    elif args.command == "qa-export":
        cmd_qa_export(ctx)
    elif args.command == "qa-import":
        cmd_qa_import(ctx, list(args.csv))
    elif args.command == "curate":
        cmd_curate(ctx)
    elif args.command == "model":
        cmd_model(ctx)
    elif args.command == "run-all":
        cmd_index(ctx)
        cmd_download(ctx)
        cmd_extract(ctx)
        cmd_curate(ctx)
        cmd_qa_export(ctx)
        cmd_model(ctx)
    else:
        raise SystemExit(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

