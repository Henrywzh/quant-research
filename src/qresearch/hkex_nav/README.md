# HKEX NAV Pipeline (v1)

This package implements a point-in-time, auditable NAV pipeline for:

- `00016` SHKP
- `01113` CK Asset
- `00012` Henderson Land

## Quick Start

This repo uses a `src/` layout and is not packaged yet, so run with:

```bash
PYTHONPATH=src python -m qresearch.hkex_nav.cli index
```

## Seed Filing Index Input (v1)

Place manual seed files in:

- `data/raw/hkex_nav/filing_index/seeds/00016.csv`
- `data/raw/hkex_nav/filing_index/seeds/01113.csv`
- `data/raw/hkex_nav/filing_index/seeds/00012.csv`

CSV columns (minimum):

- `title`
- `publish_datetime_hk` (ISO-like string)
- `pdf_url`

Optional:

- `doc_type`
- `period_end`
- `issuer_name`
- `hkex_url`
- `doc_id`

## Commands

```bash
PYTHONPATH=src python -m qresearch.hkex_nav.cli index
PYTHONPATH=src python -m qresearch.hkex_nav.cli download
PYTHONPATH=src python -m qresearch.hkex_nav.cli extract
PYTHONPATH=src python -m qresearch.hkex_nav.cli qa-export
PYTHONPATH=src python -m qresearch.hkex_nav.cli curate
PYTHONPATH=src python -m qresearch.hkex_nav.cli model
```

## Notes

- `download` supports local file paths in `pdf_url` for offline testing.
- PDF parsing uses `pdfplumber` if installed; otherwise extraction artifacts will be sparse/empty and can be completed via QA overrides.
- Curated/model stages require `duckdb`.

