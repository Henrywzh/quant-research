from __future__ import annotations

import pandas as pd

from qresearch.hkex_nav.index_hkex import classify_doc_type, parse_period_end_from_title
from qresearch.hkex_nav.normalize import parse_currency, parse_unit_scale, canonical_line_item
from qresearch.hkex_nav.pricing import next_trading_close_after_publish
from qresearch.hkex_nav.model_nav import compute_nav_outputs


def test_classify_doc_type():
    assert classify_doc_type("Annual Report 2024") == "annual_report"
    assert classify_doc_type("Interim Results for the six months ended 30 June 2024") == "interim_results"
    assert classify_doc_type("Notice of Annual General Meeting") is None


def test_parse_period_end_from_title():
    assert parse_period_end_from_title("Interim Results for the six months ended 30 June 2024") == "2024-06-30"
    assert parse_period_end_from_title("Annual Report 2024/06/30") == "2024-06-30"


def test_unit_currency_mapping():
    assert parse_currency("HK$ million") == "HKD"
    assert parse_unit_scale("HK$ million") == "millions"
    assert canonical_line_item("Equity attributable to owners of the Company") == "equity_attributable_to_owners"


def test_next_trading_close_after_publish():
    idx = pd.to_datetime(["2024-06-28", "2024-07-01", "2024-07-02"])
    px = pd.DataFrame({"0016.HK": [80.0, 81.0, 82.0]}, index=idx)
    d, p = next_trading_close_after_publish("2024-06-30T18:00:00", "0016.HK", px)
    assert d == "2024-07-01"
    assert p == 81.0


def test_book_nav_formula():
    filings = pd.DataFrame([{
        "filing_id": "f1", "stock_code": "00016", "period_end": "2024-06-30",
        "publish_datetime_hk": "2024-09-05T12:00:00"
    }])
    sli = pd.DataFrame([
        {"fact_id":"1","filing_id":"f1","stock_code":"00016","period_end":"2024-06-30","statement_type":"balance_sheet","scope":"group","line_item_code":"equity_attributable_to_owners","line_item_label_raw":"","value":1000.0,"currency":"HKD","unit_scale":"millions","as_reported_sign":1,"page_no":1,"source_kind":"table_extract","extract_confidence":1.0,"raw_table_id":"t1","row_idx":1,"col_idx":1},
        {"fact_id":"2","filing_id":"f1","stock_code":"00016","period_end":"2024-06-30","statement_type":"balance_sheet","scope":"group","line_item_code":"cash_and_bank_balances","line_item_label_raw":"","value":100.0,"currency":"HKD","unit_scale":"millions","as_reported_sign":1,"page_no":1,"source_kind":"table_extract","extract_confidence":1.0,"raw_table_id":"t1","row_idx":2,"col_idx":1},
        {"fact_id":"3","filing_id":"f1","stock_code":"00016","period_end":"2024-06-30","statement_type":"balance_sheet","scope":"group","line_item_code":"total_borrowings_current","line_item_label_raw":"","value":50.0,"currency":"HKD","unit_scale":"millions","as_reported_sign":1,"page_no":1,"source_kind":"table_extract","extract_confidence":1.0,"raw_table_id":"t1","row_idx":3,"col_idx":1},
        {"fact_id":"4","filing_id":"f1","stock_code":"00016","period_end":"2024-06-30","statement_type":"balance_sheet","scope":"group","line_item_code":"total_borrowings_noncurrent","line_item_label_raw":"","value":150.0,"currency":"HKD","unit_scale":"millions","as_reported_sign":1,"page_no":1,"source_kind":"table_extract","extract_confidence":1.0,"raw_table_id":"t1","row_idx":4,"col_idx":1},
    ])
    cap = pd.DataFrame([{
        "capital_fact_id":"c1","filing_id":"f1","stock_code":"00016","period_end":"2024-06-30",
        "shares_outstanding_period_end":100.0,"weighted_avg_shares_basic":None,"weighted_avg_shares_diluted":None,
        "treasury_shares":None,"page_no":1,"source_kind":"manual_override","extract_confidence":1.0,"notes":""
    }])
    ipn = pd.DataFrame(columns=["ip_note_fact_id","filing_id","stock_code","period_end","metric_code","metric_label_raw","value_num","value_text","currency","property_segment","geography","page_no","source_kind","extract_confidence"])
    px = pd.DataFrame({"0016.HK":[12.0]}, index=pd.to_datetime(["2024-09-06"]))
    nav, _ = compute_nav_outputs(filings, sli, cap, ipn, price_df=px)
    book = nav[nav["scenario"] == "book"].iloc[0]
    assert round(book["nav_per_share"], 6) == 10.0
    assert round(book["discount_premium_pct"], 6) == 0.2

