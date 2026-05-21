from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QB = ROOT / "quant_brain"


REQUIRED_FILES = [
    "manifest.md",
    "workspace/current_focus.md",
    "workspace/active_projects.md",
    "workspace/session_handoff.md",
    "workspace/inbox.md",
    "memory/research_log.md",
    "memory/strategies/smh_regime_filter/strategy.md",
    "memory/strategies/smh_regime_filter/experiments.md",
    "memory/strategies/smh_regime_filter/findings.md",
    "memory/strategies/smh_regime_filter/failures.md",
    "memory/strategies/smh_regime_filter/iteration_queue.md",
    "memory/strategies/smh_regime_filter/artifacts.md",
    "memory/datasets/yfinance_watchlists.md",
    "memory/knowledge/large_vs_small_caps.md",
    "memory/knowledge/sector_relative_performance.md",
    "memory/knowledge/hard_assets_and_bitcoin.md",
    "memory/knowledge/sovereign_duration_tracks.md",
    "expectations/strategy_expectations/smh_regime_filter.md",
    "expectations/market_views/daily_expectations.md",
    "expectations/market_views/weekly_expectations.md",
    "expectations/market_views/monthly_expectations.md",
    "expectations/prediction_audit_queue.md",
    "evaluations/daily_reality_checks.md",
    "evaluations/weekly_review.md",
    "evaluations/monthly_review.md",
    "evaluations/prediction_scoreboard.md",
    "evaluations/surprise_log.md",
    "automations/automation_manifest.md",
    "automations/daily_market_brief.md",
    "automations/weekly_market_review.md",
    "automations/monthly_market_review.md",
    "automations/expectation_audit.md",
    "indexes/strategy_index.md",
    "indexes/topic_index.md",
    "indexes/finding_index.md",
    "indexes/automation_index.md",
    "templates/strategy_template.md",
    "templates/experiment_entry_template.md",
    "templates/daily_observation_template.md",
    "templates/weekly_review_template.md",
    "templates/monthly_review_template.md",
    "templates/expectation_template.md",
    "templates/evaluation_template.md",
]


def _read(rel_path: str) -> str:
    return (QB / rel_path).read_text(encoding="utf-8")


def test_quant_brain_required_structure_exists():
    for rel_path in REQUIRED_FILES:
        assert (QB / rel_path).exists(), rel_path


def test_strategy_file_contains_canonical_sections():
    text = _read("memory/strategies/smh_regime_filter/strategy.md")
    for heading in [
        "## Strategy",
        "## Linked Experiments",
        "## Known Failure Modes",
        "## Open Questions",
    ]:
        assert heading in text
    assert "Status:" in text
    assert "Scope:" in text


def test_expectation_entries_are_explicit_and_scoreable():
    text = _read("expectations/strategy_expectations/smh_regime_filter.md")
    for required in [
        "- Source:",
        "- Timestamp:",
        "- Horizon:",
        "- Claim:",
        "- Measurable Check:",
        "- Confidence:",
        "- Status:",
    ]:
        assert required in text


def test_observation_templates_contain_required_sections():
    for rel_path in [
        "templates/daily_observation_template.md",
        "templates/weekly_review_template.md",
        "templates/monthly_review_template.md",
    ]:
        text = _read(rel_path)
        for heading in [
            "## Market Summary",
            "## Leadership",
            "## Breadth",
            "## Rates/Vol/Macro",
            "## Surprises",
            "## Links",
        ]:
            assert heading in text


def test_research_log_seed_contains_json_and_critic_evaluation():
    text = _read("memory/research_log.md")
    assert "```json" in text
    assert "### Critic Evaluation" in text
    assert "### Failure Points" in text
    assert "### Next Iteration" in text


def test_prediction_scoreboard_maps_expectation_to_outcome():
    text = _read("evaluations/prediction_scoreboard.md")
    for required in [
        "- Expectation Ref:",
        "- Outcome:",
        "- Verdict:",
        "- What Was Wrong:",
        "- Next Adjustment:",
    ]:
        assert required in text


def test_automation_manifest_and_prompts_keep_writes_constrained():
    manifest = _read("automations/automation_manifest.md")
    assert "must not directly rewrite" in manifest
    for rel_path in [
        "automations/daily_market_brief.md",
        "automations/weekly_market_review.md",
        "automations/monthly_market_review.md",
        "automations/expectation_audit.md",
    ]:
        text = _read(rel_path)
        assert "do not" in text.lower()
        assert "durable memory" in text.lower() or "durable findings" in text.lower()


def test_focus_files_capture_initial_research_universes():
    large_small = _read("memory/knowledge/large_vs_small_caps.md")
    for ticker in ["DIA", "SPY", "QQQ", "IWM", "1321.T", "069500.KS", "2800.HK", "510050.SS", "510300.SS", "510500.SS", "159845.SZ"]:
        assert f"`{ticker}`" in large_small

    sectors = _read("memory/knowledge/sector_relative_performance.md")
    for ticker in ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU", "SMH", "IGV"]:
        assert f"`{ticker}`" in sectors

    macro = _read("memory/knowledge/hard_assets_and_bitcoin.md")
    for ticker in ["GC=F", "SI=F", "HG=F", "CL=F", "BTC-USD"]:
        assert f"`{ticker}`" in macro

    rates = _read("memory/knowledge/sovereign_duration_tracks.md")
    for ticker in ["SHY", "IEF", "TLT", "IGLS.L", "VGOV.L", "GLTL.L", "2510.T", "236A.T", "2561.T"]:
        assert f"`{ticker}`" in rates


def test_canonical_yfinance_watchlist_file_exists_and_is_referenced():
    watchlists = _read("memory/datasets/yfinance_watchlists.md")
    for ticker in ["DIA", "SPY", "QQQ", "IWM", "1321.T", "069500.KS", "2800.HK", "510050.SS", "510300.SS", "510500.SS", "159845.SZ", "SMH", "IGV", "BTC-USD", "TLT", "GLTL.L", "236A.T"]:
        assert f"`{ticker}`" in watchlists

    for rel_path in [
        "automations/daily_market_brief.md",
        "automations/weekly_market_review.md",
        "automations/monthly_market_review.md",
        "automations/expectation_audit.md",
    ]:
        text = _read(rel_path)
        assert "yfinance_watchlists.md" in text
