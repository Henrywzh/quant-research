import pandas as pd
from qresearch.universe.filters import UniverseFilterConfig, build_universe_eligible
from qresearch.universe.hsci import build_hsci_member_mask


def build_final_universe_eligible(
    close: pd.DataFrame,
    *,
    events: pd.DataFrame,
    seed_members: set[str] | None,
    cfg: UniverseFilterConfig,
    mcap: pd.DataFrame | None = None,
    effective_on_close: bool = True,
) -> pd.DataFrame:
    """
    Build the final formation-date universe eligibility mask:

        final_eligible[t,i] = hsci_member[t,i] & investable_filters[t,i]

    No lookahead:
    - hsci_member uses effective_date <= t (applied on t close, per your rule)
    - investable filters use rolling stats up to t (inclusive)
    """
    # 1) membership gate (survivorship bias fix)
    hsci_member = build_hsci_member_mask(
        dates=close.index,
        tickers=close.columns,
        events=events,
        seed_members=seed_members,
        effective_on_close=effective_on_close
    )
    hsci_member = _align_bool(hsci_member, close)

    # 2) investable filters (price floor / IPO age / mcap floor)
    investable = build_universe_eligible(close=close, cfg=cfg, mcap=mcap)
    investable = _align_bool(investable, close)

    # 3) final gate
    final_eligible = hsci_member & investable
    final_eligible = _align_bool(final_eligible, close)

    return final_eligible


def prepare_universe_bundle(
    close: pd.DataFrame,
    *,
    events: pd.DataFrame,
    seed_members: set[str] | None,
    cfg: UniverseFilterConfig,
    mcap: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Convenience: return intermediate masks for debugging/plots.
    """
    hsci_member = build_hsci_member_mask(
        dates=close.index,
        tickers=close.columns,
        events=events,
        seed_members=seed_members,
        effective_on_close=True,
    )
    investable = build_universe_eligible(close=close, cfg=cfg, mcap=mcap)
    investable = _align_bool(investable, close)
    hsci_member = _align_bool(hsci_member, close)

    return {
        "hsci_member": hsci_member,
        "investable": investable,
        "final_eligible": hsci_member & hsci_member,
    }


def _align_bool(mask: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """Reindex mask to match `like` and fill missing with False."""
    return mask.reindex(index=like.index, columns=like.columns).fillna(False).astype(bool)
