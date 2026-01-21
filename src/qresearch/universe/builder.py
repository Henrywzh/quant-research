import pandas as pd
from qresearch.universe.filters import UniverseFilterConfig, build_universe_eligible
from qresearch.universe.hsci import build_hsci_member_mask


def build_final_universe_eligible(
    close: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    seed_members: set[str] | None = None,
    cfg: UniverseFilterConfig,
    mcap: pd.DataFrame | None = None,
    effective_on_close: bool = True,
) -> pd.DataFrame:
    """
    final_eligible[t,i] = member_gate[t,i] & investable_filters[t,i]

    If events is None:
      member_gate := True for all tickers/dates (i.e., no membership gating)
    """
    # 1) membership gate
    if events is None:
        hsci_member = pd.DataFrame(True, index=close.index, columns=close.columns)
    else:
        hsci_member = build_hsci_member_mask(
            dates=close.index,
            tickers=close.columns,
            events=events,
            seed_members=seed_members,
            effective_on_close=effective_on_close,
        )
        hsci_member = _align_bool(hsci_member, close)

    # 2) investable filters
    investable = build_universe_eligible(close=close, cfg=cfg, mcap=mcap)
    investable = _align_bool(investable, close)

    # 3) final
    return _align_bool(hsci_member & investable, close)


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
