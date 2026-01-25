import numpy as np
import pandas as pd

from qresearch.data.types import MarketData


def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Time-series rank within rolling window for each ticker.
    Output in [0,1]. Uses average rank for ties.
    """
    def _rank_last(a: np.ndarray) -> float:
        # rank of last element within window
        s = pd.Series(a)
        r = s.rank(pct=True).iloc[-1]
        return float(r)

    return x.rolling(window).apply(_rank_last, raw=False)


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank per date across tickers, output in [0,1]."""
    return x.rank(axis=1, pct=True)


def rolling_cov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window).cov(y)


def rolling_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window).corr(y)


def ref(x: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    return x.shift(k)


def ewma(x: pd.DataFrame, span: int) -> pd.DataFrame:
    return x.ewm(span=span, adjust=False).mean()





def _safe_div(numer: pd.DataFrame, denom: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    denom2 = denom.where(denom.abs() > eps, np.nan)
    return numer / denom2


def _typical_price(md: "MarketData") -> pd.DataFrame:
    return (md.high + md.low + md.close) / 3.0


def _rolling_vwap_from_ohlcv(md: "MarketData", window: int) -> pd.DataFrame:
    """
    OHLCV-only rolling VWAP proxy:
    vwap_n = sum(tp * vol, n) / sum(vol, n)
    """
    tp = _typical_price(md)
    pv = tp * md.volume
    num = pv.rolling(window).sum()
    den = md.volume.rolling(window).sum()
    return _safe_div(num, den)


def _amount_proxy(md: "MarketData", lookback: int, mode: str = "tp") -> pd.DataFrame:
    """
    Build a value-traded proxy when md.amount is unavailable.

    mode:
      - "tp": amount_proxy = typical_price * volume  (recommended default)
      - "rvwap": amount_proxy = rolling_vwap(lookback) * volume  (smoothed price)
    """
    if mode == "tp":
        px = _typical_price(md)
    elif mode == "rvwap":
        px = _rolling_vwap_from_ohlcv(md, window=lookback)
    else:
        raise ValueError("mode must be 'tp' or 'rvwap'")

    return px * md.volume


def _turnover_from_volume_and_shares(md: "MarketData", lot_size: int = 1) -> pd.DataFrame:
    # If volume is in lots, set lot_size accordingly (A-share: 100).
    vol_shares = md.volume * lot_size
    return _safe_div(vol_shares, md.shares_outstanding)
