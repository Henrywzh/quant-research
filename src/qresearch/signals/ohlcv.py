from __future__ import annotations

import numpy as np
import pandas as pd
from .registry import register_signal
from qresearch.data.types import MarketData


# tug of war
@register_signal(
    "on_minus_id",
    description="(overnight momentum) - (intraday momentum) over same lookback",
    defaults={"lookback": 21},
    requires=("open", "close"),
)
def on_minus_id(md: MarketData, lookback: int = 20) -> pd.DataFrame:
    r_on = md.open / md.close.shift(1) - 1.0
    r_id = md.close / md.open - 1.0
    on = np.exp(np.log1p(r_on).rolling(lookback).sum()) - 1.0
    id_ = np.exp(np.log1p(r_id).rolling(lookback).sum()) - 1.0
    return on - id_


@register_signal(
    "rsi",
    description="rsi, default lookback: 14",
    defaults={"lookback": 14},
    requires=("close",),
)
def rsi(md: MarketData, lookback: int = 14) -> pd.DataFrame:
    """
    RSI computed per ticker from Close prices.
    Returns RSI in [0, 100]. Higher RSI = stronger recent gains.
    """
    close = md.close.sort_index()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing: EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/lookback, adjust=False, min_periods=lookback).mean()
    avg_loss = loss.ewm(alpha=1/lookback, adjust=False, min_periods=lookback).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


@register_signal(
    "overnight_mom",
    description="Momentum on overnight returns: cumprod(1+r_on) over lookback - 1",
    defaults={"lookback": 21},
    requires=("open", "close"),
)
def overnight_mom(md: MarketData, lookback: int = 20) -> pd.DataFrame:
    r_on = md.open / md.close.shift(1) - 1.0
    return (1.0 + r_on).rolling(lookback).apply(np.prod, raw=True) - 1.0

@register_signal(
    "intraday_mom",
    description="Momentum on intraday returns: cumprod(1+r_id) over lookback - 1",
    defaults={"lookback": 21},
    requires=("open", "close"),
)
def intraday_mom(md: MarketData, lookback: int = 20) -> pd.DataFrame:
    r_id = md.close / md.open - 1.0
    return (1.0 + r_id).rolling(lookback).apply(np.prod, raw=True) - 1.0


@register_signal(
    "trend_annret_r2",
    description="Trend score = annualised return * R^2 (rolling OLS on log price)",
    defaults={"lookback": 63, "ann_factor": 252, 'skip': 0},
)
def trend_annret_r2(md: MarketData, lookback: int = 126, ann_factor: int = 252, skip: int = 0) -> pd.DataFrame:
    return trend_score_annret_r2(md.close, lookback=lookback, ann_factor=ann_factor, skip=skip)


@register_signal("mom_ret", description="Return over lookback with optional skip", defaults={"lookback": 21, "skip": 0})
def mom_ret(md: MarketData, lookback: int = 21, skip: int = 0) -> pd.DataFrame:
    p1 = md.close.shift(skip)
    p0 = md.close.shift(skip + lookback)
    return p1 / p0 - 1.0


@register_signal("mom_12_1", description="12-1 momentum (252 lookback, 21 skip)", defaults={"lookback": 252, "skip": 21})
def mom_12_1(md: MarketData, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    p1 = md.close.shift(skip)
    p0 = md.close.shift(skip + lookback)
    return p1 / p0 - 1.0


@register_signal("ma_diff", description="ma and price difference", defaults={"lookback": 21, "skip": 0, 'sign': 1})
def ma_diff(md: MarketData, lookback: int, skip: int = 0, sign: int = 1) -> pd.DataFrame:
    """
    mom_{L,skip}(t) = price(t-skip)/price(t-skip-L) - 1
    Computed using close[t] information only, no lookahead.
    """
    p1 = md.close.shift(skip)
    p0 = md.close.shift(skip + lookback)
    return sign * p1 / p0 - 1.0


@register_signal(
    "ma_angle",
    description="ma_angle, default lookback: 20",
    defaults={"lookback": 20},
    requires=("close",),
)
def ma_angle(md: MarketData, lookback: int = 20) -> pd.DataFrame:
    """
    Calculates the angle of the Moving Average (SMA) in degrees.

    Formula:
        MA = Simple Moving Average(Close, lookback)
        Slope = MA_current - MA_prev
        Angle = arctan(Slope) * (180 / PI)

    Returns:
        DataFrame with angles in degrees [-90, 90].
    """
    # 1. Prepare Close prices
    close = md.close.sort_index()

    # 2. Calculate Moving Average (SMA)
    ma = close.rolling(window=lookback).mean()

    # 3. Calculate Slope (Rise over Run, where Run = 1)
    # Using raw price difference:
    slope = ma.diff()

    # ALTERNATIVE: Use this line for price-normalized angles (better for comparing stocks)
    # slope = ma.pct_change() * 100

    # 4. Calculate Angle
    # arctan returns radians; we convert to degrees for interpretability
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def trend_score_annret_r2(
    prices: pd.DataFrame,
    lookback: int,
    skip: int = 21,
    ann_factor: int = 252,
) -> pd.DataFrame:
    """
    Trend score = annualised return * R^2 from linear regression of log(price) on time.

    Computes slope (b) and R^2 over a rolling window of length `lookback` on log(prices),
    but **skips** the most recent `skip` closes to avoid short-term reversal / microstructure noise.

    Concretely, the regression window at date t uses log(price) from:
        [t - skip - lookback + 1, ..., t - skip]

    Parameters
    ----------
    prices : pd.DataFrame
        Wide DataFrame (date × ticker) of close prices.
    lookback : int
        Number of observations in the regression window (>= 2).
    skip : int, default 21
        Number of most recent closes to exclude (lag the input by `skip` days).
    ann_factor : int, default 252
        Annualisation factor for converting daily slope to annualised return.

    Returns
    -------
    pd.DataFrame
        Trend score (date × ticker): (annualised return) * (R^2), aligned to `prices`.
        NaNs during warmup (lookback + skip).
    """
    px = prices.replace([np.inf, -np.inf], np.nan).ffill()

    L = int(lookback)
    if L < 2:
        raise ValueError("lookback must be >= 2")

    s = int(skip)
    if s < 0:
        raise ValueError("skip must be >= 0")

    # Skip the most recent `skip` closes by shifting the series back.
    # At time t, we regress using data ending at t-s.
    y = np.log(px).shift(s)

    x = np.arange(L, dtype=float)
    x_mean = x.mean()
    var_x = ((x - x_mean) ** 2).sum()  # constant

    sum_y = y.rolling(L, min_periods=L).sum()
    sum_y2 = (y * y).rolling(L, min_periods=L).sum()

    def dot_x(arr: np.ndarray) -> float:
        return float(np.dot(x, arr))

    sum_xy = y.rolling(L, min_periods=L).apply(dot_x, raw=True)

    y_mean = sum_y / L
    cov_xy = sum_xy - L * x_mean * y_mean
    b = cov_xy / var_x  # slope per day

    var_y = (sum_y2 / L) - (y_mean ** 2)
    var_y = var_y.clip(lower=0.0)

    denom = var_x * (L * var_y)
    r2 = (cov_xy ** 2) / denom
    r2 = r2.clip(lower=0.0, upper=1.0)

    ann_ret = np.exp(b * ann_factor) - 1.0
    return ann_ret * r2


@register_signal(
    "fourline_open_score",
    description=(
        "Score for 三线/四线初开 + 量比倍增. "
        "Implements: adhesion (MA5/10/20/30 up+converged) + initial_open (bull4 just formed) + VR>=2, "
        "then computes a 0–100 score and optional trade mask."
    ),
    defaults={
        # ---- MA / slope params ----
        "slope_L": 5,              # slope lookback for MA5/10/20/30 "up" test
        "adh_contract_L": 10,      # how far back to check spread contraction
        "spread_eps": 0.012,       # adhesion threshold on spread (1.2%)
        "spread_contract_min": 0.003,  # required contraction over adh_contract_L (0.3%)

        # ---- initial open windows ----
        "D_open": 5,               # "just formed" window for bull4
        "D_pre": 10,               # adhesion must have happened within last D_pre days

        # ---- volume ratio ----
        "vr_N": 5,                 # volume mean window (prior N days)
        "vr_thresh": 2.0,          # "量比倍增" threshold
        "vr_full": 5.0,            # VR that maps to full score_vol

        # ---- score weights / scaling ----
        "w_adh": 20.0,
        "w_open": 25.0,
        "w_trend": 25.0,           # used as max in strong regime
        "w_vol": 20.0,
        "w_px": 10.0,
        "gap_full": 0.01,          # gap at which score_open reaches max (1% of price)

        # ---- trend classification ----
        "trend_L": 20,             # slope lookback for MA50/120
        "strong_score": 25.0,      # score_trend in strong regime
        "neutral_score": 10.0,     # score_trend in neutral regime
        "rebound_score": -10.0,    # score_trend in rebound regime

        # ---- price confirmation ----
        "breakout_L": 20,          # breakout lookback on close (prior highs)

        # ---- trade rule ----
        "trade_threshold": 70.0,
        "output": "score",         # "score" or "trade" (trade returns 1/0 mask)
    },
    requires=("close", "volume"),
)
def fourline_open_score(
    md: MarketData,
    slope_L: int = 5,
    adh_contract_L: int = 10,
    spread_eps: float = 0.012,
    spread_contract_min: float = 0.003,
    D_open: int = 5,
    D_pre: int = 10,
    vr_N: int = 5,
    vr_thresh: float = 2.0,
    vr_full: float = 5.0,
    w_adh: float = 20.0,
    w_open: float = 25.0,
    w_trend: float = 25.0,
    w_vol: float = 20.0,
    w_px: float = 10.0,
    gap_full: float = 0.01,
    trend_L: int = 20,
    strong_score: float = 25.0,
    neutral_score: float = 10.0,
    rebound_score: float = -10.0,
    breakout_L: int = 20,
    trade_threshold: float = 70.0,
    output: str = "score",
) -> pd.DataFrame:
    """
    Implements the user's pseudocode signal:

    Inputs: Close, Volume
    Compute MA5,10,20,30,50,120,250

    spread = (max(MA5,MA10,MA20,MA30)-min(...)) / Close
    up_slopes = slopes_of(MA5,10,20,30 over slope_L) > 0

    adhesion = (spread <= spread_eps)
               AND (at least 3 of up_slopes true)
               AND (spread has contracted over last adh_contract_L days by spread_contract_min)

    bull4 = MA5>MA10>MA20>MA30
    initial_open = bull4
                   AND (bull4 was false at least once in last D_open days)
                   AND (adhesion happened in last D_pre days)

    VR = Volume / mean(Volume last vr_N days)  (uses prior vr_N days, no lookahead)
    vol_double = VR >= vr_thresh

    if not (initial_open AND vol_double): score=0
    else score = clip(score_adh + score_open + score_trend + score_vol + score_px, 0, 100)

    Trade if score >= trade_threshold
    """
    # -------- helpers --------
    def _as_df(x: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
        if isinstance(x, pd.Series):
            return x.to_frame(name=name)
        return x

    def _clip(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
        return df.clip(lower=lo, upper=hi)

    def _rowwise_max(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        out = dfs[0].copy()
        for d in dfs[1:]:
            out = np.maximum(out, d)
        return _as_df(out, "tmp")

    def _rowwise_min(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        out = dfs[0].copy()
        for d in dfs[1:]:
            out = np.minimum(out, d)
        return _as_df(out, "tmp")

    # -------- 1) Inputs --------
    close = _as_df(md.close.sort_index(), "close").astype(float)
    vol = _as_df(md.volume.sort_index(), "volume").astype(float)

    # -------- 2) Moving averages --------
    ma5 = close.rolling(5, min_periods=5).mean()
    ma10 = close.rolling(10, min_periods=10).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma30 = close.rolling(30, min_periods=30).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    ma120 = close.rolling(120, min_periods=120).mean()
    ma250 = close.rolling(250, min_periods=250).mean()

    # -------- 3) Adhesion (粘合 + 向上 + 收敛) --------
    ma_list = [ma5, ma10, ma20, ma30]
    ma_max = _rowwise_max(ma_list)
    ma_min = _rowwise_min(ma_list)

    spread = (ma_max - ma_min) / close

    # slopes over slope_L (per bar), positive => upward
    s5 = (ma5 - ma5.shift(slope_L)) / slope_L
    s10 = (ma10 - ma10.shift(slope_L)) / slope_L
    s20 = (ma20 - ma20.shift(slope_L)) / slope_L
    s30 = (ma30 - ma30.shift(slope_L)) / slope_L

    up_cnt = (s5 > 0).astype(int) + (s10 > 0).astype(int) + (s20 > 0).astype(int) + (s30 > 0).astype(int)
    up_slopes_ok = up_cnt >= 3

    spread_contracted = (spread.shift(adh_contract_L) - spread) >= spread_contract_min

    adhesion = (spread <= spread_eps) & up_slopes_ok & spread_contracted

    # -------- 4) Initial open (初开) --------
    bull4 = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma30)

    # "bull4 was false at least once in last D_open days"
    bull4_was_false = (~bull4).rolling(D_open, min_periods=1).max().astype(bool)

    # "adhesion happened in last D_pre days"
    adhesion_recent = adhesion.rolling(D_pre, min_periods=1).max().astype(bool)

    initial_open = bull4 & bull4_was_false & adhesion_recent

    # -------- 5) Volume ratio (量比) --------
    vol_mean_prev = vol.rolling(vr_N, min_periods=vr_N).mean().shift(1)  # use prior days only
    VR = vol / vol_mean_prev
    vol_double = VR >= vr_thresh

    gate = initial_open & vol_double

    # -------- 6) Score components --------
    # score_adh: tighter spread => higher (0..w_adh)
    score_adh = w_adh * _clip((spread_eps - spread) / spread_eps, 0.0, 1.0)

    # score_open: "opening gap" clarity (0..w_open)
    gap = ((ma5 - ma10) + (ma10 - ma20) + (ma20 - ma30)) / close
    score_open = w_open * _clip(gap / gap_full, 0.0, 1.0)

    # score_trend: strong vs rebound classification using MA50/120/250 and slopes
    s50 = (ma50 - ma50.shift(trend_L)) / trend_L
    s120 = (ma120 - ma120.shift(trend_L)) / trend_L

    strong = (close > ma50) & (ma50 > ma120) & (ma120 > ma250) & (s50 > 0) & (s120 > 0)
    rebound = (close < ma50) & ((s50 < 0) | (s120 < 0) | (ma50 < ma120))

    score_trend = pd.DataFrame(neutral_score, index=close.index, columns=close.columns)
    score_trend = score_trend.mask(strong, strong_score)
    score_trend = score_trend.mask(rebound, rebound_score)

    # score_vol: stronger VR => higher (0..w_vol)
    denom = max(vr_full - vr_thresh, 1e-9)
    score_vol = w_vol * _clip((VR - vr_thresh) / denom, 0.0, 1.0)

    # score_px: confirmations (0..w_px)
    #  - close above MA30 (half weight)
    #  - close breaks above prior breakout_L-day high (half weight)
    px_above_ma30 = (close > ma30).astype(float)

    prior_high = close.shift(1).rolling(breakout_L, min_periods=breakout_L).max()
    px_breakout = (close > prior_high).astype(float)

    score_px = (w_px * 0.5) * px_above_ma30 + (w_px * 0.5) * px_breakout

    # -------- 7) Total score with gate --------
    score = score_adh + score_open + score_trend + score_vol + score_px
    score = _clip(score, 0.0, 100.0)

    # if gate is false -> score = 0
    score = score.where(gate, 0.0)

    # -------- 8) Output mode --------
    if output.lower() == "trade":
        trade = (score >= trade_threshold).astype(int)
        return trade

    # default: score
    return score
