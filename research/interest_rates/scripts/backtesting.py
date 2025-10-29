import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from matplotlib import pyplot as plt


# =========================
# Helpers & Metrics
# =========================

@dataclass
class RegimeConfig:
    # weighting
    lambda_prob: float = 0.7   # blend between prob and hysteresis state
    p_ewma_halflife: int = 3   # smoothing for probability
    up: float = 0.6            # hysteresis upper switch
    down: float = 0.5          # hysteresis lower switch
    confirm: int = 2           # consecutive months required
    # bond momentum / cash filter
    mom_lookback_m: int = 6    # lookback months for bond momentum
    use_cash_filter: bool = True
    # volatility targeting (optional; simple)
    vol_target_ann: float = None  # e.g., 0.10; None to disable
    vol_lb_m: int = 36
    max_leverage: float = 2.0


def annualized_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return np.sqrt(12) * r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan

def max_drawdown_and_duration(r: pd.Series) -> Tuple[float, float]:
    # r: monthly returns
    w = (1 + r.fillna(0)).cumprod()
    peak = w.cummax()
    dd = w/peak - 1
    mdd = dd.min()
    # duration in years = longest time not at a new high
    cur = best = 0
    for i in range(len(w)):
        if w.iloc[i] == peak.iloc[i]:
            best = max(best, cur); cur = 0
        else:
            cur += 1
    best = max(best, cur)
    return float(mdd), float(best/12)

def turnover(weights: pd.Series) -> float:
    """Average |Δw| per month for the bond weight series."""
    w = weights.dropna()
    return w.diff().abs().dropna().mean() if len(w) >= 2 else np.nan

# =========================
# Core building blocks
# =========================

def ewma(x: pd.Series, halflife: int = 3) -> pd.Series:
    return x.ewm(halflife=halflife, adjust=False).mean()

def hysteresis_state(
    p: pd.Series,
    up: float = 0.6,
    down: float = 0.5,
    confirm: int = 2
) -> pd.Series:
    """
    Binary regime with confirmation:
      - need 'confirm' consecutive p >= up to switch to risk-off (1)
      - need 'confirm' consecutive p <= down to switch to risk-on  (0)
    Returns a series in {0,1}.
    """
    p = p.copy()
    state = pd.Series(0, index=p.index, dtype=int)
    cur = 0
    streak_up = streak_down = 0
    for t in p.index:
        if p.loc[t] >= up:
            streak_up += 1
            streak_down = 0
            if cur == 0 and streak_up >= confirm:
                cur = 1
        elif p.loc[t] <= down:
            streak_down += 1
            streak_up = 0
            if cur == 1 and streak_down >= confirm:
                cur = 0
        else:
            # between thresholds: no change, reset streaks
            streak_up = streak_down = 0
        state.loc[t] = cur
    return state

def bond_momentum_signal(ret_bond: pd.Series, lookback_m: int = 6) -> pd.Series:
    """
    Simple momentum proxy: rolling compounded return over 'lookback_m' months (as arithmetic approx).
    If < 0 => negative momentum.
    """
    # arithmetic approx: sum of monthly returns (close enough for small r)
    roll_sum = ret_bond.rolling(lookback_m).sum()
    return roll_sum

def build_bond_weight(
    df: pd.DataFrame,
    cfg: RegimeConfig
) -> pd.Series:
    """
    Bond weight w_t in [0,1], equity weight = 1 - w_t.
    Combines:
      - smoothed probability (continuous sizing)
      - hysteresis state (confirmation)
      - convex blend: w = λ*prob_smoothed + (1-λ)*state
      - optional cash filter: if state=1 (risk-off) and bond momentum<0 ⇒ divert that portion to cash
    """
    p = df["p_recession"].clip(0, 1)
    p_sm = ewma(p, halflife=cfg.p_ewma_halflife)
    state = hysteresis_state(p, up=cfg.up, down=cfg.down, confirm=cfg.confirm)
    w_prob = p_sm
    w_state = state.astype(float)

    w = cfg.lambda_prob * w_prob + (1 - cfg.lambda_prob) * w_state
    w = w.clip(0, 1)

    # Optional vol targeting will be applied at portfolio level; here we keep raw weights in [0,1].
    return w

def apply_cash_filter(
    w_bond: pd.Series,
    ret_bond: pd.Series,
    ret_cash: pd.Series,
    df: pd.DataFrame,
    cfg: RegimeConfig
) -> Tuple[pd.Series, pd.Series]:
    """
    If risk-off state (state=1) but bond momentum < 0, then for that fraction (w_bond) use cash instead of bonds.
    Returns (effective_bond_return_series, effective_cash_return_series_factor)
    We implement by creating an 'effective bond sleeve return':
       r_bond_eff = I(mom>=0)*ret_bond + I(mom<0)*0  (the 0 part is handled by shifting its weight to cash)
    and a 'cash boost factor' for that diverted weight.
    """
    state = hysteresis_state(df["p_recession"], up=cfg.up, down=cfg.down, confirm=cfg.confirm)
    mom = bond_momentum_signal(ret_bond, lookback_m=cfg.mom_lookback_m)
    # mask where we are risk-off and momentum is negative
    mask_divert = (state == 1) & (mom < 0)

    # Effective bond return: zero out the portion that would have gone to bonds on those months
    r_bond_eff = ret_bond.copy()
    r_bond_eff[mask_divert] = 0.0

    # The diverted bond weight goes to cash; we’ll account for it when building portfolio returns.
    # For convenience, return a series of whether we divert in each month:
    divert_indicator = mask_divert.astype(int)

    return r_bond_eff, divert_indicator

def vol_target_scaler(r: pd.Series, ann_target: float, lookback: int, max_lev: float) -> pd.Series:
    if ann_target is None:
        return pd.Series(1.0, index=r.index)
    m_target = ann_target / np.sqrt(12)
    sigma = r.rolling(lookback).std()
    scale = (m_target / sigma).clip(upper=max_lev)
    return scale.fillna(0.0)


def apply_regime_strategy(
        prob: pd.Series,
        ret_equity: pd.Series,
        ret_bond: pd.Series,
        threshold=0.6,
        lag_months=1,
        cost_bps=5
):
    prob = prob.sort_index().copy()
    signal = (prob.shift(lag_months) >= threshold).astype(int)  # 1=risk-off
    pos_eq = 1 - signal
    pos_bd = signal
    strat_ret = pos_eq*ret_equity + pos_bd * ret_bond
    # turn cost
    switches = signal.diff().abs().fillna(0)
    cost = switches * (cost_bps/10000.0)
    strat_ret = strat_ret - cost
    equity_curve = (1 + strat_ret).cumprod()

    return strat_ret, equity_curve, signal


# =========================
# Portfolio construction
# =========================

def build_portfolio_returns(
    df: pd.DataFrame,
    cfg: RegimeConfig
) -> Dict[str, pd.Series]:
    """
    Build portfolio returns under the enhanced regime design.
    Weights:
        w_bond = λ * smoothed_prob + (1-λ) * hysteresis_state
        w_eq   = 1 - w_bond
    Cash filter:
        if state=1 and bond momentum<0 ⇒ the w_bond slice goes to cash instead of bonds.
    Vol targeting (optional):
        scale the whole portfolio to target ann vol, capped by max_leverage.
    """
    p = df["p_recession"].clip(0,1)
    ret_eq, ret_bond = df["ret_eq"], df["ret_bond"]
    ret_cash = df.get("ret_cash", pd.Series(0.0, index=df.index))

    # 1) base bond weight and hysteresis state
    w_bond = build_bond_weight(df, cfg)
    w_eq = 1.0 - w_bond
    state = hysteresis_state(p, up=cfg.up, down=cfg.down, confirm=cfg.confirm)

    # 2) cash filter (optional)
    if cfg.use_cash_filter:
        r_bond_eff, divert_indicator = apply_cash_filter(w_bond, ret_bond, ret_cash, df, cfg)
        # portion w_bond is split: if divert_indicator=1, ALL of w_bond goes to cash (simple rule)
        # You can refine to partial diversion based on signal strength later.
        r_bond_for_port = r_bond_eff
        w_cash_extra = (w_bond * divert_indicator).astype(float)
    else:
        r_bond_for_port = ret_bond
        w_cash_extra = pd.Series(0.0, index=df.index)

    # 3) raw (unscaled) portfolio return
    # cash weight = baseline 0 + any diverted bond slice
    w_cash = w_cash_extra
    # when we divert to cash, we still keep w_eq unchanged; the diverted amount comes from w_bond
    # ensure weights add to 1: w_eq + (w_bond - diverted) + diverted == 1
    w_bond_eff = w_bond - w_cash
    w_bond_eff = w_bond_eff.clip(0,1)

    r_port_raw = (w_eq * ret_eq) + (w_bond_eff * r_bond_for_port) + (w_cash * ret_cash)

    # 4) simple portfolio realized vol targeting (optional)
    scaler = vol_target_scaler(r_port_raw, cfg.vol_target_ann, cfg.vol_lb_m, cfg.max_leverage)
    r_port = scaler.shift(1).fillna(1.0) * r_port_raw  # trade at month-end for next month

    out = {
        "r_port": r_port,
        "r_raw": r_port_raw,
        "w_bond": w_bond,
        "w_eq": w_eq,
        "w_cash": w_cash,
        "state": state,
    }
    return out


# =========================
# Evaluation & Frontier
# =========================

def evaluate(r: pd.Series, w_bond: pd.Series, freq=12) -> Dict[str, float]:
    sh = annualized_sharpe(r)
    mdd, mdd_yrs = max_drawdown_and_duration(r)
    to = turnover(w_bond)
    ann_ret = (1 + r).prod() ** (freq / len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq)
    # Sortino
    downside = r.where(r < 0, 0)
    dvol = downside.std() * np.sqrt(freq)
    sortino = ann_ret / (dvol + 1e-12)

    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sh,
        "sortino": sortino,
        "max_drawdown": mdd,
        "max_dd_years": mdd_yrs,
        "turnover": to,
    }


def run_param_grid(df: pd.DataFrame, grid: List[RegimeConfig]) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(grid):
        res = build_portfolio_returns(df, cfg)
        mets = evaluate(res["r_port"], res["w_bond"])
        rows.append({
            "i": i,
            "lambda_prob": cfg.lambda_prob,
            "p_halflife": cfg.p_ewma_halflife,
            "up": cfg.up,
            "down": cfg.down,
            "confirm": cfg.confirm,
            "mom_lb": cfg.mom_lookback_m,
            "cash_filter": cfg.use_cash_filter,
            "vol_target": cfg.vol_target_ann,
            **mets
        })
    return pd.DataFrame(rows).set_index("i")


def plot_frontier(df_params: pd.DataFrame, title="Sharpe vs MaxDD vs Turnover"):
    fig, ax = plt.subplots(figsize=(7, 5))
    s = 300 * (1 - df_params["turnover"].clip(0, 0.5))  # bigger point = lower turnover
    sc = ax.scatter(-df_params["max_drawdown"], df_params["sharpe"], s=s, alpha=0.8)
    ax.set_xlabel("−Max Drawdown (larger is better)")
    ax.set_ylabel("Sharpe")
    ax.set_title(title)
    for i, row in df_params.iterrows():
        ax.annotate(str(i), (-row["max_drawdown"], row["sharpe"]), fontsize=8, alpha=0.7)
    plt.show()


