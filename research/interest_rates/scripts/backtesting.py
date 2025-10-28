import pandas as pd
import numpy as np

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
