import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1) DATA
# ============================================================
def download_data(ticker: str = "SPY", period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(col).strip() for col in df.columns.values]

    df = df.dropna()
    return df


def _find_close_col(df: pd.DataFrame) -> str:
    # Prefer "Close" (not "Adj Close"), tolerate flattened names like "Close SPY"
    close_col = next((c for c in df.columns if ("Close" in c and "Adj" not in c)), None)
    if close_col is None:
        raise ValueError("Could not find a non-adjusted Close column in downloaded data.")
    return close_col


# ============================================================
# 2) FEATURES + FORWARD RETURNS (no lookahead)
# ============================================================
def engineer_features(df: pd.DataFrame,
                      mom_window: int = 10,
                      vol_window: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Creates features at time t using data up to close(t).
    Creates forward close-to-close return as ret_fwd[t] = Close[t+1]/Close[t] - 1.

    Returns:
      features_df: indexed by date, columns = [log_return, momentum, volatility]
      close: close series
      ret_fwd: forward returns aligned to features index (last date will be NaN and removed)
    """
    close_col = _find_close_col(df)
    close = df[close_col].astype(float).copy()

    feats = pd.DataFrame(index=df.index)
    feats["log_return"] = np.log(close / close.shift(1))
    feats["momentum"] = close.pct_change(mom_window)
    feats["volatility"] = feats["log_return"].rolling(vol_window).std()

    # Forward returns: position decided at t (after close), applied to (t -> t+1)
    ret_fwd = close.pct_change().shift(-1)

    out = feats.join(ret_fwd.rename("ret_fwd")).dropna()
    features_df = out[["log_return", "momentum", "volatility"]]
    ret_fwd = out["ret_fwd"]
    close = close.reindex(out.index)

    return features_df, close, ret_fwd


# ============================================================
# 3) HMM FITTING + POSTERIOR FOR "LAST STEP" USING HISTORY
# ============================================================
def fit_hmm(features: pd.DataFrame,
            n_states: int,
            covariance_type: str = "full",
            n_iter: int = 2000,
            random_state: int = 42) -> tuple[GaussianHMM, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(X)
    return model, scaler


def posterior_last(model: GaussianHMM,
                   scaler: StandardScaler,
                   features_hist: pd.DataFrame,
                   lookback: int = 100) -> np.ndarray:
    """
    Compute posterior over states for the LAST observation using an HMM over a
    recent historical window ending today (inclusive). This uses transition dynamics
    without peeking into the future.
    """
    window = features_hist.tail(lookback)
    X = scaler.transform(window.values)

    # score_samples returns (logprob, posteriors) with one posterior per row
    _, post = model.score_samples(X)
    return post[-1]  # posterior for the last time step


def argmax_state(posterior: np.ndarray) -> tuple[int, float]:
    s = int(np.argmax(posterior))
    conf = float(posterior[s])
    return s, conf


# ============================================================
# 4) STATE "MEANING" FROM IN-SAMPLE (label by empirical stats)
# ============================================================
def summarize_states(model: GaussianHMM,
                     scaler: StandardScaler,
                     features: pd.DataFrame,
                     ret_fwd: pd.Series) -> pd.DataFrame:
    """
    Decode state sequence on the given feature set and compute empirical stats per state.
    Works for any feature columns (e.g., price_feats OR vol_feats).
    """
    X = scaler.transform(features.values)
    states = model.predict(X)

    tmp = features.copy()
    tmp["state"] = states
    tmp["ret_fwd"] = ret_fwd.reindex(features.index).values

    g = tmp.groupby("state")

    # Base stats
    stats = pd.DataFrame({
        "count": g.size(),
        "mean_fwd_ret": g["ret_fwd"].mean(),
        "ann_fwd_ret_approx": g["ret_fwd"].mean() * 252,
    })

    # Add mean of each feature column that exists
    for col in features.columns:
        stats[f"mean_{col}"] = g[col].mean()

    return stats.sort_index()


# ============================================================
# 5) WALK-FORWARD OOS BACKTEST
# ============================================================
def walkforward_backtest(features: pd.DataFrame,
                         ret_fwd: pd.Series,
                         *,
                         train_years: float = 3.0,
                         price_states: int = 5,
                         vol_states: int = 4,
                         refit_every: int = 21,
                         posterior_lookback: int = 120,
                         vol_riskoff_top_k: int = 1,
                         tc_bps: float = 2.0,
                         verbose: bool = True) -> dict:
    """
    Walk-forward backtest:
      - Fit models on initial train window
      - Then for each OOS day:
          * optionally refit models on expanding window up to t-1 (every refit_every days)
          * compute posterior state for day t using history ending at t
          * translate states -> position for day t (applied to ret_fwd[t])
      - Transaction cost applied on position changes (simple turnover model)

    Trading rule (simple, transparent):
      - Risk-on if:
          * price_state has positive in-sample mean forward return
          * and vol_state is NOT among the top-k highest-volatility states (risk-off bucket)
      - Else: cash (0 exposure)

    Returns:
      dict with equity curves, positions, diagnostics, and state summary.
    """
    if len(features) != len(ret_fwd) or not features.index.equals(ret_fwd.index):
        raise ValueError("features and ret_fwd must be aligned on the same index.")

    idx = features.index

    # Define train/test split by date
    split_date = idx[0] + pd.DateOffset(days=int(365.25 * train_years))
    split_date = idx[idx.get_indexer([split_date], method="nearest")[0]]
    train_mask = idx < split_date
    test_mask = ~train_mask

    if train_mask.sum() < 300:
        raise ValueError("Training window is too small; increase period or train_years.")

    # Storage
    pos = pd.Series(index=idx, dtype=float)
    pos[:] = 0.0
    price_state_series = pd.Series(index=idx, dtype="Int64")
    vol_state_series = pd.Series(index=idx, dtype="Int64")
    price_conf_series = pd.Series(index=idx, dtype=float)
    vol_conf_series = pd.Series(index=idx, dtype=float)

    # Convert bps to decimal cost per 1.0 turnover
    tc = tc_bps / 1e4

    # Helper that refits and refreshes state maps
    def refit_and_refresh(train_end_idx: int):
        train_feats = features.iloc[:train_end_idx]
        train_ret = ret_fwd.iloc[:train_end_idx]

        # Separate feature sets like your original idea
        price_feats = train_feats[["log_return", "momentum"]]
        vol_feats = train_feats[["volatility"]]

        p_model, p_scaler = fit_hmm(price_feats, n_states=price_states)
        v_model, v_scaler = fit_hmm(vol_feats, n_states=vol_states)

        # Empirical state summaries for rule mapping
        # Empirical state summaries for rule mapping
        p_stats = summarize_states(p_model, p_scaler, price_feats, train_ret)
        v_stats = summarize_states(v_model, v_scaler, vol_feats, train_ret)

        # Price "good states": positive mean forward return
        good_price_states = set(p_stats.index[p_stats["mean_fwd_ret"] > 0].tolist())

        # Vol "risk-off states": top-k by mean volatility
        vol_mean_col = "mean_volatility"
        if vol_mean_col not in v_stats.columns:
            raise ValueError(f"Expected {vol_mean_col} in v_stats, got columns={v_stats.columns.tolist()}")

        riskoff_vol_states = set(
            v_stats.sort_values(vol_mean_col, ascending=False)
            .head(vol_riskoff_top_k)
            .index.tolist()
        )

        return (p_model, p_scaler, v_model, v_scaler, p_stats, v_stats, good_price_states, riskoff_vol_states)

    # Initial fit using the initial train window
    train_end = train_mask.sum()
    (p_model, p_scaler, v_model, v_scaler,
     p_stats, v_stats, good_price_states, riskoff_vol_states) = refit_and_refresh(train_end)

    if verbose:
        print(f"Train/Test split date: {split_date.date()} (train obs={train_end}, test obs={test_mask.sum()})")
        print("\nIn-sample PRICE state stats:")
        print(p_stats)
        print("\nIn-sample VOL state stats:")
        print(v_stats)
        print(f"\nGood price states (mean_fwd_ret>0): {sorted(good_price_states)}")
        print(f"Risk-off vol states (top {vol_riskoff_top_k} by vol): {sorted(riskoff_vol_states)}")

    # Walk forward through the test period
    test_indices = np.where(test_mask)[0]

    for k, i in enumerate(test_indices):
        # Refit every refit_every steps on expanding window up to i-1
        if k > 0 and (k % refit_every == 0):
            # Fit using data up to (current day - 1)
            (p_model, p_scaler, v_model, v_scaler,
             p_stats, v_stats, good_price_states, riskoff_vol_states) = refit_and_refresh(i)

        # Compute posteriors for day i using history up to i (inclusive)
        price_hist = features.iloc[: i + 1][["log_return", "momentum"]]
        vol_hist = features.iloc[: i + 1][["volatility"]]

        p_post = posterior_last(p_model, p_scaler, price_hist, lookback=posterior_lookback)
        v_post = posterior_last(v_model, v_scaler, vol_hist, lookback=posterior_lookback)

        p_state, p_conf = argmax_state(p_post)
        v_state, v_conf = argmax_state(v_post)

        price_state_series.iloc[i] = p_state
        vol_state_series.iloc[i] = v_state
        price_conf_series.iloc[i] = p_conf
        vol_conf_series.iloc[i] = v_conf

        # Rule -> position for day i (applies to forward return ret_fwd[i])
        risk_on = (p_state in good_price_states) and (v_state not in riskoff_vol_states)
        pos.iloc[i] = 1.0 if risk_on else 0.0

    # Ensure train period positions are 0 (pure OOS)
    pos.loc[train_mask] = 0.0

    # Transaction costs on position changes (turnover)
    turnover = pos.diff().abs().fillna(0.0)
    strat_ret = pos * ret_fwd - tc * turnover

    # Benchmark buy & hold (only on OOS portion for fair comparison)
    bh_ret = ret_fwd.copy()
    bh_ret.loc[train_mask] = 0.0

    eq = (1.0 + strat_ret.fillna(0.0)).cumprod()
    eq_bh = (1.0 + bh_ret.fillna(0.0)).cumprod()

    out = {
        "split_date": split_date,
        "pos": pos,
        "turnover": turnover,
        "strategy_ret": strat_ret,
        "bh_ret": bh_ret,
        "eq": eq,
        "eq_bh": eq_bh,
        "price_state": price_state_series,
        "vol_state": vol_state_series,
        "price_conf": price_conf_series,
        "vol_conf": vol_conf_series,
        "price_state_stats_last_fit": p_stats,
        "vol_state_stats_last_fit": v_stats,
        "good_price_states_last_fit": good_price_states,
        "riskoff_vol_states_last_fit": riskoff_vol_states,
    }
    return out


# ============================================================
# 6) METRICS + PLOTS
# ============================================================
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_table(ret: pd.Series, equity: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 2:
        return {}

    ann_ret = (equity.iloc[-1] ** (252 / len(r))) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(252)
    sharpe = (r.mean() / r.std(ddof=0) * np.sqrt(252)) if r.std(ddof=0) > 0 else np.nan
    mdd = max_drawdown(equity)
    hit = float((r > 0).mean())
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "hit_rate": hit,
    }


def plot_results(res: dict, title: str = "HMM Regime OOS Backtest"):
    eq = res["eq"]
    eq_bh = res["eq_bh"]
    split_date = res["split_date"]

    plt.figure(figsize=(11, 5))
    plt.plot(eq.index, eq.values, label="Strategy")
    plt.plot(eq_bh.index, eq_bh.values, label="Buy & Hold (OOS)")
    plt.axvline(split_date, linestyle="--", linewidth=1, label="Train/Test split")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Drawdown
    peak = eq.cummax()
    dd = eq / peak - 1.0
    plt.figure(figsize=(11, 3))
    plt.plot(dd.index, dd.values)
    plt.axvline(split_date, linestyle="--", linewidth=1)
    plt.title("Strategy Drawdown (OOS)")
    plt.grid(True)
    plt.show()


# ============================================================
# 7) RUN
# ============================================================
if __name__ == "__main__":
    df = download_data("SPY", period="10y")  # use longer history so OOS is meaningful
    features, close, ret_fwd = engineer_features(df, mom_window=10, vol_window=5)

    res = walkforward_backtest(
        features,
        ret_fwd,
        train_years=5.0,          # initial in-sample
        price_states=5,
        vol_states=4,
        refit_every=21,           # refit monthly
        posterior_lookback=120,   # use ~6 months history for posterior
        vol_riskoff_top_k=1,
        tc_bps=2.0,               # 2 bps per turnover
        verbose=True
    )

    # Metrics on OOS only
    split = res["split_date"]
    oos = res["strategy_ret"].loc[res["strategy_ret"].index >= split]
    oos_eq = res["eq"].loc[res["eq"].index >= split]

    bh_oos = res["bh_ret"].loc[res["bh_ret"].index >= split]
    bh_eq = res["eq_bh"].loc[res["eq_bh"].index >= split]

    print("\n=== Strategy OOS Performance ===")
    print(perf_table(oos, oos_eq))

    print("\n=== Buy & Hold OOS Performance ===")
    print(perf_table(bh_oos, bh_eq))

    plot_results(res, title="HMM Regime Engine — Walk-Forward OOS Backtest")
