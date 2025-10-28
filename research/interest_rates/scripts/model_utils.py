from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


def oos_prob(df, X_cols, y_col, start_idx=180, min_pos=5, min_neg=60, C=1.0):
    dates = df.index
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", C=C, solver="lbfgs", class_weight="balanced", max_iter=500))
    ])
    out = []
    for t in range(start_idx, len(dates)):
        tr = slice(0, t)
        Xtr = df[X_cols].iloc[tr].values
        ytr = df[y_col].iloc[tr].astype(int).values

        if (ytr==1).sum() < min_pos or (ytr==0).sum() < min_neg:
            continue

        pipe.fit(Xtr, ytr)
        xte = df[X_cols].iloc[[t]].values
        phat = pipe.predict_proba(xte)[0, 1].item()
        out.append((dates[t], phat, int(df[y_col].iloc[t])))

    oos = pd.DataFrame(out, columns=["date","phat","y"]).set_index("date")

    return oos, {"AUC": roc_auc_score(oos["y"], oos["phat"]), "AP": average_precision_score(oos["y"], oos["phat"])}


def oos_multivariate(df_in: pd.DataFrame, X_cols, y_col,
                     start_idx=180, min_pos=5, min_neg=60,
                     C=1.0, max_iter=500):
    dates = df_in.index
    phat, y, dts = [], [], []

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            penalty="l2", C=C, solver="lbfgs",
            class_weight="balanced", max_iter=max_iter, n_jobs=None
        ))
    ])

    for t in range(start_idx, len(dates)):
        tr = slice(0, t)
        Xtr = df_in[X_cols].iloc[tr].values
        ytr = df_in[y_col].iloc[tr].astype(int).values

        # require both classes and some minimum support
        if (ytr==1).sum() < min_pos or (ytr==0).sum() < min_neg:
            continue

        pipe.fit(Xtr, ytr)
        xte = df_in[X_cols].iloc[[t]].values
        prob = pipe.predict_proba(xte)[0,1].item()

        phat.append(prob)
        y.append(int(df_in[y_col].iloc[t]))
        dts.append(dates[t])

    oos = pd.DataFrame({"phat": phat, "y": y}, index=pd.DatetimeIndex(dts))
    return oos, pipe