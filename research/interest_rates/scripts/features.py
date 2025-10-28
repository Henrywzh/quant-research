import pandas as pd


def build_feature_table(
        data_m: pd.DataFrame,
        factors: pd.DataFrame
) -> pd.DataFrame:
    df = pd.concat([data_m, factors], axis=1).sort_index().asfreq("ME")
    # ensure derived features exist
    if "spread_10y_2y" not in df and {"DGS10","DGS2"}.issubset(df.columns):
        df["spread_10y_2y"] = df["DGS10"] - df["DGS2"]

    return df.ffill()

