from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def write_run_artifacts(
    out_dir: str | Path,
    config: dict,
    returns: pd.DataFrame,
    equity: pd.DataFrame,
    stats: pd.DataFrame,
    weights: pd.DataFrame,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    returns.to_csv(out_dir / "returns.csv")
    equity.to_csv(out_dir / "equity.csv")
    stats.to_csv(out_dir / "stats.csv")
    weights.to_parquet(out_dir / "weights.parquet")
    return out_dir
