from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BuildConfig:
    raw_csv: Path = PROJECT_ROOT / "data/raw/yellow_taxi_2023_jan_mar.csv"
    out_csv: Path = PROJECT_ROOT / "data/processed/daily_trip_count.csv"
    pickup_col: str = "tpep_pickup_datetime"


def build_daily_series(cfg: BuildConfig) -> pd.DataFrame:
    if not cfg.raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found: {cfg.raw_csv}")

    df = pd.read_csv(cfg.raw_csv, usecols=[cfg.pickup_col])

    df[cfg.pickup_col] = pd.to_datetime(df[cfg.pickup_col], errors="coerce")
    df = df.dropna(subset=[cfg.pickup_col])

    daily = (
        df.assign(ds=df[cfg.pickup_col].dt.floor("D"))
          .groupby("ds", as_index=False)
          .size()
          .rename(columns={"size": "y"})
          .sort_values("ds")
    )

    daily["ds"] = pd.to_datetime(daily["ds"])
    daily["y"] = daily["y"].astype(float)

    return daily


def main() -> None:
    cfg = BuildConfig()
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)

    daily = build_daily_series(cfg)
    daily.to_csv(cfg.out_csv, index=False)

    print(f"[build] saved -> {cfg.out_csv} (rows={len(daily)})")
    print("[build] head:")
    print(daily.head())
    print("[build] tail:")
    print(daily.tail())


if __name__ == "__main__":
    main()
