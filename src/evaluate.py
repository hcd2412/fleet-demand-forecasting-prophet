from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalConfig:
    series_csv: Path = PROJECT_ROOT / "data/processed/daily_trip_count.csv"
    test_days: int = 15
    out_metrics_csv: Path = PROJECT_ROOT / "reports/metrics/backtest_metrics.csv"
    out_plot: Path = PROJECT_ROOT / "reports/figures/backtest_actual_vs_pred.png"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def main() -> None:
    cfg = EvalConfig()
    if not cfg.series_csv.exists():
        raise FileNotFoundError(f"Missing time series: {cfg.series_csv}")

    df = pd.read_csv(cfg.series_csv)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")

    if len(df) <= cfg.test_days + 10:
        raise ValueError("Not enough data for backtest. Increase date range or reduce test_days.")

    train = df.iloc[:-cfg.test_days].copy()
    test = df.iloc[-cfg.test_days:].copy()

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=cfg.test_days, freq="D")
    fcst = model.predict(future)[["ds", "yhat"]]

    pred = test.merge(fcst, on="ds", how="left").dropna(subset=["yhat"])

    mae = float(np.mean(np.abs(pred["y"].values - pred["yhat"].values)))
    mape_pct = mape(pred["y"].values, pred["yhat"].values)

    cfg.out_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_plot.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(
        [{
            "train_start": train["ds"].min().date(),
            "train_end": train["ds"].max().date(),
            "test_start": test["ds"].min().date(),
            "test_end": test["ds"].max().date(),
            "test_days": int(cfg.test_days),
            "mae": mae,
            "mape_percent": mape_pct,
        }]
    )
    metrics.to_csv(cfg.out_metrics_csv, index=False)
    print(f"[eval] saved metrics -> {cfg.out_metrics_csv}")
    print(f"[eval] MAE={mae:,.2f}  MAPE={mape_pct:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(pred["ds"], pred["y"], label="actual")
    plt.plot(pred["ds"], pred["yhat"], label="predicted")
    plt.xticks(rotation=45)
    plt.xlabel("date")
    plt.ylabel("daily trips")
    plt.title("Backtest: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_plot)
    print(f"[eval] saved plot -> {cfg.out_plot}")


if __name__ == "__main__":
    main()
