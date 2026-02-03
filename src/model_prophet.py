from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from prophet import Prophet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelConfig:
    series_csv: Path = PROJECT_ROOT / "data/processed/daily_trip_count.csv"
    horizon_days: int = 30
    out_forecast_csv: Path = PROJECT_ROOT / "data/processed/forecast_30d.csv"
    out_plot_forecast: Path = PROJECT_ROOT / "reports/figures/prophet_forecast.png"
    out_plot_components: Path = PROJECT_ROOT / "reports/figures/prophet_components.png"


def train_and_forecast(cfg: ModelConfig):
    if not cfg.series_csv.exists():
        raise FileNotFoundError(f"Time series CSV not found: {cfg.series_csv}")

    df = pd.read_csv(cfg.series_csv)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=cfg.horizon_days, freq="D")
    forecast = model.predict(future)

    return df, model, forecast


def main() -> None:
    cfg = ModelConfig()
    cfg.out_forecast_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_plot_forecast.parent.mkdir(parents=True, exist_ok=True)

    df, model, forecast = train_and_forecast(cfg)

    keep_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    forecast[keep_cols].to_csv(cfg.out_forecast_csv, index=False)
    print(f"[model] saved forecast -> {cfg.out_forecast_csv}")

    fig1 = model.plot(forecast)
    fig1.savefig(cfg.out_plot_forecast, bbox_inches="tight")
    print(f"[model] saved plot -> {cfg.out_plot_forecast}")

    fig2 = model.plot_components(forecast)
    fig2.savefig(cfg.out_plot_components, bbox_inches="tight")
    print(f"[model] saved components -> {cfg.out_plot_components}")

    print(f"[model] train rows={len(df)}, forecast rows={len(forecast)}")


if __name__ == "__main__":
    main()
