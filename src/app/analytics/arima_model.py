from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima

from app.analytics.datasets import load_btc_daily_returns


def fit_arima_model(
    series: pd.Series,
    seasonal: bool = False,
) -> Tuple[Any, pd.Series]:
    clean = series.dropna()
    model = auto_arima(
        clean,
        seasonal=seasonal,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model, clean


def forecast_arima(
    model: Any,
    periods: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    forecast_values, conf_int = model.predict(
        n_periods=periods,
        return_conf_int=True,
    )
    fc_values = np.asarray(forecast_values, dtype=float)
    fc_ci = np.asarray(conf_int, dtype=float)
    return fc_values, fc_ci


def plot_arima_forecast(
    series: pd.Series,
    forecast: np.ndarray,
    conf_int: np.ndarray,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "arima_forecast.png"

    history_x = series.index.to_numpy() 
    history_y = series.to_numpy(dtype=float)

    last_index = series.index[-1]
    future_index = pd.date_range(
        last_index,
        periods=len(forecast) + 1,
        freq="D",
    )[1:]
    future_x = future_index.to_numpy()  

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_x, history_y, label="History")
    ax.plot(future_x, forecast, label="Forecast")

    ax.fill_between(
        future_x,
        conf_int[:, 0],
        conf_int[:, 1],
        alpha=0.3,
        label="95% CI",
    )

    ax.set_title("ARIMA Forecast of BTC Daily Log-Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def run_arima_analysis() -> None:
    returns = load_btc_daily_returns()
    model, clean = fit_arima_model(returns)

    print("ARIMA model summary:")
    print(model.summary())

    forecast, conf_int = forecast_arima(model, periods=14)

    output_dir = Path(__file__).resolve().parent / "data"
    plot_path = plot_arima_forecast(clean, forecast, conf_int, output_dir=output_dir)
    print(f"Saved ARIMA forecast plot to: {plot_path}")
