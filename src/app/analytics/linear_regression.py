from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from app.analytics.dataset import load_btc_daily_returns


def fit_ar1_regression(series: pd.Series) -> Tuple[Any, pd.DataFrame, pd.Series]:
    clean = series.dropna()
    y = clean.iloc[1:]
    x_lag = clean.shift(1).iloc[1:]

    x = pd.DataFrame(
        {
            "const": 1.0,
            "lag1": x_lag.to_numpy(dtype=float),
        },
        index=x_lag.index,
    )

    model = sm.OLS(y, x).fit()
    return model, x, y


def forecast_ar1(model: Any, last_value: float) -> float:
    params = model.params.to_numpy(dtype=float)
    if params.size < 2:
        raise ValueError("AR(1) model must have constant and lag coefficient")
    const = params[0]
    coef = params[1]
    return float(const + coef * last_value)


def plot_ar1_fit(
    y: pd.Series,
    y_pred: np.ndarray,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "ar1_fitted.png"

    x_dates = list(y.index)
    y_true = y.to_numpy(dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_dates, y_true, label="Actual")
    ax.plot(x_dates, y_hat, label="Fitted")
    ax.set_title("AR(1) Regression Fit for BTC Daily Log-Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def run_linear_regression_analysis() -> None:
    returns = load_btc_daily_returns()
    model, x, y = fit_ar1_regression(returns)

    print("AR(1) regression summary:")
    print(model.summary())

    y_pred = model.predict(x)

    output_dir = Path(__file__).resolve().parent / "data"
    plot_path = plot_ar1_fit(y, y_pred, output_dir=output_dir)
    print(f"Saved AR(1) regression plot to: {plot_path}")

    last_value = float(returns.dropna().iloc[-1])
    fc_next = forecast_ar1(model, last_value)
    print(f"One-step ahead forecast: {fc_next}")
