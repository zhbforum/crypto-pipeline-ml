from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd


ANALYTICS_ROOT = Path(__file__).resolve().parent
DEFAULT_DAILY_CSV = ANALYTICS_ROOT / "data" / "btc_daily_close.csv"


def load_btc_daily_close(
    csv_path: Optional[str | Path] = None,
) -> pd.Series:
    path = Path(csv_path) if csv_path is not None else DEFAULT_DAILY_CSV
    if not path.is_file():
        raise FileNotFoundError(f"CSV file with data not found: {path}")

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("Column 'date' is required in daily CSV")
    if "close" not in df.columns:
        raise ValueError("Column 'close' is required in daily CSV")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    daily_close = df.set_index("date")["close"].astype(float)
    daily_close.name = "close"

    return daily_close


def load_btc_daily_returns(
    csv_path: Optional[str | Path] = None,
    log_returns: bool = True,
) -> pd.Series:
    close = load_btc_daily_close(csv_path)
    close = close.sort_index()

    if log_returns:
        ratio = close / close.shift(1)
        returns = ratio.apply(
            lambda x: math.log(x) if (pd.notnull(x) and x > 0) else float("nan")
        )
        returns.name = "log_return"
    else:
        returns = close.pct_change()
        returns.name = "return"

    returns = returns.dropna()
    return returns
