from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from app.analytics.dataset import load_btc_daily_returns


DATA_DIR = Path(__file__).resolve().parent / "data"
FED_RATE_EVENTS_PATH = DATA_DIR / "fed_rate_events.jsonl"


@dataclass
class FedRateAnovaResult:
    levels: List[str]
    group_sizes: List[int]
    group_means: List[float]
    f_stat: float
    p_value: float


def load_fed_rate_events(path: Path | str = FED_RATE_EVENTS_PATH) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    df["event_date"] = pd.to_datetime(df["event_date"])

    df["rate"] = df["actual"].astype(float)
    mask = df["rate"].isna()
    df.loc[mask, "rate"] = df.loc[mask, "forecast"].astype(float)

    df = df.sort_values("event_date").reset_index(drop=True)
    return df[["event_date", "rate"]]


def build_daily_fed_rate(events: pd.DataFrame,
                         index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(
        events["rate"].to_numpy(dtype=float),
        index=pd.to_datetime(events["event_date"]).dt.normalize(),
    )

    s = s[~s.index.duplicated(keep="last")]

    daily = s.reindex(index, method="ffill")
    return daily


def categorize_rate(rate: float) -> str:
    if rate <= 1.0:
        return "Low (≤1%)"
    elif rate <= 3.0:
        return "Medium (1–3%)"
    return "High (>3%)"


def compute_fed_rate_anova(
    returns: pd.Series,
) -> tuple[FedRateAnovaResult, pd.DataFrame]:
    clean = returns.dropna()
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError("Індекс ряду дохідностей повинен бути DatetimeIndex")

    events = load_fed_rate_events()
    daily_rate = build_daily_fed_rate(events, clean.index)

    df = pd.DataFrame(
        {
            "return": clean.to_numpy(dtype=float),
            "fed_rate": daily_rate.to_numpy(dtype=float),
        },
        index=clean.index,
    ).dropna(subset=["fed_rate"])

    df["rate_regime"] = df["fed_rate"].apply(categorize_rate)

    summary = (
        df.groupby("rate_regime")["return"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("rate_regime")
    )

    groups: list[np.ndarray] = []
    levels: List[str] = []

    for _, row in summary.iterrows():
        name = str(row["rate_regime"])
        data = df.loc[df["rate_regime"] == name, "return"].to_numpy(dtype=float)

        if data.size >= 2:
            levels.append(name)
            groups.append(data)

    if len(groups) < 2:
        raise ValueError("Для ANOVA потрібно принаймні дві групи з даними")

    f_stat, p_val = stats.f_oneway(*groups)

    group_sizes = [g.size for g in groups]
    group_means = [float(np.mean(g)) for g in groups]

    result = FedRateAnovaResult(
        levels=levels,
        group_sizes=group_sizes,
        group_means=group_means,
        f_stat=float(f_stat),
        p_value=float(p_val),
    )

    summary = summary.set_index("rate_regime")
    return result, summary


def run_fed_rate_anova_analysis() -> None:
    returns = load_btc_daily_returns()
    result, summary = compute_fed_rate_anova(returns)

    print("Однофакторний дисперсійний аналіз (ANOVA) за режимами ключової ставки ФРС")
    print(f"F-статистика: {result.f_stat:.4f}")
    print(f"p-value:      {result.p_value:.4g}")
    print("\nЗведена таблиця за режимами ставки:")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    run_fed_rate_anova_analysis()
