from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from app.analytics.dataset import load_btc_daily_returns


DATA_DIR = Path(__file__).resolve().parent / "data"
FED_RATE_EVENTS_PATH = DATA_DIR / "fed_rate_events.jsonl"

ImpactMode = Literal["event_day", "post_k_days", "pre_k_days"]


@dataclass
class FedRateAnovaResult:
    levels: List[str]
    group_sizes: List[int]
    group_means: List[float]
    f_stat: float
    p_value: float
    levene_p_value: float | None = None
    eta2: float | None = None
    mode: ImpactMode | None = None
    k: int | None = None


def load_fed_rate_events(path: Path | str = FED_RATE_EVENTS_PATH) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()

    df["actual"] = pd.to_numeric(df.get("actual"), errors="coerce")
    df["forecast"] = pd.to_numeric(df.get("forecast"), errors="coerce")
    df["previous"] = pd.to_numeric(df.get("previous"), errors="coerce")

    df["rate"] = df["actual"]
    na_mask = df["rate"].isna()
    df.loc[na_mask, "rate"] = df.loc[na_mask, "forecast"]

    df["delta"] = df["rate"] - df["previous"]

    def classify(d: float) -> str:
        if pd.isna(d):
            return "unknown"
        if d > 0:
            return "hike"
        if d < 0:
            return "cut"
        return "hold"

    df["decision_class"] = df["delta"].apply(classify)

    df = df.sort_values("event_date").reset_index(drop=True)
    return df[["event_date", "rate", "previous", "delta", "decision_class"]]


def _cum_return(returns: pd.Series, start_i: int, end_i: int) -> float:
    if start_i > end_i:
        return np.nan
    seg = returns.iloc[start_i : end_i + 1].astype(float)
    seg = seg.dropna()
    if seg.empty:
        return np.nan
    return float((1.0 + seg).prod() - 1.0)


def build_event_impacts(
    returns: pd.Series,
    *,
    mode: ImpactMode = "post_k_days",
    k: int = 3,
    require_full_window: bool = True,
) -> pd.DataFrame:
    clean = returns.dropna().copy()
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError("Індекс ряду дохідностей повинен бути DatetimeIndex")

    clean.index = pd.to_datetime(clean.index).normalize()
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]

    events = load_fed_rate_events()
    events = events[events["decision_class"].isin(["hike", "cut", "hold"])].copy()

    idx_map = {d: i for i, d in enumerate(clean.index.to_list())}

    rows: list[dict] = []
    n = len(clean)

    for _, e in events.iterrows():
        d = pd.to_datetime(e["event_date"]).normalize()
        if d not in idx_map:
            continue

        i = idx_map[d]

        if mode == "event_day":
            impact = float(clean.iloc[i])

        elif mode == "post_k_days":
            start_i = i + 1
            end_i = i + k
            if require_full_window and (start_i >= n or end_i >= n):
                continue
            start_i = min(start_i, n - 1)
            end_i = min(end_i, n - 1)
            impact = _cum_return(clean, start_i, end_i)

        elif mode == "pre_k_days":
            start_i = i - k
            end_i = i - 1
            if require_full_window and (start_i < 0 or end_i < 0):
                continue
            start_i = max(start_i, 0)
            end_i = max(end_i, 0)
            impact = _cum_return(clean, start_i, end_i)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if np.isnan(impact):
            continue

        rows.append(
            {
                "event_date": d,
                "group": str(e["decision_class"]),
                "impact": float(impact),
                "delta": float(e["delta"]) if pd.notna(e["delta"]) else np.nan,
                "rate": float(e["rate"]) if pd.notna(e["rate"]) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("event_date").reset_index(drop=True)
    return out


def compute_fed_rate_anova(
    returns: pd.Series,
    *,
    mode: ImpactMode = "post_k_days",
    k: int = 3,
    require_full_window: bool = True,
) -> Tuple[FedRateAnovaResult, pd.DataFrame, pd.DataFrame]:
    
    impacts = build_event_impacts(
        returns,
        mode=mode,
        k=k,
        require_full_window=require_full_window,
    )

    if impacts.empty:
        raise ValueError("Немає подій/даних для побудови impact")

    summary = (
        impacts.groupby("group")["impact"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("group")
    )

    groups: list[np.ndarray] = []
    levels: List[str] = []
    for g in ["hike", "cut", "hold"]:
        data = impacts.loc[impacts["group"] == g, "impact"].to_numpy(dtype=float)
        if data.size >= 2:
            levels.append(g)
            groups.append(data)

    if len(groups) < 2:
        raise ValueError("Для ANOVA потрібно принаймні дві групи з >=2 спостереженнями")

    levene_p = None
    try:
        _, levene_p = stats.levene(*groups)
        levene_p = float(levene_p)
    except Exception:
        levene_p = None

    f_stat, p_val = stats.f_oneway(*groups)

    grand_mean = float(np.mean(impacts["impact"].to_numpy(dtype=float)))
    ss_between = 0.0
    for g, arr in zip(levels, groups):
        ss_between += float(arr.size) * (float(np.mean(arr)) - grand_mean) ** 2
    ss_total = float(np.sum((impacts["impact"].to_numpy(dtype=float) - grand_mean) ** 2))
    eta2 = (ss_between / ss_total) if ss_total > 0 else np.nan

    result = FedRateAnovaResult(
        levels=levels,
        group_sizes=[int(a.size) for a in groups],
        group_means=[float(np.mean(a)) for a in groups],
        f_stat=float(f_stat),
        p_value=float(p_val),
        levene_p_value=levene_p,
        eta2=float(eta2) if not np.isnan(eta2) else None,
        mode=mode,
        k=int(k),
    )

    return result, summary.set_index("group"), impacts


def run_fed_rate_anova_analysis() -> None:
    returns = load_btc_daily_returns()

    result, summary, impacts = compute_fed_rate_anova(returns, mode="post_k_days", k=3)

    print("ANOVA (Fed rate decision: hike/cut/hold) → impact BTC")
    print(f"Mode: {result.mode}, k={result.k}")
    print(f"F-статистика: {result.f_stat:.4f}")
    print(f"p-value:      {result.p_value:.4g}")
    if result.levene_p_value is not None:
        print(f"Levene p:     {result.levene_p_value:.4g}")
    if result.eta2 is not None:
        print(f"eta²:         {result.eta2:.4g}")

    print("\nЗведена таблиця за групами:")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    run_fed_rate_anova_analysis()
