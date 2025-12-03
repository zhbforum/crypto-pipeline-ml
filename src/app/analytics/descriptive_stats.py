from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import pandas as pd

from app.analytics.dataset import load_btc_daily_returns


@dataclass
class DescriptiveStats:
    n: int
    mean: float
    variance: float
    std: float
    skewness: float
    kurtosis: float
    ci_lower: float
    ci_upper: float
    alpha: float = 0.05


def compute_descriptive_stats(
    series: pd.Series,
    alpha: float = 0.05,
) -> DescriptiveStats:
    clean = series.dropna()
    n = len(clean)
    if n < 2:
        raise ValueError("Not enough data for statistical analysis")

    mean = cast(float, clean.mean())
    variance = cast(float, clean.var(ddof=1))
    std = cast(float, clean.std(ddof=1))
    skewness = cast(float, clean.skew())
    kurtosis = cast(float, clean.kurt())

    z = 1.96
    margin = z * std / math.sqrt(n)
    ci_lower = mean - margin
    ci_upper = mean + margin

    return DescriptiveStats(
        n=n,
        mean=mean,
        variance=variance,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
    )


def run_descriptive_analysis(
    csv_path: Optional[str | Path] = None,
    log_returns: bool = True,
) -> DescriptiveStats:
    returns = load_btc_daily_returns(csv_path=csv_path, log_returns=log_returns)
    stats = compute_descriptive_stats(returns)

    print("Descriptive statistics for BTC daily returns")
    print(f"Number of observations (n): {stats.n}")
    print(f"Mean:       {stats.mean:.8f}")
    print(f"Variance:   {stats.variance:.8f}")
    print(f"Std dev:    {stats.std:.8f}")
    print(f"Skewness:   {stats.skewness:.8f}")
    print(f"Kurtosis:   {stats.kurtosis:.8f}")
    print(
        f"95% CI for mean: [{stats.ci_lower:.8f}; {stats.ci_upper:.8f}] "
        f"(alpha = {stats.alpha})"
    )

    return stats
