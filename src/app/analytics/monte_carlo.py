from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.analytics.setdata import load_btc_daily_close, load_btc_daily_returns
from app.analytics.distributions import fit_student_t


@dataclass
class MonteCarloResult:
    horizon_days: int
    n_paths: int
    final_returns: np.ndarray
    prob_drawdown_30: float
    var_95: float
    scenario_name: str


def simulate_paths_student_t(
    s0: float,
    df: float,
    loc: float,
    scale: float,
    horizon_days: int,
    n_paths: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    daily_log_r = rng.standard_t(df, size=(horizon_days, n_paths))
    daily_log_r = loc + scale * daily_log_r

    log_price = np.log(s0) + np.cumsum(daily_log_r, axis=0)
    prices = np.vstack([np.full((1, n_paths), s0), np.exp(log_price)])

    cum_log_return = log_price[-1, :] - np.log(s0)
    cum_return = np.exp(cum_log_return) - 1.0

    return prices, cum_return


def analyze_scenario(
    scenario_name: str,
    s0: float,
    df: float,
    loc: float,
    scale: float,
    horizon_days: int = 30,
    n_paths: int = 10_000,
) -> MonteCarloResult:
    _, cum_returns = simulate_paths_student_t(
        s0=s0,
        df=df,
        loc=loc,
        scale=scale,
        horizon_days=horizon_days,
        n_paths=n_paths,
        random_state=42,
    )

    prob_drawdown_30 = float(np.mean(cum_returns < -0.30))
    var_95 = float(np.quantile(cum_returns, 0.05))

    return MonteCarloResult(
        horizon_days=horizon_days,
        n_paths=n_paths,
        final_returns=cum_returns,
        prob_drawdown_30=prob_drawdown_30,
        var_95=var_95,
        scenario_name=scenario_name,
    )


def plot_monte_carlo_hist(
    baseline: MonteCarloResult,
    improved: MonteCarloResult,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "monte_carlo_30d_returns.png"

    r_base = baseline.final_returns
    r_impr = improved.final_returns

    q_low = float(min(np.quantile(r_base, 0.005), np.quantile(r_impr, 0.005)))
    q_high = float(max(np.quantile(r_base, 0.995), np.quantile(r_impr, 0.995)))

    r_base_clipped = r_base[(r_base >= q_low) & (r_base <= q_high)]
    r_impr_clipped = r_impr[(r_impr >= q_low) & (r_impr <= q_high)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        r_base_clipped,
        bins=80,
        alpha=0.5,
        density=True,
        label=f"{baseline.scenario_name}",
    )
    ax.hist(
        r_impr_clipped,
        bins=80,
        alpha=0.5,
        density=True,
        label=f"{improved.scenario_name}",
    )

    ax.set_title(
        "Distribution of 30-day returns (Monte Carlo)\n"
        "(central 99% of simulated outcomes)"
    )
    ax.set_xlabel("30-day return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(q_low, q_high)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def run_monte_carlo_analysis(
    horizon_days: int = 30,
    n_paths: int = 10_000,
    vol_reduction: float = 0.20,
) -> None:
    close = load_btc_daily_close()
    s0 = float(close.dropna().iloc[-1])

    returns = load_btc_daily_returns()
    df_t, loc_t, scale_t = fit_student_t(returns)

    baseline = analyze_scenario(
        scenario_name="Baseline (current volatility)",
        s0=s0,
        df=df_t,
        loc=loc_t,
        scale=scale_t,
        horizon_days=horizon_days,
        n_paths=n_paths,
    )

    scale_improved = scale_t * (1.0 - vol_reduction)
    improved = analyze_scenario(
        scenario_name=f"Reduced volatility ({int(vol_reduction * 100)}%)",
        s0=s0,
        df=df_t,
        loc=loc_t,
        scale=scale_improved,
        horizon_days=horizon_days,
        n_paths=n_paths,
    )

    output_dir = Path(__file__).resolve().parent / "data"
    plot_path = plot_monte_carlo_hist(baseline, improved, output_dir=output_dir)

    print("Monte Carlo analysis for 30-day BTC returns")
    print(f"Initial price S0 = {s0:.2f} USDT\n")

    for res in (baseline, improved):
        print(f"Scenario: {res.scenario_name}")
        print(f"  P(R_30 < -30%) = {res.prob_drawdown_30:.4f}")
        print(f"  95% VaR (quantile 5%) = {res.var_95:.4f}")
        print("")

    print(f"Saved Monte Carlo histogram to: {plot_path}")
