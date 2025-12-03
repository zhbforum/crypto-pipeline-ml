from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from app.analytics.dataset import load_btc_daily_returns


def fit_normal(series: pd.Series) -> Tuple[float, float]:
    clean = series.dropna().to_numpy()
    if clean.size < 2:
        raise ValueError("Not enough data to fit normal distribution")
    mu = float(np.mean(clean))
    sigma = float(np.std(clean, ddof=1))
    if sigma <= 0:
        raise ValueError("Non-positive sigma for normal fit")
    return mu, sigma


def fit_student_t(series: pd.Series) -> Tuple[float, float, float]:
    clean = series.dropna().to_numpy()
    if clean.size < 3:
        raise ValueError("Not enough data to fit t-distribution")
    df, loc, scale = stats.t.fit(clean)
    if scale <= 0:
        raise ValueError("Non-positive scale for t fit")
    return float(df), float(loc), float(scale)


def compute_empirical_tail_prob(series: pd.Series, threshold: float) -> float:
    clean = series.dropna().to_numpy()
    if clean.size == 0:
        raise ValueError("Empty series for tail probability")
    count_tail = np.sum(np.abs(clean) > threshold)
    return float(count_tail / clean.size)


def compute_model_tail_prob_normal(mu: float, sigma: float, threshold: float) -> float:
    if sigma <= 0:
        raise ValueError("Non-positive sigma")
    z1 = (threshold - mu) / sigma
    z2 = (-threshold - mu) / sigma
    tail = stats.norm.sf(z1) + stats.norm.cdf(z2)
    return float(tail)


def compute_model_tail_prob_t(
    df: float,
    loc: float,
    scale: float,
    threshold: float,
) -> float:
    if scale <= 0:
        raise ValueError("Non-positive scale")
    z1 = (threshold - loc) / scale
    z2 = (-threshold - loc) / scale
    tail = stats.t.sf(z1, df) + stats.t.cdf(z2, df)
    return float(tail)


def kolmogorov_smirnov_tests(series: pd.Series) -> dict[str, Tuple[float, float]]:
    clean = series.dropna().to_numpy()
    if clean.size < 10:
        raise ValueError("Not enough data for K-S test")

    mu, sigma = fit_normal(series)
    df_t, loc_t, scale_t = fit_student_t(series)

    ks_norm = stats.kstest(clean, "norm", args=(mu, sigma))

    def cdf_t(x: np.ndarray) -> np.ndarray:
        return stats.t.cdf(x, df_t, loc=loc_t, scale=scale_t)

    ks_t = stats.kstest(clean, cdf_t)

    return {
        "normal": (float(ks_norm.statistic), float(ks_norm.pvalue)),
        "student_t": (float(ks_t.statistic), float(ks_t.pvalue)),
    }


def chi_square_normal(
    series: pd.Series,
    bins: int = 40,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    clean = series.dropna().to_numpy()
    n = clean.size
    if n < bins * 2:
        raise ValueError("Not enough data for chi-square test")

    mu, sigma = fit_normal(series)
    counts, bin_edges = np.histogram(clean, bins=bins)

    cdf_edges = stats.norm.cdf(bin_edges, loc=mu, scale=sigma)
    expected_probs = np.diff(cdf_edges)
    expected_counts = expected_probs * n

    mask = expected_counts >= 5
    if np.count_nonzero(mask) < 3:
        raise ValueError("Too few bins with expected count >= 5 for chi-square test")

    observed_masked = counts[mask]
    expected_masked = expected_counts[mask]

    k_effective = np.count_nonzero(mask)
    df_chi = k_effective - 1 - 2
    if df_chi <= 0:
        raise ValueError("Non-positive degrees of freedom for chi-square test")

    chi2_stat = float(np.sum((observed_masked - expected_masked) ** 2 / expected_masked))
    p_value = float(1.0 - stats.chi2.cdf(chi2_stat, df_chi))

    return chi2_stat, p_value, counts, expected_counts, bin_edges


def plot_histogram_with_fits(
    series: pd.Series,
    output_dir: Optional[Path] = None,
    bins: int = 80,
) -> Path:
    clean = series.dropna().to_numpy()
    if clean.size < 10:
        raise ValueError("Not enough data for histogram")

    mu, sigma = fit_normal(series)
    df_t, loc_t, scale_t = fit_student_t(series)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hist_returns_normal_t.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bin_edges, _ = ax.hist(
        clean,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Empirical returns",
    )

    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    x = np.linspace(x_min, x_max, 1000)

    pdf_norm = stats.norm.pdf(x, loc=mu, scale=sigma)
    pdf_t = stats.t.pdf((x - loc_t) / scale_t, df_t) / scale_t

    ax.plot(x, pdf_norm, linewidth=2, label=f"Normal(mu={mu:.4f}, sigma={sigma:.4f})")
    ax.plot(
        x,
        pdf_t,
        linewidth=2,
        linestyle="--",
        label=f"Student t(df={df_t:.2f}, loc={loc_t:.4f}, scale={scale_t:.4f})",
    )

    ax.set_xlabel("Daily log returns")
    ax.set_ylabel("Density")
    ax.set_title("BTC/USDT daily log returns: histogram and fitted distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_qq(
    series: pd.Series,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    clean = series.dropna().to_numpy()
    if clean.size < 10:
        raise ValueError("Not enough data for QQ-plot")

    mu, sigma = fit_normal(series)
    df_t, loc_t, scale_t = fit_student_t(series)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    path_norm = output_dir / "qq_normal.png"
    fig_norm, ax_norm = plt.subplots(figsize=(6, 6))
    stats.probplot(clean, dist="norm", sparams=(mu, sigma), plot=ax_norm)
    ax_norm.set_title("QQ-plot vs Normal")
    fig_norm.tight_layout()
    fig_norm.savefig(path_norm)
    plt.close(fig_norm)

    path_t = output_dir / "qq_t.png"
    fig_t, ax_t = plt.subplots(figsize=(6, 6))

    sorted_data = np.sort(clean)
    n = sorted_data.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    quantiles_t = stats.t.ppf(probs, df_t, loc=loc_t, scale=scale_t)

    ax_t.scatter(quantiles_t, sorted_data, s=5)
    min_val = min(np.min(quantiles_t), np.min(sorted_data))
    max_val = max(np.max(quantiles_t), np.max(sorted_data))
    ax_t.plot([min_val, max_val], [min_val, max_val], linewidth=1)
    ax_t.set_xlabel("Theoretical quantiles (Student t)")
    ax_t.set_ylabel("Empirical quantiles")
    ax_t.set_title("QQ-plot vs Student t")
    fig_t.tight_layout()
    fig_t.savefig(path_t)
    plt.close(fig_t)

    return path_norm, path_t


def plot_chi_square_observed_vs_expected(
    counts: np.ndarray,
    expected_counts: np.ndarray,
    bin_edges: np.ndarray,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "chi_square_observed_vs_expected.png"

    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        centers,
        counts,
        width=widths,
        alpha=0.6,
        align="center",
        label="Observed counts",
    )
    ax.bar(
        centers,
        expected_counts,
        width=widths,
        alpha=0.4,
        align="center",
        label="Expected counts (Normal)",
    )

    ax.set_xlabel("Daily log returns bins")
    ax.set_ylabel("Counts")
    ax.set_title("Chi-square: observed vs expected bin counts (Normal fit)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def run_distribution_analysis() -> None:
    returns = load_btc_daily_returns()
    mu, sigma = fit_normal(returns)
    df_t, loc_t, scale_t = fit_student_t(returns)

    print("Distribution analysis for BTC daily log returns")
    print(f"Normal fit: mu={mu:.6f}, sigma={sigma:.6f}")
    print(f"Student t fit: df={df_t:.3f}, loc={loc_t:.6f}, scale={scale_t:.6f}")

    threshold = 0.10
    emp_tail = compute_empirical_tail_prob(returns, threshold)
    norm_tail = compute_model_tail_prob_normal(mu, sigma, threshold)
    t_tail = compute_model_tail_prob_t(df_t, loc_t, scale_t, threshold)

    print(f"Empirical P(|R| > {threshold:.2f}): {emp_tail:.6f}")
    print(f"Normal model P(|R| > {threshold:.2f}): {norm_tail:.6f}")
    print(f"Student t model P(|R| > {threshold:.2f}): {t_tail:.6f}")

    ks_results = kolmogorov_smirnov_tests(returns)
    ks_norm_D, ks_norm_p = ks_results["normal"]
    ks_t_D, ks_t_p = ks_results["student_t"]

    print("\nKolmogorov–Smirnov tests:")
    print(f"  Normal:    D = {ks_norm_D:.4f}, p-value = {ks_norm_p:.4g}")
    print(f"  Student t: D = {ks_t_D:.4f}, p-value = {ks_t_p:.4g}")

    chi2_stat, chi2_p, counts, exp_counts, bin_edges = chi_square_normal(returns, bins=40)
    print("\nChi-square goodness-of-fit vs Normal:")
    print(f"  chi2 = {chi2_stat:.2f}, p-value = {chi2_p:.4g}")

    output_dir = Path(__file__).resolve().parent / "data"
    hist_path = plot_histogram_with_fits(returns, output_dir=output_dir)
    qq_norm_path, qq_t_path = plot_qq(returns, output_dir=output_dir)
    chi_plot_path = plot_chi_square_observed_vs_expected(
        counts, exp_counts, bin_edges, output_dir=output_dir
    )

    print(f"\nSaved histogram with fits to: {hist_path}")
    print(f"Saved QQ-plot vs normal to: {qq_norm_path}")
    print(f"Saved QQ-plot vs Student t to: {qq_t_path}")
    print(f"Saved chi-square observed vs expected plot to: {chi_plot_path}")
