from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "app").is_dir():
        ROOT = parent
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analytics.setdata import load_btc_daily_close, load_btc_daily_returns
from app.analytics.descriptive_stats import compute_descriptive_stats
from app.analytics.distributions import fit_normal, fit_student_t
from app.analytics.linear_regression import fit_ar1_regression
from app.analytics.arima_model import fit_arima_model, forecast_arima
from app.analytics.monte_carlo import analyze_scenario


st.set_page_config(page_title="BTC Analytics", layout="wide")
st.title("BTC/USDT Statistical Analysis Dashboard")

page = st.sidebar.radio(
    "Section",
    (
        "Descriptive stats",
        "Distribution & tails",
        "AR(1) regression",
        "ARIMA forecast",
        "Monte Carlo",
    ),
)

horizon_mc = st.sidebar.slider(
    "Monte Carlo horizon (days)",
    min_value=7,
    max_value=90,
    value=30,
    step=1,
)
n_paths_mc = st.sidebar.slider(
    "Monte Carlo paths",
    min_value=2000,
    max_value=50000,
    value=10000,
    step=2000,
)
vol_reduction_mc = st.sidebar.slider(
    "MC volatility reduction",
    min_value=0.0,
    max_value=0.5,
    value=0.2,
    step=0.05,
)


@st.cache_data
def load_data():
    close = load_btc_daily_close()
    returns = load_btc_daily_returns()
    return close, returns


close, returns = load_data()


if page == "Descriptive stats":
    st.header("Descriptive statistics of daily log-returns")

    stats = compute_descriptive_stats(returns)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Basic moments")
        st.write(f"Number of observations: **{stats.n}**")
        st.write(f"Mean: **{stats.mean:.8f}**")
        st.write(f"Variance: **{stats.variance:.8f}**")
        st.write(f"Std dev: **{stats.std:.8f}**")

    with col2:
        st.markdown("### Shape")
        st.write(f"Skewness: **{stats.skewness:.4f}**")
        st.write(f"Kurtosis: **{stats.kurtosis:.4f}**")
        st.write(
            f"95% CI for mean: **[{stats.ci_lower:.8f}; {stats.ci_upper:.8f}]**"
        )

    st.markdown("---")
    st.markdown("### Histogram of daily log-returns")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(returns, bins=80, alpha=0.7)
    ax.set_xlabel("Daily log-return")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


elif page == "Distribution & tails":
    st.header("Distribution fitting and tail probabilities")

    mu, sigma = fit_normal(returns)
    df_t, loc_t, scale_t = fit_student_t(returns)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Normal fit")
        st.write(f"mu = **{mu:.6f}**, sigma = **{sigma:.6f}**")
    with col2:
        st.markdown("### Student t fit")
        st.write(
            f"df = **{df_t:.2f}**, loc = **{loc_t:.6f}**, scale = **{scale_t:.6f}**"
        )

    clean = returns.dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    counts, bin_edges, _ = ax.hist(
        clean,
        bins=80,
        density=True,
        alpha=0.5,
        label="Empirical returns",
    )
    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    x = np.linspace(x_min, x_max, 1000)

    from scipy import stats as sp_stats

    pdf_norm = sp_stats.norm.pdf(x, loc=mu, scale=sigma)
    pdf_t = sp_stats.t.pdf((x - loc_t) / scale_t, df_t) / scale_t

    ax.plot(x, pdf_norm, linewidth=2, label="Normal")
    ax.plot(x, pdf_t, linewidth=2, linestyle="--", label="Student t")
    ax.set_xlabel("Daily log-return")
    ax.set_ylabel("Density")
    ax.legend()
    st.markdown("### Histogram with fitted Normal and Student t")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### QQ-plots")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sp_stats.probplot(clean, dist="norm", plot=ax2)
    ax2.set_title("QQ vs Normal")

    sorted_data = np.sort(clean)
    n = sorted_data.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    quantiles_t = sp_stats.t.ppf(probs, df_t, loc=loc_t, scale=scale_t)

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.scatter(quantiles_t, sorted_data, s=5)
    min_val = min(np.min(quantiles_t), np.min(sorted_data))
    max_val = max(np.max(quantiles_t), np.max(sorted_data))
    ax3.plot([min_val, max_val], [min_val, max_val], linewidth=1)
    ax3.set_title("QQ vs Student t")

    col_qq1, col_qq2 = st.columns(2)
    with col_qq1:
        st.pyplot(fig2)
    with col_qq2:
        st.pyplot(fig3)


elif page == "AR(1) regression":
    st.header("AR(1) regression for daily log-returns")

    model, x, y = fit_ar1_regression(returns)
    y_pred = model.predict(x)

    st.markdown("### Key coefficients")
    params = model.params
    pvalues = model.pvalues

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"const = **{params.get('const', np.nan):.6f}**")
        st.write(f"lag1  = **{params.get('lag1', np.nan):.6f}**")
    with col2:
        st.write(f"p-value const = **{pvalues.get('const', np.nan):.4f}**")
        st.write(f"p-value lag1  = **{pvalues.get('lag1', np.nan):.4f}**")

    st.write(f"R² = **{model.rsquared:.4f}**")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index, y.to_numpy(), label="Actual", linewidth=1)
    ax.plot(y.index, y_pred, label="Fitted", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-return")
    ax.set_title("AR(1) fit")
    ax.legend()
    st.markdown("### Actual vs fitted returns")
    st.pyplot(fig)


elif page == "ARIMA forecast":
    st.header("ARIMA forecast for daily log-returns")

    forecast_horizon = st.slider(
        "Forecast horizon (days)",
        min_value=7,
        max_value=60,
        value=14,
        step=1,
    )

    model, clean = fit_arima_model(returns, seasonal=False)
    fc_values, fc_ci = forecast_arima(model, periods=forecast_horizon)

    history_x = clean.index.to_numpy()
    history_y = clean.to_numpy(dtype=float)

    last_index = clean.index[-1]
    future_index = (
        Path  # dummy to keep type checkers happy, replaced below
    )
    import pandas as pd

    future_index = pd.date_range(
        last_index,
        periods=len(fc_values) + 1,
        freq="D",
    )[1:]
    future_x = future_index.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history_x, history_y, label="History", linewidth=1)
    ax.plot(future_x, fc_values, label="Forecast", linewidth=2)

    ax.fill_between(
        future_x,
        fc_ci[:, 0],
        fc_ci[:, 1],
        alpha=0.3,
        label="95% CI",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Log-return")
    ax.set_title("ARIMA forecast")
    ax.legend()

    st.markdown("### ARIMA forecast with 95% confidence interval")
    st.pyplot(fig)


elif page == "Monte Carlo":
    st.header("Monte Carlo simulation for BTC 30-day returns")

    s0 = float(close.dropna().iloc[-1])

    baseline = analyze_scenario(
        scenario_name="Baseline (current volatility)",
        s0=s0,
        df=fit_student_t(returns)[0],
        loc=fit_student_t(returns)[1],
        scale=fit_student_t(returns)[2],
        horizon_days=horizon_mc,
        n_paths=n_paths_mc,
    )

    df_t, loc_t, scale_t = fit_student_t(returns)
    scale_improved = scale_t * (1.0 - vol_reduction_mc)
    improved = analyze_scenario(
        scenario_name=f"Reduced volatility ({int(vol_reduction_mc * 100)}%)",
        s0=s0,
        df=df_t,
        loc=loc_t,
        scale=scale_improved,
        horizon_days=horizon_mc,
        n_paths=n_paths_mc,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Baseline scenario")
        st.write(f"S₀ = {s0:,.2f} USDT")
        st.write(
            f"P(R{horizon_mc} < -30%) = {baseline.prob_drawdown_30:.4%}"
        )
        st.write(f"VaR95 = {baseline.var_95:.2%}")
    with col2:
        st.markdown(f"### Reduced volatility ({int(vol_reduction_mc * 100)}%)")
        st.write(
            f"P(R{horizon_mc} < -30%) = {improved.prob_drawdown_30:.4%}"
        )
        st.write(f"VaR95 = {improved.var_95:.2%}")

    r_base = baseline.final_returns
    r_impr = improved.final_returns

    q_low = float(min(np.quantile(r_base, 0.005), np.quantile(r_impr, 0.005)))
    q_high = float(max(np.quantile(r_base, 0.995), np.quantile(r_impr, 0.995)))

    r_base_clipped = r_base[(r_base >= q_low) & (r_base <= q_high)]
    r_impr_clipped = r_impr[(r_impr >= q_low) & (r_impr <= q_high)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(
        r_base_clipped,
        bins=80,
        alpha=0.5,
        density=True,
        label="Baseline",
    )
    ax.hist(
        r_impr_clipped,
        bins=80,
        alpha=0.5,
        density=True,
        label="Reduced volatility",
    )
    ax.set_xlabel(f"{horizon_mc}-day return")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Distribution of {horizon_mc}-day returns (central 99%)"
    )
    ax.legend()

    st.markdown("### Simulated distribution of cumulative returns")
    st.pyplot(fig)
