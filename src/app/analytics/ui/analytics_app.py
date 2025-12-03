from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "app").is_dir():
        ROOT = parent
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analytics.dataset import load_btc_daily_close, load_btc_daily_returns
from app.analytics.descriptive_stats import compute_descriptive_stats
from app.analytics.distributions import (
    fit_normal,
    fit_student_t,
    kolmogorov_smirnov_tests,
    chi_square_normal,
)
from app.analytics.linear_regression import fit_ar1_regression
from app.analytics.arima_model import fit_arima_model, forecast_arima
from app.analytics.monte_carlo import analyze_scenario


st.set_page_config(page_title="Панель аналізу BTC", layout="wide")
st.title("Панель статистичного аналізу BTC/USDT")

page = st.sidebar.radio(
    "Розділ",
    (
        "Описова статистика",
        "Розподіл і хвости",
        "AR(1)-регресія",
        "Прогноз ARIMA",
        "Моделювання Монте-Карло",
    ),
)

horizon_mc = st.sidebar.slider(
    "Горизонт Монте-Карло (днів)",
    min_value=7,
    max_value=90,
    value=30,
    step=1,
)
n_paths_mc = st.sidebar.slider(
    "Кількість траєкторій Монте-Карло",
    min_value=2000,
    max_value=50000,
    value=10000,
    step=2000,
)
vol_reduction_mc = st.sidebar.slider(
    "Зменшення волатильності для сценарію (%)",
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


if page == "Описова статистика":
    st.header("Описова статистика денних логарифмічних дохідностей")

    stats = compute_descriptive_stats(returns)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Основні моменти")
        st.write(f"Кількість спостережень (n): **{stats.n}**")
        st.write(f"Середнє значення (μ): **{stats.mean:.8f}**")
        st.write(f"Дисперсія (σ²): **{stats.variance:.8f}**")
        st.write(f"Середньоквадратичне відхилення (σ): **{stats.std:.8f}**")

    with col2:
        st.markdown("### Форма розподілу")
        st.write(f"Асиметрія (γ₁): **{stats.skewness:.4f}**")
        st.write(f"Ексцес (γ₂): **{stats.kurtosis:.4f}**")
        st.write(
            f"95% довірчий інтервал для середнього (μ): "
            f"**[{stats.ci_lower:.8f}; {stats.ci_upper:.8f}]**"
        )

    st.markdown("---")
    st.markdown("### Гістограма денних логарифмічних дохідностей")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(returns, bins=80, alpha=0.7)
    ax.set_xlabel("Денна логарифмічна дохідність")
    ax.set_ylabel("Частота")
    st.pyplot(fig)


elif page == "Розподіл і хвости":
    st.header("Підбір розподілу та ймовірність хвостів")

    mu, sigma = fit_normal(returns)
    df_t, loc_t, scale_t = fit_student_t(returns)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Нормальний розподіл")
        st.write(f"μ = **{mu:.6f}**, σ = **{sigma:.6f}**")
    with col2:
        st.markdown("### t-розподіл Стьюдента")
        st.write(
            f"df = **{df_t:.2f}**, loc = **{loc_t:.6f}**, scale = **{scale_t:.6f}**"
        )

    ks_results = kolmogorov_smirnov_tests(returns)
    ks_norm_D, ks_norm_p = ks_results["normal"]
    ks_t_D, ks_t_p = ks_results["student_t"]

    chi2_stat, chi2_p, counts_chi, exp_counts_chi, bin_edges_chi = chi_square_normal(
        returns, bins=40
    )

    st.markdown("---")
    st.markdown("### Перевірка узгодженості (goodness-of-fit)")

    col_ks, col_chi = st.columns(2)

    with col_ks:
        st.markdown("#### Критерій Колмогорова–Смирнова")
        st.write(f"Нормальний: D = **{ks_norm_D:.4f}**, p-value = **{ks_norm_p:.4g}**")
        st.write(f"t-розподіл: D = **{ks_t_D:.4f}**, p-value = **{ks_t_p:.4g}**")

    with col_chi:
        st.markdown("#### Критерій χ² Пірсона (проти Normal)")
        st.write(f"χ² = **{chi2_stat:.2f}**, p-value = **{chi2_p:.4g}**")

    clean = returns.dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    counts, bin_edges, _ = ax.hist(
        clean,
        bins=80,
        density=True,
        alpha=0.5,
        label="Емпіричні дохідності",
    )
    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    x = np.linspace(x_min, x_max, 1000)

    from scipy import stats as sp_stats

    pdf_norm = sp_stats.norm.pdf(x, loc=mu, scale=sigma)
    pdf_t = sp_stats.t.pdf((x - loc_t) / scale_t, df_t) / scale_t

    ax.plot(x, pdf_norm, linewidth=2, label="Нормальний розподіл")
    ax.plot(
        x,
        pdf_t,
        linewidth=2,
        linestyle="--",
        label="t-розподіл Стьюдента",
    )
    ax.set_xlabel("Денна логарифмічна дохідність")
    ax.set_ylabel("Щільність")
    ax.legend()
    st.markdown("---")
    st.markdown("### Гістограма з підігнаними Normal та t-розподілом")
    st.pyplot(fig)

    centers_chi = 0.5 * (bin_edges_chi[1:] + bin_edges_chi[:-1])
    widths_chi = np.diff(bin_edges_chi)

    fig_chi, ax_chi = plt.subplots(figsize=(10, 4))
    ax_chi.bar(
        centers_chi,
        counts_chi,
        width=widths_chi,
        alpha=0.6,
        align="center",
        label="Спостережені частоти",
    )
    ax_chi.bar(
        centers_chi,
        exp_counts_chi,
        width=widths_chi,
        alpha=0.4,
        align="center",
        label="Очікувані частоти (Normal)",
    )
    ax_chi.set_xlabel("Біни денних логарифмічних дохідностей")
    ax_chi.set_ylabel("Кількість")
    ax_chi.set_title("Критерій χ²: спостережені vs очікувані частоти (Normal)")
    ax_chi.legend()

    st.markdown("### Chi-square: спостережені та очікувані частоти")
    st.pyplot(fig_chi)

    st.markdown("---")
    st.markdown("### QQ-графіки")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sp_stats.probplot(clean, dist="norm", sparams=(mu, sigma), plot=ax2)
    ax2.set_title("QQ-графік відносно Normal")

    sorted_data = np.sort(clean)
    n = sorted_data.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    quantiles_t = sp_stats.t.ppf(probs, df_t, loc=loc_t, scale=scale_t)

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.scatter(quantiles_t, sorted_data, s=5)
    min_val = min(np.min(quantiles_t), np.min(sorted_data))
    max_val = max(np.max(quantiles_t), np.max(sorted_data))
    ax3.plot([min_val, max_val], [min_val, max_val], linewidth=1)
    ax3.set_title("QQ-графік відносно t-розподілу")

    col_qq1, col_qq2 = st.columns(2)
    with col_qq1:
        st.pyplot(fig2)
    with col_qq2:
        st.pyplot(fig3)


elif page == "AR(1)-регресія":
    st.header("AR(1)-регресія для денних логарифмічних дохідностей")

    model, x, y = fit_ar1_regression(returns)
    y_pred = model.predict(x)

    st.markdown("### Ключові коефіцієнти моделі")
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
    ax.plot(y.index, y.to_numpy(), label="Фактичні значення", linewidth=1)
    ax.plot(y.index, y_pred, label="Оцінені значення (fitted)", linewidth=1)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Логарифмічна дохідність")
    ax.set_title("AR(1)-регресія")
    ax.legend()
    st.markdown("### Фактичні vs оцінені дохідності")
    st.pyplot(fig)


elif page == "Прогноз ARIMA":
    st.header("Прогноз ARIMA для денних логарифмічних дохідностей")

    forecast_horizon = st.slider(
        "Горизонт прогнозу (днів)",
        min_value=1,
        max_value=14,
        value=7,
        step=1,
    )

    model, clean = fit_arima_model(returns, seasonal=False)

    order = model.order
    arima_res = model.arima_res_

    param_names = list(arima_res.param_names)
    param_values = arima_res.params
    param_pvalues = arima_res.pvalues

    params_df = pd.DataFrame(
        {
            "Параметр": param_names,
            "Оцінка коефіцієнта": param_values,
            "p-value": param_pvalues,
        }
    )

    st.markdown("### Підібрана модель ARIMA")
    st.write(
        f"Порядок ARIMA (p, d, q): **({order[0]}, {order[1]}, {order[2]})**"
    )

    st.markdown("### Оцінені коефіцієнти (ваги) моделі")
    st.dataframe(
        params_df.style.format(
            {
                "Оцінка коефіцієнта": "{:.6f}",
                "p-value": "{:.4f}",
            }
        )
    )

    fc_values, fc_ci = forecast_arima(model, periods=forecast_horizon)

    history = clean.tail(250)
    history_x = history.index.to_numpy()
    history_y_pct = history.to_numpy(dtype=float) * 100.0

    last_index = clean.index[-1]

    future_index = pd.date_range(
        last_index,
        periods=len(fc_values) + 1,
        freq="D",
    )[1:]
    future_x = future_index.to_numpy()

    fc_mean_pct = fc_values * 100.0
    fc_ci_pct = fc_ci * 100.0

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(history_x, history_y_pct, label="Історія (останні 250 днів)", linewidth=1)
    ax.plot(future_x, fc_mean_pct, label="Прогноз середньої дохідності", linewidth=2)

    ax.fill_between(
        future_x,
        fc_ci_pct[:, 0],
        fc_ci_pct[:, 1],
        alpha=0.3,
        label="95% довірчий інтервал",
    )

    ax.axvline(last_index, color="grey", linestyle="--", linewidth=1)
    ax.text(
        last_index,
        ax.get_ylim()[1],
        " Початок прогнозу",
        rotation=90,
        va="top",
        ha="left",
        fontsize=8,
    )

    ax.set_xlabel("Дата")
    ax.set_ylabel("Денна логарифмічна дохідність, %")
    ax.set_title("Прогноз ARIMA для денних дохідностей (у відсотках)")
    ax.legend()

    st.markdown("### Прогноз ARIMA з 95% довірчим інтервалом (у відсотках)")
    st.pyplot(fig)

    df_forecast = pd.DataFrame(
        {
            "Дата": future_index,
            "Прогноз, %": fc_mean_pct,
            "Нижня межа 95% CI, %": fc_ci_pct[:, 0],
            "Верхня межа 95% CI, %": fc_ci_pct[:, 1],
        }
    )

    st.markdown("### Таблиця прогнозованих денних дохідностей (%)")
    st.dataframe(
        df_forecast.set_index("Дата").style.format(
            {
                "Прогноз, %": "{:.4f}",
                "Нижня межа 95% CI, %": "{:.4f}",
                "Верхня межа 95% CI, %": "{:.4f}",
            }
        )
    )


elif page == "Моделювання Монте-Карло":
    st.header("Моделювання Монте-Карло для k-денних дохідностей BTC")

    s0 = float(close.dropna().iloc[-1])

    df_t, loc_t, scale_t = fit_student_t(returns)

    baseline = analyze_scenario(
        scenario_name="Базовий сценарій (поточна волатильність)",
        s0=s0,
        df=df_t,
        loc=loc_t,
        scale=scale_t,
        horizon_days=horizon_mc,
        n_paths=n_paths_mc,
    )

    scale_improved = scale_t * (1.0 - vol_reduction_mc)
    improved = analyze_scenario(
        scenario_name=f"Знижена волатильність ({int(vol_reduction_mc * 100)}%)",
        s0=s0,
        df=df_t,
        loc=loc_t,
        scale=scale_improved,
        horizon_days=horizon_mc,
        n_paths=n_paths_mc,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Базовий сценарій")
        st.write(f"S₀ = {s0:,.2f} USDT")
        st.write(
            f"P(R{horizon_mc} < -30%) = {baseline.prob_drawdown_30:.4%}"
        )
        st.write(f"VaR95 = {baseline.var_95:.2%}")
    with col2:
        st.markdown(
            f"### Сценарій зі зниженою волатильністю ({int(vol_reduction_mc * 100)}%)"
        )
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
        label="Базовий сценарій",
    )
    ax.hist(
        r_impr_clipped,
        bins=80,
        alpha=0.5,
        density=True,
        label="Знижена волатильність",
    )
    ax.set_xlabel(f"{horizon_mc}-денна дохідність")
    ax.set_ylabel("Щільність")
    ax.set_title(
        f"Розподіл {horizon_mc}-денних сукупних дохідностей (центральні 99%)"
    )
    ax.legend()

    st.markdown("### Імітований розподіл сукупних дохідностей")
    st.pyplot(fig)
