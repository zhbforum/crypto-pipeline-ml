import csv
import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import matplotlib
import numpy as np
import pandas as pd
import torch
from streamlit.testing.v1 import AppTest

from app.analytics.dataset import load_btc_daily_close, load_btc_daily_returns
from app.analytics.distributions import (
    chi_square_normal,
    plot_histogram_with_fits,
    plot_qq,
)
from app.analytics.fed_rate_anova import compute_fed_rate_anova
from app.analytics.monte_carlo import simulate_paths_student_t
from app.exchange.binance_client import BinanceClient
from app.jobs import s3_monthly_forecast as forecasts
from app.services.collector import CollectorService
from app.sinks.csv_sink import CsvSink
from app.tweets import truth_factbase_fetch, truth_parser

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


class CollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_ticker_cycle_appends_csv_with_one_header(self):
        def respond(request):
            self.assertEqual(request.url.path, "/api/v3/ticker/price")
            return httpx.Response(200, json=[{"symbol": "BTCUSDT", "price": "100"}])

        client = BinanceClient("https://example.test", 2)
        await client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(respond)
        )
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "ticker.csv"
                sink = CsvSink(path, ["ts", "iso_ts", "symbol", "price"])
                service = CollectorService(client)
                for _ in range(2):
                    ok, fail, written, rows = await service.one_cycle_ticker(
                        ["BTCUSDT", "MISSING"], sink
                    )
                    self.assertEqual((ok, fail, written), (1, 1, 1))
                    self.assertEqual(rows[0]["price"], 100.0)
                with path.open(encoding="utf-8", newline="") as stream:
                    saved = list(csv.DictReader(stream))
                self.assertEqual(len(saved), 2)
                self.assertEqual(saved[1]["symbol"], "BTCUSDT")
        finally:
            await client.aclose()

    async def test_kline_cycle_counts_failed_symbols(self):
        client = SimpleNamespace(last_kline=Mock())

        async def kline(symbol, interval):
            if symbol == "MISSING":
                return None
            return {"ts": 0, "symbol": symbol, "interval": interval, "close": 100.0}

        client.last_kline = kline
        with tempfile.TemporaryDirectory() as folder:
            sink = CsvSink(Path(folder) / "kline.csv", ["ts", "symbol", "close"])
            ok, fail, written, rows = await CollectorService(client).one_cycle_kline(
                ["BTCUSDT", "MISSING"], "1m", sink, 2
            )
        self.assertEqual((ok, fail, written), (1, 1, 1))
        self.assertEqual(rows[0]["iso_ts"], "1970-01-01T00:00:00+00:00")


class StartupTests(unittest.TestCase):
    def test_scheduler_honors_output_directory_without_creating_it_on_import(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "custom-output"
            result = subprocess.run(
                [sys.executable, "-c", "from app.scheduler import DATA_DIR; print(DATA_DIR)"],
                cwd=folder,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "OUT_DIR": str(output)},
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(Path(result.stdout.strip()), output)
            self.assertFalse(output.exists())
            self.assertFalse((Path(folder) / "data").exists())

    def test_model_downloader_import_does_not_download(self):
        with patch("huggingface_hub.snapshot_download") as download:
            module = importlib.import_module("app.scripts.download_model")
            importlib.reload(module)
            download.assert_not_called()

    def test_fetcher_output_matches_parser_input(self):
        self.assertEqual(Path(truth_factbase_fetch.OUTPUT_PATH), truth_parser.load_config().jsonl_path)


class AnalyticsTests(unittest.TestCase):
    def test_returns_and_refactored_plots_use_bundled_data(self):
        close = load_btc_daily_close()
        returns = load_btc_daily_returns()
        np.testing.assert_allclose(returns, np.log(close.to_numpy()[1:] / close.to_numpy()[:-1]))
        statistic, p_value, counts, expected, edges = chi_square_normal(returns)
        self.assertGreaterEqual(statistic, 0)
        self.assertTrue(0 <= p_value <= 1)
        self.assertEqual(int(counts.sum()), len(returns))
        self.assertEqual(len(expected), len(edges) - 1)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            paths = [plot_histogram_with_fits(returns, output), *plot_qq(returns, output)]
            for path in paths:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_fed_anova_has_consistent_groups(self):
        returns = load_btc_daily_returns(log_returns=False)
        for mode in ("event_day", "pre_k_days", "post_k_days"):
            with self.subTest(mode=mode):
                result, summary, impacts = compute_fed_rate_anova(returns, mode=mode, k=3)
                self.assertEqual(sum(result.group_sizes), len(impacts))
                self.assertEqual(int(summary["count"].sum()), len(impacts))
                self.assertTrue(0 <= result.p_value <= 1)
                self.assertTrue(0 <= result.eta2 <= 1)

    def test_monte_carlo_is_reproducible_and_preserves_initial_price(self):
        args = dict(s0=100.0, df=5.0, loc=0.001, scale=0.02,
                    horizon_days=7, n_paths=100, random_state=42)
        prices, returns = simulate_paths_student_t(**args)
        second_prices, second_returns = simulate_paths_student_t(**args)
        np.testing.assert_array_equal(prices, second_prices)
        np.testing.assert_array_equal(returns, second_returns)
        self.assertEqual(prices.shape, (8, 100))
        np.testing.assert_array_equal(prices[0], 100.0)
        np.testing.assert_allclose(returns, prices[-1] / prices[0] - 1)


class ForecastTests(unittest.TestCase):
    def test_period_forecast_uses_only_history_before_target(self):
        frame = pd.DataFrame({"symbol": ["BTCUSDT"] * 12,
                              "ds": pd.date_range("2025-01-01", periods=12),
                              "close": np.arange(12, dtype=float) + 100})
        seen = []

        def fit(history):
            seen.append(history.to_numpy().copy())
            return float(history.iloc[-1])

        with patch.dict(os.environ, {"FORECAST_START_DS": "2025-01-11",
                                     "FORECAST_END_DS": "2025-01-12"}), \
                patch.object(forecasts, "_fit_arima_and_forecast_next", side_effect=fit):
            result = forecasts.forecast_for_period(frame)
        self.assertEqual([len(history) for history in seen], [10, 11])
        self.assertEqual(result["y_hat_close"].tolist(), [109.0, 110.0])
        self.assertEqual(result["ds"].astype(str).tolist(), ["2025-01-11", "2025-01-12"])

    def test_empty_period_selects_next_day_forecast(self):
        frame = pd.DataFrame({"symbol": ["BTCUSDT"], "ds": ["2025-01-01"], "close": [100.0]})
        with patch.dict(os.environ, {"FORECAST_START_DS": "", "FORECAST_END_DS": ""}), \
                patch.object(forecasts, "_fit_arima_and_forecast_next", return_value=101.0):
            result = forecasts.forecast_for_period(frame)
        self.assertEqual(str(result["ds"].iloc[0]), "2025-01-02")
        self.assertEqual(result["y_hat_close"].iloc[0], 101.0)


class SentimentTests(unittest.TestCase):
    def test_chunked_prediction_averages_all_batches(self):
        sentiment = truth_parser.FinBertSentiment.__new__(truth_parser.FinBertSentiment)
        sentiment.device = torch.device("cpu")
        sentiment.id2label = {0: "negative", 1: "positive", 2: "neutral"}
        sentiment.tokenizer = Mock(return_value={
            "input_ids": torch.ones((20, 3), dtype=torch.long),
            "attention_mask": torch.ones((20, 3), dtype=torch.long),
        })
        def infer(input_ids, attention_mask):
            # Unequal batches must contribute by chunk count, not batch count.
            probabilities = [0.1, 0.7, 0.2] if input_ids.size(0) == 16 else [0.7, 0.1, 0.2]
            return SimpleNamespace(
                logits=torch.tensor([probabilities]).log().repeat(input_ids.size(0), 1)
            )

        sentiment.model = Mock(side_effect=infer)
        result = sentiment.predict("Bitcoin and interest rates")
        self.assertEqual(sentiment.model.call_count, 2)
        self.assertAlmostEqual(result["sentiment_score"], 0.45, places=6)
        self.assertEqual(result["sentiment_label"], "positive")
        self.assertEqual(result["sentiment_index"], 1)


class DashboardTests(unittest.TestCase):
    def test_all_six_sections_render(self):
        app = AppTest.from_file(str(ROOT / "src/app/analytics/ui/analytics_app.py"),
                               default_timeout=120).run()
        self.assertEqual(list(app.exception), [])
        sections = app.sidebar.radio[0].options
        self.assertEqual(len(sections), 6)
        for section in sections:
            with self.subTest(section=section):
                app.sidebar.radio[0].set_value(section).run()
                self.assertEqual(list(app.exception), [])


if __name__ == "__main__":
    unittest.main()
