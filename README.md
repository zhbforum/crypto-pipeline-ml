# crypto-pipeline-ml

Bitcoin data collection, Spark batch processing, and statistical analysis in Python.
The repository includes a local Streamlit dashboard, Binance collectors, S3 processing
jobs, ARIMA forecasts, and a Truth Social / FinBERT event enrichment workflow.

## Try the dashboard

Use Python 3.11 (the version checked in CI). Run all commands from the repository root.

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m streamlit run src/app/analytics/ui/analytics_app.py
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe -m streamlit run src/app/analytics/ui/analytics_app.py
```

The dashboard uses the included BTC/USDT daily close dataset
(2017-08-17 through 2025-11-29), so its initial run needs no exchange account,
AWS credentials, Kafka cluster, or FinBERT model. The interface is in Ukrainian.

It contains descriptive statistics, normal and Student-t distribution fits,
AR(1) regression, ARIMA forecasts of daily log returns, Fed rate decision ANOVA,
and Monte Carlo scenarios. Research notes and example figures are under
[`src/app/analytics/docs/`](src/app/analytics/docs/) and
[`src/app/analytics/data/`](src/app/analytics/data/).

To refresh the local price dataset from Binance, with `PYTHONPATH=src` set:

```bash
python -m app.analytics.jobs.download_btc_daily_binance
```

This overwrites the bundled CSV. Other analysis entrypoints include:

```bash
python -m app.analytics.jobs.stats_uni_project
python -m app.analytics.jobs.distribution_analysis
python -m app.analytics.jobs.linear_regression_analysis
python -m app.analytics.jobs.arima_analysis
python -m app.analytics.jobs.fed_rate_anova
python -m app.analytics.jobs.monte_carlo_analysis
```

Plotting jobs write figures under `src/app/analytics/data/`.

## Binance collector

```bash
python -m app
```

The asynchronous HTTPX collector polls Binance and appends rows to local CSV files.
The default mode is `kline`, with a `1m` interval, for `BTCUSDT` and `ETHUSDT`.
These pairs are defined in [`src/app/constants.py`](src/app/constants.py).

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MODE` | `kline` | `kline` OHLCV candles or `ticker` last-trade prices |
| `INTERVAL` | `1m` | Candle interval; supported values are in `constants.py` |
| `OUT_DIR` | `data` | Output directory |
| `EVERY_SEC` | `5` | Delay between ticker cycles |
| `CONCURRENCY` | `8` | Concurrent requests / connection limit |
| `BINANCE_BASE` | `https://api.binance.com` | Exchange API base URL |

The collector reads exported environment variables; it does not load a `.env`
file automatically. Its outputs are `binance_ticker.csv` or
`binance_kline_<interval>.csv`. It requests the latest candle, which can still
be open. Each cycle reports successful requests, failures, and CSV rows written.

A [Kafka writer](src/app/sinks/kafka_writer.py) and a
[configuration template](src/config/client.properties.example) are included,
but the current scheduler writes CSV only. Setting `KAFKA_ENABLED` does not
connect the scheduler to Kafka.

## Spark and S3 jobs

These are separate batch entrypoints requiring configured infrastructure.
They are not automatically chained together.

```text
Binance API -> HTTPX collector -> local CSV
Binance API / monthly archives -> Spark backfill -> S3 CSV

Existing Kafka topic -> kafka_daily_to_s3 -> S3 raw messages
S3 raw messages -> s3_daily_aggregate -> S3 daily OHLCV (Parquet)
Configured daily CSV -> s3_monthly_forecast -> S3 ARIMA predictions (CSV)

FactBase posts -> local JSONL -> FinBERT + Spark -> S3 event CSV
Manual macro events -> Spark -> S3 events -> rule-based sentiment
```

| Module under `app.jobs` | Input and output |
| --- | --- |
| `binance_vision_backfill_raw` | Monthly Binance archive ZIPs to S3 CSV by symbol/month |
| `binance_api_backfill` | Binance REST klines to S3 CSV by symbol/month |
| `binance_silver_validate` | Validates prices, timestamps and volume; removes duplicates; writes partitioned CSV |
| `kafka_daily_to_s3` | Reads a Kafka time window and writes raw message envelopes to S3 by topic/date |
| `s3_daily_aggregate` | Parses JSON/CSV message payloads; aggregates OHLCV by Europe/Berlin day into gold Parquet |
| `s3_monthly_forecast` | Reads CSV with `iso_ts, symbol, close`; fits auto-ARIMA per symbol; writes predictions and optional errors |
| `macro_event_to_s3` | Reads manually prepared `data/manual_macro_events.jsonl` and writes event CSV |
| `sentiment_macro_event` | Scores changes in macro values: increase = -1, decrease = +1, unchanged/missing = 0 |

Spark execution requires Java and the Hadoop S3A / Kafka connector JARs appropriate
for your Spark installation. Several scripts contain fixed connector versions or
paths; review each job's configuration before running it against your bucket.
The Python CI checks do not validate a live Spark cluster or cloud integration.

Common settings include `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`, `S3_BUCKET`, and `SPARK_PACKAGES`. Kafka ingestion also
requires `KAFKA_BOOTSTRAP`, `CONFLUENT_API_KEY`, and `CONFLUENT_API_SECRET`.
Keep credentials in ignored local configuration.

Examples, after configuring credentials and connectors:

```bash
python -m app.jobs.kafka_daily_to_s3 --env-file .env --date 2025-11-01
python -m app.jobs.s3_daily_aggregate --env-file .env --date 2025-11-01
python -m app.jobs.s3_monthly_forecast
```

The forecast job loads `src/app/jobs/.env`, falling back to the root `.env`.
Set `DAILY_AGG_PATH` and `DAILY_FORECAST_PATH` for your own S3 locations.
`FORECAST_START_DS` / `FORECAST_END_DS` default to November 2025; set both to
empty strings for a next-day forecast. A date range produces rolling one-step
predictions from the available preceding history.

The gold aggregation outputs `day` in Parquet, while the forecast job expects
`iso_ts` in CSV. An explicit conversion or a suitable daily CSV input is needed
between those jobs. Backfill and validation paths also need alignment in
[`parser_settings/constants.py`](src/app/parser_settings/constants.py).

## Truth Social and macro events

FinBERT dependencies are optional for the local dashboard. To install them with
CPU-only PyTorch and download the model:

```bash
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-sentiment.txt
python -m app.scripts.download_model
python -m app.tweets.truth_factbase_fetch
python -m app.tweets.truth_parser
```

The download script saves `ProsusAI/finbert` to `models/finbert`.
The FactBase fetcher requests Truth Social posts from Roll Call's endpoint with
a `last_year` filter and a lower cutoff of 2025-01-20. It saves `date` and `text`
records to `data/trump_truthsocial_since_2025-01-20.jsonl`; availability and
historical coverage depend on that external service.

The parser reads that JSONL, filters economic/crypto posts, classifies text with
the local model, and writes **CSV** to S3, partitioned by `event_date`.
Its default output is under
`silver/kline=4h/event/source=truthsocial/user=realDonaldTrump/`.
This step requires AWS credentials and a working Spark/S3A configuration.

Macroeconomic events are supplied manually in JSONL. There is no automated
Investing.com scraper in the current repository.

## Development and CI

Install the core and sentiment dependencies above, then:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m prospector src
python -m compileall -q src
python -m unittest discover -s tests -v
```

Keep `PYTHONPATH=src` set. [GitHub Actions](.github/workflows/ci.yml) runs these
checks plus import smoke tests on Python 3.11. Regression tests use local data,
mocked network/model responses, and Streamlit's application test runner.
They do not download model weights or access Binance, Kafka, or AWS.

## Scope

This is a portfolio / research project. The local dashboard and individual
processing components are implemented; deployment, unified orchestration,
Athena table definitions, and Tableau dashboards are not included.
The bundled analyses describe historical data and model assumptions; forecast
accuracy and production operation are not established by passing CI.

Licensed under the [MIT License](LICENSE).
