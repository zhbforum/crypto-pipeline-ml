# **crypto-pipeline-ml**

End-to-end data & ML pipeline for Bitcoin.

* collects tick or kline data from **Binance**,
* stores them locally and in **Kafka**,
* aggregates intraday klines into daily candles in **S3** using **Spark**,
* builds **ARIMA forecasts** for BTC price,
* parses POTUS posts and macroeconomic events from Investing.com,
* computes sentiment of POTUS posts and macro events,
* stores enriched datasets in **AWS S3**,
* prepares analytics-ready tables for **Athena / Tableau** dashboards.

---

# **High-level architecture**

```text
Binance API  ──► Async collector (Python, aiohttp)
              │
              ├─► CSV on disk (optional)
              └─► Kafka topic
                    │
                    ▼
           S3 raw layer: raw/topic=.../date=YYYY-MM-DD
                    │
                    ▼
        Spark jobs (src/app/jobs/*.py)
        - s3_daily_aggregate: intraday → daily OHLCV
        - s3_monthly_forecast: ARIMA forecasts
                    │
                    ▼
           S3 gold/silver layer (daily + predictions)
                    │
                    ▼
            Athena / Tableau dashboards
```

---

# **Core Components**

---

## **1. Real-time Binance Collector**

**Module:** `src/app/scheduler.py`
**Entrypoint:** `python -m app`

The collector operates in two independent ingestion modes:

### **Modes**

* `MODE=ticker` — fetches last-trade prices for selected symbols
* `MODE=kline` — fetches OHLCV candlesticks (intervals: `1m, 3m, 5m, …, 1d`)

### **Outputs**

* writes to local **CSV** directory (`OUT_DIR`),
* optionally publishes messages to **Kafka** (`KAFKA_ENABLED=1`).

### **Logging**

Each batch cycle logs:

* successes / failures,
* number of CSV rows written,
* number of Kafka messages produced.

---

## **2. Spark job: `s3_daily_aggregate.py`**

**Path:** `src/app/jobs/s3_daily_aggregate.py`
**Purpose:** Convert intraday Kafka messages stored in S3 raw layer into **daily OHLCV candles** in Europe/Berlin timezone.

### **Inputs**

* S3 raw data:
  `s3a://{S3_BUCKET}/{S3_PREFIX}/topic=.../date=YYYY-MM-DD`
* Supports JSON or CSV payloads.

### **Outputs**

* Gold-layer daily OHLCV parquet dataset:

```
s3a://{S3_BUCKET}/{S3_PREFIX_GOLD}/ohlcv_daily/
  └── day=YYYY-MM-DD/
       └── symbol=BTCUSDT/
```

Columns include:
`day, symbol, open, high, low, close, volume, rows`

### **Functionality**

* Converts timestamps to Berlin timezone
* Computes open/high/low/close/volume for each symbol/day
* Writes optimized parquet partitions

---

## **3. Spark job: `s3_monthly_forecast.py`**

**Path:** `src/app/jobs/s3_monthly_forecast.py`
**Purpose:** Build ARIMA-based forecasts for each asset using daily historical price data.

### **Input**

Daily dataset:

```
DAILY_AGG_PATH = s3a://.../silver/kline=1d/symbol=BTCUSDT/
```

Expected columns:
`iso_ts`, `symbol`, `close`

### **Output**

Forecast dataset:

```
DAILY_FORECAST_PATH = s3a://.../silver/predictions_monthly/
```

Columns include:
`symbol, ds, y_hat_close, created_at, actual_close?, abs_error?, ape?`

### **Functionality**

* Converts `iso_ts` to date (`ds`)
* Groups time series by `symbol`
* Trains **auto-ARIMA**
* Predicts either:

  * the next day, or
  * a full date range (`FORECAST_START_DS`, `FORECAST_END_DS`)
* Optionally computes forecast errors
* Writes partitioned results to S3

---

# **4. Twitter/X (POTUS tweets) Pipeline – with file-level explanations**

**Folder:** `src/app/tweets/`
This subproject produces an “information event layer” based on POTUS tweets, which is later joined with BTC price/volume for impact analysis.

It consists of two main scripts:

---

## **4.1 `truth_factbase_fetch.py` — Tweet/FactBase Fetcher**

**Location:**
`src/app/tweets/truth_factbase_fetch.py`

### **Purpose**

Download or read the dataset of POTUS posts (tweets or TruthSocial reposts, depending on your configuration) and normalize them into a unified intermediate format.

### **Key functions**

* loads raw posts (FactBase, archive exports, JSON/CSV sources)
* normalizes:

  * `post_id`
  * `created_at` timestamp
  * `text` field
  * author metadata
* filters out duplicates / malformed entries
* writes cleaned posts to a structured local file for later sentiment analysis

---

## **4.2 `truth_parser.py` — Local Tweet/TruthSocial Sentiment Annotator**

**Location:**
`src/app/tweets/truth_parser.py`

### **Purpose**

Parse normalized posts, run them through **FinBERT** sentiment classifier, and write the final enriched dataset into the event-layer in S3.

### **What it does**

* Loads normalized posts produced by `truth_factbase_fetch.py`
* Tokenizes text using HuggingFace tokenizer
* Runs FinBERT inference locally
* Computes:

  * sentiment score
  * sentiment class (`Positive / Neutral / Negative`)
* Adds metadata:

  * economic keyword flags
  * crypto relevance flags
* Writes JSONL/Parquet event records that Athena/Tableau can join with BTC time series.

---

### **Important: FinBERT model must be installed locally**

`truth_parser.py` **requires a local FinBERT model directory**.

To download the model, run:

```
python src/app/scripts/download_model.py
```

This script:

* downloads the pretrained **ProsusAI/FinBERT** model,
* stores it under the expected project path,
* makes the model available for local inference without external API calls.

---

### **Spark jobs**

Require AWS credentials, S3 paths, partitions, etc.
Full list preserved exactly as in your original description.

---

# **Roadmap**

* Real-time Binance ingestion → CSV + Kafka ✔️
* Daily OHLCV aggregation to S3 gold layer ✔️
* ARIMA forecasting ✔️
* TruthSocial + FinBERT sentiment pipeline ✔️
* Macro events + BTC feature engineering for Tableau ✔️
* Unified orchestrated pipeline (Airflow / Prefect / GitHub Actions) 🚧

---