from __future__ import annotations

from datetime import datetime, timezone


BINANCE_BASE_URL = "https://data.binance.vision"
DEFAULT_BINANCE_INTERVAL = "4h"
DEFAULT_BINANCE_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
]

DEFAULT_BINANCE_START_DATE = "2025-11-01"
DEFAULT_BINANCE_END_DATE = "" 


# --- AWS / S3 ---

DEFAULT_AWS_REGION = "eu-north-1"
DEFAULT_S3_BUCKET = "crypto-pipeline-ml"
DEFAULT_S3_PREFIX = "silver"

# --- Spark / App ---

DEFAULT_APP_NAME = "silver_validate"
DEFAULT_DOWNLOAD_WORKERS = 5

VALIDATE_START_DATE="2025-11-01"
VALIDATE_END_DATE="2025-11-23"
S3_PREFIX_SILVER="silver"

# -- invest macro event parse
DEFAULT_EVENTS_MIN_YEAR=2023

# -- Keyword for post (feature use loughran-McDonald_dict in resources/)

ECONOMIC_CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "crypto", "cryptocurrency", "blockchain", "exchange", "etf",
    "token", "tokens", "mining", "miners", "halving", "stablecoin", "stablecoins",
]

ECONOMIC_MACRO_KEYWORDS = [
    "fed", "federal reserve", "interest rate", "rates", "cpi", "inflation",
    "unemployment", "jobs", "gdp", "recession", "treasury", "bond", "bonds",
    "tariff", "tariffs", "trade", "sanctions", "deficit", "debt", "dollar",
]

TRUTH_PLATFORM = "truthsocial"
TRUTH_SOURCE = "truthsocial"
TRUTH_USER_HANDLE = "realDonaldTrump"
TRUTH_EVENT_TYPE = "TRUMP_TRUTH_ECON"

START_DATE_UTC = datetime(2025, 1, 20, tzinfo=timezone.utc)

TRUTH_JSONL_REL_PATH = "data/trump_truthsocial_since_2025-01-20.jsonl"
FINBERT_REL_DIR = "models/finbert"

LOW_INFO_MIN_WORDS = 4
LOW_INFO_MIN_ALPHA_CHARS = 20

FINBERT_MAX_LEN = 512
FINBERT_STRIDE = 64
FINBERT_BATCH_SIZE = 16

SPARK_APP_NAME = "trump_truths_finbert_sentiment"
SPARK_WRITE_FORMAT = "json"
SPARK_WRITE_MODE = "append"
S3_EVENTS_PATH = "event/source=truthsocial/user=realDonaldTrump"
BATCH_WRITE_SIZE = 2000

NEUTRAL_LOWER = -0.3
NEUTRAL_UPPER = 0.3