# --- Binance ---

BINANCE_BASE_URL = "https://data.binance.vision"
DEFAULT_BINANCE_INTERVAL = "1d"
DEFAULT_BINANCE_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
]

DEFAULT_BINANCE_START_DATE = "2025-11-01"
DEFAULT_BINANCE_END_DATE = "" 


# --- AWS / S3 ---

DEFAULT_AWS_REGION = "eu-north-1"
DEFAULT_S3_BUCKET = "crypto-pipeline-ml"
DEFAULT_S3_PREFIX = "raw"

# --- Spark / App ---

DEFAULT_APP_NAME = "silver_validate"
DEFAULT_DOWNLOAD_WORKERS = 5

VALIDATE_START_DATE="2025-11-01"
VALIDATE_END_DATE="2025-11-23"
S3_PREFIX_SILVER="silver"

# -- invest macro event parse
DEFAULT_EVENTS_MIN_YEAR=2023

# -- Keyword for post (feature use loughran-McDonald_dict in resources/)

ECONOMIC_CRYPTO_KEYWORDS = {
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "crypto",
    "cryptocurrency",
    "solana",
    "sol ",
    "doge",
    "dogecoin",
    "xrp",
    "etf",
    "spot etf",
}

ECONOMIC_MACRO_KEYWORDS = {
    "tariff",
    "tariffs",
    "sanction",
    "sanctions",
    "trade war",
    "trade deal",
    "china",
    "chinese",
    "tax",
    "taxes",
    "tax cut",
    "tariff on",
    "inflation",
    "cpi",
    "pce",
    "jobs report",
    "unemployment",
    "interest rate",
    "interest rates",
    "federal reserve",
    "fed",
    "sec",
    "regulation",
    "regulations",
    "stock market",
    "dow jones",
    "nasdaq",
    "s&p",
    "economy",
    "economic",
    "growth",
    "gdp",
    "budget",
    "deficit",
    "debt",
    "oil",
    "gas prices",
    "energy prices",
    "tariffs on china",
}