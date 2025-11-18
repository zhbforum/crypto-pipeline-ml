# --- Binance ---

BINANCE_BASE_URL = "https://data.binance.vision"
DEFAULT_BINANCE_INTERVAL = "3m"
DEFAULT_BINANCE_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
]

DEFAULT_BINANCE_START_DATE = "2022-12-31"
DEFAULT_BINANCE_END_DATE = "" 


# --- AWS / S3 ---

DEFAULT_AWS_REGION = "eu-north-1"
DEFAULT_S3_BUCKET = "crypto-pipeline-ml"
DEFAULT_S3_PREFIX = "raw"

# --- Spark / App ---

DEFAULT_APP_NAME = "binance_vision_backfill_monthly_csv"
DEFAULT_DOWNLOAD_WORKERS = 5
