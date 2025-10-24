from typing import Dict

MODE = "kline"

INTERVAL = "1m"

PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    # "BNBUSDT",
]

EVERY_SEC = 5

OUT_DIR = "data"

CONCURRENCY = 8

BINANCE_BASE = "https://api.binance.com"

_INTERVAL_MAP_SECONDS: Dict[str, int] = {
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "6h": 6 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
}
