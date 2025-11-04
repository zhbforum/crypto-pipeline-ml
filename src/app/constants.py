from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Final, List
import pytz

BERLIN: Final = pytz.timezone("Europe/Berlin")
UTC: Final = pytz.utc

MODE: Final[str] = os.getenv("MODE", "kline")
INTERVAL: Final[str] = os.getenv("INTERVAL", "1m")

PAIRS: Final[List[str]] = [
    "BTCUSDT",
    "ETHUSDT",
]

EVERY_SEC: Final[int] = int(os.getenv("EVERY_SEC", "5"))
OUT_DIR: Final[str] = os.getenv("OUT_DIR", "data")
CONCURRENCY: Final[int] = int(os.getenv("CONCURRENCY", "8"))
BINANCE_BASE: Final[str] = os.getenv("BINANCE_BASE", "https://api.binance.com")

SINK: Final[str] = os.getenv("SINK", "csv+kafka")

KAFKA_TOPIC: Final[str] = os.getenv("KAFKA_TOPIC", "topic_0")
CLIENT_PROPERTIES_PATH: Final[str] = os.getenv("CLIENT_PROPERTIES_PATH", "./config/client.properties")

_env_flag = os.getenv("KAFKA_ENABLED")
KAFKA_ENABLED: Final[bool] = (
    (_env_flag == "1")
    or ("kafka" in SINK)
    or (Path(CLIENT_PROPERTIES_PATH).exists() and _env_flag != "0")
)

_INTERVAL_MAP_SECONDS: Final[Dict[str, int]] = {
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

def interval_seconds(interval: str) -> int:
    val = _INTERVAL_MAP_SECONDS.get(interval)
    if val is None:
        raise ValueError(f"Unsupported interval: {interval}")
    return val
