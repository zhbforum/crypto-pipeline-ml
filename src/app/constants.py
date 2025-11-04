from __future__ import annotations
import os
from typing import Dict, Final, List
import pytz

BERLIN: Final = pytz.timezone("Europe/Berlin")
UTC: Final = pytz.utc

MODE: Final[str] = "kline"
INTERVAL: Final[str] = "1m"
PAIRS: Final[List[str]] = ["BTCUSDT", "ETHUSDT"]

EVERY_SEC: Final[int] = 5
OUT_DIR: Final[str] = "data"
CONCURRENCY: Final[int] = 8
BINANCE_BASE: Final[str] = "https://api.binance.com"

SINK: Final[str] = "csv+kafka"

KAFKA_TOPIC: Final[str] = "topic_0"

_env_flag = os.getenv("KAFKA_ENABLED")
KAFKA_ENABLED: Final[bool] = (_env_flag == "1") or ("kafka" in SINK)



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


KAFKA_CONFIG: Final[Dict[str, str]] = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP"),
    "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
    "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
    "sasl.username": os.getenv("KAFKA_USERNAME"),
    "sasl.password": os.getenv("KAFKA_PASSWORD"),
    "session.timeout.ms": os.getenv("KAFKA_SESSION_TIMEOUT_MS", "45000"),
    "client.id": os.getenv("KAFKA_CLIENT_ID", "binance-collector"),
}
