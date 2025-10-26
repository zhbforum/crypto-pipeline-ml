from typing import Dict
import os
from pathlib import Path


MODE = os.getenv("MODE", "kline")          
INTERVAL = os.getenv("INTERVAL", "1m")

PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    # "BNBUSDT",
]

EVERY_SEC = int(os.getenv("EVERY_SEC", "5"))
OUT_DIR = os.getenv("OUT_DIR", "data")
CONCURRENCY = int(os.getenv("CONCURRENCY", "8"))
BINANCE_BASE = os.getenv("BINANCE_BASE", "https://api.binance.com")

SINK = os.getenv("SINK", "csv+kafka")  

KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "binance_events")
CLIENT_PROPERTIES_PATH = os.getenv("CLIENT_PROPERTIES_PATH", "./config/client.properties")

_env_flag = os.getenv("KAFKA_ENABLED")  
KAFKA_ENABLED = (
    (_env_flag == "1")
    or ("kafka" in SINK)
    or (Path(CLIENT_PROPERTIES_PATH).exists() and _env_flag != "0")
)

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
