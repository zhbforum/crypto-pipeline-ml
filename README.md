# crypto-pipeline-ml
# Config (quick reference)

* `MODE` — `"ticker"` or `"kline"`. Data collection mode.
* `INTERVAL` — kline timeframe (used only when `MODE="kline"`). Allowed: `1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d`.
* `PAIRS` — list of Binance symbols to collect (e.g., `["BTCUSDT", "ETHUSDT"]`).
* `EVERY_SEC` — pause between cycles in **ticker** mode (seconds). Ignored in kline mode.
* `OUT_DIR` — folder for CSV output (auto-created if missing).
* `CONCURRENCY` — number of concurrent requests in **kline** mode.
