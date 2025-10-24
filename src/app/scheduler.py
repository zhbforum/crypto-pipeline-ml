from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from app.timeutil import interval_seconds, sleep_until_next_boundary
from app.exchange.binance_client import BinanceClient
from app.sinks.csv_sink import CsvSink
from app.services.collector import CollectorService
from app.constants import MODE, INTERVAL, PAIRS, EVERY_SEC, OUT_DIR, CONCURRENCY, BINANCE_BASE


async def run() -> None:
    client = BinanceClient(base_url=BINANCE_BASE, max_concurrency=CONCURRENCY, user_agent="binance-collector/csv-const-0.7")
    svc = CollectorService(client)

    if MODE == "ticker":
        sink = CsvSink(Path(OUT_DIR) / "binance_ticker.csv", ["ts","iso_ts","symbol","price"])
    else:
        sink = CsvSink(
            Path(OUT_DIR) / f"binance_kline_{INTERVAL}.csv",
            ["ts","iso_ts","symbol","interval","open","high","low","close","volume"],
        )

    cycle = 0
    if MODE == "kline":
        await sleep_until_next_boundary(interval_seconds(INTERVAL))

    try:
        while True:
            cycle += 1
            t0 = asyncio.get_event_loop().time()
            if MODE == "ticker":
                ok, fail, wrote = await svc.one_cycle_ticker(PAIRS, sink)
            elif MODE == "kline":
                ok, fail, wrote = await svc.one_cycle_kline(PAIRS, INTERVAL, sink, CONCURRENCY)
            else:
                ok, fail, wrote = 0, len(PAIRS), 0

            dt = asyncio.get_event_loop().time() - t0
            extra = f" {INTERVAL}" if MODE == "kline" else ""
            print(f"{datetime.now(timezone.utc).isoformat()} | {MODE}{extra} | pairs={len(PAIRS)} ok={ok} fail={fail} wrote={wrote} dt={dt:.2f}s")

            if MODE == "ticker":
                await asyncio.sleep(EVERY_SEC)
            else:
                await sleep_until_next_boundary(interval_seconds(INTERVAL))
    finally:
        await client.aclose()
