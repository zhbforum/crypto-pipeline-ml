from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from app.timeutil import interval_seconds, sleep_until_next_boundary
from app.exchange.binance_client import BinanceClient
from app.services.collector import CollectorService
from app.sinks.kafka_writer import KafkaWriter
from app.constants import (
    MODE,
    INTERVAL,
    PAIRS,
    EVERY_SEC,
    CONCURRENCY,
    BINANCE_BASE,
    KAFKA_ENABLED,
    KAFKA_TOPIC,
)

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

async def run() -> None:
    client = BinanceClient(
        base_url=BINANCE_BASE,
        max_concurrency=CONCURRENCY,
        user_agent="binance-collector/csv+kafka-1.0",
    )
    svc = CollectorService(client)

    # if MODE == "ticker":
    #     sink = CsvSink(
    #         Path(OUT_DIR) / "binance_ticker.csv",
    #         ["ts", "iso_ts", "symbol", "price"],
    #     )
    # else:
    #     sink = CsvSink(
    #         Path(OUT_DIR) / f"binance_kline_{INTERVAL}.csv",
    #         ["ts", "iso_ts", "symbol", "interval", "open", "high", "low", "close", "volume"],
    #     )
    sink = None # Temporary disable CSV sink
    kafka_writer = KafkaWriter.from_env(KAFKA_TOPIC) if KAFKA_ENABLED else None

    cycle = 0
    if MODE == "kline":
        await sleep_until_next_boundary(interval_seconds(INTERVAL))

    try:
        while True:
            cycle += 1
            t0 = asyncio.get_event_loop().time()

            if MODE == "ticker":
                ok, fail, wrote_csv, rows = await svc.one_cycle_ticker(PAIRS, sink)
            elif MODE == "kline":
                ok, fail, wrote_csv, rows = await svc.one_cycle_kline(PAIRS, INTERVAL, sink, CONCURRENCY)
            else:
                ok, fail, wrote_csv, rows = 0, len(PAIRS), 0, []

            sent_to_kafka = 0
            if kafka_writer and rows:
                sent_to_kafka = kafka_writer.send_batch(rows)

            dt = asyncio.get_event_loop().time() - t0
            extra = f" {INTERVAL}" if MODE == "kline" else ""
            print(
                f"{datetime.now(timezone.utc).isoformat()} | {MODE}{extra} | "
                f"pairs={len(PAIRS)} ok={ok} fail={fail} wrote_csv={wrote_csv} "
                f"kafka={sent_to_kafka} dt={dt:.2f}s"
            )

            if MODE == "ticker":
                await asyncio.sleep(EVERY_SEC)
            else:
                await sleep_until_next_boundary(interval_seconds(INTERVAL))
    finally:
        await client.aclose()
