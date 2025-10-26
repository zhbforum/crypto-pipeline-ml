from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from app.exchange.binance_client import BinanceClient
import asyncio

class CollectorService:
    def __init__(self, client: BinanceClient):
        self.client = client

    async def one_cycle_ticker(
        self, pairs: List[str], sink
    ) -> Tuple[int, int, int, List[Dict[str, Any]]]:
        ok = 0
        rows: List[Dict[str, Any]] = []
        try:
            all_prices = await self.client.bulk_tickers()
            now_ms = int(time.time() * 1000)
            iso = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).isoformat()
            for sym in pairs:
                price = all_prices.get(sym)
                if price is None:
                    continue
                rows.append(
                    {
                        "ts": now_ms,
                        "iso_ts": iso,
                        "symbol": sym,
                        "price": price,
                        "event_type": "ticker",
                        "source": "binance",
                    }
                )
                ok += 1
        except Exception:
            pass
        wrote = await sink.write(rows)
        return ok, len(pairs) - ok, wrote, rows

    async def one_cycle_kline(
        self, pairs: List[str], interval: str, sink, concurrency: int
    ) -> Tuple[int, int, int, List[Dict[str, Any]]]:
        sem = asyncio.Semaphore(concurrency)

        async def worker(sym: str) -> Optional[Dict[str, Any]]:
            async with sem:
                return await self.client.last_kline(sym, interval)

        res = await asyncio.gather(*[worker(s) for s in pairs])
        rows: List[Dict[str, Any]] = []
        for r in res:
            if not r:
                continue
            rows.append(
                {
                    **r,
                    "iso_ts": datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).isoformat(),
                    "event_type": "kline",
                    "source": "binance",
                }
            )
        wrote = await sink.write(rows)
        ok = len(rows)
        fail = len(pairs) - ok
        return ok, fail, wrote, rows
