from __future__ import annotations
import httpx
from typing import Any, Dict, Optional


class BinanceClient:
    def __init__(self, base_url: str, max_concurrency: int, user_agent: str = "crypto-pipeline/1.0"):
        limits = httpx.Limits(max_keepalive_connections=max_concurrency, max_connections=max_concurrency)
        self._client = httpx.AsyncClient(base_url=base_url, limits=limits, headers={"User-Agent": user_agent})

    async def bulk_tickers(self) -> Dict[str, float]:
        r = await self._client.get("/api/v3/ticker/price", timeout=10)
        r.raise_for_status()
        return {it["symbol"]: float(it["price"]) for it in r.json()}

    async def last_kline(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        try:
            r = await self._client.get(
                "/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": 1},
                timeout=10,
            )
            r.raise_for_status()
            arr = r.json()
            if not arr:
                return None
            o = arr[0]
            return {
                "ts": int(o[0]),
                "symbol": symbol,
                "interval": interval,
                "open": float(o[1]),
                "high": float(o[2]),
                "low": float(o[3]),
                "close": float(o[4]),
                "volume": float(o[5]),
            }
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    async def aclose(self) -> None:
        await self._client.aclose()
