from __future__ import annotations

import asyncio
import csv
import math
import time
import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from constants import MODE, INTERVAL, PAIRS, EVERY_SEC, OUT_DIR, CONCURRENCY, BINANCE_BASE, _INTERVAL_MAP_SECONDS


def interval_seconds(interval: str) -> int:
    
    if interval not in _INTERVAL_MAP_SECONDS:
        raise ValueError(f"Unsupported INTERVAL: {interval}")
    return _INTERVAL_MAP_SECONDS[interval]


async def sleep_until_next_boundary(period_sec: int) -> None:
    
    now = time.time()
    next_edge = (math.floor(now / period_sec) + 1) * period_sec
    sleep_s = max(0.0, next_edge - now + 0.5)
    await asyncio.sleep(sleep_s)


async def binance_bulk_tickers(client: httpx.AsyncClient) -> Dict[str, float]:
    
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    r = await client.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    return {it["symbol"]: float(it["price"]) for it in data}


async def binance_kline_last(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
) -> Optional[Dict[str, Any]]:
    
    url = f"{BINANCE_BASE}/api/v3/klines"
    try:
        r = await client.get(
            url,
            params={"symbol": symbol, "interval": interval, "limit": 1},
            timeout=10,
        )
        r.raise_for_status()
        arr = r.json()
        if not arr:
            return None
        o = arr[0]
        open_time, open_, high, low, close, volume = o[0], o[1], o[2], o[3], o[4], o[5]
        return {
            "ts": int(open_time),
            "symbol": symbol,
            "interval": interval,
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> int:
    
    if not rows:
        return 0
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_rows_ticker(out_dir: Path, rows: List[Dict[str, Any]]) -> int:
    
    cols = ["ts", "iso_ts", "symbol", "price"]
    norm: List[Dict[str, Any]] = []
    for r in rows:
        iso = datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).isoformat()
        norm.append({
            "ts": r["ts"],
            "iso_ts": iso,
            "symbol": r["symbol"],
            "price": r["price"],
        })
    out_path = out_dir / "binance_ticker.csv"
    return _write_csv(out_path, cols, norm)


def write_rows_kline(out_dir: Path, interval: str, rows: List[Dict[str, Any]]) -> int:
    
    cols = ["ts", "iso_ts", "symbol", "interval", "open", "high", "low", "close", "volume"]
    norm: List[Dict[str, Any]] = []
    for r in rows:
        iso = datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).isoformat()
        norm.append({
            "ts": r["ts"],
            "iso_ts": iso,
            "symbol": r["symbol"],
            "interval": r["interval"],
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"],
        })
    out_path = out_dir / f"binance_kline_{interval}.csv"
    return _write_csv(out_path, cols, norm)


async def one_cycle_ticker(
    pairs: List[str],
    client: httpx.AsyncClient,
    out_dir: Path,
) -> Tuple[int, int, int]:
    
    ok = 0
    rows: List[Dict[str, Any]] = []
    try:
        all_prices = await binance_bulk_tickers(client)
        now_ms = int(time.time() * 1000)
        for sym in pairs:
            price = all_prices.get(sym)
            if price is None:
                continue
            rows.append({"ts": now_ms, "symbol": sym, "price": price})
            ok += 1
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    written = write_rows_ticker(out_dir, rows)
    fail = len(pairs) - ok
    return ok, fail, written


async def one_cycle_kline(
    pairs: List[str],
    interval: str,
    client: httpx.AsyncClient,
    out_dir: Path,
    concurrency: int,
) -> Tuple[int, int, int]:
    
    sem = asyncio.Semaphore(concurrency)

    async def worker(sym: str) -> Optional[Dict[str, Any]]:
        async with sem:
            return await binance_kline_last(client, sym, interval)

    tasks = [asyncio.create_task(worker(s)) for s in pairs]
    res = await asyncio.gather(*tasks)
    rows = [r for r in res if r]
    written = write_rows_kline(out_dir, interval, rows)
    ok = len(rows)
    fail = len(pairs) - ok
    return ok, fail, written


async def run() -> None:
    
    out_dir = Path(OUT_DIR)
    limits = httpx.Limits(max_keepalive_connections=CONCURRENCY, max_connections=CONCURRENCY)
    headers = {"User-Agent": "binance-collector/csv-const-0.7"}

    async with httpx.AsyncClient(limits=limits, headers=headers) as client:
        cycle = 0
        if MODE == "kline":
            period = interval_seconds(INTERVAL)
            await sleep_until_next_boundary(period)
        while True:
            cycle += 1
            t0 = time.monotonic()
            if MODE == "ticker":
                ok, fail, wrote = await one_cycle_ticker(PAIRS, client, out_dir)
            elif MODE == "kline":
                ok, fail, wrote = await one_cycle_kline(PAIRS, INTERVAL, client, out_dir, CONCURRENCY)
            else:
                ok, fail, wrote = 0, len(PAIRS), 0
            dt = time.monotonic() - t0
            ts_iso = datetime.now(timezone.utc).isoformat()
            extra = f" {INTERVAL}" if MODE == "kline" else ""
            print(f"{ts_iso} | {MODE}{extra} | pairs={len(PAIRS)} ok={ok} fail={fail} wrote={wrote} dt={dt:.2f}s")
            if MODE == "ticker":
                await asyncio.sleep(EVERY_SEC)
            else:
                await sleep_until_next_boundary(interval_seconds(INTERVAL))


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("[STOP]")
