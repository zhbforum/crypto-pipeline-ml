from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import List

import requests


BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"


def fetch_klines(
    start_time_ms: int,
    end_time_ms: int,
    limit: int = 1000,
) -> list:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": limit,
    }
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected response from Binance API")
    return data


def main() -> None:
    utc = dt.timezone.utc
    start_dt = dt.datetime(2017, 1, 1, tzinfo=utc)
    end_dt = dt.datetime.now(tz=utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_rows: List[tuple[str, float]] = []

    current_ms = start_ms
    max_iterations = 1000
    iterations = 0

    while current_ms < end_ms and iterations < max_iterations:
        data = fetch_klines(current_ms, end_ms)
        if not data:
            break

        for entry in data:
            open_time_ms = int(entry[0])
            close_price_str = entry[4]
            close_price = float(close_price_str)

            open_dt = dt.datetime.fromtimestamp(
                open_time_ms / 1000.0,
                tz=utc,
            )
            date_str = open_dt.date().isoformat()
            all_rows.append((date_str, close_price))

        last_open_time_ms = int(data[-1][0])
        current_ms = last_open_time_ms + 1
        iterations += 1
        time.sleep(0.1)

    if not all_rows:
        raise RuntimeError("No data fetched from Binance API")

    seen_dates = set()
    dedup_rows: List[tuple[str, float]] = []
    for date_str, close_price in all_rows:
        if date_str in seen_dates:
            continue
        seen_dates.add(date_str)
        dedup_rows.append((date_str, close_price))

    dedup_rows.sort(key=lambda x: x[0])

    jobs_dir = Path(__file__).resolve().parent
    analytics_dir = jobs_dir.parent
    data_dir = analytics_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "btc_daily_close.csv"

    lines = ["date,close"]
    for date_str, close_price in dedup_rows:
        lines.append(f"{date_str},{close_price}")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {len(dedup_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
