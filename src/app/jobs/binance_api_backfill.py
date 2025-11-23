from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import requests
from dotenv import load_dotenv
from pyspark import SparkConf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    LongType,
    DoubleType,
)

from app.parser_settings.constants import (
    DEFAULT_AWS_REGION,
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_PREFIX,
    DEFAULT_BINANCE_INTERVAL,
    DEFAULT_BINANCE_SYMBOLS,
    DEFAULT_BINANCE_START_DATE,
    DEFAULT_BINANCE_END_DATE,
    DEFAULT_APP_NAME,
    DEFAULT_DOWNLOAD_WORKERS,
)


def _load_project_env() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        env_path = parent / ".env"
        if env_path.is_file():
            print(f"[info] loaded dotenv from {env_path}")
            load_dotenv(env_path, override=True)
            return
    print("[warn] .env not found while walking up from script dir")


_load_project_env()


BINANCE_API_BASE_URL = "https://api.binance.com"

ROWS_SCHEMA = StructType(
    [
        StructField("ts", LongType(), False),
        StructField("iso_ts", StringType(), False),
        StructField("symbol", StringType(), False),
        StructField("interval", StringType(), False),
        StructField("open", DoubleType(), False),
        StructField("high", DoubleType(), False),
        StructField("low", DoubleType(), False),
        StructField("close", DoubleType(), False),
        StructField("volume", DoubleType(), False),
    ]
)


def _get_env_str(
    key: str,
    default: Optional[str] = None,
    *,
    required: bool = False,
) -> str:
    val = os.getenv(key, default)
    if (val is None or val == "") and required:
        raise SystemExit(f"Missing required env var: {key}")
    return "" if val is None else val


def _build_spark(app_name: str) -> SparkSession:
    conf = SparkConf().setAppName(app_name).set("spark.sql.session.timeZone", "UTC")

    region = os.getenv("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION or "eu-north-1")
    conf = (
        conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .set(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .set(
            "spark.hadoop.fs.s3a.access.key",
            _get_env_str("AWS_ACCESS_KEY_ID", required=True),
        )
        .set(
            "spark.hadoop.fs.s3a.secret.key",
            _get_env_str("AWS_SECRET_ACCESS_KEY", required=True),
        )
        .set("spark.hadoop.fs.s3a.endpoint", f"s3.{region}.amazonaws.com")
    )

    return SparkSession.builder.config(conf=conf).getOrCreate()


def _interval_to_ms(interval: str) -> int:
    mapping = {
        "1m": 1 * 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]


def _fetch_symbol_klines_api(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> List[Tuple]:
    url = BINANCE_API_BASE_URL + "/api/v3/klines"
    rows: List[Tuple] = []

    step_ms = _interval_to_ms(interval)
    cur = start_ms

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"[WARN] request failed for {symbol}: {exc}")
            break

        if resp.status_code != 200:
            print(
                f"[WARN] non-200 for {symbol}: status={resp.status_code}, body={resp.text[:200]}"
            )
            break

        data = resp.json()
        if not data:
            break

        last_open_time = None

        for entry in data:
            open_time_ms = int(entry[0])
            if open_time_ms >= end_ms:
                break

            dt_utc = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
            iso_ts = dt_utc.isoformat().replace("+00:00", "Z")

            o = float(entry[1])
            h = float(entry[2])
            l = float(entry[3])
            c = float(entry[4])
            vol = float(entry[5])

            rows.append(
                (
                    open_time_ms,
                    iso_ts,
                    symbol,
                    interval,
                    o,
                    h,
                    l,
                    c,
                    vol,
                )
            )
            last_open_time = open_time_ms

        if last_open_time is None:
            break

        next_start = last_open_time + step_ms
        if next_start >= end_ms:
            break

        cur = next_start
        time.sleep(0.1)

    print(f"[info] fetched {len(rows)} rows from API for {symbol}")
    return rows


def _flush_month_to_s3(
    spark: SparkSession,
    rows_buffer: List[Tuple],
    raw_base: str,
    year_month: str,
) -> None:
    if not rows_buffer:
        print(f"[info] no rows to flush for {year_month}")
        return

    print(f"[info] flushing month {year_month} to S3 (rows={len(rows_buffer)})")

    df = spark.createDataFrame(rows_buffer, schema=ROWS_SCHEMA)
    symbols = sorted({row[2] for row in rows_buffer})

    for sym in symbols:
        sub_df: DataFrame = (
            df.filter(F.col("symbol") == sym)
            .select(
                "ts",
                "iso_ts",
                "symbol",
                "interval",
                "open",
                "high",
                "low",
                "close",
                "volume",
            )
            .orderBy("ts")
        )

        out_path = f"{raw_base}/symbol={sym}/month={year_month}"
        print(f"[info] writing {sym} {year_month} to {out_path}")

        (
            sub_df.coalesce(1)
            .write.mode("overwrite")
            .option("header", "true")
            .option("compression", "snappy")
            .csv(out_path)
        )

    print(f"[ok] written monthly CSVs for {year_month} to {raw_base}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill Binance klines from API into "
            "s3a://BUCKET/PREFIX/symbol=SYM/month=YYYY-MM as CSV (snappy)"
        )
    )
    parser.add_argument("--start-date", help="YYYY-MM-DD", default=None)
    parser.add_argument("--end-date", help="YYYY-MM-DD", default=None)
    parser.add_argument("--symbols", help="Comma-separated: BTCUSDT,ETHUSDT,...", default=None)
    parser.add_argument(
        "--interval",
        help="Binance kline interval (1m,3m,5m...). Default: from constants / env",
        default=None,
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=base_dir / ".env", override=True)
    print(f"[info] loaded dotenv from {base_dir / '.env'}")

    bucket = _get_env_str("S3_BUCKET", DEFAULT_S3_BUCKET, required=True)
    s3_prefix = os.getenv("S3_PREFIX", DEFAULT_S3_PREFIX) or DEFAULT_S3_PREFIX
    raw_base = f"s3a://{bucket}/{s3_prefix}"

    interval = (
        args.interval
        or os.getenv("BINANCE_INTERVAL", DEFAULT_BINANCE_INTERVAL)
        or DEFAULT_BINANCE_INTERVAL
    )

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        env_syms = os.getenv("BINANCE_SYMBOLS", "")
        symbols = (
            [s.strip() for s in env_syms.split(",") if s.strip()]
            if env_syms
            else DEFAULT_BINANCE_SYMBOLS
        )

    download_workers = int(os.getenv("DOWNLOAD_WORKERS", str(DEFAULT_DOWNLOAD_WORKERS)))

    start_str = (
        args.start_date
        or os.getenv("BINANCE_START_DATE", DEFAULT_BINANCE_START_DATE)
        or DEFAULT_BINANCE_START_DATE
    )
    end_env = os.getenv("BINANCE_END_DATE", DEFAULT_BINANCE_END_DATE)
    end_str = args.end_date or (end_env if end_env else None)

    start_day = date.fromisoformat(start_str)
    end_day = date.fromisoformat(end_str) if end_str else date.today()

    start_dt = datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc)
    end_dt = datetime(end_day.year, end_day.month, end_day.day, tzinfo=timezone.utc) + timedelta(
        days=1
    )
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    app_name = os.getenv("APP_NAME", DEFAULT_APP_NAME) or DEFAULT_APP_NAME
    spark = _build_spark(app_name)

    print(f"[info] app_name: {app_name}")
    print(f"[info] raw_base: {raw_base}")
    print(f"[info] symbols: {symbols}")
    print(f"[info] interval: {interval}")
    print(f"[info] date range (days): {start_day} .. {end_day}")
    print(f"[info] download_workers (not used yet): {download_workers}")

    rows_by_month: Dict[str, List[Tuple]] = defaultdict(list)

    for sym in symbols:
        sym_rows = _fetch_symbol_klines_api(sym, interval, start_ms, end_ms)
        for row in sym_rows:
            ts = row[0]
            dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            ym = dt_utc.strftime("%Y-%m")
            rows_by_month[ym].append(row)

    for ym, rows in rows_by_month.items():
        _flush_month_to_s3(spark, rows, raw_base, ym)

    spark.stop()
    print("[info] done")


if __name__ == "__main__":
    main()
