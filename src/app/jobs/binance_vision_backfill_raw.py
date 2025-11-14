from __future__ import annotations

import argparse
import csv
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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

BINANCE_BASE_URL = "https://data.binance.vision"


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
    conf = (
        SparkConf()
        .setAppName(app_name)
        .set("spark.sql.session.timeZone", "UTC")
    )

    region = _get_env_str("AWS_DEFAULT_REGION", "eu-central-1")

    conf = (
        conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .set(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .set("spark.hadoop.fs.s3a.access.key", _get_env_str("AWS_ACCESS_KEY_ID", ""))
        .set(
            "spark.hadoop.fs.s3a.secret.key",
            _get_env_str("AWS_SECRET_ACCESS_KEY", ""),
        )
        .set("spark.hadoop.fs.s3a.endpoint", f"s3.{region}.amazonaws.com")
    )

    return SparkSession.builder.config(conf=conf).getOrCreate()


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


def _iter_days(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _fetch_binance_zip(symbol: str, day: date) -> Optional[bytes]:
    path = f"/data/spot/daily/klines/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"
    url = BINANCE_BASE_URL + path
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"[WARN] request failed for {symbol} {day}: {exc}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] no data for {symbol} {day} (status={resp.status_code})")
        return None

    return resp.content


def _parse_kline_csv(symbol: str, content: bytes) -> List[Tuple]:
    out: List[Tuple] = []

    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as e:
        print(f"[WARN] bad zip for {symbol}: {e}")
        return out

    with zf:
        names = zf.namelist()
        if not names:
            return out

        with zf.open(names[0]) as f:
            reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
            for i, row in enumerate(reader):
                if not row:
                    continue

                try:
                    open_time_raw = int(row[0])
                except (ValueError, TypeError) as e:
                    print(f"[WARN] bad open_time for {symbol} row={i}: {row} ({e})")
                    continue

                if open_time_raw > 10**13:
                    open_time_ms = open_time_raw // 1000
                else:
                    open_time_ms = open_time_raw

                if open_time_ms < 1483228800000 or open_time_ms > 4102444800000:
                    print(
                        f"[WARN] out-of-range open_time_ms={open_time_ms} "
                        f"for {symbol} row={i}: {row}"
                    )
                    continue

                try:
                    iso_ts = (
                        datetime.utcfromtimestamp(open_time_ms / 1000)
                        .isoformat()
                        + "Z"
                    )
                except OSError as e:
                    print(
                        f"[WARN] utcfromtimestamp failed for {symbol} "
                        f"row={i} ts={open_time_ms}: {e}"
                    )
                    continue

                try:
                    o, h, l, c = map(float, row[1:5])
                    vol = float(row[5])
                except (ValueError, TypeError, IndexError) as e:
                    print(
                        f"[WARN] bad OHLCV for {symbol} row={i}: {row} ({e})"
                    )
                    continue

                out.append(
                    (
                        open_time_ms,
                        iso_ts,
                        symbol,
                        "1m",
                        o,
                        h,
                        l,
                        c,
                        vol,
                    )
                )

    return out


def _load_day_for_symbols(
    symbols: Iterable[str],
    day: date,
    max_workers: int = 5,
) -> List[Tuple]:
    def fetch_for_symbol(sym: str) -> List[Tuple]:
        print(f"[info] fetching {sym} {day}")
        content = _fetch_binance_zip(sym, day)
        if content is None:
            return []
        return _parse_kline_csv(sym, content)

    all_rows: List[Tuple] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_for_symbol, s): s for s in symbols}
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)

    return all_rows


def _flush_buffer_to_s3(
    spark: SparkSession,
    rows_buffer: List[Tuple],
    raw_base: str,
    topic: str,
) -> None:
    if not rows_buffer:
        return

    num_rows = len(rows_buffer)
    print(f"[info] flushing buffer to S3 (rows={num_rows})")

    df = spark.createDataFrame(rows_buffer, schema=ROWS_SCHEMA)

    ts_utc = (F.col("ts") / F.lit(1000)).cast("timestamp")
    ts_berlin = F.from_utc_timestamp(ts_utc, "Europe/Berlin")
    df = df.withColumn("date", F.to_date(ts_berlin))

    df = df.withColumn("topic", F.lit(topic))
    df = df.withColumn(
        "value",
        F.to_json(
            F.struct(
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
        ),
    )

    out_df: DataFrame = df.select("topic", "date", "value")

    (
        out_df.write.format("parquet")
        .mode("append")
        .partitionBy("topic", "date")
        .save(raw_base)
    )

    print(f"[ok] written chunk to {raw_base} (rows={num_rows})")

    rows_buffer.clear()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Binance Vision 1m klines into raw/topic=.../date=..."
    )
    parser.add_argument("--start-date", help="YYYY-MM-DD", default=None)
    parser.add_argument("--end-date", help="YYYY-MM-DD", default=None)
    parser.add_argument(
        "--symbols",
        help="Comma-separated list: BTCUSDT,ETHUSDT,...",
        default=None,
    )
    parser.add_argument(
        "--topic",
        help="topic partition name in raw (e.g. topic_0)",
        default=None,
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    dotenv_path = base_dir / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"[info] loaded dotenv from {dotenv_path}")

    bucket = _get_env_str("S3_BUCKET", required=True)
    s3_prefix = _get_env_str("S3_PREFIX", "raw") or "raw"
    raw_base = f"s3a://{bucket}/{s3_prefix}"

    topic = args.topic or _get_env_str("KAFKA_TOPIC", "topic_0") or "topic_0"

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        env_syms = _get_env_str("BINANCE_SYMBOLS", "")
        if env_syms:
            symbols = [s.strip() for s in env_syms.split(",") if s.strip()]
        else:
            symbols = [
                "BTCUSDT",
                "ETHUSDT",
                "XRPUSDT",
                "SOLUSDT",
                "BNBUSDT",
                "DOGEUSDT",
                "LTCUSDT",
                "ADAUSDT",
                "TRXUSDT",
                "LINKUSDT",
            ]

    download_workers = int(os.getenv("DOWNLOAD_WORKERS", "5"))

    start_str = (
        args.start_date
        or _get_env_str("BINANCE_START_DATE", "2022-12-31")
        or "2022-12-31"
    )
    end_str = args.end_date or _get_env_str("BINANCE_END_DATE", "") or None

    start_day = date.fromisoformat(start_str)
    end_day = date.fromisoformat(end_str) if end_str else date.today()

    app_name = _get_env_str("APP_NAME", "binance_vision_backfill_raw") or "binance_vision_backfill_raw"
    spark = _build_spark(app_name)

    print(f"[info] raw_base: {raw_base}")
    print(f"[info] topic: {topic}")
    print(f"[info] symbols: {symbols}")
    print(f"[info] date range: {start_day} .. {end_day}")
    print(f"[info] download_workers: {download_workers}")

    rows_buffer: List[Tuple] = []

    current_year = start_day.year
    current_month = start_day.month

    for day in _iter_days(start_day, end_day):
        rows = _load_day_for_symbols(symbols, day, max_workers=download_workers)
        if rows:
            rows_buffer.extend(rows)

        next_day = day + timedelta(days=1)
        month_changed = (next_day.month != current_month or next_day.year != current_year)
        is_last_day = day >= end_day

        if month_changed or is_last_day:
            _flush_buffer_to_s3(spark, rows_buffer, raw_base, topic)
            current_year = next_day.year
            current_month = next_day.month

    spark.stop()


if __name__ == "__main__":
    main()