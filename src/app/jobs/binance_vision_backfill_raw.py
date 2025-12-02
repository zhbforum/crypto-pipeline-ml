from __future__ import annotations

import argparse
import csv
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from pyspark import SparkConf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, StringType, LongType, DoubleType

from app.parser_settings.constants import (
    BINANCE_BASE_URL,
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

    region = os.getenv("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION)
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


def _iter_months(start: date, end: date) -> Iterable[date]:
    cur = start.replace(day=1)
    end_m = end.replace(day=1)
    while cur <= end_m:
        yield cur
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)


def _fetch_binance_month_zip(
    symbol: str, month_day: date, interval: str
) -> Optional[bytes]:
    ym = month_day.strftime("%Y-%m")
    path = f"/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    url = BINANCE_BASE_URL + path
    print(f"[info] request: {url}")
    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"[WARN] request failed for {symbol} {ym}: {exc}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] no data for {symbol} {ym} (status={resp.status_code})")
        return None

    return resp.content


def _parse_single_kline_row(
    symbol: str,
    interval: str,
    row: List[str],
    row_index: int,
) -> Optional[Tuple]:
    if not row:
        return None

    try:
        open_time_raw = int(row[0])
    except (ValueError, TypeError) as exc:
        print(f"[WARN] bad open_time for {symbol} row={row_index}: {row} ({exc})")
        return None

    open_time_ms = open_time_raw // 1000 if open_time_raw > 10**13 else open_time_raw

    if not (1483228800000 <= open_time_ms <= 4102444800000):
        print(
            f"[WARN] out-of-range open_time_ms={open_time_ms} "
            f"for {symbol} row={row_index}: {row}"
        )
        return None

    try:
        dt_utc = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
        iso_ts = dt_utc.isoformat().replace("+00:00", "Z")
    except OSError as exc:
        print(
            f"[WARN] fromtimestamp failed for {symbol} "
            f"row={row_index} ts={open_time_ms}: {exc}"
        )
        return None

    try:
        o, h, l, c = map(float, row[1:5])
        vol = float(row[5])
    except (ValueError, TypeError, IndexError) as exc:
        print(f"[WARN] bad OHLCV for {symbol} row={row_index}: {row} ({exc})")
        return None

    return (
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


def _parse_kline_csv(symbol: str, content: bytes, interval: str) -> List[Tuple]:
    out: List[Tuple] = []

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            if not names:
                return out

            with zf.open(names[0]) as f:
                reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
                for i, row in enumerate(reader):
                    parsed = _parse_single_kline_row(symbol, interval, row, i)
                    if parsed is not None:
                        out.append(parsed)
    except zipfile.BadZipFile as exc:
        print(f"[WARN] bad zip for {symbol}: {exc}")
        return out

    return out


def _load_month_for_symbols(
    symbols: Iterable[str],
    month_day: date,
    interval: str,
    max_workers: int = 5,
) -> List[Tuple]:
    ym = month_day.strftime("%Y-%m")

    def fetch_for_symbol(sym: str) -> List[Tuple]:
        print(f"[info] fetching {sym} {ym} ({interval})")
        content = _fetch_binance_month_zip(sym, month_day, interval)
        return [] if content is None else _parse_kline_csv(sym, content, interval)

    all_rows: List[Tuple] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_for_symbol, s): s for s in symbols}
        for fut in as_completed(futures):
            all_rows.extend(fut.result())

    return all_rows


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

        out_path = f"{raw_base}/kline=4h/symbol={sym}/month={year_month}"
        print(f"[info] writing {sym} {year_month} to {out_path}")

        (
            sub_df.coalesce(1)
            .write.mode("overwrite")
            .option("header", "true")
            .csv(out_path)
        )

    print(f"[ok] written monthly CSVs for {year_month} to {raw_base}")


@dataclass
class VisionBackfillConfig:
    app_name: str
    raw_base: str
    interval: str
    symbols: List[str]
    start_day: date
    end_day: date
    download_workers: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill Binance monthly klines into "
            "s3a://BUCKET/PREFIX/symbol=SYM/month=YYYY-MM as CSV (snappy)"
        )
    )
    parser.add_argument("--start-date", help="YYYY-MM-DD", default=None)
    parser.add_argument("--end-date", help="YYYY-MM-DD", default=None)
    parser.add_argument(
        "--symbols", help="Comma-separated: BTCUSDT,ETHUSDT,...", default=None
    )
    parser.add_argument(
        "--interval",
        help="Binance kline interval (1m,3m,5m...). Default: from constants / env",
        default=None,
    )
    return parser.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [s.strip() for s in args.symbols.split(",") if s.strip()]

    env_syms = os.getenv("BINANCE_SYMBOLS", "")
    if env_syms:
        return [s.strip() for s in env_syms.split(",") if s.strip()]

    return DEFAULT_BINANCE_SYMBOLS


def _resolve_date_range(args: argparse.Namespace) -> Tuple[date, date]:
    start_str = (
        args.start_date
        or os.getenv("BINANCE_START_DATE", DEFAULT_BINANCE_START_DATE)
        or DEFAULT_BINANCE_START_DATE
    )
    end_env = os.getenv("BINANCE_END_DATE", DEFAULT_BINANCE_END_DATE)
    end_str = args.end_date or (end_env if end_env else None)

    start_day = date.fromisoformat(start_str)
    end_day = date.fromisoformat(end_str) if end_str else date.today()
    return start_day, end_day


def _build_backfill_config(args: argparse.Namespace) -> VisionBackfillConfig:
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

    symbols = _resolve_symbols(args)
    download_workers = int(os.getenv("DOWNLOAD_WORKERS", str(DEFAULT_DOWNLOAD_WORKERS)))

    start_day, end_day = _resolve_date_range(args)
    app_name = os.getenv("APP_NAME", DEFAULT_APP_NAME) or DEFAULT_APP_NAME

    return VisionBackfillConfig(
        app_name=app_name,
        raw_base=raw_base,
        interval=interval,
        symbols=symbols,
        start_day=start_day,
        end_day=end_day,
        download_workers=download_workers,
    )


def main() -> None:
    args = _parse_args()
    config = _build_backfill_config(args)

    spark = _build_spark(config.app_name)

    print(f"[info] app_name: {config.app_name}")
    print(f"[info] raw_base: {config.raw_base}")
    print(f"[info] symbols: {config.symbols}")
    print(f"[info] interval: {config.interval}")
    print(f"[info] date range (months): {config.start_day} .. {config.end_day}")
    print(f"[info] download_workers: {config.download_workers}")

    for month_day in _iter_months(config.start_day, config.end_day):
        ym_str = month_day.strftime("%Y-%m")
        print(f"[info] processing month {ym_str}")

        rows = _load_month_for_symbols(
            config.symbols,
            month_day,
            config.interval,
            max_workers=config.download_workers,
        )
        if not rows:
            print(f"[info] no rows for month {ym_str}, skipping")
            continue

        _flush_month_to_s3(spark, rows, config.raw_base, ym_str)

    spark.stop()
    print("[info] done")


if __name__ == "__main__":
    main()
