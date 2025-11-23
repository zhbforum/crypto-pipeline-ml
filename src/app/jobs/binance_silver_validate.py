from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List

from dotenv import load_dotenv
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
)
from pyspark.errors.exceptions.base import AnalysisException

from app.parser_settings.constants import (
    DEFAULT_APP_NAME,
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_PREFIX,
    S3_PREFIX_SILVER,
    DEFAULT_BINANCE_SYMBOLS,
    VALIDATE_START_DATE,
    VALIDATE_END_DATE,
)

RAW_ROWS_SCHEMA = StructType(
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
    key: str, default: str | None = None, *, required: bool = False
) -> str:
    val = os.getenv(key, default)
    if (val is None or val == "") and required:
        raise SystemExit(f"Missing required env variable: {key}")
    return "" if val is None else val


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@dataclass
class MonthBatch:
    year: int
    month: int
    batch_start: date
    batch_end: date


def month_batches(start: date, end: date) -> List[MonthBatch]:
    batches: List[MonthBatch] = []
    cur = date(start.year, start.month, 1)

    while cur <= end:
        if cur.month == 12:
            next_month = date(cur.year + 1, 1, 1)
        else:
            next_month = date(cur.year, cur.month + 1, 1)

        batch_start = max(start, cur)
        batch_end = min(end, next_month - timedelta(days=1))

        batches.append(
            MonthBatch(
                year=cur.year,
                month=cur.month,
                batch_start=batch_start,
                batch_end=batch_end,
            )
        )
        cur = next_month

    return batches


def validate_kline_schema(df: DataFrame) -> DataFrame:
    required_cols = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in raw df: {missing}")

    df = df.select(
        "ts",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    df = df.withColumn("open_time", F.col("ts").cast("long"))
    df = df.withColumn("open_ts", F.to_timestamp(F.col("open_time") / 1000.0))
    df = df.withColumn("date", F.to_date("open_ts"))

    for col_name in [
        "symbol",
        "open_time",
        "open_ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df = df.filter(F.col(col_name).isNotNull())

    df = df.filter(
        (F.col("open") > 0)
        & (F.col("high") > 0)
        & (F.col("low") > 0)
        & (F.col("close") > 0)
        & (F.col("high") >= F.col("low"))
        & (F.col("volume") >= 0)
    )

    df = df.dropDuplicates(["symbol", "open_ts"])

    cols = ["symbol", "open_ts", "open", "high", "low", "close", "volume", "date"]
    return df.select(*cols)


def _configure_s3_credentials(spark: SparkSession) -> None:
    # pylint: disable=protected-access
    jsc = spark.sparkContext._jsc
    assert jsc is not None
    hconf = jsc.hadoopConfiguration()  # type: ignore[call-arg]

    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_token = os.getenv("AWS_SESSION_TOKEN")

    if aws_key and aws_secret:
        hconf.set("fs.s3a.access.key", aws_key)
        hconf.set("fs.s3a.secret.key", aws_secret)
    if aws_token:
        hconf.set("fs.s3a.session.token", aws_token)


def build_spark() -> SparkSession:
    load_dotenv()

    app_name = DEFAULT_APP_NAME
    master = _get_env_str("SPARK_MASTER", "local[*]")
    local_dir = _get_env_str("SPARK_LOCAL_DIR", r"F:\temp")

    conf = (
        SparkConf()
        .setAppName(app_name)
        .set("spark.master", master)
        .set("spark.local.dir", local_dir)
        .set("spark.hadoop.parquet.enable.summary-metadata", "false")
        .set("spark.hadoop.fs.s3a.path.style.access", "true")
        .set("spark.hadoop.fs.s3a.fast.upload", "true")
        .set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .set("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    )

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    _configure_s3_credentials(spark)

    return spark


def process_month_for_symbol(
    spark: SparkSession,
    raw_base_path: str,
    silver_base_path: str,
    symbol: str,
    batch: MonthBatch,
) -> None:
    ym = f"{batch.year:04d}-{batch.month:02d}"
    path = f"{raw_base_path}/symbol={symbol}/month={ym}"

    min_d = batch.batch_start.isoformat()
    max_d = batch.batch_end.isoformat()

    print(
        f"[info] reading RAW for {symbol} from: {path} "
        f"(fmt=csv, days={min_d}..{max_d})"
    )

    try:
        df_raw = (
            spark.read.format("csv")
            .option("header", "true")
            .schema(RAW_ROWS_SCHEMA)
            .load(path)
        )
    except AnalysisException:
        print(f"[info] {symbol} {ym}: no raw data, skip")
        return

    df_valid = validate_kline_schema(df_raw)

    df_valid = df_valid.filter((F.col("date") >= min_d) & (F.col("date") <= max_d))

    df_valid = (
        df_valid.withColumn("year", F.lit(batch.year).cast("int"))
        .withColumn("month", F.lit(batch.month).cast("int"))
        .coalesce(1)
    )

    (
        df_valid.write.mode("append")
        .format("csv")
        .option("header", "true")
        .option("compression", "snappy")
        .partitionBy("symbol", "year", "month")
        .save(silver_base_path)
    )

    print(
        f"[info] written SILVER for {symbol} {ym} "
        f"into {silver_base_path} (partitioned by symbol/year/month, csv+snappy)"
    )


@dataclass
class SilverJobConfig:
    raw_base_path: str
    silver_base_path: str
    symbols: List[str]
    batches: List[MonthBatch]


def _build_silver_job_config() -> SilverJobConfig:
    start = parse_date(VALIDATE_START_DATE)
    end = parse_date(VALIDATE_END_DATE)
    if end < start:
        raise SystemExit("VALIDATE_END_DATE must be >= VALIDATE_START_DATE")

    bucket = DEFAULT_S3_BUCKET
    raw_prefix = DEFAULT_S3_PREFIX
    silver_prefix = S3_PREFIX_SILVER

    raw_base_path = f"s3a://{bucket}/{raw_prefix}"
    silver_base_path = f"s3a://{bucket}/{silver_prefix}/kline_1m"

    symbols = list(DEFAULT_BINANCE_SYMBOLS)
    batches = month_batches(start, end)

    print(
        f"[info] total month batches: {len(batches)} "
        f"from {batches[0].batch_start} to {batches[-1].batch_end}"
    )
    print(f"[info] raw_base_path    = {raw_base_path}")
    print(f"[info] silver_base_path = {silver_base_path}")
    print(f"[info] symbols          = {symbols}")

    return SilverJobConfig(
        raw_base_path=raw_base_path,
        silver_base_path=silver_base_path,
        symbols=symbols,
        batches=batches,
    )


def main() -> None:
    config = _build_silver_job_config()
    spark = build_spark()
    try:
        for batch in config.batches:
            print(
                f"[info] === processing batch {batch.year:04d}-{batch.month:02d} "
                f"({batch.batch_start}..{batch.batch_end}) ==="
            )
            for symbol in config.symbols:
                process_month_for_symbol(
                    spark=spark,
                    raw_base_path=config.raw_base_path,
                    silver_base_path=config.silver_base_path,
                    symbol=symbol,
                    batch=batch,
                )
    finally:
        spark.stop()
        print("[info] spark stopped")


if __name__ == "__main__":
    main()
