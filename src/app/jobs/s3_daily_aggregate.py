from __future__ import annotations

import argparse
import os
from typing import Optional

from dotenv import load_dotenv
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    LongType,
    TimestampType,
)
from pyspark import SparkConf


def _get_env_str(
    key: str, default: Optional[str] = None, *, required: bool = False
) -> str:
    val = os.getenv(key, default)
    if (val is None or val == "") and required:
        raise SystemExit(f"Missing required env variable: {key}")
    return "" if val is None else val


def _build_spark(app_name: str) -> SparkSession:
    conf = (
        SparkConf()
        .setAppName(app_name)
        .set("spark.sql.session.timeZone", "UTC")
        .set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    )

    pkgs = os.getenv("SPARK_PACKAGES")
    if pkgs:
        conf = conf.set("spark.jars.packages", pkgs)

    region = _get_env_str("AWS_DEFAULT_REGION", "eu-north-1")
    conf = (
        conf.set(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .set(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .set(
            "spark.hadoop.fs.s3a.access.key",
            _get_env_str("AWS_ACCESS_KEY_ID", ""),
        )
        .set(
            "spark.hadoop.fs.s3a.secret.key",
            _get_env_str("AWS_SECRET_ACCESS_KEY", ""),
        )
        .set("spark.hadoop.fs.s3a.endpoint", f"s3.{region}.amazonaws.com")
    )
    return SparkSession.builder.config(conf=conf).getOrCreate()


PAYLOAD_SCHEMA = StructType(
    [
        StructField("ts", LongType(), True),
        StructField("iso_ts", StringType(), True),
        StructField("symbol", StringType(), True),
        StructField("interval", StringType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
    ]
)

PAYLOAD_SCHEMA_DDL = (
    "ts BIGINT, iso_ts STRING, symbol STRING, interval STRING, "
    "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE"
)


def _read_raw_for_day(
    spark: SparkSession, raw_base: str, day: str, topic: Optional[str]
) -> DataFrame:
    fmt = _get_env_str("WRITE_FORMAT", "parquet") or "parquet"

    if topic:
        path = f"{raw_base}/topic={topic}/date={day}"
    else:
        path = f"{raw_base}/topic=*/date={day}"

    df = spark.read.format(fmt).option("basePath", raw_base).load(path)

    df = df.withColumn("payload_str", F.col("value").cast("string"))

    json_struct = F.from_json(F.col("payload_str"), PAYLOAD_SCHEMA).alias("pj")
    csv_struct = F.from_csv(
        F.col("payload_str"),
        PAYLOAD_SCHEMA_DDL,
        {
            "header": "false",
            "multiLine": "false",
            "quote": '"',
            "escape": "\\",
            "sep": ",",
        },
    ).alias("pc")

    df = df.withColumns(
        {
            "pj": json_struct,
            "pc": csv_struct,
        }
    )

    def pick(field: str):
        return F.coalesce(F.col(f"pj.{field}"), F.col(f"pc.{field}")).alias(
            field
        )

    df = df.select(
        pick("ts"),
        pick("iso_ts"),
        pick("symbol"),
        pick("interval"),
        pick("open"),
        pick("high"),
        pick("low"),
        pick("close"),
        pick("volume"),
        F.col("date").alias("raw_date"),
    )

    ts_utc = (F.col("ts") / F.lit(1000)).cast(TimestampType())
    ts_berlin = F.from_utc_timestamp(ts_utc, "Europe/Berlin")
    df = df.withColumn("day_berlin", F.to_date(ts_berlin))

    return df.filter(F.col("day_berlin") == F.lit(day))


def _aggregate_daily(df: DataFrame) -> DataFrame:
    ts_utc = (F.col("ts") / F.lit(1000)).cast(TimestampType())
    ts_berlin = F.from_utc_timestamp(ts_utc, "Europe/Berlin")
    return (
        df.withColumn("ts_berlin", ts_berlin)
        .groupBy("symbol", "day_berlin")
        .agg(
            F.min_by(F.col("open"), F.col("ts_berlin")).alias("open"),
            F.max(F.col("high")).alias("high"),
            F.min(F.col("low")).alias("low"),
            F.max_by(F.col("close"), F.col("ts_berlin")).alias("close"),
            F.sum(F.col("volume")).alias("volume"),
            F.count(F.lit(1)).alias("rows"),
        )
        .withColumnRenamed("day_berlin", "day")
        .select(
            "day", "symbol", "open", "high", "low", "close", "volume", "rows"
        )
    )


def _write_gold_for_day(
    df: DataFrame, gold_root: str, coalesce_n: int
) -> None:
    (
        df.coalesce(coalesce_n)
        .write.format("parquet")
        .mode("overwrite")
        .option("compression", "snappy")
        .partitionBy("day", "symbol")
        .save(gold_root)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate intraday OHLCV into daily gold."
    )
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--all-topics",
        action="store_true",
        help="aggregate across all topics (topic=*)",
    )
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    spark = _build_spark(_get_env_str("APP_NAME", "s3_daily_aggregate"))
    bucket = _get_env_str("S3_BUCKET", required=True)
    raw_prefix = _get_env_str("S3_PREFIX", "raw") or "raw"
    gold_prefix = _get_env_str("S3_PREFIX_GOLD", "gold") or "gold"
    day = args.date or _get_env_str("DATE", required=True)
    topic = None if args.all_topics else _get_env_str("KAFKA_TOPIC", None)
    coalesce_n = int(_get_env_str("COALESCE", "1") or "1")

    raw_base = f"s3a://{bucket}/{raw_prefix}"
    gold_root = f"s3a://{bucket}/{gold_prefix}/ohlcv_daily"

    df_raw = _read_raw_for_day(spark, raw_base, day, topic)

    needed = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    missing = needed.difference(df_raw.columns)
    if missing:
        raise SystemExit(f"Parsed dataset missing columns: {sorted(missing)}")

    df_daily = _aggregate_daily(df_raw)
    _write_gold_for_day(df_daily, gold_root, coalesce_n)

    print(
        f"[ok] gold written: {gold_root} (day={day}, rows={df_daily.count()})"
    )
    spark.stop()


if __name__ == "__main__":
    main()
