from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from app.parser_settings.constants import (
    DEFAULT_AWS_REGION,
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_PREFIX,
    DEFAULT_APP_NAME,
)

CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[2]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_project_env() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[sentiment_macro_event] Loaded .env from {env_path}")
            return
    print("[sentiment_macro_event] WARNING: .env not found anywhere above")


_load_project_env()


def _build_spark() -> SparkSession:
    endpoint = f"s3.{DEFAULT_AWS_REGION}.amazonaws.com"

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    builder = (
        SparkSession.builder
        .appName(DEFAULT_APP_NAME + "-sentiment-macro-event")
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
    )

    if access_key and secret_key:
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.access.key", access_key)
            .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        )
    else:
        print(
            "[sentiment_macro_event] WARNING: "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set in env"
        )

    spark = builder.getOrCreate()
    print(f"[sentiment_macro_event] Spark S3A endpoint = {endpoint}")
    return spark


def _ensure_event_type(df: DataFrame) -> DataFrame:
    if "event_type" in df.columns:
        return df

    df = df.withColumn(
        "event_type",
        F.when(
            F.col("event_id").isNotNull(),
            F.regexp_extract(F.col("event_id"), r"^([^-]+)", 1),
        ).otherwise(F.lit("unknown")),
    )
    return df


def _with_event_sentiment(df: DataFrame) -> DataFrame:
    actual = F.col("actual").cast("double")
    previous = F.col("previous").cast("double")
    delta = actual - previous

    sentiment = (
        F.when(actual.isNull() | previous.isNull(), F.lit(0))
        .when(delta > 0, F.lit(-1))
        .when(delta < 0, F.lit(1))
        .otherwise(F.lit(0))
    )

    return (
        df.withColumn("event_delta", delta)
          .withColumn("event_sentiment", sentiment)
    )


def main() -> None:
    spark = _build_spark()

    base_path = f"s3a://{DEFAULT_S3_BUCKET}"
    prefix = (DEFAULT_S3_PREFIX or "").strip("/")
    if prefix:
        base_path += f"/{prefix}"

    events_root = base_path + "/events"
    manual_macro_path = events_root + "/source=manual_macro"

    print(f"[sentiment_macro_event] Reading from: {manual_macro_path}")
    df = spark.read.json(manual_macro_path)

    df = df.withColumn("event_date", F.to_date("event_date"))
    df = _ensure_event_type(df)

    df = (
        df
        .withColumn("event_year", F.year("event_date"))
        .withColumn("event_month", F.month("event_date"))
        .withColumn("source", F.lit("macro_sentiment"))
    )

    df = _with_event_sentiment(df)

    total = df.count()
    print(f"[sentiment_macro_event] Total events to write (after sentiment + dedup): {total}")

    print(f"[sentiment_macro_event] Writing to: {events_root}")
    (
        df.write
        .mode("append")
        .partitionBy("source", "event_type", "event_year", "event_month")
        .json(events_root)
    )

    print(
        "[sentiment_macro_event] Written "
        f"{total} events under source=macro_sentiment to {events_root}"
    )

    spark.stop()


if __name__ == "__main__":
    main()