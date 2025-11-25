from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql import SparkSession
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
            print(f"[manual_macro] Loaded .env from {env_path}")
            return
    print("[manual_macro] WARNING: .env not found anywhere above")


_load_project_env()

def _build_spark() -> SparkSession:
    endpoint = f"s3.{DEFAULT_AWS_REGION}.amazonaws.com"

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    builder = (
        SparkSession.builder
        .appName(DEFAULT_APP_NAME + "-manual-macro")
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
    )

    if access_key and secret_key:
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.access.key", access_key)
            .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        )
    else:
        print("[manual_macro] WARNING: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set in env")

    spark = builder.getOrCreate()
    print(f"[manual_macro] Spark S3A endpoint = {endpoint}")
    return spark


def main():
    spark = _build_spark()

    events_path = (CURRENT_FILE.parents[3] / "data" / "manual_macro_events.jsonl").as_posix()
    print(f"[manual_macro] Reading events from {events_path}")

    df = spark.read.json(events_path)

    df = df.withColumn("event_date", F.to_date("event_date"))

    df = (
        df
        .withColumn("event_year", F.year("event_date"))
        .withColumn("event_month", F.month("event_date"))
        .withColumn(
            "event_id",
            F.concat_ws(
                "-",
                F.col("event_type"),
                F.date_format("event_date", "yyyy-MM-dd"),
            )
        )
    )

    base_path = f"s3a://{DEFAULT_S3_BUCKET}"
    prefix = (DEFAULT_S3_PREFIX or "").strip("/")
    if prefix:
        base_path += f"/{prefix}"
    base_path += "/events"

    (
        df.write
        .mode("overwrite")
        .partitionBy("source", "event_type", "event_year", "event_month")
        .json(base_path)
    )

    total = df.count()
    print(f"[manual_macro] Written {total} events to {base_path}")

    spark.stop()


if __name__ == "__main__":
    main()
