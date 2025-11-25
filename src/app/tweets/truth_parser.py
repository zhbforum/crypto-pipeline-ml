import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType,
    DoubleType, IntegerType, ArrayType,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.parser_settings.constants import (
    DEFAULT_AWS_REGION,
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_PREFIX,
    ECONOMIC_CRYPTO_KEYWORDS,
    ECONOMIC_MACRO_KEYWORDS,
)

TRUMP_HANDLE = "@realDonaldTrump"
START_DATE = datetime(2025, 1, 20, tzinfo=timezone.utc)

NEUTRAL_LOWER = -0.3
NEUTRAL_UPPER = 0.3

analyzer = SentimentIntensityAnalyzer()

CURRENT_FILE = Path(__file__).resolve()
LOCAL_JSONL_PATH = CURRENT_FILE.parents[3] / "data" / "trump_truthsocial_since_2025-01-20.jsonl"

EVENT_SCHEMA = StructType([
    StructField("event_type",            StringType(),  False),
    StructField("platform",              StringType(),  False),
    StructField("user_handle",           StringType(),  False),
    StructField("truth_id",              StringType(),  True),
    StructField("truth_url",             StringType(),  True),
    StructField("created_at",            StringType(),  False),
    StructField("event_date",            StringType(),  False),
    StructField("raw_text",              StringType(),  False),
    StructField("language",              StringType(),  False),
    StructField("is_economic",           BooleanType(), False),
    StructField("economic_categories",   ArrayType(StringType()), False),
    StructField("sentiment_score",       DoubleType(),  False),
    StructField("sentiment_label",       StringType(),  False),
    StructField("sentiment_index",       IntegerType(), False),
    StructField("market_bias",           StringType(),  False),
    StructField("ingested_at",           StringType(),  False),
])


def init_env() -> None:
    load_dotenv()


def init_spark() -> SparkSession:
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)

    if not aws_key or not aws_secret:
        raise RuntimeError("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set")

    return (
        SparkSession.builder
        .appName("trump_truths_sentiment")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_region}.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.access.key", aws_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret)
        .getOrCreate()
    )


def parse_date_utc(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_economic_post(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ECONOMIC_CRYPTO_KEYWORDS) or any(k in t for k in ECONOMIC_MACRO_KEYWORDS)


def economic_categories(text: str) -> List[str]:
    t = text.lower()
    cats: List[str] = []
    if any(k in t for k in ECONOMIC_CRYPTO_KEYWORDS):
        cats.append("crypto")
    if any(k in t for k in ECONOMIC_MACRO_KEYWORDS):
        cats.append("macro")
    return cats or ["other_econ"]


def compute_sentiment(text: str) -> Dict[str, Any]:
    score = float(analyzer.polarity_scores(text)["compound"])
    if score >= NEUTRAL_UPPER:
        return {"sentiment_score": score, "sentiment_label": "positive", "sentiment_index": 1}
    if score <= NEUTRAL_LOWER:
        return {"sentiment_score": score, "sentiment_label": "negative", "sentiment_index": -1}
    return {"sentiment_score": score, "sentiment_label": "neutral", "sentiment_index": 0}


def compute_market_bias(sentiment_index: int) -> str:
    if sentiment_index > 0:
        return "bullish"
    if sentiment_index < 0:
        return "bearish"
    return "neutral"


def transform_record(date_str: str, text: str) -> Dict[str, Any]:
    created_at = parse_date_utc(date_str)
    sent = compute_sentiment(text)
    return {
        "event_type": "TRUMP_TRUTH_ECON",
        "platform": "truthsocial",
        "user_handle": TRUMP_HANDLE.lstrip("@"),
        "truth_id": None,
        "truth_url": None,
        "created_at": created_at.isoformat(),
        "event_date": created_at.date().isoformat(),
        "raw_text": text,
        "language": "en",
        "is_economic": True,
        "economic_categories": economic_categories(text),
        "sentiment_score": sent["sentiment_score"],
        "sentiment_label": sent["sentiment_label"],
        "sentiment_index": sent["sentiment_index"],
        "market_bias": compute_market_bias(sent["sentiment_index"]),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def iter_jsonl_records(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            date_str = obj.get("date")
            text = obj.get("text")
            if date_str and text:
                yield date_str, text


def build_events_and_stats(records: Iterable[Tuple[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "total": 0,
        "after_date": 0,
        "has_tariff": 0,
        "econ_total": 0,
        "tariff_and_econ": 0,
    }
    events: List[Dict[str, Any]] = []

    for date_str, text in records:
        stats["total"] += 1

        t = text.lower()
        has_tariff = ("tariff" in t) or ("tariffs" in t)
        if has_tariff:
            stats["has_tariff"] += 1

        try:
            created_at = parse_date_utc(date_str)
        except Exception:
            continue

        if created_at < START_DATE:
            continue

        stats["after_date"] += 1

        econ = is_economic_post(text)
        if econ:
            stats["econ_total"] += 1
            if has_tariff:
                stats["tariff_and_econ"] += 1
            try:
                events.append(transform_record(date_str, text))
            except Exception:
                continue

    return events, stats


def build_output_path() -> str:
    prefix = (DEFAULT_S3_PREFIX or "").strip("/")
    base = f"s3a://{DEFAULT_S3_BUCKET}"
    if prefix:
        base += f"/{prefix}"
    return base + "/events/source=truthsocial/user=realDonaldTrump"


def write_events_to_s3(spark: SparkSession, events: List[Dict[str, Any]]) -> str:
    df = spark.createDataFrame(events, schema=EVENT_SCHEMA)
    output_path = build_output_path()

    (df.write
       .mode("append")
       .partitionBy("event_date")
       .format("json")
       .save(output_path))

    return output_path


def main() -> None:
    init_env()

    if not LOCAL_JSONL_PATH.exists():
        raise FileNotFoundError(f"Local jsonl file not found: {LOCAL_JSONL_PATH}")

    spark = init_spark()

    events, stats = build_events_and_stats(iter_jsonl_records(LOCAL_JSONL_PATH))
    print(f"[DEBUG] {stats}")

    if not events:
        print("No economic posts found, nothing to write.")
        return

    output_path = write_events_to_s3(spark, events)
    print(f"[Spark] Written {len(events)} events to {output_path} partitioned by event_date")


if __name__ == "__main__":
    main()
