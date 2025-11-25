import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType,
    DoubleType,
    IntegerType,
    ArrayType,
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

S3_BUCKET = DEFAULT_S3_BUCKET
S3_PREFIX = f"{DEFAULT_S3_PREFIX}/events/source=truthsocial/user=realDonaldTrump"

NEUTRAL_LOWER = -0.3
NEUTRAL_UPPER = 0.3

analyzer = SentimentIntensityAnalyzer()

CURRENT_FILE = Path(__file__).resolve()
LOCAL_JSONL_PATH = CURRENT_FILE.parents[3] / "data" / "trump_truthsocial_since_2025-01-20.jsonl"


print(f"[DEBUG] Using JSONL file: {LOCAL_JSONL_PATH}")

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

    spark = (
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

    return spark



def is_economic_post(text: str) -> bool:
    t = text.lower()

    if any(k in t for k in ECONOMIC_CRYPTO_KEYWORDS):
        return True

    if any(k in t for k in ECONOMIC_MACRO_KEYWORDS):
        return True

    return False


def economic_categories(text: str) -> List[str]:
    t = text.lower()
    cats: List[str] = []

    if any(k in t for k in ECONOMIC_CRYPTO_KEYWORDS):
        cats.append("crypto")

    if any(k in t for k in ECONOMIC_MACRO_KEYWORDS):
        cats.append("macro")

    if cats:
        return cats

    return ["other_econ"]


def compute_sentiment(text: str) -> Dict[str, Any]:
    scores = analyzer.polarity_scores(text)
    score = float(scores["compound"])

    if score >= NEUTRAL_UPPER:
        label = "positive"
        index = 1
    elif score <= NEUTRAL_LOWER:
        label = "negative"
        index = -1
    else:
        label = "neutral"
        index = 0

    return {
        "sentiment_score": score,
        "sentiment_label": label,
        "sentiment_index": index,
    }


def compute_market_bias(sentiment_index: int) -> str:
    if sentiment_index > 0:
        return "bullish"
    if sentiment_index < 0:
        return "bearish"
    return "neutral"


def parse_date_utc(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def transform_record(date_str: str, text: str) -> Dict[str, Any]:
    created_at = parse_date_utc(date_str)
    event_date = created_at.date().isoformat()

    if not is_economic_post(text):
        raise RuntimeError("Non-economic post passed to transform_record")

    sent = compute_sentiment(text)
    market_bias = compute_market_bias(sent["sentiment_index"])
    econ_cats = economic_categories(text)

    event = {
        "event_type": "TRUMP_TRUTH_ECON",
        "platform": "truthsocial",
        "user_handle": TRUMP_HANDLE.lstrip("@"),
        "truth_id": None,
        "truth_url": None,
        "created_at": created_at.isoformat(),
        "event_date": event_date,
        "raw_text": text,
        "language": "en",
        "is_economic": True,
        "economic_categories": econ_cats,
        "sentiment_score": sent["sentiment_score"],
        "sentiment_label": sent["sentiment_label"],
        "sentiment_index": sent["sentiment_index"],
        "market_bias": market_bias,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    return event


def main():
    init_env()
    spark = init_spark()

    if not LOCAL_JSONL_PATH.exists():
        raise FileNotFoundError(f"Local jsonl file not found: {LOCAL_JSONL_PATH}")

    print(f"[DEBUG] Using JSONL file: {LOCAL_JSONL_PATH}")

    events: List[Dict[str, Any]] = []

    total = 0
    after_date = 0
    has_tariff = 0
    tariff_and_econ = 0
    econ_total = 0

    with LOCAL_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse JSON line: {e}")
                continue

            if total == 1:
                print(f"[DEBUG] first obj keys = {list(obj.keys())[:10]}")

            date_str = obj.get("date")
            text = obj.get("text")

            if not date_str or not text:
                continue

            t = text.lower()
            if "tariff" in t or "tariffs" in t:
                has_tariff += 1

            try:
                created_at = parse_date_utc(date_str)
                if created_at < START_DATE:
                    continue
                after_date += 1

                if is_economic_post(text):
                    econ_total += 1
                    if "tariff" in t or "tariffs" in t:
                        tariff_and_econ += 1

                if not is_economic_post(text):
                    continue

                event = transform_record(date_str, text)
                events.append(event)
            except Exception as e:
                print(f"[WARN] Failed to transform record: {e}")
                continue

    print(
        f"[DEBUG] total={total}, after_date={after_date}, "
        f"has_tariff={has_tariff}, econ_total={econ_total}, "
        f"tariff_and_econ={tariff_and_econ}"
    )

    if not events:
        print("No economic posts found, nothing to write.")
        return

    df = spark.createDataFrame(events, schema=EVENT_SCHEMA)


    output_path = f"s3a://{S3_BUCKET}/{S3_PREFIX}"
    (
        df.write
        .mode("append")
        .partitionBy("event_date")
        .format("json")
        .save(output_path)
    )

    print(f"[Spark] Written {df.count()} events to {output_path} partitioned by event_date")


if __name__ == "__main__":
    main()