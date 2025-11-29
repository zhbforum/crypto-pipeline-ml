from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import torch
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.parser_settings.constants import (
    BATCH_WRITE_SIZE,
    DEFAULT_AWS_REGION,
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_PREFIX,
    ECONOMIC_CRYPTO_KEYWORDS,
    ECONOMIC_MACRO_KEYWORDS,
    FINBERT_BATCH_SIZE,
    FINBERT_MAX_LEN,
    FINBERT_REL_DIR,
    FINBERT_STRIDE,
    LOW_INFO_MIN_ALPHA_CHARS,
    LOW_INFO_MIN_WORDS,
    S3_EVENTS_PATH,
    SPARK_APP_NAME,
    SPARK_WRITE_FORMAT,
    SPARK_WRITE_MODE,
    START_DATE_UTC,
    TRUTH_EVENT_TYPE,
    TRUTH_PLATFORM,
    TRUTH_USER_HANDLE,
    TRUTH_JSONL_REL_PATH,
    NEUTRAL_LOWER,
    NEUTRAL_UPPER,
)

logger = logging.getLogger(__name__)

_URL_ONLY = re.compile(
    r"""(?ix)
    ^(?:https?://)?
    (?:www\.)?
    [a-z0-9-]+(\.[a-z0-9-]+)+
    (?:/.*)?$
"""
)

_JUNK = re.compile(r"[@#]\w+|[^\w\s]", re.UNICODE)

EVENT_SCHEMA = StructType(
    [
        StructField("event_type", StringType(), False),
        StructField("platform", StringType(), False),
        StructField("user_handle", StringType(), False),
        StructField("truth_id", StringType(), True),
        StructField("truth_url", StringType(), True),
        StructField("created_at", StringType(), False),
        StructField("event_date", StringType(), False),
        StructField("raw_text", StringType(), False),
        StructField("language", StringType(), False),
        StructField("is_economic", BooleanType(), False),
        StructField("economic_categories", ArrayType(StringType()), False),
        StructField("sentiment_score", DoubleType(), False),
        StructField("sentiment_label", StringType(), False),
        StructField("sentiment_index", IntegerType(), False),
        StructField("market_bias", StringType(), False),
        StructField("ingested_at", StringType(), False),
    ]
)


@dataclass(frozen=True)
class AppConfig:
    jsonl_path: Path
    finbert_dir: Path
    s3_bucket: str
    s3_prefix: str
    aws_region: str
    batch_write_size: int


def load_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[3]
    jsonl_path = project_root / TRUTH_JSONL_REL_PATH
    finbert_dir = project_root / FINBERT_REL_DIR
    aws_region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    s3_bucket = os.getenv("S3_BUCKET", DEFAULT_S3_BUCKET)
    s3_prefix = os.getenv("S3_PREFIX", DEFAULT_S3_PREFIX)
    return AppConfig(
        jsonl_path=jsonl_path,
        finbert_dir=finbert_dir,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        aws_region=aws_region,
        batch_write_size=int(os.getenv("BATCH_WRITE_SIZE", str(BATCH_WRITE_SIZE))),
    )


def score_to_label(score: float, low: float = NEUTRAL_LOWER, up: float = NEUTRAL_UPPER) -> Tuple[str, int]:
    if score <= low:
        return "negative", -1
    if score >= up:
        return "positive", 1
    return "neutral", 0


class FinBertSentiment:
    def __init__(self, model_dir: Path) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(f"FinBERT dir not found: {model_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.model.to(self.device)
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=FINBERT_MAX_LEN,
            stride=FINBERT_STRIDE,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        probs_all: List[torch.Tensor] = []
        for i in range(0, input_ids.size(0), FINBERT_BATCH_SIZE):
            ids = input_ids[i : i + FINBERT_BATCH_SIZE].to(self.device)
            mask = attention_mask[i : i + FINBERT_BATCH_SIZE].to(self.device)
            logits = self.model(input_ids=ids, attention_mask=mask).logits
            probs_all.append(torch.softmax(logits, dim=-1).cpu())

        probs_mean = torch.cat(probs_all, dim=0).mean(dim=0)

        probs_dict: Dict[str, float] = {
            self.id2label.get(idx, str(idx)): float(p)
            for idx, p in enumerate(probs_mean.tolist())
        }

        pos = float(probs_dict.get("positive", 0.0))
        neg = float(probs_dict.get("negative", 0.0))

        denom = pos + neg
        score = float((pos - neg) / denom) if denom > 0.0 else 0.0

        label, sidx = score_to_label(score)

        return {
            "sentiment_score": score,
            "sentiment_label": label,
            "sentiment_index": sidx,
        }


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


def compute_market_bias(sentiment_index: int) -> str:
    return "bullish" if sentiment_index > 0 else "bearish" if sentiment_index < 0 else "neutral"


def is_low_info_text(text: str) -> bool:
    if not text:
        return True
    raw = text.strip()
    if not raw:
        return True
    collapsed = re.sub(r"\s+", "", raw)
    if _URL_ONLY.match(collapsed):
        return True
    t = raw
    t = re.sub(r"(?i)\bhttps?://\s*", " ", t)
    t = re.sub(r"(?i)\bwww\.\s*", " ", t)
    t = re.sub(r"(?i)\b[a-z0-9-]+(\.[a-z0-9-]+)+\S*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    cleaned = _JUNK.sub(" ", t)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    alpha_chars = sum(ch.isalpha() for ch in cleaned)
    return (len(words) < LOW_INFO_MIN_WORDS) or (alpha_chars < LOW_INFO_MIN_ALPHA_CHARS)


def iter_jsonl_records(path: Path) -> Iterator[Tuple[str, str]]:
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
            if date_str and isinstance(text, str) and text.strip():
                yield date_str, text


def build_output_path(bucket: str, prefix: str) -> str:
    p = (prefix or "").strip("/")
    base = f"s3a://{bucket}"
    if p:
        base += f"/{p}"
    return base + f"/{S3_EVENTS_PATH}"


def init_spark(aws_region: str) -> SparkSession:
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_token = os.getenv("AWS_SESSION_TOKEN")
    if not aws_key or not aws_secret:
        raise RuntimeError("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set")

    b = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.ip", "127.0.0.1")
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .config("spark.executor.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_region}.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.access.key", aws_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret)
    )
    if aws_token:
        b = b.config("spark.hadoop.fs.s3a.session.token", aws_token)

    return b.getOrCreate()


def transform_record(created_at: datetime, text: str, finbert: FinBertSentiment) -> Dict[str, Any]:
    sent = finbert.predict(text)
    return {
        "event_type": TRUTH_EVENT_TYPE,
        "platform": TRUTH_PLATFORM,
        "user_handle": TRUTH_USER_HANDLE,
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


def iter_economic_events(
    records: Iterable[Tuple[str, str]],
    finbert: FinBertSentiment,
) -> Tuple[Iterator[Dict[str, Any]], Dict[str, int]]:
    stats = {"written_events": 0}

    def _gen() -> Iterator[Dict[str, Any]]:
        for date_str, text in records:
            try:
                created_at = parse_date_utc(date_str)
            except Exception:
                continue
            if created_at < START_DATE_UTC:
                continue
            if is_low_info_text(text):
                continue
            if not is_economic_post(text):
                continue
            try:
                ev = transform_record(created_at, text, finbert)
            except Exception:
                continue
            stats["written_events"] += 1
            yield ev

    return _gen(), stats


def write_events_to_s3_in_batches(
    spark: SparkSession,
    events: Iterator[Dict[str, Any]],
    output_path: str,
    batch_size: int,
) -> int:
    buf: List[Dict[str, Any]] = []
    total_written = 0

    def flush() -> int:
        nonlocal buf
        if not buf:
            return 0
        df = spark.createDataFrame(buf, schema=EVENT_SCHEMA)
        (
            df.write.mode(SPARK_WRITE_MODE)
            .partitionBy("event_date")
            .format(SPARK_WRITE_FORMAT)
            .save(output_path)
        )
        n = len(buf)
        buf = []
        return n

    for ev in events:
        buf.append(ev)
        if len(buf) >= batch_size:
            total_written += flush()

    total_written += flush()
    return total_written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    load_dotenv()

    cfg = load_config()
    if not cfg.jsonl_path.exists():
        raise FileNotFoundError(f"Local jsonl file not found: {cfg.jsonl_path}")
    if not cfg.finbert_dir.exists():
        raise FileNotFoundError(f"FinBERT dir not found: {cfg.finbert_dir}")

    finbert = FinBertSentiment(cfg.finbert_dir)
    spark = init_spark(cfg.aws_region)
    output_path = build_output_path(cfg.s3_bucket, cfg.s3_prefix)

    events_iter, stats = iter_economic_events(iter_jsonl_records(cfg.jsonl_path), finbert)
    written = write_events_to_s3_in_batches(
        spark=spark,
        events=events_iter,
        output_path=output_path,
        batch_size=cfg.batch_write_size,
    )

    logger.info("written_events=%s", stats["written_events"])
    logger.info("spark_written=%s", written)
    logger.info("output_path=%s", output_path)


if __name__ == "__main__":
    main()
