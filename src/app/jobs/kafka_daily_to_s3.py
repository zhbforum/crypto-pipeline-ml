from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import pytz
from confluent_kafka import Consumer, TopicPartition, KafkaException
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format


BERLIN = pytz.timezone("Europe/Berlin")
UTC = pytz.utc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kafka_daily_to_s3")


def get_env_str(key: str, *, required: bool = False, default: Optional[str] = None) -> str:
    val = os.getenv(key, default)
    if (val is None or val == "") and required:
        raise SystemExit(f"Missing required env variable: {key}")
    return "" if val is None else val


def get_env_opt_str(key: str) -> Optional[str]:
    val = os.getenv(key)
    return None if (val is None or val == "") else val


def get_env_int(key: str, *, default: int = 0) -> int:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError as exc:
        raise SystemExit(f"Env {key} must be int, got '{val}'") from exc


def ensure_winutils() -> None:
    if platform.system() != "Windows":
        return

    hadoop_home = os.getenv("HADOOP_HOME", r"C:\hadoop")
    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ["hadoop.home.dir"] = hadoop_home

    bin_path = os.path.join(hadoop_home, "bin")
    exe_path = os.path.join(bin_path, "winutils.exe")
    if not os.path.exists(exe_path):
        raise SystemExit(
            f"winutils.exe not found: {exe_path}. "
            f"Download it for Hadoop 3.3.x and place it in {bin_path}."
        )

    if bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{bin_path};{os.environ['PATH']}"


def get_day_bounds_utc_ms(day_str: Optional[str]) -> Tuple[int, int, str]:
    if day_str:
        local_day = BERLIN.localize(datetime.strptime(day_str, "%Y-%m-%d"))
    else:
        now_local = datetime.now(BERLIN)
        local_day = BERLIN.localize(datetime(now_local.year, now_local.month, now_local.day))

    start_local = local_day.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)

    start_ms = int(start_local.astimezone(UTC).timestamp() * 1000)
    end_ms = int(end_local.astimezone(UTC).timestamp() * 1000)
    return start_ms, end_ms, start_local.strftime("%Y-%m-%d")


def compute_offsets_by_time(
    bootstrap: str,
    security_conf: Dict[str, str],
    topic: str,
    start_ms: int,
    end_ms: int,
    group_id: str = "offset_lookup",
) -> Tuple[str, str]:
    consumer_conf: Dict[str, object] = {
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "security.protocol": security_conf.get("security.protocol", "SASL_SSL"),
        "sasl.mechanism": security_conf.get("sasl.mechanism", "PLAIN"),
        "sasl.username": security_conf["sasl.username"],
        "sasl.password": security_conf["sasl.password"],
    }

    consumer = Consumer(consumer_conf)
    try:
        meta = consumer.list_topics(topic=topic, timeout=10.0)
        if topic not in meta.topics:
            raise RuntimeError(f"Topic '{topic}' not found in Kafka.")

        partitions = sorted(meta.topics[topic].partitions.keys())
        start_tps = [TopicPartition(topic, p, start_ms) for p in partitions]
        end_tps = [TopicPartition(topic, p, end_ms) for p in partitions]

        start_offsets = consumer.offsets_for_times(start_tps, timeout=10.0)
        end_offsets = consumer.offsets_for_times(end_tps, timeout=10.0)

        start_dict: Dict[str, int] = {}
        end_dict: Dict[str, int] = {}

        for p in partitions:
            tp = TopicPartition(topic, p)
            low, high = consumer.get_watermark_offsets(tp, timeout=10.0)

            s_match = next(x for x in start_offsets if x.partition == p)
            e_match = next(x for x in end_offsets if x.partition == p)

            s_off = s_match.offset if (s_match and s_match.offset is not None and s_match.offset >= 0) else low
            e_off = e_match.offset if (e_match and e_match.offset is not None and e_match.offset >= 0) else high
            if e_off < s_off:
                e_off = s_off

            start_dict[str(p)] = s_off
            end_dict[str(p)] = e_off

        return (
            json.dumps({topic: start_dict}, separators=(",", ":")),
            json.dumps({topic: end_dict}, separators=(",", ":")),
        )
    except KafkaException as e:
        raise RuntimeError(f"Kafka error: {e}") from e
    finally:
        consumer.close()


def build_spark(app_name: str, aws_region: Optional[str]) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config(
            "spark.jars.packages",
            ",".join(
                [
                    "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1",
                    "org.apache.hadoop:hadoop-aws:3.3.4",
                    "com.amazonaws:aws-java-sdk-bundle:1.12.316",
                ]
            ),
        )
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000")
        .config("spark.hadoop.fs.s3a.socket.timeout", "60000")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )

    jsc = getattr(spark, "_jsc", None)
    if jsc is None:
        raise RuntimeError("Spark _jsc gateway is not available.")
    hconf = jsc.hadoopConfiguration()

    hconf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
    hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hconf.set("fs.s3a.path.style.access", "true")
    if aws_region:
        hconf.set("fs.s3a.endpoint", f"s3.{aws_region}.amazonaws.com")

    return spark


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Kafka → S3 batch job")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument(
        "--date",
        help="Day in YYYY-MM-DD (Europe/Berlin). If omitted, use DATE from .env or today.",
    )
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file)

    bootstrap: str = get_env_str("KAFKA_BOOTSTRAP", required=True)
    topic: str = get_env_str("KAFKA_TOPIC", required=True)
    sasl_username: str = get_env_str("CONFLUENT_API_KEY", required=True)
    sasl_password: str = get_env_str("CONFLUENT_API_SECRET", required=True)

    s3_bucket: str = get_env_str("S3_BUCKET", required=True)
    s3_prefix: str = get_env_str("S3_PREFIX", default="raw")
    app_name: str = get_env_str("APP_NAME", default="kafka_daily_to_s3")
    write_format: str = get_env_str("WRITE_FORMAT", default="parquet")
    coalesce_parts: int = get_env_int("COALESCE", default=0)
    aws_region: Optional[str] = get_env_opt_str("AWS_DEFAULT_REGION")
    date_str: Optional[str] = args.date or get_env_opt_str("DATE")

    ensure_winutils()

    start_ms, end_ms, ymd = get_day_bounds_utc_ms(date_str)
    log.info("Day: %s (Europe/Berlin) | UTC ms: %s .. %s", ymd, start_ms, end_ms)

    security_conf: Dict[str, str] = {
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "PLAIN",
        "sasl.username": sasl_username,
        "sasl.password": sasl_password,
        "ssl.endpoint.identification.algorithm": "https",
        "request.timeout.ms": "20000",
        "session.timeout.ms": "45000",
    }

    start_json, end_json = compute_offsets_by_time(
        bootstrap=bootstrap,
        security_conf=security_conf,
        topic=topic,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    log.info("startingOffsets: %s", start_json)
    log.info("endingOffsets  : %s", end_json)

    spark = build_spark(app_name=app_name, aws_region=aws_region)

    kafka_opts: Dict[str, str] = {
        "kafka.bootstrap.servers": bootstrap,
        "subscribe": topic,
        "startingOffsets": start_json,
        "endingOffsets": end_json,
        "failOnDataLoss": "false",
        "kafka.security.protocol": security_conf["security.protocol"],
        "kafka.sasl.mechanism": security_conf["sasl.mechanism"],
        "kafka.sasl.jaas.config": (
            "org.apache.kafka.common.security.plain.PlainLoginModule required "
            f"username='{sasl_username}' password='{sasl_password}';"
        ),
        "kafka.ssl.endpoint.identification.algorithm": security_conf["ssl.endpoint.identification.algorithm"],
        "kafka.request.timeout.ms": security_conf["request.timeout.ms"],
        "kafka.session.timeout.ms": security_conf["session.timeout.ms"],
    }

    df = spark.read.format("kafka").options(**kafka_opts).load()

    out_df = (
        df.select(
            col("topic"),
            col("partition"),
            col("offset"),
            col("timestamp"),
            col("timestampType"),
            col("key").cast("string").alias("key"),
            col("value").cast("string").alias("value"),
        )
        .where(date_format(col("timestamp"), "yyyy-MM-dd") == ymd)
        .withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd"))
    )

    target_path = f"s3a://{s3_bucket}/{s3_prefix}/topic={topic}/"

    if coalesce_parts > 0:
        out_df = out_df.coalesce(coalesce_parts)

    writer = out_df.write.mode("append").partitionBy("date")
    fmt = write_format.strip().lower()
    if fmt not in ("parquet", "json"):
        raise SystemExit("WRITE_FORMAT must be 'parquet' or 'json'.")
    writer = writer.format(fmt)

    jsc = getattr(spark, "_jsc", None)
    if jsc is None:
        raise RuntimeError("Spark _jsc gateway is not available.")
    hconf = jsc.hadoopConfiguration()
    for k in (
        "fs.s3a.connection.timeout",
        "fs.s3a.connection.request.timeout",
        "fs.s3a.socket.timeout",
        "fs.s3a.connection.establish.timeout",
    ):
        log.info("%s = %s", k, hconf.get(k))

    writer.save(target_path)
    spark.stop()

    log.info("Written to %s for date=%s", target_path, ymd)
    log.info("Done.")


if __name__ == "__main__":
    main()
