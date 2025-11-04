from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from confluent_kafka import Consumer, TopicPartition, KafkaException
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format

from src.app.constants import KAFKA_TOPIC, BERLIN, UTC
from src.app.lib.logger import get_logger

log = get_logger("kafka_daily_to_s3")


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
        raise SystemExit(f"winutils.exe not found: {exe_path}. Download it for Hadoop 3.4.x and place it in {bin_path}.")
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


@dataclass(frozen=True)
class Window:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Security:
    protocol: str
    mechanism: str
    username: str
    password: str
    ssl_endpoint_algo: str
    request_timeout_ms: str
    session_timeout_ms: str


@dataclass(frozen=True)
class AppConfig:
    bootstrap: str
    topic: str
    security: Security
    s3_bucket: str
    s3_prefix: str
    app_name: str
    write_format: str
    coalesce_parts: int
    aws_region: Optional[str]
    ymd: str
    window: Window


def _k_consumer_conf(bootstrap: str, s: Security, group_id: str) -> Dict[str, object]:
    return {
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "security.protocol": s.protocol,
        "sasl.mechanism": s.mechanism,
        "sasl.username": s.username,
        "sasl.password": s.password,
    }


def _offsets_map(consumer: Consumer, topic: str, parts: list[int], ts: int) -> Dict[int, Optional[int]]:
    tps = [TopicPartition(topic, p, ts) for p in parts]
    res = consumer.offsets_for_times(tps, timeout=10.0)
    return {x.partition: (x.offset if (x is not None and x.offset is not None and x.offset >= 0) else None) for x in res}


def _watermarks(consumer: Consumer, topic: str, parts: list[int]) -> Dict[int, Tuple[int, int]]:
    return {p: consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=10.0) for p in parts}


def _as_int(v: Optional[int], fallback: int) -> int:
    return fallback if v is None else v


def compute_offsets_by_time(
    bootstrap: str,
    security: Security,
    topic: str,
    window: Window,
    group_id: str = "offset_lookup",
) -> Tuple[str, str]:
    consumer = Consumer(_k_consumer_conf(bootstrap, security, group_id))
    try:
        meta = consumer.list_topics(topic=topic, timeout=10.0)
        if topic not in meta.topics:
            raise RuntimeError(f"Topic '{topic}' not found in Kafka.")

        parts: list[int] = sorted(meta.topics[topic].partitions.keys())

        start_map = _offsets_map(consumer, topic, parts, window.start_ms)
        end_map = _offsets_map(consumer, topic, parts, window.end_ms)
        watermarks = _watermarks(consumer, topic, parts)

        start_dict: Dict[str, int] = {
            str(p): (_as_int(start_map.get(p), watermarks[p][0]))
            for p in parts
        }

        end_dict: Dict[str, int] = {
            str(p): max(
                _as_int(end_map.get(p), watermarks[p][1]),
                start_dict[str(p)],
            )
            for p in parts
        }

        return (
            json.dumps({topic: start_dict}, separators=(",", ":")),
            json.dumps({topic: end_dict}, separators=(",", ":")),
        )
    except KafkaException as e:
        raise RuntimeError(f"Kafka error: {e}") from e
    finally:
        consumer.close()


def _make_spark(app_name: str) -> SparkSession:
    py = sys.executable
    pkgs_env = os.getenv("SPARK_PACKAGES", "")
    if "software.amazon.awssdk" in pkgs_env:
        log.warning("Detected AWS SDK v2 in SPARK_PACKAGES")
        pkgs_env = ""
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.pyspark.python", py)
        .config("spark.pyspark.driver.python", py)
        .config("spark.executorEnv.PYSPARK_PYTHON", py)
        .config("spark.ui.showConsoleProgress", "true")
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000")
        .config("spark.hadoop.fs.s3a.socket.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000")
    )
    if pkgs_env:
        builder = builder.config("spark.jars.packages", pkgs_env)
    else:
        builder = builder.config(
            "spark.jars.packages",
            ",".join(
                [
                    "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1",
                    "org.apache.hadoop:hadoop-aws:3.4.1",
                    "com.amazonaws:aws-java-sdk-bundle:1.12.774",
                ]
            ),
        )
    return builder.getOrCreate()


def _hconf(spark: SparkSession):
    jsc = getattr(spark, "_jsc", None)
    if jsc is None:
        raise RuntimeError("Spark _jsc gateway is not available.")
    return jsc.hadoopConfiguration()


def _apply_s3a_base(hconf) -> None:
    hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hconf.set("fs.s3a.path.style.access", "true")
    for k, v in (
        ("fs.s3a.connection.timeout", "60000"),
        ("fs.s3a.connection.request.timeout", "60000"),
        ("fs.s3a.socket.timeout", "60000"),
        ("fs.s3a.connection.establish.timeout", "30000"),
    ):
        hconf.set(k, v)


def _apply_s3a_creds(hconf, aws_region: Optional[str]) -> None:
    access = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    token = os.getenv("AWS_SESSION_TOKEN")
    if access and secret:
        hconf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        hconf.set("fs.s3a.access.key", access)
        hconf.set("fs.s3a.secret.key", secret)
        if token:
            hconf.set("fs.s3a.session.token", token)
    else:
        hconf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
    if aws_region:
        hconf.set("fs.s3a.endpoint", f"s3.{aws_region}.amazonaws.com")


def build_spark(app_name: str, aws_region: Optional[str]) -> SparkSession:
    spark = _make_spark(app_name)
    hconf = _hconf(spark)
    _apply_s3a_base(hconf)
    _apply_s3a_creds(hconf, aws_region)
    return spark


def _security_from_env() -> Security:
    return Security(
        protocol="SASL_SSL",
        mechanism="PLAIN",
        username=get_env_str("CONFLUENT_API_KEY", required=True),
        password=get_env_str("CONFLUENT_API_SECRET", required=True),
        ssl_endpoint_algo="https",
        request_timeout_ms="20000",
        session_timeout_ms="45000",
    )


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    bootstrap = get_env_str("KAFKA_BOOTSTRAP", required=True)
    topic = os.getenv("KAFKA_TOPIC", KAFKA_TOPIC)
    s3_bucket = get_env_str("S3_BUCKET", required=True)
    s3_prefix = get_env_str("S3_PREFIX", default="raw")
    app_name = get_env_str("APP_NAME", default="kafka_daily_to_s3")
    write_format = get_env_str("WRITE_FORMAT", default="parquet").strip().lower()
    coalesce_parts = get_env_int("COALESCE", default=0)
    aws_region = get_env_opt_str("AWS_DEFAULT_REGION")
    date_str = args.date or get_env_opt_str("DATE")
    start_ms, end_ms, ymd = get_day_bounds_utc_ms(date_str)
    return AppConfig(
        bootstrap=bootstrap,
        topic=topic,
        security=_security_from_env(),
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        app_name=app_name,
        write_format=write_format,
        coalesce_parts=coalesce_parts,
        aws_region=aws_region,
        ymd=ymd,
        window=Window(start_ms=start_ms, end_ms=end_ms),
    )


def _kafka_reader_options(cfg: AppConfig, start_json: str, end_json: str) -> Dict[str, str]:
    jaas = (
        "org.apache.kafka.common.security.plain.PlainLoginModule required "
        f"username='{cfg.security.username}' password='{cfg.security.password}';"
    )
    return {
        "kafka.bootstrap.servers": cfg.bootstrap,
        "subscribe": cfg.topic,
        "startingOffsets": start_json,
        "endingOffsets": end_json,
        "failOnDataLoss": "false",
        "kafka.security.protocol": cfg.security.protocol,
        "kafka.sasl.mechanism": cfg.security.mechanism,
        "kafka.sasl.jaas.config": jaas,
        "kafka.ssl.endpoint.identification.algorithm": cfg.security.ssl_endpoint_algo,
        "kafka.request.timeout.ms": cfg.security.request_timeout_ms,
        "kafka.session.timeout.ms": cfg.security.session_timeout_ms,
    }


def _write_to_s3(spark: SparkSession, cfg: AppConfig, kafka_opts: Dict[str, str]) -> None:
    df = spark.read.format("kafka").options(**kafka_opts).load()
    out = (
        df.select(
            col("topic"),
            col("partition"),
            col("offset"),
            col("timestamp"),
            col("timestampType"),
            col("key").cast("string").alias("key"),
            col("value").cast("string").alias("value"),
        )
        .where(date_format(col("timestamp"), "yyyy-MM-dd") == cfg.ymd)
        .withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd"))
    )
    if cfg.coalesce_parts > 0:
        out = out.coalesce(cfg.coalesce_parts)
    target = f"s3a://{cfg.s3_bucket}/{cfg.s3_prefix}/topic={cfg.topic}/"
    jsc = getattr(spark, "_jsc", None)
    if jsc is None:
        raise RuntimeError("Spark _jsc gateway is not available.")
    hconf = jsc.hadoopConfiguration()
    for k in (
        "fs.s3a.aws.credentials.provider",
        "fs.s3a.endpoint",
        "fs.s3a.connection.timeout",
        "fs.s3a.connection.request.timeout",
        "fs.s3a.socket.timeout",
        "fs.s3a.connection.establish.timeout",
    ):
        log.info("%s = %s", k, hconf.get(k))
    out.write.mode("append").partitionBy("date").format(cfg.write_format).save(target)
    log.info("Written to %s for date=%s", target, cfg.ymd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Kafka → S3 batch job")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    ensure_winutils()
    cfg = _config_from_args(args)
    log.info("Day: %s (Europe/Berlin) | UTC ms: %s .. %s", cfg.ymd, cfg.window.start_ms, cfg.window.end_ms)
    start_json, end_json = compute_offsets_by_time(cfg.bootstrap, cfg.security, cfg.topic, cfg.window)
    log.info("startingOffsets: %s", start_json)
    log.info("endingOffsets  : %s", end_json)
    spark = build_spark(cfg.app_name, cfg.aws_region)
    try:
        if cfg.write_format not in ("parquet", "json"):
            raise SystemExit("WRITE_FORMAT must be 'parquet' or 'json'.")
        _write_to_s3(spark, cfg, _kafka_reader_options(cfg, start_json, end_json))
        log.info("Done.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
