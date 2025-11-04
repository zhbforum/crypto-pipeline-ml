from __future__ import annotations
import argparse
import json
import logging
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import pytz
from confluent_kafka import Consumer, TopicPartition, KafkaException
from dotenv import load_dotenv
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, date_format

BERLIN = pytz.timezone("Europe/Berlin")
UTC = pytz.utc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kafka_daily_to_s3")


@dataclass(frozen=True)
class KafkaSecurity:
    username: str
    password: str
    protocol: str = "SASL_SSL"
    mechanism: str = "PLAIN"
    ssl_algorithm: str = "https"
    request_timeout_ms: str = "20000"
    session_timeout_ms: str = "45000"


@dataclass(frozen=True)
class KafkaConn:
    bootstrap: str
    topic: str
    group_id: str = "offset_lookup"


@dataclass(frozen=True)
class JobCfg:
    s3_bucket: str
    s3_prefix: str
    app_name: str
    write_format: str
    coalesce_parts: int
    aws_region: Optional[str]


@dataclass(frozen=True)
class DayWindow:
    ymd: str
    start_ms: int
    end_ms: int


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
        raise SystemExit(f"winutils.exe not found: {exe_path}. Download it and place it in {bin_path}.")
    if bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{bin_path};{os.environ['PATH']}"


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


def get_kafka_consumer_conf(conn: KafkaConn, sec: KafkaSecurity) -> Dict[str, object]:
    return {
        "bootstrap.servers": conn.bootstrap,
        "group.id": conn.group_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "security.protocol": sec.protocol,
        "sasl.mechanism": sec.mechanism,
        "sasl.username": sec.username,
        "sasl.password": sec.password,
    }


def _resolve_partition_bounds(consumer: Consumer, topic: str, p: int, s_match, e_match) -> Tuple[int, int]:
    low, high = consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=10.0)
    s_off = s_match.offset if (s_match and s_match.offset is not None and s_match.offset >= 0) else low
    e_off_raw = e_match.offset if (e_match and e_match.offset is not None and e_match.offset >= 0) else high
    return s_off, max(e_off_raw, s_off)


def compute_offsets_by_time(conn: KafkaConn, sec: KafkaSecurity, start_ms: int, end_ms: int) -> Tuple[str, str]:
    consumer = Consumer(get_kafka_consumer_conf(conn, sec))
    try:
        topic_md = consumer.list_topics(topic=conn.topic, timeout=10.0).topics.get(conn.topic)
        if topic_md is None:
            raise RuntimeError(f"Topic '{conn.topic}' not found in Kafka.")
        partitions = sorted(topic_md.partitions.keys())

        def offsets_for(ts_ms: int):
            return consumer.offsets_for_times([TopicPartition(conn.topic, p, ts_ms) for p in partitions], timeout=10.0)

        start_offsets = offsets_for(start_ms)
        end_offsets = offsets_for(end_ms)

        def bounds(p: int) -> Tuple[int, int]:
            s = next((x for x in start_offsets if x.partition == p), None)
            e = next((x for x in end_offsets if x.partition == p), None)
            return _resolve_partition_bounds(consumer, conn.topic, p, s, e)

        pairs = {str(p): bounds(p) for p in partitions}  
        start_dict = {k: v[0] for k, v in pairs.items()}
        end_dict = {k: v[1] for k, v in pairs.items()}

        return (
            json.dumps({conn.topic: start_dict}, separators=(",", ":")),
            json.dumps({conn.topic: end_dict}, separators=(",", ":")),
        )
    except KafkaException as e:
        raise RuntimeError(f"Kafka error: {e}") from e
    finally:
        consumer.close()



def get_kafka_options(conn: KafkaConn, sec: KafkaSecurity, *, start_json: str, end_json: str) -> Dict[str, str]:
    return {
        "kafka.bootstrap.servers": conn.bootstrap,
        "subscribe": conn.topic,
        "startingOffsets": start_json,
        "endingOffsets": end_json,
        "failOnDataLoss": "false",
        "kafka.security.protocol": sec.protocol,
        "kafka.sasl.mechanism": sec.mechanism,
        "kafka.sasl.jaas.config": (
            "org.apache.kafka.common.security.plain.PlainLoginModule required "
            f"username='{sec.username}' password='{sec.password}';"
        ),
        "kafka.ssl.endpoint.identification.algorithm": sec.ssl_algorithm,
        "kafka.request.timeout.ms": sec.request_timeout_ms,
        "kafka.session.timeout.ms": sec.session_timeout_ms,
    }


def build_writer_and_save(df: DataFrame, *, write_format: str, target_path: str, coalesce_parts: int) -> None:
    out_df = df.coalesce(coalesce_parts) if coalesce_parts > 0 else df
    fmt = write_format.strip().lower()
    if fmt not in ("parquet", "json"):
        raise SystemExit("WRITE_FORMAT must be 'parquet' or 'json'.")
    out_df.write.mode("append").format(fmt).partitionBy("date").save(target_path)


def resolve_day(cli_date: Optional[str]) -> DayWindow:
    start_ms, end_ms, ymd = get_day_bounds_utc_ms(cli_date or get_env_opt_str("DATE"))
    log.info("Day: %s (Europe/Berlin) | UTC ms: %s .. %s", ymd, start_ms, end_ms)
    return DayWindow(ymd=ymd, start_ms=start_ms, end_ms=end_ms)


def load_cfg() -> JobCfg:
    return JobCfg(
        s3_bucket=get_env_str("S3_BUCKET", required=True),
        s3_prefix=get_env_str("S3_PREFIX", default="raw"),
        app_name=get_env_str("APP_NAME", default="kafka_daily_to_s3"),
        write_format=get_env_str("WRITE_FORMAT", default="parquet"),
        coalesce_parts=get_env_int("COALESCE", default=0),
        aws_region=get_env_opt_str("AWS_DEFAULT_REGION"),
    )


def log_s3a_timeouts(spark: SparkSession) -> None:
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


def kafka_to_df(spark: SparkSession, kafka_opts: Dict[str, str], ymd: str) -> DataFrame:
    df = spark.read.format("kafka").options(**kafka_opts).load()
    return (
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


def run_job(*, conn: KafkaConn, sec: KafkaSecurity, cfg: JobCfg, day: DayWindow) -> None:
    start_json, end_json = compute_offsets_by_time(conn, sec, day.start_ms, day.end_ms)
    log.info("startingOffsets: %s", start_json)
    log.info("endingOffsets  : %s", end_json)
    spark = build_spark(app_name=cfg.app_name, aws_region=cfg.aws_region)
    try:
        kafka_opts = get_kafka_options(conn, sec, start_json=start_json, end_json=end_json)
        out_df = kafka_to_df(spark, kafka_opts, day.ymd)
        target_path = f"s3a://{cfg.s3_bucket}/{cfg.s3_prefix}/topic={conn.topic}/"
        log_s3a_timeouts(spark)
        build_writer_and_save(
            out_df,
            write_format=cfg.write_format,
            target_path=target_path,
            coalesce_parts=cfg.coalesce_parts,
        )
        log.info("Written to %s for date=%s", target_path, day.ymd)
    finally:
        spark.stop()
        log.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Kafka → S3 batch job")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--date", help="Day in YYYY-MM-DD (Europe/Berlin). If omitted, use DATE from .env or today.")
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file)

    ensure_winutils()

    conn = KafkaConn(
        bootstrap=get_env_str("KAFKA_BOOTSTRAP", required=True),
        topic=get_env_str("KAFKA_TOPIC", required=True),
    )
    sec = KafkaSecurity(
        username=get_env_str("CONFLUENT_API_KEY", required=True),
        password=get_env_str("CONFLUENT_API_SECRET", required=True),
    )
    cfg = load_cfg()
    day = resolve_day(args.date)
    run_job(conn=conn, sec=sec, cfg=cfg, day=day)


if __name__ == "__main__":
    main()
