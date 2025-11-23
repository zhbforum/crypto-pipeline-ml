from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pmdarima import auto_arima
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as F, types as T

from app.lib.logger import get_logger


def _load_project_env() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        env_path = parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
            break


_load_project_env()
logger = get_logger(__name__)


def _get_env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


DAILY_AGG_PATH = os.getenv(
    "DAILY_AGG_PATH",
    "s3a://crypto-pipeline-ml/silver/kline_1m/",
)

DAILY_FORECAST_PATH = os.getenv(
    "DAILY_FORECAST_PATH",
    "s3a://crypto-pipeline-ml/gold/predictions_montly/",
)


def build_spark(app_name: str = "s3-daily-forecast") -> SparkSession:
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
        conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
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

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    logger.info("Spark version: %s", spark.version)
    return spark


FORECAST_SCHEMA = T.StructType(
    [
        T.StructField("symbol", T.StringType(), nullable=False),
        T.StructField("ds", T.DateType(), nullable=False),
        T.StructField("y_hat_close", T.DoubleType(), nullable=False),
    ]
)


def _fit_arima_and_forecast_next(y: pd.Series) -> Optional[float]:
    y_clean = y.astype(float).dropna()
    if len(y_clean) < 10:
        return None
    try:
        model = auto_arima(
            y_clean.to_numpy(),
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        return float(model.predict(n_periods=1)[0])
    except Exception as exc:
        logger.exception("ARIMA fit/predict failed: %s", exc)
        return None


def forecast_next_day(pdf: pd.DataFrame) -> pd.DataFrame:
    if pdf.empty:
        return pd.DataFrame(columns=["symbol", "ds", "y_hat_close"])

    pdf = pdf.sort_values("ds")
    symbol = str(pdf["symbol"].iloc[0])
    y_hat = _fit_arima_and_forecast_next(pdf["close"])

    if y_hat is None:
        return pd.DataFrame(columns=["symbol", "ds", "y_hat_close"])

    next_ds = (pd.to_datetime(pdf["ds"].iloc[-1]) + pd.Timedelta(days=1)).date()

    return pd.DataFrame(
        {
            "symbol": [symbol],
            "ds": [next_ds],
            "y_hat_close": [y_hat],
        }
    )


def run() -> None:
    spark = build_spark()
    logger.info("Reading minute klines from %s", DAILY_AGG_PATH)

    raw_df = (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load(DAILY_AGG_PATH)
    )

    logger.info("Loaded columns: %s", raw_df.columns)

    required_cols = {"date", "symbol", "close"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing}, got: {raw_df.columns}")

    df = raw_df.select("date", "symbol", "close").withColumn(
        "ds", F.col("date").cast("date")
    )

    daily_df = (
        df.dropna(subset=["symbol", "ds", "close"])
        .groupBy("symbol", "ds")
        .agg(F.last("close", ignorenulls=True).alias("close"))
    )

    logger.info("Running ARIMA forecasts via pandas UDF")

    forecast_df = (
        daily_df.groupBy("symbol")
        .applyInPandas(forecast_next_day, schema=FORECAST_SCHEMA)
        .withColumn("created_at", F.current_timestamp())
    )

    logger.info("Writing forecasts to %s", DAILY_FORECAST_PATH)
    (
        forecast_df.repartition("ds", "symbol")
        .write.mode("append")
        .partitionBy("ds", "symbol")
        .parquet(DAILY_FORECAST_PATH)
    )

    logger.info("Forecast job finished successfully.")


if __name__ == "__main__":
    run()
