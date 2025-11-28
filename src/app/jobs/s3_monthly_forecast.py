from __future__ import annotations

from typing import Final
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pmdarima import auto_arima
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as F, types as T

from app.lib.logger import get_logger


def _load_project_env() -> None:
    root = Path.cwd()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        print(f"[WARN] .env not found in {root}, AWS creds may be empty")


_load_project_env()

from app.constants import (  
    DAILY_AGG_PATH,
    DAILY_FORECAST_PATH,
    SPARK_PACKAGES,
    AWS_DEFAULT_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    FORECAST_START_DS,
    FORECAST_END_DS,
)

logger = get_logger(__name__)


def build_spark(app_name: str = "s3-daily-forecast") -> SparkSession:
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError(
            "AWS credentials are empty. "
            "Check .env in project root (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)."
        )

    conf = (
        SparkConf()
        .setAppName(app_name)
        .set("spark.sql.session.timeZone", "UTC")
        .set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    )

    if SPARK_PACKAGES:
        conf = conf.set("spark.jars.packages", SPARK_PACKAGES)

    conf = (
        conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .set(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .set("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)
        .set("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
        .set("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_DEFAULT_REGION}.amazonaws.com")
    )

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    logger.info("Spark version: %s", spark.version)
    return spark


FORECAST_SCHEMA: Final = T.StructType(
    [
        T.StructField("symbol", T.StringType(), nullable=False),
        T.StructField("ds", T.DateType(), nullable=False),
        T.StructField("y_hat_close", T.DoubleType(), nullable=False),
    ]
)


def _fit_arima_and_forecast_next(y: pd.Series) -> float | None:
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


def forecast_for_period(pdf: pd.DataFrame) -> pd.DataFrame:
    if pdf.empty:
        return pd.DataFrame(columns=["symbol", "ds", "y_hat_close"])

    pdf = pdf.sort_values("ds").copy()
    symbol = str(pdf["symbol"].iloc[0])

    if not FORECAST_START_DS or not FORECAST_END_DS:
        return forecast_next_day(pdf)

    try:
        start_date = pd.to_datetime(FORECAST_START_DS).date()
        end_date = pd.to_datetime(FORECAST_END_DS).date()
    except Exception as exc:  
        logger.exception("Invalid FORECAST_*_DS values: %s", exc)
        return pd.DataFrame(columns=["symbol", "ds", "y_hat_close"])

    pdf["ds"] = pd.to_datetime(pdf["ds"]).dt.date

    rows: list[dict[str, object]] = []

    for target_ts in pd.date_range(start_date, end_date):
        target_ds = target_ts.date()
        history = pdf[pdf["ds"] < target_ds]

        if history.empty:
            continue

        y_hat = _fit_arima_and_forecast_next(history["close"])
        if y_hat is None:
            continue

        rows.append(
            {
                "symbol": symbol,
                "ds": target_ds,
                "y_hat_close": y_hat,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["symbol", "ds", "y_hat_close"])

    return pd.DataFrame(rows)


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
        .applyInPandas(forecast_for_period, schema=FORECAST_SCHEMA)
        .withColumn("created_at", F.current_timestamp())
    )

    if FORECAST_START_DS and FORECAST_END_DS:
        logger.info(
            "Joining forecasts with actual close for period %s to %s",
            FORECAST_START_DS,
            FORECAST_END_DS,
        )

        actuals_df = (
            daily_df.filter(
                (F.col("ds") >= F.lit(FORECAST_START_DS).cast("date"))
                & (F.col("ds") <= F.lit(FORECAST_END_DS).cast("date"))
            )
            .select("symbol", "ds", "close")
            .withColumnRenamed("close", "actual_close")
        )

        forecast_df = (
            forecast_df.alias("f")
            .join(actuals_df.alias("a"), ["symbol", "ds"], "left")
            .withColumn(
                "abs_error",
                F.when(
                    F.col("a.actual_close").isNull(),
                    F.lit(None).cast("double"),
                ).otherwise(
                    F.abs(F.col("f.y_hat_close") - F.col("a.actual_close"))
                ),
            )
            .withColumn(
                "ape",
                F.when(
                    F.col("a.actual_close").isNull(),
                    F.lit(None).cast("double"),
                ).otherwise(
                    F.abs(
                        (F.col("f.y_hat_close") - F.col("a.actual_close"))
                        / F.col("a.actual_close")
                    )
                ),
            )
        )

    logger.info("Writing forecasts to %s", DAILY_FORECAST_PATH)

    (
        forecast_df.repartition("ds", "symbol")
        .write.mode("append")
        .option("header", True)
        .option("compression", "none")
        .partitionBy("ds", "symbol")
        .csv(DAILY_FORECAST_PATH)
    )

    logger.info("Forecast job finished successfully.")


if __name__ == "__main__":
    run()
