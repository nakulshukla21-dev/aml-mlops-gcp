"""Append batch and online predictions to a normalized BigQuery audit log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import bigquery

from src.automl_utils import automl_config, bq_table_id
from src.config import PROJECT_ROOT
from src.deploy_views import render_sql
from src.predictions import fraud_score_expr, prediction_column, predicted_positive_expr

SCHEMA_PATH = PROJECT_ROOT / "schemas" / "prediction_log.json"
DDL_PATH = PROJECT_ROOT / "sql" / "logging" / "create_prediction_log_table.sql"


def prediction_log_table_name(config: dict) -> str:
    return config["bigquery"].get("prediction_log_table", "prediction_log")


def prediction_log_table_id(config: dict) -> str:
    return bq_table_id(config, prediction_log_table_name(config))


def load_schema(path: Path) -> list[bigquery.SchemaField]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [bigquery.SchemaField.from_api_repr(field) for field in raw]


def ensure_prediction_log_table(client: bigquery.Client, config: dict) -> str:
    """Create the prediction log table if missing."""
    table_id = prediction_log_table_id(config)
    gcp = config["gcp"]
    bq = config["bigquery"]
    variables = {
        "project_id": gcp["project_id"],
        "dataset": bq["dataset"],
        "prediction_log_table": prediction_log_table_name(config),
    }
    ddl = render_sql(DDL_PATH.read_text(encoding="utf-8"), variables)
    client.query(ddl).result()
    print(f"Prediction log table ready: {table_id}")
    return table_id


def count_existing_log_rows(
    client: bigquery.Client,
    config: dict,
    raw_predictions_table: str,
) -> int:
    table_id = prediction_log_table_id(config)
    query = f"""
        SELECT COUNT(*) AS n_rows
        FROM `{table_id}`
        WHERE raw_predictions_table = @raw_predictions_table
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "raw_predictions_table", "STRING", raw_predictions_table
            )
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    return int(rows[0].n_rows) if rows else 0


def log_batch_predictions(
    client: bigquery.Client,
    config: dict,
    predictions_table: str,
    *,
    model_resource_name: str,
    prediction_source: str = "batch_eval",
    model_display_name: str | None = None,
    batch_job_display_name: str | None = None,
    endpoint_resource_name: str | None = None,
    feature_view: str | None = None,
    force: bool = False,
) -> int:
    """
    Insert normalized rows from a Vertex batch prediction table into prediction_log.

    Idempotent per raw_predictions_table unless force=True.
    """
    ensure_prediction_log_table(client, config)

    if not force:
        existing = count_existing_log_rows(client, config, predictions_table)
        if existing > 0:
            print(
                f"Skipping prediction log: {existing:,} rows already logged "
                f"from {predictions_table}"
            )
            return 0

    pred_table = client.get_table(predictions_table)
    automl = automl_config(config)
    target_column = automl.get("target_column", "is_fraud")
    predicted_col = prediction_column(pred_table.schema, target_column)
    predicted_positive = predicted_positive_expr(pred_table.schema, predicted_col, alias="p")
    fraud_score = fraud_score_expr(pred_table.schema, predicted_col, alias="p")

    bq = config["bigquery"]
    profile = config.get("profile", "train")
    log_table_id = prediction_log_table_id(config)
    eval_view_id = bq_table_id(config, bq["features_eval_view"])
    feature_view = feature_view or bq.get("features_test_view")

    scored_at = datetime.now(timezone.utc)
    insert_sql = f"""
        INSERT INTO `{log_table_id}` (
          prediction_id,
          transaction_id,
          scored_at,
          prediction_source,
          model_resource_name,
          model_display_name,
          batch_job_display_name,
          endpoint_resource_name,
          profile,
          feature_view,
          predicted_is_fraud,
          fraud_score,
          actual_is_fraud,
          raw_predictions_table,
          logged_at
        )
        SELECT
          GENERATE_UUID() AS prediction_id,
          p.transaction_id,
          @scored_at AS scored_at,
          @prediction_source AS prediction_source,
          @model_resource_name AS model_resource_name,
          @model_display_name AS model_display_name,
          @batch_job_display_name AS batch_job_display_name,
          @endpoint_resource_name AS endpoint_resource_name,
          @profile AS profile,
          @feature_view AS feature_view,
          {predicted_positive} AS predicted_is_fraud,
          {fraud_score} AS fraud_score,
          e.{target_column} AS actual_is_fraud,
          @raw_predictions_table AS raw_predictions_table,
          CURRENT_TIMESTAMP() AS logged_at
        FROM `{predictions_table}` AS p
        LEFT JOIN `{eval_view_id}` AS e
          USING (transaction_id)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("scored_at", "TIMESTAMP", scored_at),
            bigquery.ScalarQueryParameter("prediction_source", "STRING", prediction_source),
            bigquery.ScalarQueryParameter(
                "model_resource_name", "STRING", model_resource_name
            ),
            bigquery.ScalarQueryParameter(
                "model_display_name", "STRING", model_display_name
            ),
            bigquery.ScalarQueryParameter(
                "batch_job_display_name", "STRING", batch_job_display_name
            ),
            bigquery.ScalarQueryParameter(
                "endpoint_resource_name", "STRING", endpoint_resource_name
            ),
            bigquery.ScalarQueryParameter("profile", "STRING", profile),
            bigquery.ScalarQueryParameter("feature_view", "STRING", feature_view),
            bigquery.ScalarQueryParameter(
                "raw_predictions_table", "STRING", predictions_table
            ),
        ]
    )
    job = client.query(insert_sql, job_config=job_config)
    job.result()
    inserted = int(job.num_dml_affected_rows or 0)
    print(f"Logged {inserted:,} predictions to {log_table_id}")
    return inserted
