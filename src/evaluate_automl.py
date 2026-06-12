"""Evaluate a trained AutoML model on the test split with per-typology metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import aiplatform, bigquery

from src.automl_utils import (
    artifact_path,
    automl_config,
    bq_dataset_id,
    bq_dataset_uri,
    bq_source_uri,
    bq_table_id,
    eval_artifact_path,
    load_run_artifact,
    metrics_artifact_path,
    save_run_artifact,
)
from src.config import add_config_arguments, load_config_from_args
from src.metrics import compute_metrics, print_metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-predict on test data and report fraud metrics by typology."
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Path to automl run artifact JSON. Defaults to artifacts/automl_<profile>.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Vertex model resource name. Overrides artifact model_resource_name.",
    )
    parser.add_argument(
        "--skip-batch-predict",
        action="store_true",
        help="Reuse the latest batch prediction table in the dataset instead of running a new job.",
    )
    return parser.parse_args()


def prediction_column(schema: list[bigquery.SchemaField], target_column: str) -> str:
    candidates = [
        f"predicted_{target_column}",
        f"predicted_{target_column}.bool",
        "predicted_label",
    ]
    names = {field.name for field in schema}
    for candidate in candidates:
        if candidate in names:
            return candidate
    predicted = [name for name in names if name.startswith("predicted_")]
    if len(predicted) == 1:
        return predicted[0]
    raise ValueError(f"Could not find prediction column in schema: {sorted(names)}")


def predicted_positive_expr(
    schema: list[bigquery.SchemaField],
    predicted_col: str,
    alias: str = "p",
) -> str:
    """SQL boolean expression for the model's positive-class prediction."""
    field = next((item for item in schema if item.name == predicted_col), None)
    if field is None:
        raise ValueError(f"Prediction column not found in schema: {predicted_col}")
    if field.field_type in {"BOOLEAN", "BOOL"}:
        return f"{alias}.{predicted_col}"
    if field.field_type == "RECORD":
        # AutoML batch output: STRUCT<classes ARRAY<STRING>, scores ARRAY<FLOAT64>>
        return f"""(
          SELECT AS VALUE LOWER(class) IN ('true', '1')
          FROM UNNEST({alias}.{predicted_col}.classes) AS class WITH OFFSET i
          JOIN UNNEST({alias}.{predicted_col}.scores) AS score WITH OFFSET j
            ON i = j
          ORDER BY score DESC
          LIMIT 1
        )"""
    raise ValueError(
        f"Unsupported prediction column type for {predicted_col}: {field.field_type}"
    )


def find_latest_prediction_table(
    client: bigquery.Client,
    dataset_id: str,
    prefix: str,
) -> str | None:
    query = f"""
        SELECT table_name
        FROM `{client.project}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
        WHERE table_name LIKE '{prefix}%'
        ORDER BY creation_time DESC
        LIMIT 1
    """
    rows = list(client.query(query).result())
    if not rows:
        return None
    return f"{client.project}.{dataset_id}.{rows[0].table_name}"


def normalize_predictions_table(predictions_table: str, config: dict) -> str:
    if predictions_table.count(".") >= 2:
        return predictions_table
    return bq_table_id(config, predictions_table)


def run_batch_predict(
    model: aiplatform.Model,
    config: dict,
    job_display_name: str,
) -> str:
    automl = automl_config(config)
    bq = config["bigquery"]
    test_view = bq["features_test_view"]
    destination_prefix = bq_dataset_uri(config)

    print(f"Batch predict source: {bq_source_uri(config, test_view)}")
    machine_type = automl.get("batch_predict_machine_type", "n1-standard-4")
    batch_job = model.batch_predict(
        job_display_name=job_display_name,
        bigquery_source=bq_source_uri(config, test_view),
        instances_format="bigquery",
        predictions_format="bigquery",
        bigquery_destination_prefix=destination_prefix,
        machine_type=machine_type,
        sync=True,
    )

    output_table = batch_job.output_info.bigquery_output_table
    if not output_table:
        raise RuntimeError("Batch prediction completed without a BigQuery output table.")
    output_table = normalize_predictions_table(output_table, config)
    print(f"Predictions written to: {output_table}")
    return output_table


def evaluate_and_report(
    client: bigquery.Client,
    predictions_table: str,
    config: dict,
    model_resource_name: str,
) -> dict:
    table = client.get_table(predictions_table)
    target_column = automl_config(config).get("target_column", "is_fraud")
    predicted_col = prediction_column(table.schema, target_column)
    predicted_positive = predicted_positive_expr(table.schema, predicted_col, alias="p")

    metrics = compute_metrics(
        client,
        predictions_table,
        config,
        predicted_col=predicted_col,
        predicted_positive=predicted_positive,
    )
    metrics["model_resource_name"] = model_resource_name
    print_metrics_report(metrics)
    save_run_artifact(metrics_artifact_path(config), metrics)
    return metrics


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    print(f"Profile: {profile} ({config_path.name})")

    artifact_file = args.artifact or artifact_path(config)
    artifact = load_run_artifact(artifact_file)
    model_name = args.model or artifact["model_resource_name"]
    region = artifact["region"]
    project_id = artifact["project_id"]

    aiplatform.init(project=project_id, location=region)
    model = aiplatform.Model(model_name)

    client = bigquery.Client(project=project_id)
    automl = automl_config(config)
    prefix = automl.get("predictions_table_prefix", "automl_predictions")

    eval_path = eval_artifact_path(config)
    if args.skip_batch_predict:
        if eval_path.exists():
            predictions_table = normalize_predictions_table(
                load_run_artifact(eval_path)["predictions_table"],
                config,
            )
            print(f"Reusing predictions table: {predictions_table}")
        else:
            predictions_table = find_latest_prediction_table(
                client, bq_dataset_id(config), prefix
            )
            if not predictions_table:
                raise RuntimeError(
                    "No existing prediction table found. Run without --skip-batch-predict."
                )
            print(f"Reusing latest predictions table: {predictions_table}")
    else:
        job_name = f"{automl.get('display_name_prefix', 'aml-fraud')}-eval-{profile}"
        predictions_table = run_batch_predict(model, config, job_name)
        save_run_artifact(
            eval_path,
            {
                "profile": profile,
                "model_resource_name": model_name,
                "predictions_table": predictions_table,
            },
        )

    evaluate_and_report(client, predictions_table, config, model_name)


if __name__ == "__main__":
    main()
