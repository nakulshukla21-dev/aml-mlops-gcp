"""Evaluate a trained AutoML model on the test split with per-typology metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import aiplatform, bigquery

from src.automl_utils import (
    ARTIFACTS_DIR,
    artifact_path,
    automl_config,
    bq_dataset_id,
    bq_dataset_uri,
    bq_source_uri,
    bq_table_id,
    load_run_artifact,
    save_run_artifact,
)
from src.config import add_config_arguments, load_config_from_args


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


def print_metrics(client: bigquery.Client, predictions_table: str, config: dict) -> None:
    automl = automl_config(config)
    bq = config["bigquery"]
    target_column = automl.get("target_column", "is_fraud")
    eval_view = bq_table_id(config, bq["features_eval_view"])

    table = client.get_table(predictions_table)
    predicted_col = prediction_column(table.schema, target_column)
    predicted_positive = predicted_positive_expr(table.schema, predicted_col, alias="p")

    overall_query = f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNTIF(e.{target_column}) AS fraud_rows,
          COUNTIF({predicted_positive}) AS predicted_fraud_rows,
          COUNTIF(e.{target_column} AND {predicted_positive}) AS true_positives,
          COUNTIF(NOT e.{target_column} AND NOT ({predicted_positive})) AS true_negatives,
          COUNTIF(e.{target_column} AND NOT ({predicted_positive})) AS false_negatives,
          COUNTIF(NOT e.{target_column} AND {predicted_positive}) AS false_positives
        FROM `{predictions_table}` AS p
        JOIN `{eval_view}` AS e
          USING (transaction_id)
    """
    row = list(client.query(overall_query).result())[0]
    precision = (
        row.true_positives / (row.true_positives + row.false_positives)
        if (row.true_positives + row.false_positives)
        else 0.0
    )
    recall = (
        row.true_positives / (row.true_positives + row.false_negatives)
        if (row.true_positives + row.false_negatives)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    print("\nOverall test metrics:")
    print(f"  Rows: {row.n_rows:,}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(
        f"  Confusion: TP={row.true_positives:,} FP={row.false_positives:,} "
        f"TN={row.true_negatives:,} FN={row.false_negatives:,}"
    )

    typology_query = f"""
        SELECT
          e.typology,
          COUNT(*) AS n_rows,
          COUNTIF(e.{target_column}) AS fraud_rows,
          COUNTIF({predicted_positive}) AS predicted_fraud_rows,
          COUNTIF(e.{target_column} AND {predicted_positive}) AS true_positives,
          COUNTIF(e.{target_column} AND NOT ({predicted_positive})) AS false_negatives,
          COUNTIF(NOT e.{target_column} AND {predicted_positive}) AS false_positives
        FROM `{predictions_table}` AS p
        JOIN `{eval_view}` AS e
          USING (transaction_id)
        GROUP BY e.typology
        ORDER BY n_rows DESC
    """
    print("\nPer-typology test metrics:")
    print(f"  {'typology':<18} {'n':>8} {'fraud':>8} {'prec':>8} {'recall':>8} {'f1':>8}")
    for typology_row in client.query(typology_query).result():
        tp = typology_row.true_positives
        fp = typology_row.false_positives
        fn = typology_row.false_negatives
        typ_precision = tp / (tp + fp) if (tp + fp) else 0.0
        typ_recall = tp / (tp + fn) if (tp + fn) else 0.0
        typ_f1 = (
            2 * typ_precision * typ_recall / (typ_precision + typ_recall)
            if (typ_precision + typ_recall)
            else 0.0
        )
        print(
            f"  {typology_row.typology:<18} "
            f"{typology_row.n_rows:>8,} "
            f"{typology_row.fraud_rows:>8,} "
            f"{typ_precision:>8.3f} "
            f"{typ_recall:>8.3f} "
            f"{typ_f1:>8.3f}"
        )


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

    eval_artifact_path = ARTIFACTS_DIR / f"eval_{profile}.json"
    if args.skip_batch_predict:
        if eval_artifact_path.exists():
            predictions_table = normalize_predictions_table(
                load_run_artifact(eval_artifact_path)["predictions_table"],
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
            eval_artifact_path,
            {
                "profile": profile,
                "model_resource_name": model_name,
                "predictions_table": predictions_table,
            },
        )

    print_metrics(client, predictions_table, config)


if __name__ == "__main__":
    main()
