"""Export evaluation metrics JSON from an existing batch predictions table."""

from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import bigquery

from src.automl_utils import (
    artifact_path,
    automl_config,
    bq_dataset_id,
    eval_artifact_path,
    load_run_artifact,
    metrics_artifact_path,
    save_run_artifact,
)
from src.config import add_config_arguments, load_config_from_args
from src.evaluate_automl import (
    find_latest_prediction_table,
    normalize_predictions_table,
    prediction_column,
    predicted_positive_expr,
)
from src.metrics import compute_metrics, print_metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute test metrics from predictions and write artifacts/metrics_<profile>.json."
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--eval-artifact",
        type=Path,
        default=None,
        help="Path to eval artifact JSON. Defaults to artifacts/eval_<profile>.json.",
    )
    parser.add_argument(
        "--predictions-table",
        type=str,
        default=None,
        help="Fully-qualified BigQuery predictions table. Overrides eval artifact.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Vertex model resource name to include in metrics JSON.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Only write JSON; do not print metrics to stdout.",
    )
    return parser.parse_args()


def resolve_predictions_table(
    client: bigquery.Client,
    config: dict,
    args: argparse.Namespace,
) -> str:
    if args.predictions_table:
        return normalize_predictions_table(args.predictions_table, config)

    eval_path = args.eval_artifact or eval_artifact_path(config)
    if eval_path.exists():
        table = load_run_artifact(eval_path)["predictions_table"]
        return normalize_predictions_table(table, config)

    automl = automl_config(config)
    prefix = automl.get("predictions_table_prefix", "automl_predictions")
    table = find_latest_prediction_table(client, bq_dataset_id(config), prefix)
    if not table:
        raise RuntimeError(
            "No predictions table found. Run evaluate_automl first or pass --predictions-table."
        )
    return table


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    print(f"Profile: {profile} ({config_path.name})")

    train_artifact = load_run_artifact(artifact_path(config))
    project_id = train_artifact["project_id"]
    client = bigquery.Client(project=project_id)

    predictions_table = resolve_predictions_table(client, config, args)
    print(f"Using predictions table: {predictions_table}")

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
    metrics["model_resource_name"] = args.model or train_artifact.get("model_resource_name")

    output_path = metrics_artifact_path(config)
    save_run_artifact(output_path, metrics)

    if not args.no_print:
        print_metrics_report(metrics)


if __name__ == "__main__":
    main()
