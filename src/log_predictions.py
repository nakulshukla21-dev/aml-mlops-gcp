"""Backfill prediction_log from an existing Vertex batch prediction table."""

from __future__ import annotations

import argparse

from google.cloud import aiplatform, bigquery

from src.automl_utils import (
    artifact_path,
    automl_config,
    bq_dataset_id,
    eval_artifact_path,
    load_run_artifact,
    normalize_predictions_table,
)
from src.config import add_config_arguments, load_config_from_args
from src.evaluate_automl import find_latest_prediction_table
from src.prediction_logging import log_batch_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append batch predictions to the prediction_log audit table."
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--predictions-table",
        type=str,
        default=None,
        help="Vertex batch output table. Defaults to latest eval artifact or newest automl_predictions_* table.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Vertex model resource name. Defaults to automl run artifact.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="batch_eval",
        choices=["batch_eval", "batch_score", "online"],
        help="prediction_source value written to the log.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Log even if this raw_predictions_table was already logged.",
    )
    return parser.parse_args()


def resolve_predictions_table(
    client: bigquery.Client,
    config: dict,
    args: argparse.Namespace,
) -> str:
    if args.predictions_table:
        return normalize_predictions_table(args.predictions_table, config)

    eval_path = eval_artifact_path(config)
    if eval_path.exists():
        table = load_run_artifact(eval_path).get("predictions_table")
        if table:
            return normalize_predictions_table(table, config)

    prefix = automl_config(config).get("predictions_table_prefix", "automl_predictions")
    latest = find_latest_prediction_table(client, bq_dataset_id(config), prefix)
    if latest:
        return latest

    raise RuntimeError(
        "No predictions table found. Pass --predictions-table or run evaluate_automl first."
    )


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    print(f"Profile: {profile} ({config_path.name})")

    project_id = config["gcp"]["project_id"]
    client = bigquery.Client(project=project_id)

    predictions_table = resolve_predictions_table(client, config, args)
    print(f"Source predictions table: {predictions_table}")

    train_artifact = load_run_artifact(artifact_path(config))
    model_resource_name = args.model or train_artifact["model_resource_name"]
    model_display_name = train_artifact.get("model_display_name")

    aiplatform.init(project=project_id, location=train_artifact["region"])

    log_batch_predictions(
        client,
        config,
        predictions_table,
        model_resource_name=model_resource_name,
        prediction_source=args.source,
        model_display_name=model_display_name,
        batch_job_display_name=f"{automl_config(config).get('display_name_prefix', 'aml-fraud')}-eval-{profile}",
        feature_view=config["bigquery"].get("features_test_view"),
        force=args.force,
    )


if __name__ == "__main__":
    main()
