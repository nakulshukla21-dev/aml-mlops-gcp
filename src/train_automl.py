"""Train a Vertex AI AutoML Tabular classifier on BigQuery feature views."""

from __future__ import annotations

import argparse

from google.cloud import aiplatform, bigquery

from src.automl_utils import (
    artifact_path,
    automl_config,
    bq_source_uri,
    build_column_specs,
    deploy_automl_input_view,
    excluded_columns,
    save_run_artifact,
    utc_timestamp,
)
from src.config import add_config_arguments, load_config_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Vertex AI AutoML Tabular model on temporal feature splits."
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--skip-deploy-view",
        action="store_true",
        help="Skip deploying the features_automl input view (assumes it already exists).",
    )
    parser.add_argument(
        "--budget-milli-node-hours",
        type=int,
        default=None,
        help="Override automl.budget_milli_node_hours from config.",
    )
    return parser.parse_args()


def train_model(config: dict, args: argparse.Namespace) -> dict:
    gcp = config["gcp"]
    automl = automl_config(config)
    profile = config.get("profile", args.profile)
    project_id = gcp["project_id"]
    region = gcp["region"]

    client = bigquery.Client(project=project_id)
    if not args.skip_deploy_view:
        deploy_automl_input_view(client, config)

    input_view = automl.get("automl_input_view", "features_automl")
    target_column = automl.get("target_column", "is_fraud")
    split_column = automl.get("split_column", "ml_split")
    budget = args.budget_milli_node_hours or automl.get("budget_milli_node_hours", 1000)
    optimization_objective = automl.get("optimization_objective", "maximize-au-prc")
    display_name_prefix = automl.get("display_name_prefix", "aml-fraud")
    timestamp = utc_timestamp()

    aiplatform.init(project=project_id, location=region)

    dataset_display_name = f"{display_name_prefix}-dataset-{profile}-{timestamp}"
    print(f"Creating TabularDataset from {bq_source_uri(config, input_view)}")
    dataset = aiplatform.TabularDataset.create(
        display_name=dataset_display_name,
        bq_source=bq_source_uri(config, input_view),
    )
    column_specs = build_column_specs(list(dataset.column_names), config)
    print(f"Training on {len(column_specs)} feature columns")

    training_display_name = f"{display_name_prefix}-training-{profile}-{timestamp}"
    training_job = aiplatform.AutoMLTabularTrainingJob(
        display_name=training_display_name,
        optimization_prediction_type="classification",
        optimization_objective=optimization_objective,
        column_specs=column_specs,
    )

    model_display_name = automl.get(
        "model_display_name",
        f"{display_name_prefix}-model-{profile}-{timestamp}",
    )
    print(
        f"Starting AutoML training: target={target_column}, "
        f"split={split_column}, budget_milli_node_hours={budget}"
    )
    model = training_job.run(
        dataset=dataset,
        target_column=target_column,
        predefined_split_column_name=split_column,
        budget_milli_node_hours=budget,
        model_display_name=model_display_name,
        disable_early_stopping=False,
        sync=True,
    )

    artifact = {
        "profile": profile,
        "project_id": project_id,
        "region": region,
        "target_column": target_column,
        "split_column": split_column,
        "excluded_columns": excluded_columns(config),
        "column_specs": column_specs,
        "automl_input_view": input_view,
        "dataset_resource_name": dataset.resource_name,
        "training_job_resource_name": training_job.resource_name,
        "model_resource_name": model.resource_name,
        "model_display_name": model_display_name,
        "optimization_objective": optimization_objective,
        "budget_milli_node_hours": budget,
        "trained_at": timestamp,
    }
    path = artifact_path(config)
    save_run_artifact(path, artifact)
    print(f"Model trained: {model.resource_name}")
    return artifact


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    print(f"Profile: {profile} ({config_path.name})")
    train_model(config, args)


if __name__ == "__main__":
    main()
