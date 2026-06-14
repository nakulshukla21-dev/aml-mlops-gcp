"""Refresh automl_<profile>.json column_specs from the live BigQuery feature view."""

from __future__ import annotations

import argparse

from google.cloud import bigquery

from src.automl_utils import (
    artifact_path,
    automl_config,
    bq_table_id,
    build_column_specs,
    deploy_artifact_path,
    excluded_columns,
    load_run_artifact,
    save_run_artifact,
)
from src.config import add_config_arguments, load_config_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh column_specs in automl run artifact from BigQuery schema."
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Optional path to artifact JSON. Defaults to artifacts/automl_<profile>.json.",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        help="BigQuery view to read column names from. Defaults to automl input view.",
    )
    return parser.parse_args()


def feature_columns_from_view(
    client: bigquery.Client,
    config: dict,
    view_name: str,
) -> tuple[list[str], dict[str, str]]:
    table = client.get_table(bq_table_id(config, view_name))
    column_names = [field.name for field in table.schema]
    column_specs = build_column_specs(column_names, config)
    return column_names, column_specs


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    print(f"Profile: {profile} ({config_path.name})")

    automl = automl_config(config)
    view_name = args.view or automl.get("automl_input_view", "features_automl")
    artifact_file = artifact_path(config) if args.artifact is None else args.artifact

    client = bigquery.Client(project=config["gcp"]["project_id"])
    view_columns, column_specs = feature_columns_from_view(client, config, view_name)

    if artifact_file.exists():
        artifact = load_run_artifact(artifact_file)
        previous = artifact.get("column_specs") or {}
        print(f"Existing column_specs: {len(previous)}")
    else:
        artifact = {
            "profile": profile,
            "project_id": config["gcp"]["project_id"],
            "region": config["gcp"]["region"],
            "target_column": automl.get("target_column", "is_fraud"),
            "split_column": automl.get("split_column", "ml_split"),
            "excluded_columns": excluded_columns(config),
            "automl_input_view": view_name,
        }
        previous = {}
        print(f"No artifact found at {artifact_file}; creating a new one.")

    deploy_path = deploy_artifact_path(config)
    if not artifact.get("model_resource_name") and deploy_path.exists():
        deploy_artifact = load_run_artifact(deploy_path)
        for key in (
            "model_resource_name",
            "model_display_name",
            "project_id",
            "region",
            "profile",
        ):
            if deploy_artifact.get(key) and not artifact.get(key):
                artifact[key] = deploy_artifact[key]
        print("Restored model metadata from deploy artifact.")

    added = sorted(set(column_specs) - set(previous))
    removed = sorted(set(previous) - set(column_specs))
    if added:
        print(f"Added columns ({len(added)}): {', '.join(added)}")
    if removed:
        print(f"Removed columns ({len(removed)}): {', '.join(removed)}")

    artifact["excluded_columns"] = excluded_columns(config)
    artifact["column_specs"] = column_specs
    artifact["feature_view"] = view_name
    artifact["feature_column_count"] = len(column_specs)
    artifact["view_column_count"] = len(view_columns)
    save_run_artifact(artifact_file, artifact)
    print(f"Refreshed column_specs: {len(column_specs)} feature columns from `{view_name}`")


if __name__ == "__main__":
    main()
