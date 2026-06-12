"""Shared helpers for Vertex AI AutoML training and evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import bigquery

from src.config import PROJECT_ROOT
from src.deploy_views import render_sql

AUTOML_INPUT_SQL = PROJECT_ROOT / "sql" / "training" / "automl_input.sql"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def automl_config(config: dict) -> dict:
    return config.get("automl", {})


def excluded_columns(config: dict) -> list[str]:
    automl = automl_config(config)
    return list(
        automl.get(
            "excluded_columns",
            [
                "transaction_id",
                "timestamp",
                "txn_date",
                "sender_account",
                "receiver_account",
                "ml_split",
            ],
        )
    )


def build_column_specs(
    column_names: list[str],
    config: dict,
) -> dict[str, str]:
    """Build column_specs for AutoML — omitted columns are ignored during training."""
    automl = automl_config(config)
    target_column = automl.get("target_column", "is_fraud")
    skip = set(excluded_columns(config))
    skip.add(target_column)

    return {name: "auto" for name in column_names if name not in skip}


def bq_table_id(config: dict, table_or_view: str) -> str:
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    return f"{project}.{dataset}.{table_or_view}"


def bq_dataset_id(config: dict) -> str:
    return config["bigquery"]["dataset"]


def bq_dataset_uri(config: dict) -> str:
    """BigQuery dataset URI for batch prediction output (no table name)."""
    project = config["gcp"]["project_id"]
    return f"bq://{project}.{bq_dataset_id(config)}"


def bq_source_uri(config: dict, table_or_view: str) -> str:
    return f"bq://{bq_table_id(config, table_or_view)}"


def template_variables(config: dict) -> dict[str, str]:
    gcp = config["gcp"]
    bq = config["bigquery"]
    automl = automl_config(config)
    return {
        "project_id": gcp["project_id"],
        "dataset": bq["dataset"],
        "automl_input_view": automl.get("automl_input_view", "features_automl"),
        "features_train_view": bq.get("features_train_view", "features_train"),
        "features_val_view": bq.get("features_val_view", "features_val"),
        "features_test_view": bq.get("features_test_view", "features_test"),
    }


def deploy_automl_input_view(client: bigquery.Client, config: dict) -> str:
    """Create or replace the AutoML input view in BigQuery."""
    variables = template_variables(config)
    template = AUTOML_INPUT_SQL.read_text(encoding="utf-8")
    sql = render_sql(template, variables)
    client.query(sql).result()
    view_id = bq_table_id(config, variables["automl_input_view"])
    print(f"Deployed AutoML input view: {view_id}")
    return view_id


def artifact_path(config: dict) -> Path:
    profile = config.get("profile", "train")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / f"automl_{profile}.json"


def eval_artifact_path(config: dict) -> Path:
    profile = config.get("profile", "train")
    return ARTIFACTS_DIR / f"eval_{profile}.json"


def metrics_artifact_path(config: dict) -> Path:
    profile = config.get("profile", "train")
    return ARTIFACTS_DIR / f"metrics_{profile}.json"


def deploy_artifact_path(config: dict) -> Path:
    profile = config.get("profile", "train")
    return ARTIFACTS_DIR / f"deploy_{profile}.json"


def deployment_config(config: dict) -> dict:
    return config.get("deployment", {})


def save_run_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved run artifact: {path}")


def load_run_artifact(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
