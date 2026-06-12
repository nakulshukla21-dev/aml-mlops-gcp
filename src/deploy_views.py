"""Deploy BigQuery feature and split views from sql/ templates."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from google.cloud import bigquery

from src.config import PROJECT_ROOT, add_config_arguments, load_config_from_args

SQL_DIR = PROJECT_ROOT / "sql"

VIEW_SCRIPTS = [
    SQL_DIR / "features" / "features_base.sql",
    SQL_DIR / "features" / "features_velocity.sql",
    SQL_DIR / "features" / "features_network.sql",
    SQL_DIR / "features" / "features_combined.sql",
    SQL_DIR / "features" / "features_eval.sql",
    SQL_DIR / "splits" / "temporal_splits.sql",
    SQL_DIR / "training" / "automl_input.sql",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy BigQuery views from SQL templates.")
    add_config_arguments(parser)
    return parser.parse_args()


def render_sql(template: str, variables: dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    leftover = re.findall(r"\{\{(\w+)\}\}", rendered)
    if leftover:
        raise ValueError(f"Unresolved SQL template variables: {', '.join(leftover)}")
    return rendered


def split_statements(sql: str) -> list[str]:
    parts = [part.strip() for part in sql.split(";")]
    return [part for part in parts if part]


def template_variables(config: dict) -> dict[str, str]:
    gcp = config["gcp"]
    bq = config["bigquery"]
    splits = config.get("splits", {})
    automl = config.get("automl", {})
    return {
        "project_id": gcp["project_id"],
        "dataset": bq["dataset"],
        "raw_table": bq["raw_table"],
        "dim_customer_table": bq.get("dim_customer_table", "dim_customer"),
        "dim_counterparty_table": bq.get("dim_counterparty_table", "dim_counterparty"),
        "dim_account_table": bq.get("dim_account_table", "dim_account"),
        "dim_counterparty_account_table": bq.get(
            "dim_counterparty_account_table", "dim_counterparty_account"
        ),
        "features_base_view": bq.get("features_base_view", "features_base"),
        "features_velocity_view": bq.get("features_velocity_view", "features_velocity"),
        "features_network_view": bq.get("features_network_view", "features_network"),
        "features_view": bq.get("features_view", "features_training"),
        "features_eval_view": bq.get("features_eval_view", "features_eval"),
        "features_train_view": bq.get("features_train_view", "features_train"),
        "features_val_view": bq.get("features_val_view", "features_val"),
        "features_test_view": bq.get("features_test_view", "features_test"),
        "automl_input_view": automl.get("automl_input_view", "features_automl"),
        "train_end": splits.get("train_end", "2024-09-30"),
        "val_end": splits.get("val_end", "2024-10-31"),
    }


def deploy_views(client: bigquery.Client, config: dict) -> None:
    variables = template_variables(config)

    for script_path in VIEW_SCRIPTS:
        template = script_path.read_text(encoding="utf-8")
        sql = render_sql(template, variables)
        for statement in split_statements(sql):
            client.query(statement).result()
        print(f"Deployed: {script_path.relative_to(PROJECT_ROOT)}")


def print_split_stats(client: bigquery.Client, config: dict) -> None:
    bq = config["bigquery"]
    dataset = bq["dataset"]
    project = config["gcp"]["project_id"]
    train_view = bq.get("features_train_view", "features_train")
    val_view = bq.get("features_val_view", "features_val")
    test_view = bq.get("features_test_view", "features_test")
    query = f"""
        SELECT 'train' AS split, COUNT(*) AS n_rows, COUNTIF(is_fraud) AS fraud_rows
        FROM `{project}.{dataset}.{train_view}`
        UNION ALL
        SELECT 'val', COUNT(*), COUNTIF(is_fraud)
        FROM `{project}.{dataset}.{val_view}`
        UNION ALL
        SELECT 'test', COUNT(*), COUNTIF(is_fraud)
        FROM `{project}.{dataset}.{test_view}`
        ORDER BY split
    """
    rows = list(client.query(query).result())
    print("Split stats:")
    for row in rows:
        rate = row.fraud_rows / row.n_rows if row.n_rows else 0
        print(f"  {row.split}: {row.n_rows:,} rows, {row.fraud_rows:,} fraud ({rate:.2%})")


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    print(f"Profile: {profile} ({config_path.name})")

    client = bigquery.Client(project=config["gcp"]["project_id"])
    deploy_views(client, config)
    print_split_stats(client, config)


if __name__ == "__main__":
    main()
