"""Run parity checks on multiple random test transactions."""

from __future__ import annotations

import argparse

from google.cloud import bigquery

from src.automl_utils import bq_table_id
from src.config import add_config_arguments, load_config_from_args
from src.serving.feature_builder import (
    compare_feature_rows,
    fetch_offline_features,
    model_feature_columns,
    score_features,
)
from src.serving.parity_check import load_raw_transaction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch parity check against features_training.")
    add_config_arguments(parser)
    parser.add_argument("--limit", type=int, default=5, help="Number of random test rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, _ = load_config_from_args(args)
    client = bigquery.Client(project=config["gcp"]["project_id"])
    view_id = bq_table_id(config, config["bigquery"]["features_test_view"])
    query = f"SELECT transaction_id FROM `{view_id}` ORDER BY RAND() LIMIT @limit"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", args.limit)]
    )
    ids = [row.transaction_id for row in client.query(query, job_config=job_config).result()]

    failures: list[str] = []
    for transaction_id in ids:
        txn = load_raw_transaction(client, config, transaction_id)
        online = score_features(client, config, txn)
        offline = fetch_offline_features(client, config, transaction_id)
        if offline is None:
            failures.append(transaction_id)
            print(f"{transaction_id}: missing offline row")
            continue
        cols = model_feature_columns(config, list(online.keys()))
        parity = compare_feature_rows(online, offline, compare_columns=cols)
        status = "OK" if parity["matched"] else f"FAIL delta={parity['max_feature_delta']}"
        print(f"{transaction_id}: {status}")
        if not parity["matched"]:
            failures.append(transaction_id)

    if failures:
        raise SystemExit(f"Parity failures: {failures}")
    print(f"All {len(ids)} parity checks passed.")


if __name__ == "__main__":
    main()
