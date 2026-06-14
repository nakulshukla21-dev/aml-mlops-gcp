"""CLI: compare online score_features output against features_training."""

from __future__ import annotations

import argparse
import json

from google.cloud import bigquery

from src.automl_utils import excluded_columns
from src.config import DEV_CONFIG_PATH, add_config_arguments, load_config_from_args
from src.serving.feature_builder import (
    compare_feature_rows,
    fetch_offline_features,
    model_feature_columns,
    score_features,
)
from src.serving.schemas import RawTransactionV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parity-check online feature SQL against features_training."
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--transaction-id",
        required=True,
        help="Existing transaction_id from raw_transactions to validate.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Max allowed numeric feature delta.",
    )
    return parser.parse_args()


def load_raw_transaction(
    client: bigquery.Client,
    config: dict,
    transaction_id: str,
) -> RawTransactionV2:
    raw_table = config["bigquery"]["raw_table"]
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    query = f"""
        SELECT *
        FROM `{project}.{dataset}.{raw_table}`
        WHERE transaction_id = @transaction_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("transaction_id", "STRING", transaction_id),
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        raise SystemExit(f"Transaction not found: {transaction_id}")
    payload = dict(rows[0].items())
    payload.pop("ingested_at", None)
    payload.pop("typology", None)
    payload.pop("is_fraud", None)
    txn = RawTransactionV2.from_dict(payload)
    txn.validate_xor_legs()
    return txn


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    print(f"Profile: {config.get('profile', args.profile)} ({config_path.name})")

    client = bigquery.Client(project=config["gcp"]["project_id"])
    txn = load_raw_transaction(client, config, args.transaction_id)
    online = score_features(client, config, txn)
    offline = fetch_offline_features(client, config, args.transaction_id)
    if offline is None:
        raise SystemExit(f"No offline features for transaction_id={args.transaction_id}")

    compare_keys = model_feature_columns(config, list(online.keys()))
    parity = compare_feature_rows(
        online,
        offline,
        compare_columns=compare_keys,
        tolerance=args.tolerance,
    )
    result = {
        "transaction_id": args.transaction_id,
        "parity": parity,
        "excluded_columns": excluded_columns(config),
    }
    print(json.dumps(result, indent=2, default=str))
    if not parity["matched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
