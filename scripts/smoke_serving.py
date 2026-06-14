"""Run end-to-end smoke test against local serving API."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_payload() -> dict:
    from google.cloud import bigquery

    from src.config import DEV_CONFIG_PATH, load_config

    config = load_config(DEV_CONFIG_PATH)
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    client = bigquery.Client(project=project)

    sender = list(
        client.query(
            f"""
            SELECT customer_id, account_id
            FROM `{project}.{dataset}.{config['bigquery']['dim_account_table']}`
            WHERE LOWER(status) = 'active'
            LIMIT 1
            """
        ).result()
    )[0]

    receiver = list(
        client.query(
            f"""
            SELECT cp.legal_name, cp.country, ca.account_label
            FROM `{project}.{dataset}.{config['bigquery']['dim_counterparty_account_table']}` AS ca
            JOIN `{project}.{dataset}.{config['bigquery']['dim_counterparty_table']}` AS cp
              ON ca.counterparty_id = cp.counterparty_id
            WHERE ca.account_label IS NOT NULL
            LIMIT 1
            """
        ).result()
    )[0]

    return {
        "transaction_id": f"TXN-SMOKE-{uuid.uuid4().hex[:8].upper()}",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "customer_id": int(sender.customer_id),
        "sender_account_id": sender.account_id,
        "amount": 2500.0,
        "transaction_currency": "USD",
        "transaction_type": "transfer",
        "channel": "wire",
        "payment_sender_country": "US",
        "payment_receiver_country": receiver.country or "KY",
        "settlement_currency": "USD",
        "settlement_amount": 2500.0,
        "settlement_status": "pending",
        "receiver": {
            "type": "external",
            "beneficiary_name": receiver.legal_name,
            "country": receiver.country,
            "account_reference": receiver.account_label,
        },
        "log_prediction": True,
    }


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    payload = build_payload()

    with httpx.Client(timeout=180.0) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        print("HEALTH", json.dumps(health.json(), indent=2))

        score = client.post(f"{base_url}/score", json=payload)
        print("SCORE_STATUS", score.status_code)
        print("SCORE_BODY", json.dumps(score.json(), indent=2))
        score.raise_for_status()

        txn_id = payload["transaction_id"]
        validate_payload = {
            "transaction_id": "TXN-05BE16B0E90E",
            "timestamp": "2024-11-02T09:15:00Z",
            "sender_account_id": "BAUS0000100",
            "receiver_counterparty_account_id": "CPAKY00001",
            "amount": 100.0,
            "transaction_currency": "USD",
            "transaction_type": "payment",
            "channel": "wire",
            "payment_sender_country": "US",
            "payment_receiver_country": "KY",
            "settlement_currency": "USD",
            "settlement_amount": 100.0,
            "settlement_status": "settled",
            "run_model": False,
        }
        validate = client.post(f"{base_url}/validate", json=validate_payload)
        print("VALIDATE_STATUS", validate.status_code)
        body = validate.json()
        print(
            "VALIDATE_PARITY",
            body.get("parity", {}).get("matched"),
            "delta",
            body.get("parity", {}).get("max_feature_delta"),
        )
        validate.raise_for_status()

    print(f"Smoke test passed. Scored transaction: {txn_id}")


if __name__ == "__main__":
    main()
