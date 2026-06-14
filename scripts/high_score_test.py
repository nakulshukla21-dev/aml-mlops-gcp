"""Score a payment shaped like a known fraud transaction."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from google.cloud import bigquery

from src.config import DEV_CONFIG_PATH, load_config

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
SOURCE_TXN = sys.argv[2] if len(sys.argv) > 2 else "TXN-836A9E57DA81"


def main() -> None:
    config = load_config(DEV_CONFIG_PATH)
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    client = bigquery.Client(project=project)

    raw = list(
        client.query(
            f"""
            SELECT *
            FROM `{project}.{dataset}.{config['bigquery']['raw_table']}`
            WHERE transaction_id = @transaction_id
            LIMIT 1
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("transaction_id", "STRING", SOURCE_TXN)
                ]
            ),
        ).result()
    )[0]

    sender_account_id = raw.sender_account_id
    account = list(
        client.query(
            f"""
            SELECT customer_id
            FROM `{project}.{dataset}.{config['bigquery']['dim_account_table']}`
            WHERE account_id = @account_id
            LIMIT 1
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("account_id", "STRING", sender_account_id)
                ]
            ),
        ).result()
    )[0]

    cp = list(
        client.query(
            f"""
            SELECT cp.legal_name, cp.country, cp.dba_name, ca.account_label
            FROM `{project}.{dataset}.{config['bigquery']['dim_counterparty_account_table']}` AS ca
            JOIN `{project}.{dataset}.{config['bigquery']['dim_counterparty_table']}` AS cp
              ON ca.counterparty_id = cp.counterparty_id
            WHERE ca.counterparty_account_id = @cp_account_id
            LIMIT 1
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "cp_account_id", "STRING", raw.receiver_counterparty_account_id
                    )
                ]
            ),
        ).result()
    )[0]

    payload = {
        "transaction_id": f"TXN-HIGH-{uuid.uuid4().hex[:8].upper()}",
        "timestamp": raw.timestamp.isoformat().replace("+00:00", "Z"),
        "customer_id": int(account.customer_id),
        "sender_account_id": sender_account_id,
        "amount": float(raw.amount),
        "transaction_currency": raw.transaction_currency,
        "transaction_type": raw.transaction_type,
        "channel": raw.channel,
        "payment_sender_country": raw.payment_sender_country,
        "payment_receiver_country": raw.payment_receiver_country,
        "settlement_currency": raw.settlement_currency,
        "settlement_amount": float(raw.settlement_amount),
        "settlement_status": raw.settlement_status,
        "fx_rate": float(raw.fx_rate) if raw.fx_rate is not None else 1.0,
        "settlement_date": raw.settlement_date.isoformat() if raw.settlement_date else None,
        "payment_reference": raw.payment_reference,
        "memo": raw.memo,
        "clearing_system": raw.clearing_system,
        "correspondent_bic": raw.correspondent_bic,
        "receiver": {
            "type": "external",
            "beneficiary_name": cp.legal_name,
            "dba_name": cp.dba_name,
            "country": cp.country,
            "account_reference": cp.account_label,
        },
        "log_prediction": True,
    }

    with httpx.Client(timeout=180.0) as http:
        validate = {
            "transaction_id": SOURCE_TXN,
            "timestamp": raw.timestamp.isoformat().replace("+00:00", "Z"),
            "sender_account_id": raw.sender_account_id,
            "sender_counterparty_account_id": raw.sender_counterparty_account_id,
            "receiver_account_id": raw.receiver_account_id,
            "receiver_counterparty_account_id": raw.receiver_counterparty_account_id,
            "amount": float(raw.amount),
            "transaction_currency": raw.transaction_currency,
            "transaction_type": raw.transaction_type,
            "channel": raw.channel,
            "payment_sender_country": raw.payment_sender_country,
            "payment_receiver_country": raw.payment_receiver_country,
            "settlement_currency": raw.settlement_currency,
            "settlement_amount": float(raw.settlement_amount),
            "settlement_status": raw.settlement_status,
            "fx_rate": float(raw.fx_rate) if raw.fx_rate is not None else 1.0,
            "settlement_date": raw.settlement_date.isoformat() if raw.settlement_date else None,
            "payment_reference": raw.payment_reference,
            "memo": raw.memo,
            "clearing_system": raw.clearing_system,
            "correspondent_bic": raw.correspondent_bic,
            "run_model": True,
        }
        baseline = http.post(f"{BASE_URL}/validate", json=validate)
        baseline.raise_for_status()

        score = http.post(f"{BASE_URL}/score", json=payload)
        score.raise_for_status()

    baseline_body = baseline.json()
    score_body = score.json()
    print("SOURCE_TXN", SOURCE_TXN)
    print("TYPOLOGY", raw.typology)
    print("BASELINE_VALIDATE", json.dumps(baseline_body, indent=2, default=str))
    print("HIGH_SCORE_PAYMENT", json.dumps(score_body, indent=2, default=str))


if __name__ == "__main__":
    main()
