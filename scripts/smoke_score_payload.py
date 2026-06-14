"""Build JSON payload for POST /score smoke test."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google.cloud import bigquery

from src.config import DEV_CONFIG_PATH, load_config

config = load_config(DEV_CONFIG_PATH)
project = config["gcp"]["project_id"]
dataset = config["bigquery"]["dataset"]
client = bigquery.Client(project=project)

sender_query = f"""
    SELECT a.customer_id, a.account_id
    FROM `{project}.{dataset}.{config['bigquery']['dim_account_table']}` AS a
    WHERE LOWER(a.status) = 'active'
    LIMIT 1
"""
sender = list(client.query(sender_query).result())[0]

receiver_query = f"""
    SELECT cp.legal_name, cp.country, ca.account_label, ca.counterparty_account_id
    FROM `{project}.{dataset}.{config['bigquery']['dim_counterparty_account_table']}` AS ca
    JOIN `{project}.{dataset}.{config['bigquery']['dim_counterparty_table']}` AS cp
      ON ca.counterparty_id = cp.counterparty_id
    WHERE ca.account_label IS NOT NULL
    LIMIT 1
"""
receiver = list(client.query(receiver_query).result())[0]

payload = {
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
print(json.dumps(payload))
