"""Fetch one active sender account for serving smoke tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google.cloud import bigquery

from src.config import DEV_CONFIG_PATH, load_config

config = load_config(DEV_CONFIG_PATH)
project = config["gcp"]["project_id"]
dataset = config["bigquery"]["dataset"]
account_table = config["bigquery"]["dim_account_table"]
customer_table = config["bigquery"]["dim_customer_table"]
client = bigquery.Client(project=project)
query = f"""
    SELECT a.customer_id, a.account_id
    FROM `{project}.{dataset}.{account_table}` AS a
    JOIN `{project}.{dataset}.{customer_table}` AS c
      ON a.customer_id = c.customer_id
    WHERE LOWER(a.status) = 'active'
    LIMIT 1
"""
row = list(client.query(query).result())[0]
print(f"{row.customer_id}|{row.account_id}")
