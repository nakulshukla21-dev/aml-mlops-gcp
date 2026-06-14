"""Fetch a sample transaction_id from features_test_dev for parity checks."""

from google.cloud import bigquery

from src.config import DEV_CONFIG_PATH, load_config

config = load_config(DEV_CONFIG_PATH)
project = config["gcp"]["project_id"]
dataset = config["bigquery"]["dataset"]
test_view = config["bigquery"]["features_test_view"]
client = bigquery.Client(project=project)
query = f"SELECT transaction_id FROM `{project}.{dataset}.{test_view}` LIMIT 1"
rows = list(client.query(query).result())
print(rows[0].transaction_id if rows else "")
