"""Load dimension tables from BigQuery for party resolution."""

from __future__ import annotations

from google.cloud import bigquery

from src.automl_utils import bq_table_id
from src.serving.party_resolver import DimensionTables


def load_dimension_tables(client: bigquery.Client, config: dict) -> DimensionTables:
    bq = config["bigquery"]
    table_keys = {
        "dim_account": bq["dim_account_table"],
        "dim_customer": bq["dim_customer_table"],
        "dim_counterparty": bq["dim_counterparty_table"],
        "dim_counterparty_account": bq["dim_counterparty_account_table"],
    }
    frames = {}
    for key, table_name in table_keys.items():
        table_id = bq_table_id(config, table_name)
        frames[key] = client.query(f"SELECT * FROM `{table_id}`").to_dataframe()
    return DimensionTables(
        dim_account=frames["dim_account"],
        dim_customer=frames["dim_customer"],
        dim_counterparty=frames["dim_counterparty"],
        dim_counterparty_account=frames["dim_counterparty_account"],
    )
