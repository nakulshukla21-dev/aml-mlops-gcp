"""Unit tests for BigQuery load helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from google.cloud import bigquery

from src.load_to_bigquery import (
    POST_LOAD_COLUMNS,
    SCHEMA_PATH,
    csv_load_schema,
    load_schema,
    recreate_table,
)


def test_csv_load_schema_excludes_ingested_at():
    table_schema = load_schema(SCHEMA_PATH)
    csv_schema = csv_load_schema(table_schema)

    csv_names = {field.name for field in csv_schema}
    table_names = {field.name for field in table_schema}

    assert "ingested_at" not in csv_names
    assert "ingested_at" in table_names
    assert POST_LOAD_COLUMNS == frozenset({"ingested_at"})


def test_recreate_table_drops_and_creates_with_partitioning():
    client = MagicMock()
    client.delete_table.return_value = None
    client.create_table.return_value = bigquery.Table("proj.ds.raw_transactions")

    schema = load_schema(SCHEMA_PATH)
    recreate_table(client, "proj.ds.raw_transactions", schema)

    client.delete_table.assert_called_once_with("proj.ds.raw_transactions", not_found_ok=True)
    client.create_table.assert_called_once()

    created_table = client.create_table.call_args[0][0]
    assert created_table.time_partitioning.field == "timestamp"
    assert created_table.clustering_fields == [
        "sender_account",
        "receiver_account",
        "is_fraud",
    ]
