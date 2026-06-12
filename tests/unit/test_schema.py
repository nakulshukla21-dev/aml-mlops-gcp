"""Unit tests for schema alignment between generator and BigQuery."""

from __future__ import annotations

import json
from datetime import datetime

from src.config import PROJECT_ROOT
from src.generate_synthetic_data import (
    POST_LOAD_COLUMNS,
    SyntheticAMLGenerator,
    csv_column_order,
)
from src.load_to_bigquery import csv_load_schema, load_schema

SCHEMA_PATH = PROJECT_ROOT / "schemas" / "raw_transactions_v2.json"


def test_csv_column_order_matches_bq_csv_schema():
    bq_csv_names = [field.name for field in csv_load_schema(load_schema(SCHEMA_PATH))]
    assert csv_column_order() == bq_csv_names


def test_schema_excludes_post_load_columns_from_csv_order():
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        all_fields = [field["name"] for field in json.load(f)]

    csv_columns = csv_column_order()
    assert "ingested_at" in all_fields
    assert "ingested_at" not in csv_columns
    assert set(csv_columns) == set(all_fields) - POST_LOAD_COLUMNS


def test_generator_output_columns_match_schema():
    generator = SyntheticAMLGenerator(
        n_transactions=50,
        fraud_rate=0.1,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        seed=99,
    )
    df = generator.generate()

    assert set(df.columns) == set(csv_column_order())
    assert len(df) == 50
    assert "ingested_at" not in df.columns
    assert df["transaction_id"].is_unique


def test_generator_account_xor_constraint():
    generator = SyntheticAMLGenerator(
        n_transactions=100,
        fraud_rate=0.15,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        seed=7,
    )
    df = generator.generate()

    sender_ok = df["sender_account_id"].notna() ^ df["sender_counterparty_account_id"].notna()
    receiver_ok = df["receiver_account_id"].notna() ^ df["receiver_counterparty_account_id"].notna()
    assert sender_ok.all()
    assert receiver_ok.all()
