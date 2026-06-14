"""Unit tests for serving SQL and feature builder helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.config import DEV_CONFIG_PATH, load_config
from src.deploy_views import render_sql, template_variables
from src.serving.feature_builder import (
    RAW_QUERY_FIELDS,
    build_query_parameters,
    compare_feature_rows,
    render_score_features_sql,
)
from src.serving.schemas import RawTransactionV2


def test_score_features_sql_renders_without_leftover_placeholders():
    config = load_config(DEV_CONFIG_PATH)
    rendered = render_score_features_sql(config)
    assert "{{" not in rendered
    assert "features_base_dev" in rendered
    assert "@transaction_id" in rendered


def test_render_sql_substitutes_serving_table_names():
    config = load_config(DEV_CONFIG_PATH)
    variables = template_variables(config)
    rendered = render_score_features_sql(config)
    assert variables["dim_account_table"] in rendered
    assert variables["features_base_view"] in rendered


def test_build_query_parameters_includes_all_raw_fields():
    txn = RawTransactionV2(
        transaction_id="TXN-001",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        sender_account_id="BAUS0000100",
        receiver_counterparty_account_id="CPAKY00001",
        amount=100.0,
        transaction_currency="USD",
        transaction_type="payment",
        channel="wire",
        payment_sender_country="US",
        payment_receiver_country="KY",
        settlement_currency="USD",
        settlement_amount=100.0,
        settlement_status="settled",
        settlement_date=date(2024, 6, 15),
    )
    params = build_query_parameters(txn)
    assert {param.name for param in params} == set(RAW_QUERY_FIELDS)


def test_compare_feature_rows_flags_numeric_mismatch():
    online = {"amount": 100.0, "txn_count_24h": 3}
    offline = {"amount": 100.0, "txn_count_24h": 4}
    result = compare_feature_rows(online, offline, compare_columns=["amount", "txn_count_24h"])
    assert result["matched"] is False
    assert result["max_feature_delta"] == 1.0
    assert result["mismatched_columns"][0]["column"] == "txn_count_24h"


def test_raw_transaction_xor_validation():
    txn = RawTransactionV2(
        transaction_id="TXN-001",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        sender_account_id="BAUS0000100",
        sender_counterparty_account_id="CPAKY00001",
        receiver_counterparty_account_id="CPAKY00002",
        amount=100.0,
        transaction_currency="USD",
        transaction_type="payment",
        channel="wire",
        payment_sender_country="US",
        payment_receiver_country="KY",
        settlement_currency="USD",
        settlement_amount=100.0,
        settlement_status="settled",
    )
    with pytest.raises(ValueError, match="Exactly one sender"):
        txn.validate_xor_legs()
