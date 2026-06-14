"""Build and run parameterized score_features.sql against BigQuery."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from google.cloud import bigquery

from src.automl_utils import bq_table_id, excluded_columns
from src.config import PROJECT_ROOT
from src.deploy_views import render_sql, template_variables
from src.serving.schemas import RawTransactionV2

SCORE_FEATURES_SQL = PROJECT_ROOT / "sql" / "serving" / "score_features.sql"

RAW_QUERY_FIELDS: tuple[str, ...] = (
    "transaction_id",
    "timestamp",
    "sender_account_id",
    "sender_counterparty_account_id",
    "receiver_account_id",
    "receiver_counterparty_account_id",
    "amount",
    "transaction_currency",
    "transaction_type",
    "channel",
    "channel_indicator",
    "terminal_id",
    "atm_id",
    "merchant_city",
    "merchant_state",
    "merchant_country",
    "merchant_legal_name",
    "merchant_dba_name",
    "pos_entry_mode",
    "payment_reference",
    "memo",
    "payment_sender_country",
    "payment_receiver_country",
    "settlement_currency",
    "settlement_amount",
    "fx_rate",
    "settlement_date",
    "settlement_status",
    "clearing_system",
    "correspondent_bic",
)


def _bq_param_type(field: str, value: Any) -> str:
    if field == "timestamp":
        return "TIMESTAMP"
    if field == "settlement_date":
        return "DATE"
    if field in {"amount", "settlement_amount", "fx_rate"}:
        return "FLOAT64"
    return "STRING"


def build_query_parameters(txn: RawTransactionV2) -> list[bigquery.ScalarQueryParameter]:
    params: list[bigquery.ScalarQueryParameter] = []
    payload = txn.to_param_dict()
    for field in RAW_QUERY_FIELDS:
        value = payload.get(field)
        param_type = _bq_param_type(field, value)
        params.append(bigquery.ScalarQueryParameter(field, param_type, value))
    return params


def render_score_features_sql(config: dict) -> str:
    template = SCORE_FEATURES_SQL.read_text(encoding="utf-8")
    return render_sql(template, template_variables(config))


def model_feature_columns(config: dict, all_columns: list[str]) -> list[str]:
    skip = set(excluded_columns(config))
    automl = config.get("automl", {})
    skip.add(automl.get("target_column", "is_fraud"))
    return [name for name in all_columns if name not in skip]


def score_features(
    client: bigquery.Client,
    config: dict,
    txn: RawTransactionV2,
) -> dict[str, Any]:
    """Return one feature row for the incoming transaction."""
    txn.validate_xor_legs()
    sql = render_score_features_sql(config)
    job_config = bigquery.QueryJobConfig(query_parameters=build_query_parameters(txn))
    rows = list(client.query(sql, job_config=job_config).result())
    if not rows:
        raise RuntimeError(f"No feature row produced for transaction_id={txn.transaction_id}")
    row = rows[0]
    return dict(row.items())


def fetch_offline_features(
    client: bigquery.Client,
    config: dict,
    transaction_id: str,
) -> dict[str, Any] | None:
    """Load the precomputed training feature row for parity comparison."""
    view_id = bq_table_id(config, config["bigquery"]["features_view"])
    query = f"SELECT * FROM `{view_id}` WHERE transaction_id = @transaction_id LIMIT 1"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("transaction_id", "STRING", transaction_id),
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return None
    return dict(rows[0].items())


def compare_feature_rows(
    online: dict[str, Any],
    offline: dict[str, Any],
    *,
    compare_columns: list[str] | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare online vs offline feature values; return parity summary."""
    keys = compare_columns or sorted(set(online) & set(offline))
    mismatched: list[dict[str, Any]] = []
    max_delta = 0.0

    for key in keys:
        left = online.get(key)
        right = offline.get(key)
        if left is None and right is None:
            continue
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta = abs(float(left) - float(right))
            max_delta = max(max_delta, delta)
            if delta > tolerance:
                mismatched.append({"column": key, "online": left, "offline": right, "delta": delta})
            continue
        if isinstance(left, datetime) and isinstance(right, datetime):
            if left != right:
                mismatched.append({"column": key, "online": left, "offline": right})
            continue
        if isinstance(left, date) and isinstance(right, date):
            if left != right:
                mismatched.append({"column": key, "online": left, "offline": right})
            continue
        if left != right:
            mismatched.append({"column": key, "online": left, "offline": right})

    return {
        "matched": not mismatched,
        "max_feature_delta": max_delta,
        "mismatched_columns": mismatched,
        "compared_columns": keys,
    }
