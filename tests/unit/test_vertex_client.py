"""Unit tests for Vertex client helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import DEV_CONFIG_PATH, load_config
from src.serving.vertex_client import build_prediction_instance, risk_level


def test_build_prediction_instance_selects_columns():
    feature_row = {
        "amount": 100.0,
        "txn_count_24h": 3,
        "sender_is_pep": True,
        "sender_account_age_days": 120,
        "timestamp": datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
    }
    instance = build_prediction_instance(
        feature_row, ["amount", "txn_count_24h", "sender_is_pep", "sender_account_age_days"]
    )
    assert instance["amount"] == 100.0
    assert instance["txn_count_24h"] == "3"
    assert instance["sender_is_pep"] is True
    assert instance["sender_account_age_days"] == "120"


def test_feature_column_names_prefers_refreshed_artifact(tmp_path, monkeypatch):
    from src.automl_utils import save_run_artifact
    from src.config import DEV_CONFIG_PATH, load_config
    from src.serving.vertex_client import feature_column_names

    config = load_config(DEV_CONFIG_PATH)
    artifact_file = tmp_path / "automl_dev.json"
    save_run_artifact(
        artifact_file,
        {
            "column_specs": {
                "amount": "auto",
                "sender_is_bank_client": "auto",
                "txn_count_24h": "auto",
            }
        },
    )
    monkeypatch.setattr("src.serving.vertex_client.artifact_path", lambda _: artifact_file)

    feature_row = {
        "amount": 100.0,
        "sender_is_bank_client": True,
        "txn_count_24h": 2,
        "extra_ignored": "x",
    }
    cols = feature_column_names(config, feature_row)
    assert cols == ["amount", "sender_is_bank_client", "txn_count_24h"]
