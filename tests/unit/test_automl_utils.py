"""Unit tests for AutoML helper utilities."""

from __future__ import annotations

from src.automl_utils import (
    bq_source_uri,
    bq_table_id,
    build_column_specs,
    excluded_columns,
    template_variables,
)
from src.config import load_config, CONFIG_PATH, DEV_CONFIG_PATH
from src.deploy_views import render_sql

AUTOML_INPUT_TEMPLATE = (
    "CREATE VIEW `{{project_id}}.{{dataset}}.{{automl_input_view}}` AS "
    "SELECT * FROM `{{project_id}}.{{dataset}}.{{features_train_view}}`;"
)


def test_excluded_columns_include_join_keys_and_split():
    config = load_config(CONFIG_PATH)
    excluded = excluded_columns(config)

    assert "transaction_id" in excluded
    assert "sender_account" in excluded
    assert "receiver_account" in excluded
    assert "ml_split" in excluded
    assert "is_fraud" not in excluded


def test_build_column_specs_omits_excluded_and_target():
    config = load_config(CONFIG_PATH)
    columns = [
        "transaction_id",
        "timestamp",
        "amount",
        "channel",
        "is_fraud",
        "ml_split",
    ]
    specs = build_column_specs(columns, config)

    assert set(specs) == {"amount", "channel"}
    assert all(value == "auto" for value in specs.values())


def test_bq_source_uri_format():
    config = load_config(CONFIG_PATH)
    uri = bq_source_uri(config, "features_automl")
    table_id = bq_table_id(config, "features_automl")

    assert uri == f"bq://{table_id}"
    assert uri.startswith("bq://aml-mlops-demo-498203.aml_mlops.")


def test_template_variables_include_automl_view_names():
    train_vars = template_variables(load_config(CONFIG_PATH))
    dev_vars = template_variables(load_config(DEV_CONFIG_PATH))

    assert train_vars["automl_input_view"] == "features_automl"
    assert dev_vars["automl_input_view"] == "features_automl_dev"
    assert dev_vars["features_train_view"] == "features_train_dev"


def test_automl_input_sql_renders_for_train_and_dev():
    for config_path, expected_view in (
        (CONFIG_PATH, "features_automl"),
        (DEV_CONFIG_PATH, "features_automl_dev"),
    ):
        rendered = render_sql(AUTOML_INPUT_TEMPLATE, template_variables(load_config(config_path)))
        assert expected_view in rendered
        assert "{{" not in rendered
