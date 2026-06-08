"""Unit tests for SQL view deployment helpers."""

from __future__ import annotations

import pytest

from src.deploy_views import render_sql, split_statements, template_variables
from src.config import load_config, CONFIG_PATH, DEV_CONFIG_PATH


def test_render_sql_substitutes_variables():
    template = "SELECT * FROM `{{project_id}}.{{dataset}}.{{raw_table}}`"
    rendered = render_sql(
        template,
        {
            "project_id": "my-project",
            "dataset": "aml_mlops",
            "raw_table": "raw_transactions",
        },
    )
    assert rendered == "SELECT * FROM `my-project.aml_mlops.raw_transactions`"


def test_render_sql_raises_on_unresolved_variables():
    template = "SELECT * FROM `{{project_id}}.{{dataset}}.{{missing_var}}`"
    with pytest.raises(ValueError, match="Unresolved SQL template variables: missing_var"):
        render_sql(template, {"project_id": "p", "dataset": "d"})


def test_split_statements_ignores_empty_parts():
    sql = "CREATE VIEW a AS SELECT 1;\n\n; SELECT 2;"
    assert split_statements(sql) == ["CREATE VIEW a AS SELECT 1", "SELECT 2"]


def test_template_variables_from_train_config():
    config = load_config(CONFIG_PATH)
    variables = template_variables(config)

    assert variables["project_id"] == config["gcp"]["project_id"]
    assert variables["dataset"] == config["bigquery"]["dataset"]
    assert variables["raw_table"] == "raw_transactions"
    assert variables["features_base_view"] == "features_base"
    assert variables["features_view"] == "features_training"
    assert variables["train_end"] == "2024-09-30"


def test_template_variables_from_dev_config():
    config = load_config(DEV_CONFIG_PATH)
    variables = template_variables(config)

    assert variables["raw_table"] == "raw_transactions_dev"
    assert variables["features_base_view"] == "features_base_dev"
    assert variables["features_velocity_view"] == "features_velocity_dev"
    assert variables["features_network_view"] == "features_network_dev"
    assert variables["features_view"] == "features_training_dev"


def test_feature_sql_templates_render_without_leftover_placeholders():
    from src.deploy_views import VIEW_SCRIPTS

    config = load_config(CONFIG_PATH)
    variables = template_variables(config)

    for script_path in VIEW_SCRIPTS:
        template = script_path.read_text(encoding="utf-8")
        rendered = render_sql(template, variables)
        assert "{{" not in rendered, f"Unresolved placeholders in {script_path.name}"
