"""Unit tests for prediction log helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config import load_config, CONFIG_PATH, DEV_CONFIG_PATH
from src.prediction_logging import (
    prediction_log_table_id,
    prediction_log_table_name,
)


def test_prediction_log_table_name_from_config():
    train = load_config(CONFIG_PATH)
    dev = load_config(DEV_CONFIG_PATH)
    assert prediction_log_table_name(train) == "prediction_log"
    assert prediction_log_table_name(dev) == "prediction_log_dev"


def test_prediction_log_table_id_is_fully_qualified():
    config = load_config(CONFIG_PATH)
    assert prediction_log_table_id(config) == (
        "aml-mlops-demo-498203.aml_mlops.prediction_log"
    )


@patch("src.prediction_logging.ensure_prediction_log_table")
@patch("src.prediction_logging.count_existing_log_rows", return_value=5)
def test_log_batch_predictions_skips_when_already_logged(mock_count, mock_ensure):
    from src.prediction_logging import log_batch_predictions

    client = MagicMock()
    config = load_config(CONFIG_PATH)

    inserted = log_batch_predictions(
        client,
        config,
        "aml-mlops-demo-498203.aml_mlops.automl_predictions_123",
        model_resource_name="projects/p/locations/us-central1/models/1",
    )

    assert inserted == 0
    mock_ensure.assert_called_once()
    client.query.assert_not_called()
