"""Unit tests for deployment config helpers."""

from __future__ import annotations

from src.automl_utils import deploy_artifact_path, deployment_config
from src.config import load_config, CONFIG_PATH, DEV_CONFIG_PATH


def test_deployment_config_train_profile():
    config = load_config(CONFIG_PATH)
    deploy = deployment_config(config)

    assert deploy["endpoint_display_name"] == "aml-fraud-endpoint"
    assert deploy["machine_type"] == "n1-standard-2"
    assert deploy["min_replica_count"] == 1


def test_deployment_config_dev_profile_isolated():
    config = load_config(DEV_CONFIG_PATH)
    deploy = deployment_config(config)

    assert deploy["endpoint_display_name"] == "aml-fraud-endpoint-dev"
    assert deploy_artifact_path(config).name == "deploy_dev.json"
