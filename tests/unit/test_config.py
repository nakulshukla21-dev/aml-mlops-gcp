"""Unit tests for config loading and profile resolution."""

from __future__ import annotations

import pytest

from src.config import (
    CONFIG_PATH,
    DEV_CONFIG_PATH,
    load_config,
    resolve_config_path,
)


def test_resolve_config_path_train_and_dev():
    assert resolve_config_path("train") == CONFIG_PATH
    assert resolve_config_path("dev") == DEV_CONFIG_PATH


def test_resolve_config_path_explicit_override():
    assert resolve_config_path("train", config=DEV_CONFIG_PATH) == DEV_CONFIG_PATH


def test_resolve_config_path_unknown_profile():
    with pytest.raises(ValueError, match="Unknown profile 'staging'"):
        resolve_config_path("staging")


def test_train_and_dev_profiles_use_isolated_resources():
    train = load_config(CONFIG_PATH)
    dev = load_config(DEV_CONFIG_PATH)

    assert train["profile"] == "train"
    assert dev["profile"] == "dev"

    assert train["bigquery"]["raw_table"] != dev["bigquery"]["raw_table"]
    assert train["storage"]["raw_prefix"] != dev["storage"]["raw_prefix"]
    assert dev["bigquery"]["features_view"].endswith("_dev")
    assert not train["bigquery"]["features_view"].endswith("_dev")
