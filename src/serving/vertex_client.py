"""Vertex AI endpoint client for online fraud scoring."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from google.cloud import aiplatform

from src.automl_utils import (
    artifact_path,
    automl_config,
    deploy_artifact_path,
    load_run_artifact,
)
from src.predictions import parse_online_prediction
from src.serving.feature_builder import model_feature_columns


def serving_config(config: dict) -> dict:
    return config.get("serving", {})


def feature_column_names(config: dict, feature_row: dict[str, Any]) -> list[str]:
    """Prefer refreshed automl artifact column_specs; fall back to the live feature row."""
    row_cols = model_feature_columns(config, list(feature_row.keys()))
    path = artifact_path(config)
    if path.exists():
        specs = load_run_artifact(path).get("column_specs") or {}
        if specs:
            artifact_cols = list(specs.keys())
            if all(column in feature_row for column in artifact_cols):
                return artifact_cols
    return row_cols


def _serialize_feature_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return value
    return value


def build_prediction_instance(
    feature_row: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    instance: dict[str, Any] = {}
    for column in columns:
        instance[column] = _serialize_feature_value(feature_row.get(column))
    return instance


def risk_level(score: float | None, config: dict) -> str:
    if score is None:
        return "Unknown"
    thresholds = serving_config(config).get("risk_thresholds", {})
    high = float(thresholds.get("high", 0.7))
    medium = float(thresholds.get("medium", 0.4))
    if score >= high:
        return "High"
    if score >= medium:
        return "Medium"
    return "Low"


def load_endpoint_resource_name(config: dict) -> str:
    path = deploy_artifact_path(config)
    if not path.exists():
        raise RuntimeError(
            f"No deployment artifact at {path}. Run: python -m src.deploy_model --profile {config.get('profile', 'dev')}"
        )
    deploy_artifact = load_run_artifact(path)
    endpoint = deploy_artifact.get("endpoint_resource_name")
    if not endpoint:
        raise RuntimeError(f"Deployment artifact missing endpoint_resource_name: {path}")
    return endpoint


def load_model_resource_name(config: dict) -> str:
    deploy_path = deploy_artifact_path(config)
    if deploy_path.exists():
        model = load_run_artifact(deploy_path).get("model_resource_name")
        if model:
            return model
    raise RuntimeError("Could not resolve model_resource_name from deploy artifact.")


class VertexScorer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.project_id = config["gcp"]["project_id"]
        self.region = config["gcp"]["region"]
        self.endpoint_resource_name = load_endpoint_resource_name(config)
        self.model_resource_name = load_model_resource_name(config)
        self.target_column = automl_config(config).get("target_column", "is_fraud")
        self.prediction_threshold = float(
            serving_config(config).get("prediction_threshold", 0.5)
        )
        aiplatform.init(project=self.project_id, location=self.region)
        self._endpoint = aiplatform.Endpoint(self.endpoint_resource_name)

    def predict(self, feature_row: dict[str, Any]) -> dict[str, Any]:
        columns = feature_column_names(self.config, feature_row)
        instance = build_prediction_instance(feature_row, columns)
        response = self._endpoint.predict(instances=[instance])
        predictions = getattr(response, "predictions", None) or response
        if not predictions:
            raise RuntimeError("Vertex endpoint returned no predictions.")
        predicted_is_fraud, fraud_score = parse_online_prediction(
            predictions[0],
            target_column=self.target_column,
            threshold=self.prediction_threshold,
        )
        features_used = {column: feature_row.get(column) for column in columns}
        return {
            "predicted_is_fraud": predicted_is_fraud,
            "fraud_score": fraud_score,
            "risk_level": risk_level(fraud_score, self.config),
            "features_used": features_used,
            "model_resource_name": self.model_resource_name,
            "endpoint_resource_name": self.endpoint_resource_name,
        }
