"""Unit tests for AutoML prediction column parsing."""

from __future__ import annotations

from google.cloud import bigquery

from src.predictions import (
    fraud_score_expr,
    parse_online_prediction,
    prediction_column,
    predicted_positive_expr,
)


def _struct_schema(name: str = "predicted_is_fraud") -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("transaction_id", "STRING"),
        bigquery.SchemaField(
            name,
            "RECORD",
            fields=[
                bigquery.SchemaField("classes", "STRING", mode="REPEATED"),
                bigquery.SchemaField("scores", "FLOAT", mode="REPEATED"),
            ],
        ),
    ]


def test_prediction_column_finds_struct_target():
    schema = _struct_schema()
    assert prediction_column(schema, "is_fraud") == "predicted_is_fraud"


def test_predicted_positive_expr_for_struct():
    schema = _struct_schema()
    expr = predicted_positive_expr(schema, "predicted_is_fraud", alias="p")
    assert "UNNEST(p.predicted_is_fraud.classes)" in expr
    assert "true" in expr


def test_fraud_score_expr_for_struct():
    schema = _struct_schema()
    expr = fraud_score_expr(schema, "predicted_is_fraud", alias="p")
    assert "SELECT AS VALUE score" in expr
    assert "UNNEST(p.predicted_is_fraud.scores)" in expr


def test_boolean_prediction_column_exprs():
    schema = [
        bigquery.SchemaField("transaction_id", "STRING"),
        bigquery.SchemaField("predicted_is_fraud", "BOOL"),
    ]
    assert predicted_positive_expr(schema, "predicted_is_fraud", alias="p") == "p.predicted_is_fraud"
    assert fraud_score_expr(schema, "predicted_is_fraud", alias="p") == "IF(p.predicted_is_fraud, 1.0, 0.0)"


def test_parse_online_prediction_struct():
    prediction = {
        "predicted_is_fraud": {
            "classes": ["false", "true"],
            "scores": [0.88, 0.12],
        }
    }
    predicted, score = parse_online_prediction(prediction, threshold=0.5)
    assert predicted is False
    assert score == 0.12


def test_parse_online_prediction_flat_classes_scores():
    prediction = {"classes": ["false", "true"], "scores": [0.97, 0.03]}
    predicted, score = parse_online_prediction(prediction, threshold=0.5)
    assert predicted is False
    assert score == 0.03
