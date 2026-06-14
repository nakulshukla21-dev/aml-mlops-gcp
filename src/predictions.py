"""Parse Vertex AutoML batch prediction output columns."""

from __future__ import annotations

from google.cloud import bigquery


def prediction_column(schema: list[bigquery.SchemaField], target_column: str) -> str:
    candidates = [
        f"predicted_{target_column}",
        f"predicted_{target_column}.bool",
        "predicted_label",
    ]
    names = {field.name for field in schema}
    for candidate in candidates:
        if candidate in names:
            return candidate
    predicted = [name for name in names if name.startswith("predicted_")]
    if len(predicted) == 1:
        return predicted[0]
    raise ValueError(f"Could not find prediction column in schema: {sorted(names)}")


def _schema_field(
    schema: list[bigquery.SchemaField],
    predicted_col: str,
) -> bigquery.SchemaField:
    field = next((item for item in schema if item.name == predicted_col), None)
    if field is None:
        raise ValueError(f"Prediction column not found in schema: {predicted_col}")
    return field


def predicted_positive_expr(
    schema: list[bigquery.SchemaField],
    predicted_col: str,
    alias: str = "p",
) -> str:
    """SQL boolean expression for the model's positive-class prediction."""
    field = _schema_field(schema, predicted_col)
    if field.field_type in {"BOOLEAN", "BOOL"}:
        return f"{alias}.{predicted_col}"
    if field.field_type == "RECORD":
        return f"""(
          SELECT AS VALUE LOWER(class) IN ('true', '1')
          FROM UNNEST({alias}.{predicted_col}.classes) AS class WITH OFFSET i
          JOIN UNNEST({alias}.{predicted_col}.scores) AS score WITH OFFSET j
            ON i = j
          ORDER BY score DESC
          LIMIT 1
        )"""
    raise ValueError(
        f"Unsupported prediction column type for {predicted_col}: {field.field_type}"
    )


def fraud_score_expr(
    schema: list[bigquery.SchemaField],
    predicted_col: str,
    alias: str = "p",
) -> str:
    """SQL expression for positive-class probability."""
    field = _schema_field(schema, predicted_col)
    if field.field_type in {"BOOLEAN", "BOOL"}:
        return f"IF({alias}.{predicted_col}, 1.0, 0.0)"
    if field.field_type == "RECORD":
        return f"""(
          SELECT AS VALUE score
          FROM UNNEST({alias}.{predicted_col}.classes) AS class WITH OFFSET i
          JOIN UNNEST({alias}.{predicted_col}.scores) AS score WITH OFFSET j
            ON i = j
          WHERE LOWER(class) IN ('true', '1')
          LIMIT 1
        )"""
    raise ValueError(
        f"Unsupported prediction column type for {predicted_col}: {field.field_type}"
    )


def _positive_class_index(classes: list) -> int:
    normalized = [str(item).lower() for item in classes]
    for label in ("true", "1", "yes"):
        if label in normalized:
            return normalized.index(label)
    return max(range(len(classes)), key=lambda idx: float(idx))


def parse_online_prediction(
    prediction: dict,
    *,
    target_column: str = "is_fraud",
    threshold: float = 0.5,
) -> tuple[bool, float | None]:
    """
    Parse one Vertex online prediction dict into (predicted_positive, fraud_score).

    Supports AutoML classification structs and boolean predictions.
    """
    predicted_key = f"predicted_{target_column}"
    value = prediction.get(predicted_key)
    if value is None:
        predicted_keys = [key for key in prediction if key.startswith("predicted_")]
        if len(predicted_keys) == 1:
            value = prediction[predicted_keys[0]]
        elif "classes" in prediction and "scores" in prediction:
            value = prediction
        else:
            raise ValueError(f"Could not find prediction field in response: {prediction}")

    if isinstance(value, bool):
        score = 1.0 if value else 0.0
        return value, score

    if isinstance(value, dict):
        classes = value.get("classes") or value.get("displayNames") or []
        scores = value.get("scores") or []
        if not classes or not scores:
            raise ValueError(f"Prediction struct missing classes/scores: {value}")
        idx = _positive_class_index(list(classes))
        score = float(scores[idx])
        return score >= threshold, score

    if isinstance(value, (int, float)):
        score = float(value)
        return score >= threshold, score

    raise ValueError(f"Unsupported online prediction payload: {value!r}")
