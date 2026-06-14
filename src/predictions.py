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
