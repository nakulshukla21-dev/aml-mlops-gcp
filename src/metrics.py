"""Compute and serialize model evaluation metrics."""

from __future__ import annotations

from datetime import datetime, timezone

from google.cloud import bigquery

from src.automl_utils import automl_config, bq_table_id


def classification_metrics(
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
) -> dict[str, float]:
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(
    client: bigquery.Client,
    predictions_table: str,
    config: dict,
    *,
    predicted_col: str,
    predicted_positive: str,
) -> dict:
    automl = automl_config(config)
    bq = config["bigquery"]
    target_column = automl.get("target_column", "is_fraud")
    eval_view = bq_table_id(config, bq["features_eval_view"])
    profile = config.get("profile", "train")

    overall_query = f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNTIF(e.{target_column}) AS fraud_rows,
          COUNTIF({predicted_positive}) AS predicted_fraud_rows,
          COUNTIF(e.{target_column} AND {predicted_positive}) AS true_positives,
          COUNTIF(NOT e.{target_column} AND NOT ({predicted_positive})) AS true_negatives,
          COUNTIF(e.{target_column} AND NOT ({predicted_positive})) AS false_negatives,
          COUNTIF(NOT e.{target_column} AND {predicted_positive}) AS false_positives
        FROM `{predictions_table}` AS p
        JOIN `{eval_view}` AS e
          USING (transaction_id)
    """
    row = list(client.query(overall_query).result())[0]
    overall_scores = classification_metrics(
        row.true_positives,
        row.false_positives,
        row.true_negatives,
        row.false_negatives,
    )

    typology_query = f"""
        SELECT
          e.typology,
          COUNT(*) AS n_rows,
          COUNTIF(e.{target_column}) AS fraud_rows,
          COUNTIF({predicted_positive}) AS predicted_fraud_rows,
          COUNTIF(e.{target_column} AND {predicted_positive}) AS true_positives,
          COUNTIF(e.{target_column} AND NOT ({predicted_positive})) AS false_negatives,
          COUNTIF(NOT e.{target_column} AND {predicted_positive}) AS false_positives
        FROM `{predictions_table}` AS p
        JOIN `{eval_view}` AS e
          USING (transaction_id)
        GROUP BY e.typology
        ORDER BY n_rows DESC
    """
    by_typology = []
    for typology_row in client.query(typology_query).result():
        scores = classification_metrics(
            typology_row.true_positives,
            typology_row.false_positives,
            0,
            typology_row.false_negatives,
        )
        by_typology.append(
            {
                "typology": typology_row.typology,
                "n_rows": typology_row.n_rows,
                "fraud_rows": typology_row.fraud_rows,
                "predicted_fraud_rows": typology_row.predicted_fraud_rows,
                "true_positives": typology_row.true_positives,
                "false_positives": typology_row.false_positives,
                "false_negatives": typology_row.false_negatives,
                **scores,
            }
        )

    return {
        "profile": profile,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "predictions_table": predictions_table,
        "test_view": bq["features_test_view"],
        "eval_view": bq["features_eval_view"],
        "target_column": target_column,
        "predicted_column": predicted_col,
        "overall": {
            "n_rows": row.n_rows,
            "fraud_rows": row.fraud_rows,
            "predicted_fraud_rows": row.predicted_fraud_rows,
            "true_positives": row.true_positives,
            "false_positives": row.false_positives,
            "true_negatives": row.true_negatives,
            "false_negatives": row.false_negatives,
            **overall_scores,
        },
        "by_typology": by_typology,
    }


def print_metrics_report(metrics: dict) -> None:
    overall = metrics["overall"]
    print("\nOverall test metrics:")
    print(f"  Rows: {overall['n_rows']:,}")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1:        {overall['f1']:.4f}")
    print(
        "  Confusion: "
        f"TP={overall['true_positives']:,} "
        f"FP={overall['false_positives']:,} "
        f"TN={overall['true_negatives']:,} "
        f"FN={overall['false_negatives']:,}"
    )

    print("\nPer-typology test metrics:")
    print(f"  {'typology':<18} {'n':>8} {'fraud':>8} {'prec':>8} {'recall':>8} {'f1':>8}")
    for row in metrics["by_typology"]:
        print(
            f"  {row['typology']:<18} "
            f"{row['n_rows']:>8,} "
            f"{row['fraud_rows']:>8,} "
            f"{row['precision']:>8.3f} "
            f"{row['recall']:>8.3f} "
            f"{row['f1']:>8.3f}"
        )
