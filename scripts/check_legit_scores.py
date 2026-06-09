"""Inspect fraud scores for legitimate-typology fraud rows."""

from google.cloud import bigquery

PREDICTIONS = "aml-mlops-demo-498203.aml_mlops.predictions_2026_06_09T09_19_26_025Z_833"
EVAL_VIEW = "aml-mlops-demo-498203.aml_mlops.features_eval_dev"

QUERY = f"""
WITH preds AS (
  SELECT
    e.transaction_id,
    e.typology,
    e.is_fraud,
    (
      SELECT AS VALUE score
      FROM UNNEST(p.predicted_is_fraud.classes) AS class WITH OFFSET i
      JOIN UNNEST(p.predicted_is_fraud.scores) AS score WITH OFFSET j ON i = j
      WHERE LOWER(class) IN ('true', '1')
      LIMIT 1
    ) AS fraud_score,
    (
      SELECT AS VALUE LOWER(class) IN ('true', '1')
      FROM UNNEST(p.predicted_is_fraud.classes) AS class WITH OFFSET i
      JOIN UNNEST(p.predicted_is_fraud.scores) AS score WITH OFFSET j ON i = j
      ORDER BY score DESC
      LIMIT 1
    ) AS predicted_fraud
  FROM `{PREDICTIONS}` AS p
  JOIN `{EVAL_VIEW}` AS e USING (transaction_id)
)
SELECT
  typology,
  COUNTIF(is_fraud) AS fraud_rows,
  COUNTIF(is_fraud AND predicted_fraud) AS tp,
  ROUND(MIN(IF(is_fraud, fraud_score, NULL)), 4) AS min_fraud_score,
  ROUND(MAX(IF(is_fraud, fraud_score, NULL)), 4) AS max_fraud_score,
  ROUND(AVG(IF(is_fraud, fraud_score, NULL)), 4) AS avg_fraud_score
FROM preds
GROUP BY typology
ORDER BY fraud_rows DESC
"""

LEGIT_QUERY = f"""
WITH preds AS (
  SELECT
    e.is_fraud,
    (
      SELECT AS VALUE score
      FROM UNNEST(p.predicted_is_fraud.classes) AS class WITH OFFSET i
      JOIN UNNEST(p.predicted_is_fraud.scores) AS score WITH OFFSET j ON i = j
      WHERE LOWER(class) IN ('true', '1')
      LIMIT 1
    ) AS fraud_score
  FROM `{PREDICTIONS}` AS p
  JOIN `{EVAL_VIEW}` AS e USING (transaction_id)
  WHERE e.typology = 'legitimate' AND e.is_fraud
)
SELECT
  COUNT(*) AS n,
  ROUND(MIN(fraud_score), 4) AS min_score,
  ROUND(MAX(fraud_score), 4) AS max_score,
  ROUND(AVG(fraud_score), 4) AS avg_score,
  COUNTIF(fraud_score >= 0.5) AS at_or_above_50pct,
  COUNTIF(fraud_score >= 0.1) AS at_or_above_10pct
FROM preds
"""

client = bigquery.Client(project="aml-mlops-demo-498203")
print("By typology:")
for row in client.query(QUERY).result():
    print(dict(row))

print("\nLegitimate-typology fraud rows score distribution:")
for row in client.query(LEGIT_QUERY).result():
    print(dict(row))
