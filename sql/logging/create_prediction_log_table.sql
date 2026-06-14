-- Append-only audit log for model predictions (batch and online).

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.{{prediction_log_table}}` (
  prediction_id STRING NOT NULL,
  transaction_id STRING NOT NULL,
  scored_at TIMESTAMP NOT NULL,
  prediction_source STRING NOT NULL,
  model_resource_name STRING NOT NULL,
  model_display_name STRING,
  batch_job_display_name STRING,
  endpoint_resource_name STRING,
  profile STRING,
  feature_view STRING,
  predicted_is_fraud BOOL NOT NULL,
  fraud_score FLOAT64,
  actual_is_fraud BOOL,
  raw_predictions_table STRING,
  logged_at TIMESTAMP NOT NULL
)
PARTITION BY DATE(scored_at)
CLUSTER BY transaction_id, model_resource_name, prediction_source;
