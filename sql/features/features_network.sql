-- As-of network features: fan-out/fan-in counts use only history up to each timestamp.

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_network_view}}` AS
WITH base AS (
  SELECT
    transaction_id,
    sender_account,
    receiver_account,
    timestamp,
    sender_country,
    receiver_country
  FROM `{{project_id}}.{{dataset}}.{{features_base_view}}`
)
SELECT
  current_txn.transaction_id,
  (
    SELECT COUNT(DISTINCT prior.receiver_account)
    FROM base AS prior
    WHERE prior.sender_account = current_txn.sender_account
      AND prior.timestamp <= current_txn.timestamp
  ) AS sender_fan_out_as_of,
  (
    SELECT COUNT(DISTINCT prior.sender_account)
    FROM base AS prior
    WHERE prior.receiver_account = current_txn.receiver_account
      AND prior.timestamp <= current_txn.timestamp
  ) AS receiver_fan_in_as_of,
  current_txn.sender_country IN ('KY', 'VG', 'PA', 'AE')
    AND current_txn.receiver_country IN ('KY', 'VG', 'PA', 'AE') AS both_high_risk
FROM base AS current_txn;
