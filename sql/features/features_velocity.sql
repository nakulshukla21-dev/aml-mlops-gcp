-- Point-in-time velocity features per sender_account.
-- Each row only uses transactions at or before its own timestamp.

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_velocity_view}}` AS
WITH base AS (
  SELECT
    transaction_id,
    sender_account,
    receiver_account,
    timestamp,
    amount
  FROM `{{project_id}}.{{dataset}}.{{features_base_view}}`
),
windowed AS (
  SELECT
    transaction_id,
    COUNT(*) OVER (
      PARTITION BY sender_account
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    ) AS txn_count_24h,
    COUNT(*) OVER (
      PARTITION BY sender_account
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS txn_count_7d,
    SUM(amount) OVER (
      PARTITION BY sender_account
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    ) AS total_amount_24h
  FROM base
),
distinct_receivers AS (
  SELECT
    current_txn.transaction_id,
    COUNT(DISTINCT prior_txn.receiver_account) AS unique_receivers_24h
  FROM base AS current_txn
  JOIN base AS prior_txn
    ON current_txn.sender_account = prior_txn.sender_account
   AND prior_txn.timestamp BETWEEN TIMESTAMP_SUB(current_txn.timestamp, INTERVAL 24 HOUR)
                               AND current_txn.timestamp
  GROUP BY current_txn.transaction_id
)
SELECT
  windowed.transaction_id,
  windowed.txn_count_24h,
  windowed.txn_count_7d,
  windowed.total_amount_24h,
  distinct_receivers.unique_receivers_24h
FROM windowed
JOIN distinct_receivers USING (transaction_id);
