-- Point-in-time feature row for a single incoming transaction.
-- Table names are templated via deploy_views.render_sql; row values use @query parameters.

WITH incoming_raw AS (
  SELECT
    @transaction_id AS transaction_id,
    @timestamp AS timestamp,
    DATE(@timestamp) AS txn_date,
    @sender_account_id AS sender_account_id,
    @sender_counterparty_account_id AS sender_counterparty_account_id,
    @receiver_account_id AS receiver_account_id,
    @receiver_counterparty_account_id AS receiver_counterparty_account_id,
    @amount AS amount,
    @transaction_currency AS transaction_currency,
    @transaction_type AS transaction_type,
    @channel AS channel,
    @channel_indicator AS channel_indicator,
    @terminal_id AS terminal_id,
    @atm_id AS atm_id,
    @merchant_city AS merchant_city,
    @merchant_state AS merchant_state,
    @merchant_country AS merchant_country,
    @merchant_legal_name AS merchant_legal_name,
    @merchant_dba_name AS merchant_dba_name,
    @pos_entry_mode AS pos_entry_mode,
    @payment_reference AS payment_reference,
    @memo AS memo,
    @payment_sender_country AS payment_sender_country,
    @payment_receiver_country AS payment_receiver_country,
    @settlement_currency AS settlement_currency,
    @settlement_amount AS settlement_amount,
    @fx_rate AS fx_rate,
    @settlement_date AS settlement_date,
    @settlement_status AS settlement_status,
    @clearing_system AS clearing_system,
    @correspondent_bic AS correspondent_bic
),
incoming_enriched AS (
  SELECT
    incoming_raw.*,
    COALESCE(incoming_raw.sender_account_id, incoming_raw.sender_counterparty_account_id) AS sender_account,
    COALESCE(incoming_raw.receiver_account_id, incoming_raw.receiver_counterparty_account_id) AS receiver_account,
    CASE
      WHEN UPPER(TRIM(incoming_raw.payment_sender_country)) IN ('USA', 'U.S.', 'US') THEN 'US'
      WHEN UPPER(TRIM(incoming_raw.payment_sender_country)) IN ('UK', 'U.K.', 'GB') THEN 'GB'
      WHEN UPPER(TRIM(incoming_raw.payment_sender_country)) IN ('DE', 'DEUTSCHLAND') THEN 'DE'
      WHEN UPPER(TRIM(incoming_raw.payment_sender_country)) IN ('FR', 'FRA') THEN 'FR'
      ELSE UPPER(TRIM(incoming_raw.payment_sender_country))
    END AS sender_country,
    CASE
      WHEN UPPER(TRIM(incoming_raw.payment_receiver_country)) IN ('USA', 'U.S.', 'US') THEN 'US'
      WHEN UPPER(TRIM(incoming_raw.payment_receiver_country)) IN ('UK', 'U.K.', 'GB') THEN 'GB'
      WHEN UPPER(TRIM(incoming_raw.payment_receiver_country)) IN ('DE', 'DEUTSCHLAND') THEN 'DE'
      WHEN UPPER(TRIM(incoming_raw.payment_receiver_country)) IN ('FR', 'FRA') THEN 'FR'
      ELSE UPPER(TRIM(incoming_raw.payment_receiver_country))
    END AS receiver_country,
    sa.opened_date AS sender_bank_opened_date,
    sca.first_seen_date AS sender_counterparty_first_seen,
    ra.opened_date AS receiver_bank_opened_date,
    rca.first_seen_date AS receiver_counterparty_first_seen,
    sc.risk_rating AS sender_risk_rating,
    sc.is_pep AS sender_is_pep,
    sc.is_shell_entity AS sender_is_shell_entity,
    rc.risk_rating AS receiver_risk_rating,
    rc.is_pep AS receiver_is_pep,
    rc.is_shell_entity AS receiver_is_shell_entity,
    scp.is_shell_entity AS sender_counterparty_is_shell,
    rcp.is_shell_entity AS receiver_counterparty_is_shell,
    scp.counterparty_role AS sender_counterparty_role,
    rcp.counterparty_role AS receiver_counterparty_role,
    incoming_raw.sender_account_id IS NOT NULL AS sender_is_bank_client,
    incoming_raw.receiver_account_id IS NOT NULL AS receiver_is_bank_client
  FROM incoming_raw
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_account_table}}` AS sa
    ON incoming_raw.sender_account_id = sa.account_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_customer_table}}` AS sc
    ON sa.customer_id = sc.customer_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_counterparty_account_table}}` AS sca
    ON incoming_raw.sender_counterparty_account_id = sca.counterparty_account_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_counterparty_table}}` AS scp
    ON sca.counterparty_id = scp.counterparty_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_account_table}}` AS ra
    ON incoming_raw.receiver_account_id = ra.account_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_customer_table}}` AS rc
    ON ra.customer_id = rc.customer_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_counterparty_account_table}}` AS rca
    ON incoming_raw.receiver_counterparty_account_id = rca.counterparty_account_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_counterparty_table}}` AS rcp
    ON rca.counterparty_id = rcp.counterparty_id
),
incoming_base AS (
  SELECT
    transaction_id,
    timestamp,
    txn_date,
    sender_account,
    receiver_account,
    sender_account_id,
    sender_counterparty_account_id,
    receiver_account_id,
    receiver_counterparty_account_id,
    sender_is_bank_client,
    receiver_is_bank_client,
    amount,
    settlement_amount,
    COALESCE(fx_rate, 1.0) AS fx_rate,
    transaction_currency,
    settlement_currency,
    DATE_DIFF(settlement_date, txn_date, DAY) AS settlement_lag_days,
    CASE
      WHEN UPPER(TRIM(channel)) IN ('WIRE', 'WIRE TRANSFER') THEN 'wire'
      WHEN UPPER(TRIM(channel)) IN ('ACH', 'ACH TRANSFER') THEN 'ach'
      WHEN UPPER(TRIM(channel)) = 'CARD' THEN 'card'
      WHEN UPPER(TRIM(channel)) IN ('INTERNAL', 'INTERNAL TRANSFER') THEN 'internal'
      ELSE LOWER(TRIM(channel))
    END AS channel,
    CASE
      WHEN UPPER(TRIM(transaction_type)) IN ('PAYMENT', 'PAYMENT ') THEN 'payment'
      WHEN UPPER(TRIM(transaction_type)) IN ('TRANSFER', 'XFER') THEN 'transfer'
      WHEN UPPER(TRIM(transaction_type)) = 'WITHDRAWAL' THEN 'withdrawal'
      WHEN UPPER(TRIM(transaction_type)) = 'DEPOSIT' THEN 'deposit'
      ELSE LOWER(TRIM(transaction_type))
    END AS transaction_type,
    sender_country,
    receiver_country,
    CASE
      WHEN channel_indicator IS NULL THEN NULL
      WHEN UPPER(TRIM(channel_indicator)) IN ('ONLINE', 'WEB') THEN 'Online'
      WHEN UPPER(TRIM(channel_indicator)) IN ('IN-STORE', 'IN STORE', 'INSTORE') THEN 'In-Store'
      WHEN UPPER(TRIM(channel_indicator)) IN ('MOBILE APP', 'MOBILE') THEN 'Mobile App'
      WHEN UPPER(TRIM(channel_indicator)) = 'PHONE' THEN 'Phone'
      WHEN UPPER(TRIM(channel_indicator)) IN ('ATM', 'ATM ') THEN 'ATM'
      ELSE channel_indicator
    END AS channel_indicator,
    merchant_country,
    pos_entry_mode,
    terminal_id IS NOT NULL AS has_terminal,
    atm_id IS NOT NULL AS has_atm,
    merchant_country IS NOT NULL AS has_merchant_geo,
    CASE
      WHEN sender_account_id IS NOT NULL THEN DATE_DIFF(txn_date, sender_bank_opened_date, DAY)
      WHEN sender_counterparty_account_id IS NOT NULL THEN DATE_DIFF(txn_date, sender_counterparty_first_seen, DAY)
      ELSE NULL
    END AS sender_account_age_days,
    CASE
      WHEN receiver_account_id IS NOT NULL THEN DATE_DIFF(txn_date, receiver_bank_opened_date, DAY)
      WHEN receiver_counterparty_account_id IS NOT NULL THEN DATE_DIFF(txn_date, receiver_counterparty_first_seen, DAY)
      ELSE NULL
    END AS receiver_account_age_days,
    CASE
      WHEN receiver_counterparty_account_id IS NOT NULL THEN receiver_counterparty_is_shell
      WHEN receiver_account_id IS NOT NULL THEN COALESCE(receiver_is_shell_entity, FALSE)
      ELSE NULL
    END AS receiver_is_shell_company,
    sender_risk_rating,
    sender_is_pep,
    receiver_risk_rating,
    receiver_is_pep,
    sender_country != receiver_country AS is_cross_border,
    transaction_currency = settlement_currency AS same_settlement_currency,
    settlement_status,
    clearing_system,
    payment_reference IS NOT NULL AS has_payment_reference,
    LENGTH(COALESCE(memo, '')) AS memo_length,
    COALESCE(memo, '') = '' AS memo_is_empty,
    merchant_dba_name IS NOT NULL AS has_merchant_dba
  FROM incoming_enriched
),
history AS (
  SELECT
    transaction_id,
    sender_account,
    receiver_account,
    timestamp,
    amount,
    sender_country,
    receiver_country
  FROM `{{project_id}}.{{dataset}}.{{features_base_view}}`
  WHERE timestamp <= @timestamp
    AND transaction_id != @transaction_id
),
combined_velocity AS (
  SELECT transaction_id, sender_account, receiver_account, timestamp, amount
  FROM history
  UNION ALL
  SELECT transaction_id, sender_account, receiver_account, timestamp, amount
  FROM incoming_base
),
velocity_windowed AS (
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
  FROM combined_velocity
),
distinct_receivers AS (
  SELECT
    current_txn.transaction_id,
    COUNT(DISTINCT prior_txn.receiver_account) AS unique_receivers_24h
  FROM combined_velocity AS current_txn
  JOIN combined_velocity AS prior_txn
    ON current_txn.sender_account = prior_txn.sender_account
   AND prior_txn.timestamp BETWEEN TIMESTAMP_SUB(current_txn.timestamp, INTERVAL 24 HOUR)
                               AND current_txn.timestamp
  WHERE current_txn.transaction_id = @transaction_id
  GROUP BY current_txn.transaction_id
),
velocity AS (
  SELECT
    velocity_windowed.transaction_id,
    velocity_windowed.txn_count_24h,
    velocity_windowed.txn_count_7d,
    velocity_windowed.total_amount_24h,
    distinct_receivers.unique_receivers_24h
  FROM velocity_windowed
  JOIN distinct_receivers USING (transaction_id)
  WHERE velocity_windowed.transaction_id = @transaction_id
),
combined_network AS (
  SELECT transaction_id, sender_account, receiver_account, timestamp, sender_country, receiver_country
  FROM history
  UNION ALL
  SELECT transaction_id, sender_account, receiver_account, timestamp, sender_country, receiver_country
  FROM incoming_base
),
sender_fan_out AS (
  SELECT
    current_txn.transaction_id,
    COUNT(DISTINCT prior.receiver_account) AS sender_fan_out_as_of
  FROM combined_network AS current_txn
  JOIN combined_network AS prior
    ON prior.sender_account = current_txn.sender_account
   AND prior.timestamp <= current_txn.timestamp
  WHERE current_txn.transaction_id = @transaction_id
  GROUP BY current_txn.transaction_id
),
receiver_fan_in AS (
  SELECT
    current_txn.transaction_id,
    COUNT(DISTINCT prior.sender_account) AS receiver_fan_in_as_of
  FROM combined_network AS current_txn
  JOIN combined_network AS prior
    ON prior.receiver_account = current_txn.receiver_account
   AND prior.timestamp <= current_txn.timestamp
  WHERE current_txn.transaction_id = @transaction_id
  GROUP BY current_txn.transaction_id
),
network AS (
  SELECT
    incoming_base.transaction_id,
    sender_fan_out.sender_fan_out_as_of,
    receiver_fan_in.receiver_fan_in_as_of,
    incoming_base.sender_country IN ('KY', 'VG', 'PA', 'AE')
      AND incoming_base.receiver_country IN ('KY', 'VG', 'PA', 'AE') AS both_high_risk
  FROM incoming_base
  JOIN sender_fan_out USING (transaction_id)
  JOIN receiver_fan_in USING (transaction_id)
)
SELECT
  incoming_base.*,
  velocity.txn_count_24h,
  velocity.txn_count_7d,
  velocity.total_amount_24h,
  velocity.unique_receivers_24h,
  network.sender_fan_out_as_of,
  network.receiver_fan_in_as_of,
  network.both_high_risk
FROM incoming_base
JOIN velocity USING (transaction_id)
JOIN network USING (transaction_id);
