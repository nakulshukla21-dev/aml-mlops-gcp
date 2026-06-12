-- Normalized transaction attributes with dimension joins (no velocity/network, no label).

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_base_view}}` AS
WITH base AS (
  SELECT
    *,
    DATE(timestamp) AS txn_date
  FROM `{{project_id}}.{{dataset}}.{{raw_table}}`
),
enriched AS (
  SELECT
    base.*,

    COALESCE(base.sender_account_id, base.sender_counterparty_account_id) AS sender_account,
    COALESCE(base.receiver_account_id, base.receiver_counterparty_account_id) AS receiver_account,

    CASE
      WHEN UPPER(TRIM(base.payment_sender_country)) IN ('USA', 'U.S.', 'US') THEN 'US'
      WHEN UPPER(TRIM(base.payment_sender_country)) IN ('UK', 'U.K.', 'GB') THEN 'GB'
      WHEN UPPER(TRIM(base.payment_sender_country)) IN ('DE', 'DEUTSCHLAND') THEN 'DE'
      WHEN UPPER(TRIM(base.payment_sender_country)) IN ('FR', 'FRA') THEN 'FR'
      ELSE UPPER(TRIM(base.payment_sender_country))
    END AS sender_country,

    CASE
      WHEN UPPER(TRIM(base.payment_receiver_country)) IN ('USA', 'U.S.', 'US') THEN 'US'
      WHEN UPPER(TRIM(base.payment_receiver_country)) IN ('UK', 'U.K.', 'GB') THEN 'GB'
      WHEN UPPER(TRIM(base.payment_receiver_country)) IN ('DE', 'DEUTSCHLAND') THEN 'DE'
      WHEN UPPER(TRIM(base.payment_receiver_country)) IN ('FR', 'FRA') THEN 'FR'
      ELSE UPPER(TRIM(base.payment_receiver_country))
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

    base.sender_account_id IS NOT NULL AS sender_is_bank_client,
    base.receiver_account_id IS NOT NULL AS receiver_is_bank_client
  FROM base
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_account_table}}` AS sa
    ON base.sender_account_id = sa.account_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_customer_table}}` AS sc
    ON sa.customer_id = sc.customer_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_counterparty_account_table}}` AS sca
    ON base.sender_counterparty_account_id = sca.counterparty_account_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_counterparty_table}}` AS scp
    ON sca.counterparty_id = scp.counterparty_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_account_table}}` AS ra
    ON base.receiver_account_id = ra.account_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_customer_table}}` AS rc
    ON ra.customer_id = rc.customer_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_counterparty_account_table}}` AS rca
    ON base.receiver_counterparty_account_id = rca.counterparty_account_id
  LEFT JOIN `{{project_id}}.{{dataset}}.{{dim_counterparty_table}}` AS rcp
    ON rca.counterparty_id = rcp.counterparty_id
)
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

FROM enriched;
