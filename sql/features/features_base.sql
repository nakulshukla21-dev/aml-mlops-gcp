-- Normalized transaction attributes (no velocity/network, no label).

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_base_view}}` AS
WITH base AS (
  SELECT
    *,
    DATE(timestamp) AS txn_date
  FROM `{{project_id}}.{{dataset}}.{{raw_table}}`
)
SELECT
  transaction_id,
  timestamp,
  txn_date,
  sender_account,
  receiver_account,

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

  CASE
    WHEN UPPER(TRIM(sender_country)) IN ('USA', 'U.S.', 'US') THEN 'US'
    WHEN UPPER(TRIM(sender_country)) IN ('UK', 'U.K.', 'GB') THEN 'GB'
    WHEN UPPER(TRIM(sender_country)) IN ('DE', 'DEUTSCHLAND') THEN 'DE'
    WHEN UPPER(TRIM(sender_country)) IN ('FR', 'FRA') THEN 'FR'
    ELSE UPPER(TRIM(sender_country))
  END AS sender_country,

  CASE
    WHEN UPPER(TRIM(receiver_country)) IN ('USA', 'U.S.', 'US') THEN 'US'
    WHEN UPPER(TRIM(receiver_country)) IN ('UK', 'U.K.', 'GB') THEN 'GB'
    WHEN UPPER(TRIM(receiver_country)) IN ('DE', 'DEUTSCHLAND') THEN 'DE'
    WHEN UPPER(TRIM(receiver_country)) IN ('FR', 'FRA') THEN 'FR'
    ELSE UPPER(TRIM(receiver_country))
  END AS receiver_country,

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

  sender_account_age_days,
  receiver_account_age_days,
  receiver_is_shell_company,

  sender_country != receiver_country AS is_cross_border,
  transaction_currency = settlement_currency AS same_settlement_currency,
  settlement_status,
  clearing_system,

  payment_reference IS NOT NULL AS has_payment_reference,
  LENGTH(COALESCE(memo, '')) AS memo_length,
  COALESCE(memo, '') = '' AS memo_is_empty,
  merchant_dba_name IS NOT NULL AS has_merchant_dba

FROM base;
