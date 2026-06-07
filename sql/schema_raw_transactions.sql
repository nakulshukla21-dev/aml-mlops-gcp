-- Raw transactions table. Partitioned by transaction date for temporal splits and monitoring.
-- typology is retained for evaluation/debugging; exclude from model training features.
-- channel = settlement rail (wire/ach/card/internal)
-- channel_indicator = customer initiation method (Online/In-Store/Mobile App/Phone/ATM)

CREATE TABLE IF NOT EXISTS `aml-mlops-demo-498203.aml_mlops.raw_transactions` (
  transaction_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  sender_account STRING NOT NULL,
  receiver_account STRING NOT NULL,
  amount FLOAT64 NOT NULL,
  transaction_currency STRING NOT NULL,
  transaction_type STRING NOT NULL,
  channel STRING NOT NULL,
  channel_indicator STRING,
  terminal_id STRING,
  atm_id STRING,
  merchant_city STRING,
  merchant_state STRING,
  merchant_country STRING,
  merchant_legal_name STRING,
  merchant_dba_name STRING,
  pos_entry_mode STRING,
  payment_reference STRING,
  memo STRING,
  sender_country STRING NOT NULL,
  receiver_country STRING NOT NULL,
  settlement_currency STRING NOT NULL,
  settlement_amount FLOAT64 NOT NULL,
  fx_rate FLOAT64,
  settlement_date DATE,
  settlement_status STRING NOT NULL,
  clearing_system STRING,
  correspondent_bic STRING,
  sender_account_age_days INT64,
  receiver_account_age_days INT64,
  receiver_is_shell_company BOOL,
  typology STRING,
  is_fraud BOOL NOT NULL,
  ingested_at TIMESTAMP
)
PARTITION BY DATE(timestamp)
CLUSTER BY sender_account, receiver_account, is_fraud
OPTIONS (
  description = 'Synthetic AML transaction feed. typology is for analysis only, not model input.'
);
