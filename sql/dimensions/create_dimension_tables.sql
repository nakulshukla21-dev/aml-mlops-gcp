-- Dimension and reference tables for data architecture v2.
-- Bank clients (dim_customer) are separate from external parties (dim_counterparty).

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.ref_country` (
  country_code STRING NOT NULL,
  country_name STRING NOT NULL,
  iso_alpha3 STRING,
  risk_tier STRING NOT NULL,
  is_fatf_grey BOOL,
  default_currency STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.ref_state` (
  country_code STRING NOT NULL,
  state_code STRING NOT NULL,
  state_name STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.ref_naics` (
  naics_code STRING NOT NULL,
  naics_title STRING NOT NULL,
  sector STRING NOT NULL,
  aml_risk_tier STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.ref_product` (
  product_code STRING NOT NULL,
  product_name STRING NOT NULL,
  product_type STRING NOT NULL,
  product_category STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.dim_customer` (
  customer_id INT64 NOT NULL,
  entity_type STRING NOT NULL,
  legal_name STRING NOT NULL,
  dba_name STRING,
  date_of_birth DATE,
  incorporation_date DATE,
  residence_country STRING NOT NULL,
  residence_state STRING,
  citizenship_country STRING,
  onboarding_date DATE NOT NULL,
  cip_status STRING NOT NULL,
  risk_rating STRING NOT NULL,
  is_pep BOOL,
  is_sanctioned BOOL,
  is_shell_entity BOOL,
  naics_code STRING,
  last_review_date DATE,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.dim_counterparty` (
  counterparty_id INT64 NOT NULL,
  entity_type STRING,
  counterparty_role STRING NOT NULL,
  legal_name STRING NOT NULL,
  dba_name STRING,
  country STRING,
  is_shell_entity BOOL,
  is_high_risk_jurisdiction BOOL,
  first_seen_date DATE,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.dim_account` (
  account_id STRING NOT NULL,
  customer_id INT64 NOT NULL,
  product_code STRING NOT NULL,
  status STRING NOT NULL,
  opened_date DATE NOT NULL,
  closed_date DATE,
  currency STRING NOT NULL,
  current_balance FLOAT64,
  available_balance FLOAT64,
  is_primary BOOL,
  domicile_country STRING,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.dim_counterparty_account` (
  counterparty_account_id STRING NOT NULL,
  counterparty_id INT64 NOT NULL,
  account_label STRING,
  country STRING,
  first_seen_date DATE,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset}}.beneficial_owner` (
  beneficial_owner_id INT64 NOT NULL,
  business_customer_id INT64 NOT NULL,
  owner_customer_id INT64,
  owner_counterparty_id INT64,
  owner_is_bank_client BOOL NOT NULL,
  ownership_pct FLOAT64 NOT NULL,
  control_type STRING NOT NULL,
  is_pep BOOL,
  effective_from DATE NOT NULL,
  effective_to DATE
);
