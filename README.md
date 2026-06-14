# AML MLOps Pipeline (GCP Demo)

End-to-end AML transaction risk scoring pipeline on Google Cloud Platform. Designed as a personal demo with production-shaped patterns: partitioned BigQuery tables, config-driven scripts, realistic fraud typologies, and temporal data spanning a full year.

## Architecture (target state)

1. Python generates synthetic transactions + dimension tables → Cloud Storage / BigQuery
2. BigQuery joins dimensions and engineers features (SQL)
3. Vertex AI AutoML Tabular trains a binary classifier
4. Model deployed to a Vertex AI endpoint
5. Cloud Run API wraps inference
6. Streamlit dashboard for scoring and monitoring

**GCP project:** `aml-mlops-demo-498203`  
**Region:** `us-central1`  
**Bucket:** `gs://aml-mlops-nakul`  
**Dataset:** `aml_mlops`

## Data architecture v2

Bank **clients** and external **counterparties** live in separate tables — cleaner features and model training.

```
ref_* tables
     ↓
dim_customer (bank clients, CIP/AML)  ←── beneficial_owner
     ↓
dim_account

dim_counterparty (external parties)
     ↓
dim_counterparty_account

raw_transactions (v2 schema)
  sender_account_id  XOR sender_counterparty_account_id
  receiver_account_id XOR receiver_counterparty_account_id
```

| Table | Holds |
|-------|--------|
| `dim_customer` | Bank clients only — CIP, onboarding, `risk_rating`, PEP |
| `dim_counterparty` | Wire beneficiaries, merchants, shell entities — minimal fields |
| `dim_account` | Client product accounts (`product_code` → `ref_product`) |
| `dim_counterparty_account` | External account stubs referenced by transactions |
| `beneficial_owner` | Business customer → owner (bank client or counterparty) |

Account age, shell-company flags, and client risk attributes are **not** stored on raw transactions. `features_base` derives them via joins to the dimension tables.

**Schemas:** `schemas/dim_*.json`, `raw_transactions_v2.json`, `ref_*.json`  
**DDL:** `sql/dimensions/create_dimension_tables.sql`

## Phase 1 — Data ingest

| Step | Script |
|------|--------|
| Generate transactions + dimension CSVs | `src/generate_synthetic_data.py` |
| Upload transaction CSV to GCS | `src/upload_to_gcs.py` |
| Load transactions + dimensions into BigQuery | `src/load_to_bigquery.py` |

### Fraud typologies modeled

- **Smurfing** — many sub-threshold ACH deposits into one account
- **Layering** — multi-hop wire transfers through shell companies
- **Round-tripping** — outbound cross-border transfer returning to origin
- **Funnel accounts** — fan-in of small payments, large outbound wire
- **Legitimate** — baseline non-fraud activity (~98% fraud rate target: 2%)

`typology` is retained in the raw table for evaluation and debugging. **Exclude it from model training features** to avoid label leakage.

Post-generation **noise** (missing fields, dirty enums, label noise, truncated memos) is configurable under `noise:` in config. Set `noise.enabled: false` for clean pipeline tests.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
gcloud auth application-default login
gcloud config set project aml-mlops-demo-498203
```

## Tests

Unit tests (no GCP credentials required):

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Run Phase 1

### Dev profile (25k rows — fast iteration)

Uses isolated paths: `transactions_dev.csv`, `gs://.../transactions/dev/`, BQ tables `raw_transactions_dev` and `dim_*_dev`.

```bash
python -m src.generate_synthetic_data --profile dev
python -m src.upload_to_gcs --profile dev
python -m src.load_to_bigquery --profile dev --replace --load-dimensions
```

### Train profile (200k rows — default)

```bash
python -m src.generate_synthetic_data
python -m src.upload_to_gcs
python -m src.load_to_bigquery --replace --load-dimensions
```

Generation writes:
- Transaction CSV → `data/transactions.csv` (or `data/transactions_dev.csv`)
- Dimension CSVs → `data/dimensions/` (train) or `data/dimensions_dev/` (dev)

`--load-dimensions` loads all reference and dimension tables from that directory into BigQuery (local file upload). Dimension tables must be loaded **before** deploying feature views.

`--profile train` is optional (it's the default). Use `--config path/to/custom.yaml` to override both profiles.

Use `--dated-prefix` on upload for partition-style paths (`transactions/raw/dt=YYYY-MM-DD/`).

### BigQuery load behavior

- **CSV schema** excludes post-load columns (`ingested_at`); table schema includes them.
- **`--replace`**: truncates and reloads the target table.
- **Default (append)**: loads to a staging table, MERGEs new `transaction_id`s into the target, then truncates staging.
- **Schema migration**: missing columns are added automatically on existing tables.

## Project layout

```
config/
  config.yaml      # train profile (200k)
  config.dev.yaml  # dev profile (25k, isolated tables)
data/              # Generated CSVs (gitignored)
  dimensions/      # ref_*, dim_*, beneficial_owner CSVs (train)
  dimensions_dev/  # same, dev profile
schemas/           # BigQuery table schemas (JSON)
sql/
  dimensions/      # Dimension DDL
  features/        # Feature view SQL
  splits/          # Temporal split views
  training/        # AutoML input view
src/
  generate_synthetic_data.py  # transactions + dimension CSVs
  dimension_data.py           # static ref_country, ref_product, etc.
  reference_data.py           # merchant/geo lookup for generation
  load_to_bigquery.py         # raw load + --load-dimensions
  deploy_views.py             # deploy BQ views from sql/
```

## Design choices (demo, production-shaped)

- **2% fraud rate** — realistic class imbalance
- **12-month timestamps** — supports temporal train/val/test splits
- **Partitioned BQ table** on `DATE(timestamp)` — mirrors production ingest
- **Config file** — no hardcoded project IDs in scripts
- **Separated parties** — bank clients (`dim_customer`) vs external counterparties (`dim_counterparty`); no `party_type` filtering in feature SQL
- **Derived risk attributes** — account age, shell flags, client `risk_rating` / PEP joined in `features_base`, not duplicated on raw rows
- **Payment context on txn** — channel/POS, merchant geo and names, settlement fields stay on the transaction row

### Raw transaction fields (v2)

| Group | Fields |
|-------|--------|
| Account legs | `sender_account_id` XOR `sender_counterparty_account_id`; `receiver_account_id` XOR `receiver_counterparty_account_id` |
| Payment routing | `payment_sender_country`, `payment_receiver_country` — routing/settlement countries (not customer residence) |
| Payment rail | `channel` — `wire`, `ach`, `card`, `internal` |
| Channel / POS | `channel_indicator`, `terminal_id`, `atm_id`, `pos_entry_mode` |
| Merchant (txn-level) | `merchant_city`, `merchant_state`, `merchant_country`, `merchant_legal_name`, `merchant_dba_name` |
| Payment narrative | `payment_reference`, `memo` |
| Settlement | `transaction_currency`, `settlement_currency`, `settlement_amount`, `fx_rate`, `settlement_date`, `settlement_status`, `clearing_system`, `correspondent_bic` |
| Labels | `is_fraud` (training target), `typology` (evaluation only) |

## Phase 2 — Features & splits

```bash
# Deploy feature views and temporal splits (after raw + dimension tables are loaded)
python -m src.deploy_views --profile dev
python -m src.deploy_views              # train profile
```

| View | Purpose |
|------|---------|
| `features_base` | Raw txn + dimension joins — normalized enums, settlement, POS/memo flags, derived account ages, shell flags, client risk attributes; unified `sender_account` / `receiver_account` for graph features |
| `features_velocity` | Point-in-time sender velocity — `txn_count_24h`, `txn_count_7d`, `total_amount_24h`, `unique_receivers_24h` |
| `features_network` | As-of fan-out/fan-in — `sender_fan_out_as_of`, `receiver_fan_in_as_of`, `both_high_risk` |
| `features_training` | Combined training view (base + velocity + network + `is_fraud`) |
| `features_eval` | Combined features + `typology` for evaluation only |
| `features_train` / `_val` / `_test` | Temporal splits (Jan–Sep / Oct / Nov–Dec) |

Dev profile uses `*_dev` view names so train and dev don't collide.

## Phase 3 — Vertex AI AutoML

Prerequisites:

1. Enable the [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project=aml-mlops-demo-498203) on your GCP project (one-time).
2. Load raw transactions and dimensions (`python -m src.load_to_bigquery --replace --load-dimensions`).
3. Deploy Phase 2 views (`python -m src.deploy_views`).

```bash
pip install -r requirements.txt
python -m src.deploy_views              # includes features_automl input view
python -m src.train_automl --profile dev   # fast iteration (~1 node hour)
python -m src.evaluate_automl --profile dev

# Full 200k training profile (longer, ~8 node hours)
python -m src.train_automl
python -m src.evaluate_automl
```

| Step | What it does |
|------|----------------|
| `features_automl` | Unions train/val/test splits with `ml_split` = `TRAIN` / `VALIDATE` / `TEST` |
| `train_automl.py` | Creates TabularDataset from BigQuery, runs AutoML Tabular (target: `is_fraud`, objective: `maximize-au-prc`) |
| `evaluate_automl.py` | Batch-predicts on `features_test`, reports precision/recall/F1 overall and by `typology` |
| `log_predictions.py` | Backfill `prediction_log` from an existing Vertex batch output table |

Excluded from training (config `automl.excluded_columns`): `transaction_id`, `timestamp`, `txn_date`, `sender_account`, `receiver_account`, `sender_account_id`, `sender_counterparty_account_id`, `receiver_account_id`, `receiver_counterparty_account_id`, `ml_split`.

Run metadata is saved locally to `artifacts/automl_<profile>.json` (gitignored).

### Prediction logging

Batch evaluation appends a normalized audit row per transaction to `prediction_log` (or `prediction_log_dev`):

| Column | Purpose |
|--------|---------|
| `prediction_id` | Unique log row |
| `transaction_id` | Scored transaction |
| `predicted_is_fraud`, `fraud_score` | Model output |
| `actual_is_fraud` | Ground truth when available (joined from `features_eval`) |
| `model_resource_name`, `prediction_source` | Model + channel (`batch_eval`, `batch_score`, `online`) |
| `raw_predictions_table` | Vertex batch output table (idempotency key) |

```bash
# Logged automatically when you evaluate
python -m src.evaluate_automl --skip-batch-predict   # re-log skipped if already logged

# Backfill from the latest eval artifact or newest automl_predictions_* table
python -m src.log_predictions
python -m src.log_predictions --predictions-table automl_predictions_123 --force
```

**Schema:** `schemas/prediction_log.json`  
**DDL:** `sql/logging/create_prediction_log_table.sql`

## Phase 4 — Deploy to Vertex endpoint

```bash
python -m src.deploy_model
python -m src.deploy_model --undeploy   # delete endpoint to stop serving charges
```

Deployment metadata is saved to `artifacts/deploy_<profile>.json`.

After evaluation, metrics are written to `artifacts/metrics_<profile>.json` (overall + per-typology). Re-export anytime from an existing predictions table:

```bash
python -m src.export_metrics
python -m src.export_metrics --profile dev
```

## Phase 5 — Cloud Run serving (Phase 2-lite)

**Spec:** [`docs/serving-spec.md`](docs/serving-spec.md)

Prod-shaped online scoring: authenticated customer + payment instruction → party resolution → BQ feature SQL → Vertex → `prediction_log`.

| Endpoint | Purpose |
|----------|---------|
| `POST /score` | **Primary API** — `customer_id`, `sender_account_id`, amount, `receiver` object; resolver builds canonical v2 txn |
| `POST /validate` | **Dev parity only** — full raw v2 JSON with pre-resolved IDs; compare vs `features_training_dev` |
| `GET /health` | Liveness |

Build order: `party_resolver.py` + `sql/serving/score_features.sql` + parity tests, then FastAPI + Cloud Run.

```bash
# Parity check: online score_features.sql vs features_training_dev (requires BQ + ADC)
python -m src.serving.parity_check --profile dev --transaction-id <TXN_ID>
python -m src.serving.parity_batch --profile dev --limit 10
```

## Next iterations

- [x] BigQuery SQL feature engineering views
- [x] Temporal split views (train / val / test)
- [x] Vertex AI AutoML training pipeline
- [x] Vertex AI endpoint deployment
- [x] Data arch v2 — separated client/counterparty dimensions, v2 transactions, feature joins
- [x] Prediction logging table (`prediction_log`)
- [x] Dev BQ reload + views + AutoML retrain on v2 (25k profile)
- [ ] Phase 2-lite serving API — see `docs/serving-spec.md`
- [ ] Streamlit monitoring dashboard
