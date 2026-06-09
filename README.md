# AML MLOps Pipeline (GCP Demo)

End-to-end AML transaction risk scoring pipeline on Google Cloud Platform. Designed as a personal demo with production-shaped patterns: partitioned BigQuery tables, config-driven scripts, realistic fraud typologies, and temporal data spanning a full year.

## Architecture (target state)

1. Python generates synthetic AML transactions → Cloud Storage
2. BigQuery loads and engineers features (SQL)
3. Vertex AI AutoML Tabular trains a binary classifier
4. Model deployed to a Vertex AI endpoint
5. Cloud Run API wraps inference
6. Streamlit dashboard for scoring and monitoring

**GCP project:** `aml-mlops-demo-498203`  
**Region:** `us-central1`  
**Bucket:** `gs://aml-mlops-nakul`  
**Dataset:** `aml_mlops`

## Phase 1 — Data ingest (this repo)

| Step | Script |
|------|--------|
| Generate 200k synthetic transactions | `src/generate_synthetic_data.py` |
| Upload CSV to GCS | `src/upload_to_gcs.py` |
| Load into BigQuery | `src/load_to_bigquery.py` |

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

Uses isolated paths: `transactions_dev.csv`, `gs://.../transactions/dev/`, BQ table `raw_transactions_dev`.

```bash
python -m src.generate_synthetic_data --profile dev
python -m src.upload_to_gcs --profile dev
python -m src.load_to_bigquery --profile dev --replace
```

### Train profile (200k rows — default)

```bash
python -m src.generate_synthetic_data
python -m src.upload_to_gcs
python -m src.load_to_bigquery --replace
```

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
data/            # Generated CSVs (gitignored)
schemas/         # BigQuery table schema (JSON)
sql/             # DDL and feature SQL (later phases)
src/             # Python pipeline scripts (reference_data.py = lookup tables)
```

## Design choices (demo, production-shaped)

- **2% fraud rate** — realistic class imbalance
- **12-month timestamps** — supports temporal train/val/test splits
- **Partitioned BQ table** on `DATE(timestamp)` — mirrors production ingest
- **Config file** — no hardcoded project IDs in scripts
- **Extra columns** — payment context (`channel_indicator`, `terminal_id`, `merchant_country`, `pos_entry_mode`), settlement fields, account ages, `typology`

### Transaction fields (v0.2)

| Group | Fields |
|-------|--------|
| Payment rail | `channel` — settlement rail: `wire`, `ach`, `card`, `internal` |
| Channel / POS | `channel_indicator` (`Online`, `In-Store`, `Mobile App`, `Phone`, `ATM`), `terminal_id`, `atm_id`, `pos_entry_mode` (`Chip/EMV`, `Contactless/Tap`, `Magstripe`, `Manually Keyed`) |
| Merchant geo | `merchant_city`, `merchant_state`, `merchant_country` — store or HQ location (card/ATM txs) |
| Merchant identity | `merchant_legal_name`, `merchant_dba_name` — counterparty or merchant business names |
| Payment narrative | `payment_reference`, `memo` — invoice refs, wire notes, vague or misleading text |
| Settlement | `transaction_currency`, `settlement_currency`, `settlement_amount`, `fx_rate`, `settlement_date`, `settlement_status`, `clearing_system`, `correspondent_bic` |

## Phase 2 — Features & splits

```bash
# Deploy feature views and temporal splits (after raw data is loaded)
python -m src.deploy_views --profile dev
python -m src.deploy_views              # train profile
```

| View | Purpose |
|------|---------|
| `features_base` | Normalized attributes from raw (enums, settlement, POS, memo flags; includes account IDs for downstream joins) |
| `features_velocity` | Derived from base — `txn_count_24h`, `txn_count_7d`, `total_amount_24h`, `unique_receivers_24h` |
| `features_network` | Derived from base — `sender_fan_out_as_of`, `receiver_fan_in_as_of`, `both_high_risk` |
| `features_training` | Combined training view (base + velocity + network + `is_fraud`) |
| `features_eval` | Combined features + `typology` for evaluation only |
| `features_train` / `_val` / `_test` | Temporal splits (Jan–Sep / Oct / Nov–Dec) |

Dev profile uses `*_dev` view names so train and dev don't collide.

## Phase 3 — Vertex AI AutoML

Prerequisites:

1. Enable the [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project=aml-mlops-demo-498203) on your GCP project (one-time).
2. Ensure Phase 2 views are deployed (`python -m src.deploy_views`).

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

Excluded from training (config `automl.excluded_columns`): `transaction_id`, `timestamp`, `txn_date`, `sender_account`, `receiver_account`, `ml_split`.

Run metadata is saved locally to `artifacts/automl_<profile>.json` (gitignored).

## Next iterations

- [x] BigQuery SQL feature engineering views
- [x] Temporal split views (train / val / test)
- [x] Vertex AI AutoML training pipeline
- [ ] Prediction logging table
- [ ] Cloud Run inference API with matching feature logic
- [ ] Streamlit monitoring dashboard
