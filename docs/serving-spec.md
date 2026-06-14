# Phase 2-lite serving spec

**Goal:** Score a **new payment at decision time** by resolving parties from authenticated context, building the same feature vector training uses, and calling Vertex — using **BigQuery for party master + transaction history** (not a precomputed feature-row lookup).

This spec reflects how a real bank payment flow works: the client authenticates, selects a source account, and submits a payment instruction. Dimension data is **looked up or stubbed** at score time; it is not assumed to arrive fully resolved in the request.

---

## What you learn (vs Phase 1 lookup)

| Topic | Phase 2-lite |
|-------|----------------|
| Auth context + payment instruction API | Yes |
| Party resolution (lookup / stub) | Yes |
| v2 dim joins at score time | Yes |
| Point-in-time velocity (no leakage) | Yes |
| Fan-in / fan-out as-of | Yes |
| Train/serve parity testing | Yes |
| Real HTTP serving + online logging | Yes |

Acceptable for demo: BQ latency (~1–3s), no streaming cache, auth simulated via request fields (not full OAuth).

---

## Architecture

```
POST /score  (authenticated payment instruction)
       │
       ▼
┌──────────────────────┐
│ Validate request     │  sender account ∈ customer; amount > 0; rail enums
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Party resolution     │  src/serving/party_resolver.py
│  - sender (known)    │  validate dim_account + dim_customer
│  - receiver          │  match dim_counterparty_* OR create stub
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Canonical v2 txn     │  resolved account IDs + payment fields
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ BQ: build features   │  sql/serving/score_features.sql (parameterized)
│  - dim joins         │
│  - history           │  raw_transactions_dev + features_base_dev
│  - velocity/network  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Vertex predict       │  endpoint from deploy_dev.json
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ prediction_log       │  prediction_source=online
└──────────┬───────────┘
           ▼
      JSON response
```

**`POST /validate`** (dev-only) bypasses party resolution and accepts a **full raw v2 row** (with pre-resolved account IDs) for parity testing against `features_training_dev`. Not the production API shape.

---

## API contracts

### `POST /score` — primary (prod-shaped)

Represents a logged-in customer initiating a payment. The bank already knows **who** the customer is and **which account** they selected; the request carries the payment instruction and optional receiver details.

**Request**

```json
{
  "transaction_id": "TXN-NEW-001",
  "timestamp": "2024-11-15T14:30:00Z",

  "customer_id": 1001,
  "sender_account_id": "BAUS0000100",

  "amount": 9500.0,
  "transaction_currency": "USD",
  "transaction_type": "transfer",
  "channel": "wire",

  "payment_sender_country": "US",
  "payment_receiver_country": "KY",

  "receiver": {
    "type": "external",
    "account_id": null,
    "beneficiary_name": "Global Trade Solutions",
    "dba_name": null,
    "country": "KY",
    "account_reference": "EXT-12345",
    "entity_type": "business"
  },

  "payment_reference": "INV-2024-9912",
  "memo": "Consulting fees Q4",
  "settlement_currency": "USD",
  "settlement_amount": 9500.0,
  "settlement_status": "pending",
  "fx_rate": 1.0
}
```

**`receiver.type` values**

| Type | Meaning | Resolution |
|------|---------|------------|
| `bank` | Internal transfer to another bank client | Lookup `receiver.account_id` in `dim_account` |
| `external` | Wire / ACH beneficiary outside the bank | Match or stub `dim_counterparty` + `dim_counterparty_account` |
| `merchant` | Card / POS payment | Match or stub counterparty from merchant names + country |

For `merchant`, also accept txn-level fields on the request root (same as raw v2): `merchant_legal_name`, `merchant_dba_name`, `merchant_city`, `merchant_state`, `merchant_country`, `channel_indicator`, `terminal_id`, `pos_entry_mode`, etc.

**Response**

```json
{
  "transaction_id": "TXN-NEW-001",
  "fraud_score": 0.87,
  "is_fraud_predicted": true,
  "risk_level": "High",
  "resolved_parties": {
    "sender_account_id": "BAUS0000100",
    "receiver_account_id": null,
    "receiver_counterparty_account_id": "CPAUS00999",
    "receiver_created": true
  },
  "features_used": { "amount": 9500, "txn_count_24h": 14 },
  "scored_at": "2026-06-12T18:30:00Z"
}
```

**`GET /health`** — liveness; optionally probe BQ + Vertex endpoint reachability.

---

### `POST /validate` — dev parity only

Accepts a **complete raw v2 transaction** (same fields as `raw_transactions_v2`, minus `is_fraud`, `typology`, `ingested_at`). All account IDs must already be resolved — this path **skips party resolution** and exists only to prove train/serve parity.

```json
{
  "transaction_id": "TXN-0001234",
  "timestamp": "2024-11-02T09:15:00Z",
  "sender_account_id": "BAUS0000100",
  "sender_counterparty_account_id": null,
  "receiver_counterparty_account_id": "CPAUS00123",
  "receiver_account_id": null,
  "amount": 4200.0,
  "channel": "ach",
  "payment_sender_country": "US",
  "payment_receiver_country": "US",
  ...
}
```

**Response** — online features vs `features_training_dev` row (if `transaction_id` exists), plus model score:

```json
{
  "transaction_id": "TXN-0001234",
  "parity": {
    "matched": true,
    "max_feature_delta": 0.0,
    "mismatched_columns": []
  },
  "online_score": 0.12,
  "offline_score": 0.12,
  "features_used": { ... }
}
```

Use this with rows from `raw_transactions_dev` where IDs are already correct. Do **not** document `/validate` as the client integration path.

---

## Party resolution

Implemented in `src/serving/party_resolver.py`. Runs **before** `score_features.sql`.

### Sender (always known in prod)

1. Require `customer_id` + `sender_account_id`.
2. Query `dim_account` → confirm `account_id` exists, `status = active`, and `customer_id` matches.
3. On mismatch → **403** (`sender account does not belong to customer`).
4. Join `dim_customer` for client risk attributes (used downstream in feature SQL).

The client never sends a free-text sender; the bank binds `sender_account_id` from the authenticated session.

### Receiver — `type: bank`

1. Require `receiver.account_id`.
2. Lookup in `dim_account` (must be a bank client account).
3. Set `receiver_account_id`; leave counterparty legs null.

### Receiver — `type: external`

1. Match existing counterparty account (in order):
   - Exact `account_reference` → `dim_counterparty_account.account_label`
   - Fuzzy `beneficiary_name` + `country` → `dim_counterparty.legal_name` / `dba_name`
2. If no match → **create stub** (demo: in-memory for request only; optional write-back to BQ stub tables):
   - New `counterparty_id` (next available or hash-based)
   - `first_seen_date = DATE(timestamp)`
   - `counterparty_role = wire_beneficiary`
   - `is_shell_entity = null` until enriched
   - New `counterparty_account_id` (e.g. `CPA{country}{hash}`)
3. Set `receiver_counterparty_account_id`; leave `receiver_account_id` null.
4. Derive `payment_receiver_country` from receiver country if omitted.

First-time beneficiaries score with **weaker receiver-side features** (null shell flag, account age ≈ 0). That is expected and realistic.

### Receiver — `type: merchant`

1. Match on `merchant_legal_name` / `merchant_dba_name` + `merchant_country`.
2. If no match → stub with `counterparty_role = merchant`.
3. Set `receiver_counterparty_account_id`.

### Canonical v2 row

Party resolution outputs an internal `ResolvedTransaction` dataclass — the same shape `score_features.sql` expects:

```
sender_account_id XOR sender_counterparty_account_id
receiver_account_id XOR receiver_counterparty_account_id
+ all payment / settlement / merchant fields from the request
```

Only `/validate` accepts this shape directly from the client.

---

## Feature build strategy

**Do not reimplement 40 features in Python.** Use one parameterized SQL file:

`sql/serving/score_features.sql`

Both `/score` and `/validate` converge on the same canonical v2 row, then run identical feature SQL.

1. **`incoming`** — single row from resolved v2 parameters (`@transaction_id`, `@timestamp`, `@amount`, account IDs, …).

2. **`incoming_base`** — same transforms as `features_base.sql`:
   - `COALESCE` sender/receiver account keys
   - country normalization
   - channel / `transaction_type` cleanup
   - dim joins: `dim_account_dev`, `dim_customer_dev`, `dim_counterparty_*_dev`
   - derived: account ages, shell flags, `sender_risk_rating`, `is_pep`, booleans

3. **`history`** — from `features_base_dev` where:
   - `timestamp <= @timestamp`
   - `transaction_id != @transaction_id`
   - accounts overlap sender/receiver of incoming

4. **`velocity`** — mirror `features_velocity.sql` on `UNION ALL(history, incoming_base)`:
   - `txn_count_24h`, `txn_count_7d`, `total_amount_24h`, `unique_receivers_24h`

5. **`network`** — mirror `features_network.sql`:
   - `sender_fan_out_as_of`, `receiver_fan_in_as_of`, `both_high_risk`

6. **`model_features`** — select only columns in `artifacts/automl_dev.json` → `column_specs` (load at runtime).

**Parity test:** For existing test txns, `/validate` online output must match `features_training_dev` within float tolerance.

---

## Model input columns

Load from `artifacts/automl_dev.json` at runtime — **do not hardcode feature count**.

Excluded from Vertex payload (same as training `automl.excluded_columns`):

`transaction_id`, `timestamp`, `txn_date`, `sender_account`, `receiver_account`, all four account ID columns, `ml_split`.

---

## Vertex + config

| Setting | Source |
|---------|--------|
| Profile | `dev` |
| Endpoint | `artifacts/deploy_dev.json` → `endpoint_resource_name` |
| Model columns | `artifacts/automl_dev.json` → `column_specs` |
| BQ tables | `config.dev.yaml` |

**Prerequisite:** `python -m src.deploy_model --profile dev`

Parse Vertex response with `src/predictions.py`.

**Risk levels** (add to `config.dev.yaml`):

```yaml
serving:
  risk_thresholds:
    high: 0.7
    medium: 0.4
  # Demo auth: trust customer_id + sender_account_id in body.
  # Prod: replace with JWT / session middleware; inject customer_id server-side.
  auth_mode: demo_trust_body
```

---

## Logging

New `log_online_prediction()` (pattern from `prediction_logging.py`):

| Field | Value |
|-------|-------|
| `prediction_source` | `online` |
| `actual_is_fraud` | `null` |
| `feature_view` | `serving/score_features` |
| `raw_predictions_table` | `null` |

Optionally log `resolved_parties` in a JSON metadata column or structured log line (out of scope for `prediction_log` schema v1).

---

## Repo layout

```
docs/serving-spec.md          # this file
src/serving/
  app.py                      # FastAPI: /score, /validate, /health (next)
  schemas.py                  # PaymentScoreRequest, RawTransactionV2
  party_resolver.py           # auth + receiver resolution → canonical v2
  feature_builder.py          # Run score_features.sql
  parity_check.py             # Single-transaction parity CLI
  parity_batch.py             # Random batch parity CLI
  vertex_client.py            # Predict + parse (next)
sql/serving/
  score_features.sql
  resolve_counterparty.sql    # optional: BQ lookup helpers for resolver
Dockerfile
requirements-serving.txt      # fastapi, uvicorn, google-cloud-*
```

---

## Constraints & errors

| Case | Behavior |
|------|----------|
| `sender_account_id` not owned by `customer_id` | **403** |
| Both sender IDs set (post-resolution) | **400** — XOR violation |
| Both receiver IDs set | **400** |
| Unknown sender account (data gap) | **403** in `/score` (must be onboarded client) |
| Unknown receiver (external, first time) | Resolve to stub; dim fields null or day-0 |
| No txn history for sender | velocity = 1 (current txn only); fan-out/fan-in minimal |
| BQ timeout | **503** |
| Vertex down | **503** |

---

## Validation workflow

1. Pick 10 rows from `features_test_dev`.
2. `POST /validate` with their fields from `raw_transactions_dev` (pre-resolved IDs).
3. Assert max feature delta &lt; 1e-6 vs `features_training_dev`.
4. Assert `fraud_score` ≈ batch predict for same row.

Separately, exercise `/score` with realistic payment instructions (new `transaction_id`, authenticated sender, external receiver) to confirm end-to-end flow including stub creation.

---

## Out of scope (true prod)

- Sub-second latency / feature store / Redis
- Real OAuth / session middleware (demo uses `customer_id` in body)
- Persisting scored txn + stub counterparties back to ledger tables
- Async KYC enrichment after stub creation
- Cloud Run IAM as sole auth (add API gateway in prod)

---

## Build order

1. Refresh `automl_dev.json` `column_specs` from trained model.
2. Write `party_resolver.py` + unit tests (sender validation, external match, stub).
3. Write `score_features.sql` + parity test via `/validate`.
4. `feature_builder.py` + `vertex_client.py`.
5. FastAPI `/score`, `/validate`, `/health`.
6. Online logging.
7. `deploy_model --profile dev` + Cloud Run deploy.
8. Run parity test on 10 test txns; smoke-test `/score` with new beneficiary.

**Start with steps 2–3** (resolver + SQL parity) before FastAPI — proves the hard part without endpoint cost.

---

## Effort

~2–3 days focused work. Hardest pieces: **`score_features.sql`** staying in sync with `features_*.sql`, and **receiver matching** rules that behave sensibly on synthetic data.
