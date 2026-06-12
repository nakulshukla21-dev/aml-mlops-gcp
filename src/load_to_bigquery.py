"""Create BigQuery resources and load transactions from GCS."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from google.api_core.exceptions import Conflict
from google.cloud import bigquery

from src.config import PROJECT_ROOT, add_config_arguments, load_config_from_args

SCHEMA_PATH = PROJECT_ROOT / "schemas" / "raw_transactions_v2.json"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"

# Columns populated after CSV load, not present in the source file.
POST_LOAD_COLUMNS = frozenset({"ingested_at"})
DEDUP_KEY = "transaction_id"


DIMENSION_TABLES: dict[str, str] = {
    "ref_country": "ref_country.json",
    "ref_state": "ref_state.json",
    "ref_naics": "ref_naics.json",
    "ref_product": "ref_product.json",
    "dim_customer": "dim_customer.json",
    "dim_counterparty": "dim_counterparty.json",
    "dim_account": "dim_account.json",
    "dim_counterparty_account": "dim_counterparty_account.json",
    "beneficial_owner": "beneficial_owner.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load transactions CSV into BigQuery.")
    add_config_arguments(parser)
    parser.add_argument(
        "--gcs-uri",
        type=str,
        default=None,
        help="GCS URI of CSV file. Defaults to gs://<bucket>/<prefix>/<filename>.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing table contents (WRITE_TRUNCATE). Default appends with dedup.",
    )
    parser.add_argument(
        "--dimensions-dir",
        type=Path,
        default=None,
        help="Local directory with dimension CSVs. Defaults to data/<dimensions_dir> from config.",
    )
    parser.add_argument(
        "--load-dimensions",
        action="store_true",
        help="Load dimension and reference tables from --dimensions-dir.",
    )
    return parser.parse_args()


def load_schema(path: Path) -> list[bigquery.SchemaField]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [bigquery.SchemaField.from_api_repr(field) for field in raw]


def csv_load_schema(schema: list[bigquery.SchemaField]) -> list[bigquery.SchemaField]:
    """Schema matching the CSV file (excludes post-load metadata columns)."""
    return [field for field in schema if field.name not in POST_LOAD_COLUMNS]


def ensure_dataset(client: bigquery.Client, dataset_id: str, location: str) -> bigquery.Dataset:
    dataset_ref = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset_ref.location = location
    try:
        return client.create_dataset(dataset_ref)
    except Conflict:
        return client.get_dataset(dataset_ref)


def _apply_table_options(table: bigquery.Table) -> None:
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp",
    )
    table.clustering_fields = ["sender_account_id", "receiver_account_id", "is_fraud"]


def sync_table_schema(
    client: bigquery.Client,
    table: bigquery.Table,
    desired_schema: list[bigquery.SchemaField],
) -> bigquery.Table:
    """Add any missing columns from desired_schema to an existing table."""
    existing_names = {field.name for field in table.schema}
    new_fields = [field for field in desired_schema if field.name not in existing_names]
    if not new_fields:
        return table

    table.schema = list(table.schema) + new_fields
    updated = client.update_table(table, ["schema"])
    added = ", ".join(field.name for field in new_fields)
    print(f"Added columns to {table.table_id}: {added}")
    return updated


def recreate_table(
    client: bigquery.Client,
    table_id: str,
    schema: list[bigquery.SchemaField],
) -> bigquery.Table:
    """Drop and recreate a table with the desired schema (for full reloads)."""
    client.delete_table(table_id, not_found_ok=True)
    table = bigquery.Table(table_id, schema=schema)
    _apply_table_options(table)
    created = client.create_table(table)
    print(f"Recreated table: {table_id}")
    return created


def ensure_table(
    client: bigquery.Client,
    table_id: str,
    schema: list[bigquery.SchemaField],
) -> bigquery.Table:
    table = bigquery.Table(table_id, schema=schema)
    _apply_table_options(table)

    try:
        return client.create_table(table)
    except Conflict:
        existing = client.get_table(table_id)
        return sync_table_schema(client, existing, schema)


def ensure_staging_table(
    client: bigquery.Client,
    table_id: str,
    schema: list[bigquery.SchemaField],
) -> bigquery.Table:
    """Staging table mirrors CSV columns only (no partition/cluster required)."""
    table = bigquery.Table(table_id, schema=schema)
    try:
        return client.create_table(table)
    except Conflict:
        existing = client.get_table(table_id)
        return sync_table_schema(client, existing, schema)


def default_gcs_uri(config: dict) -> str:
    storage_cfg = config["storage"]
    return f"gs://{storage_cfg['bucket']}/{storage_cfg['raw_prefix']}/{storage_cfg['output_filename']}"


def build_load_job_config(
    schema: list[bigquery.SchemaField],
    write_disposition: str,
) -> bigquery.LoadJobConfig:
    """Build a load job config. Partitioning is owned by ensure_table(), not the load job."""
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=False,
        schema=schema,
        write_disposition=write_disposition,
    )
    if write_disposition == bigquery.WriteDisposition.WRITE_APPEND:
        job_config.schema_update_options = [
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
        ]
    return job_config


def load_csv_from_gcs(
    client: bigquery.Client,
    table_id: str,
    gcs_uri: str,
    schema: list[bigquery.SchemaField],
    write_disposition: str,
) -> bigquery.LoadJob:
    job_config = build_load_job_config(schema, write_disposition)
    load_job = client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
    load_job.result()
    return load_job


def load_csv_from_file(
    client: bigquery.Client,
    table_id: str,
    csv_path: Path,
    schema: list[bigquery.SchemaField],
    write_disposition: str,
) -> bigquery.LoadJob:
    job_config = build_load_job_config(schema, write_disposition)
    with csv_path.open("rb") as handle:
        load_job = client.load_table_from_file(handle, table_id, job_config=job_config)
    load_job.result()
    return load_job


def ensure_simple_table(
    client: bigquery.Client,
    table_id: str,
    schema: list[bigquery.SchemaField],
) -> bigquery.Table:
    table = bigquery.Table(table_id, schema=schema)
    try:
        return client.create_table(table)
    except Conflict:
        existing = client.get_table(table_id)
        return sync_table_schema(client, existing, schema)


def recreate_simple_table(
    client: bigquery.Client,
    table_id: str,
    schema: list[bigquery.SchemaField],
) -> bigquery.Table:
    client.delete_table(table_id, not_found_ok=True)
    created = client.create_table(bigquery.Table(table_id, schema=schema))
    print(f"Recreated table: {table_id}")
    return created


def merge_staging_into_target(
    client: bigquery.Client,
    target_table_id: str,
    staging_table_id: str,
    columns: list[str],
) -> None:
    """Insert staging rows into target, skipping duplicates on transaction_id."""
    column_list = ", ".join(columns)
    insert_columns = ", ".join(f"S.{column}" for column in columns)
    merge_sql = f"""
        MERGE `{target_table_id}` AS T
        USING `{staging_table_id}` AS S
        ON T.{DEDUP_KEY} = S.{DEDUP_KEY}
        WHEN NOT MATCHED BY TARGET THEN
          INSERT ({column_list})
          VALUES ({insert_columns})
    """
    client.query(merge_sql).result()


def truncate_table(client: bigquery.Client, table_id: str) -> None:
    """Truncate table data while preserving schema (unlike WRITE_TRUNCATE loads)."""
    client.query(f"TRUNCATE TABLE `{table_id}`").result()


def truncate_staging_table(client: bigquery.Client, staging_table_id: str) -> None:
    truncate_table(client, staging_table_id)


def stamp_ingested_at(client: bigquery.Client, table_id: str) -> None:
    """Set ingested_at on rows loaded in the current job. No-op if column is absent."""
    table = client.get_table(table_id)
    if not any(field.name == "ingested_at" for field in table.schema):
        print(f"Skipping ingested_at stamp: column not present on {table_id}")
        return

    ingested_at = datetime.now(timezone.utc)
    update_query = f"""
        UPDATE `{table_id}`
        SET ingested_at = @ingested_at
        WHERE ingested_at IS NULL
    """
    client.query(
        update_query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ingested_at", "TIMESTAMP", ingested_at)
            ]
        ),
    ).result()


def load_transactions(
    client: bigquery.Client,
    target_table_id: str,
    staging_table_id: str,
    gcs_uri: str,
    csv_schema: list[bigquery.SchemaField],
    replace: bool,
) -> None:
    csv_columns = [field.name for field in csv_schema]

    if replace:
        # TRUNCATE preserves the full table schema (including ingested_at).
        # A WRITE_TRUNCATE load would reset schema to CSV columns only.
        truncate_table(client, target_table_id)
        load_job = load_csv_from_gcs(
            client,
            target_table_id,
            gcs_uri,
            csv_schema,
            bigquery.WriteDisposition.WRITE_APPEND,
        )
        print(f"Loaded {load_job.output_rows:,} rows into {target_table_id}")
        stamp_ingested_at(client, target_table_id)
        return

    load_job = load_csv_from_gcs(
        client,
        staging_table_id,
        gcs_uri,
        csv_schema,
        bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    print(f"Staged {load_job.output_rows:,} rows in {staging_table_id}")
    merge_staging_into_target(client, target_table_id, staging_table_id, csv_columns)
    truncate_staging_table(client, staging_table_id)
    stamp_ingested_at(client, target_table_id)


def dimensions_local_dir(config: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    rel = config.get("storage", {}).get("dimensions_dir", "dimensions")
    return PROJECT_ROOT / "data" / rel


def dimension_table_name(config: dict, logical_name: str) -> str:
    bq_cfg = config["bigquery"]
    return bq_cfg.get(f"{logical_name}_table", logical_name)


def load_dimension_tables(
    client: bigquery.Client,
    config: dict,
    dimensions_dir: Path,
    *,
    replace: bool,
) -> None:
    dataset_id = config["bigquery"]["dataset"]
    project_id = config["gcp"]["project_id"]

    for logical_name, schema_file in DIMENSION_TABLES.items():
        csv_name = f"{logical_name}.csv"
        csv_path = dimensions_dir / csv_name
        if not csv_path.exists():
            print(f"Skipping {logical_name}: {csv_path} not found")
            continue

        schema = load_schema(SCHEMAS_DIR / schema_file)
        table_name = dimension_table_name(config, logical_name)
        table_id = f"{project_id}.{dataset_id}.{table_name}"

        if replace:
            recreate_simple_table(client, table_id, schema)
        else:
            ensure_simple_table(client, table_id, schema)

        load_job = load_csv_from_file(
            client,
            table_id,
            csv_path,
            schema,
            bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        print(f"Loaded {load_job.output_rows:,} rows into {table_id} from {csv_path.name}")


def print_table_stats(client: bigquery.Client, table_id: str) -> None:
    table = client.get_table(table_id)
    has_ingested_at = any(field.name == "ingested_at" for field in table.schema)
    print(f"Loaded table: {table_id}")
    print(f"Rows: {table.num_rows:,}")

    ingested_expr = (
        "COUNTIF(ingested_at IS NOT NULL) AS rows_with_ingested_at"
        if has_ingested_at
        else "0 AS rows_with_ingested_at"
    )
    stats_query = f"""
        SELECT
          COUNT(*) AS total_rows,
          COUNTIF(is_fraud) AS fraud_rows,
          ROUND(COUNTIF(is_fraud) / COUNT(*), 4) AS fraud_rate,
          COUNT(DISTINCT typology) AS typology_count,
          {ingested_expr}
        FROM `{table_id}`
    """
    rows = list(client.query(stats_query).result())
    if rows:
        row = rows[0]
        ingested_msg = (
            f", ingested_at set: {row.rows_with_ingested_at:,}"
            if has_ingested_at
            else ""
        )
        print(
            f"Fraud rows: {row.fraud_rows:,} "
            f"({row.fraud_rate:.2%}), typologies: {row.typology_count}"
            f"{ingested_msg}"
        )


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)

    print(f"Profile: {profile} ({config_path.name})")

    gcp = config["gcp"]
    bq_cfg = config["bigquery"]
    dataset_id = bq_cfg["dataset"]
    table_name = bq_cfg["raw_table"]
    staging_table_name = bq_cfg.get("staging_table", f"{table_name}_staging")
    gcs_uri = args.gcs_uri or default_gcs_uri(config)

    client = bigquery.Client(project=gcp["project_id"])
    table_schema = load_schema(SCHEMA_PATH)
    csv_schema = csv_load_schema(table_schema)

    ensure_dataset(client, dataset_id, bq_cfg["location"])
    target_table_id = f"{gcp['project_id']}.{dataset_id}.{table_name}"
    staging_table_id = f"{gcp['project_id']}.{dataset_id}.{staging_table_name}"

    if args.replace:
        recreate_table(client, target_table_id, table_schema)
    else:
        ensure_table(client, target_table_id, table_schema)
    if not args.replace:
        ensure_staging_table(client, staging_table_id, csv_schema)

    mode = "replace" if args.replace else "append (dedup on transaction_id)"
    print(f"Loading from {gcs_uri} [{mode}]")
    load_transactions(
        client,
        target_table_id,
        staging_table_id,
        gcs_uri,
        csv_schema,
        replace=args.replace,
    )
    print_table_stats(client, target_table_id)

    if args.load_dimensions:
        dim_dir = dimensions_local_dir(config, args.dimensions_dir)
        print(f"Loading dimensions from {dim_dir}")
        load_dimension_tables(client, config, dim_dir, replace=args.replace)


if __name__ == "__main__":
    main()
