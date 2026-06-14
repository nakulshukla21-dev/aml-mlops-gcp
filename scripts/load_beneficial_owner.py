"""Load beneficial_owner CSV for a config profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import bigquery

from src.config import add_config_arguments, load_config_from_args
from src.load_to_bigquery import (
    SCHEMAS_DIR,
    dimension_table_name,
    dimensions_local_dir,
    load_csv_from_file,
    load_schema,
    recreate_simple_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load beneficial_owner dimension CSV.")
    add_config_arguments(parser)
    parser.add_argument(
        "--dimensions-dir",
        type=Path,
        default=None,
        help="Directory containing beneficial_owner.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    dim_dir = dimensions_local_dir(config, args.dimensions_dir)
    csv_path = dim_dir / "beneficial_owner.csv"

    client = bigquery.Client(project=config["gcp"]["project_id"])
    schema = load_schema(SCHEMAS_DIR / "beneficial_owner.json")
    table_id = (
        f"{config['gcp']['project_id']}."
        f"{config['bigquery']['dataset']}."
        f"{dimension_table_name(config, 'beneficial_owner')}"
    )
    recreate_simple_table(client, table_id, schema)
    job = load_csv_from_file(
        client,
        table_id,
        csv_path,
        schema,
        bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    print(f"Profile: {config.get('profile')} ({config_path.name})")
    print(f"Loaded {job.output_rows:,} rows into {table_id}")


if __name__ == "__main__":
    main()
