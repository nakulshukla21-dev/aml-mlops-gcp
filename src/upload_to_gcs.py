"""Upload generated transaction CSV to Cloud Storage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from src.config import add_config_arguments, default_data_path, load_config_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload transactions CSV to GCS.")
    add_config_arguments(parser)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local CSV file to upload. Defaults to data/<output_filename> from config.",
    )
    parser.add_argument(
        "--dated-prefix",
        action="store_true",
        help="Upload under a date partition (transactions/raw/dt=YYYY-MM-DD/).",
    )
    return parser.parse_args()


def build_blob_name(prefix: str, filename: str, dated: bool) -> str:
    if dated:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"{prefix}/dt={today}/{filename}"
    return f"{prefix}/{filename}"


def upload_file(local_path: Path, bucket_name: str, blob_name: str, project_id: str) -> str:
    if not local_path.exists():
        raise FileNotFoundError(f"Input file not found: {local_path}")

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type="text/csv")

    uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Uploaded {local_path} -> {uri}")
    return uri


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    input_path = args.input or default_data_path(config)

    print(f"Profile: {profile} ({config_path.name})")

    gcp = config["gcp"]
    storage_cfg = config["storage"]
    blob_name = build_blob_name(
        storage_cfg["raw_prefix"],
        storage_cfg["output_filename"],
        dated=args.dated_prefix,
    )

    upload_file(
        local_path=input_path,
        bucket_name=storage_cfg["bucket"],
        blob_name=blob_name,
        project_id=gcp["project_id"],
    )


if __name__ == "__main__":
    main()
