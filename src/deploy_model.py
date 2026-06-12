"""Deploy a trained AutoML model to a Vertex AI endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import aiplatform

from src.automl_utils import (
    artifact_path,
    deploy_artifact_path,
    deployment_config,
    load_run_artifact,
    save_run_artifact,
    utc_timestamp,
)
from src.config import add_config_arguments, load_config_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy AutoML model to Vertex AI endpoint.")
    add_config_arguments(parser)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Path to automl run artifact JSON. Defaults to artifacts/automl_<profile>.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Vertex model resource name. Overrides artifact model_resource_name.",
    )
    parser.add_argument(
        "--undeploy",
        action="store_true",
        help="Undeploy model and delete the endpoint recorded in artifacts/deploy_<profile>.json.",
    )
    return parser.parse_args()


def deploy_model_to_vertex(
    config: dict,
    train_artifact: dict,
    model_resource_name: str,
) -> dict:
    deploy_cfg = deployment_config(config)
    profile = config.get("profile", "train")
    project_id = train_artifact["project_id"]
    region = train_artifact["region"]
    timestamp = utc_timestamp()

    aiplatform.init(project=project_id, location=region)
    model = aiplatform.Model(model_resource_name)

    endpoint_display_name = deploy_cfg.get(
        "endpoint_display_name",
        f"aml-fraud-endpoint-{profile}",
    )
    deployed_model_display_name = deploy_cfg.get(
        "deployed_model_display_name",
        f"aml-fraud-scorer-{profile}-{timestamp}",
    )
    machine_type = deploy_cfg.get("machine_type", "n1-standard-2")
    min_replicas = deploy_cfg.get("min_replica_count", 1)
    max_replicas = deploy_cfg.get("max_replica_count", 1)

    print(f"Deploying model to Vertex endpoint: {endpoint_display_name}")
    print(f"Machine type: {machine_type} (replicas {min_replicas}-{max_replicas})")

    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
        sync=True,
    )

    payload = {
        "profile": profile,
        "project_id": project_id,
        "region": region,
        "model_resource_name": model_resource_name,
        "model_display_name": train_artifact.get("model_display_name"),
        "endpoint_resource_name": endpoint.resource_name,
        "endpoint_display_name": endpoint_display_name,
        "deployed_model_display_name": deployed_model_display_name,
        "machine_type": machine_type,
        "min_replica_count": min_replicas,
        "max_replica_count": max_replicas,
        "deployed_at": timestamp,
    }
    save_run_artifact(deploy_artifact_path(config), payload)
    print(f"Endpoint ready: {endpoint.resource_name}")
    return payload


def undeploy_endpoint(config: dict) -> None:
    deploy_path = deploy_artifact_path(config)
    if not deploy_path.exists():
        raise RuntimeError(
            f"No deployment artifact found at {deploy_path}. Nothing to undeploy."
        )

    deploy_artifact = load_run_artifact(deploy_path)
    aiplatform.init(
        project=deploy_artifact["project_id"],
        location=deploy_artifact["region"],
    )
    endpoint = aiplatform.Endpoint(deploy_artifact["endpoint_resource_name"])
    print(f"Deleting endpoint: {endpoint.resource_name}")
    endpoint.delete(force=True)
    deploy_path.unlink()
    print("Endpoint deleted and deploy artifact removed.")


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    print(f"Profile: {profile} ({config_path.name})")

    if args.undeploy:
        undeploy_endpoint(config)
        return

    train_artifact = load_run_artifact(args.artifact or artifact_path(config))
    model_resource_name = args.model or train_artifact["model_resource_name"]
    deploy_model_to_vertex(config, train_artifact, model_resource_name)


if __name__ == "__main__":
    main()
