"""Load project configuration and resolve dev/train profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEV_CONFIG_PATH = CONFIG_DIR / "config.dev.yaml"

CONFIG_PROFILES: dict[str, Path] = {
    "train": CONFIG_PATH,
    "dev": DEV_CONFIG_PATH,
}


def load_config(path: Path | None = None) -> dict:
    config_path = path or CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(profile: str = "train", config: Path | None = None) -> Path:
    """Return config file path. Explicit --config overrides --profile."""
    if config is not None:
        return config
    try:
        return CONFIG_PROFILES[profile]
    except KeyError as exc:
        valid = ", ".join(CONFIG_PROFILES)
        raise ValueError(f"Unknown profile '{profile}'. Choose from: {valid}") from exc


def default_data_path(config: dict) -> Path:
    filename = config["storage"]["output_filename"]
    return PROJECT_ROOT / "data" / filename


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=sorted(CONFIG_PROFILES),
        default="train",
        help="Config profile: dev (25k, isolated paths) or train (200k). Default: train.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a config YAML file. Overrides --profile.",
    )


def load_config_from_args(args: argparse.Namespace) -> tuple[dict, Path]:
    config_path = resolve_config_path(args.profile, args.config)
    config = load_config(config_path)
    return config, config_path
