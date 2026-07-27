from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def bronze_manifest(project_root: Path) -> dict[str, Any]:
    return read_json(
        project_root
        / "data"
        / "bronze"
        / "metadata"
        / "accepted_loans_manifest.json"
    )


@pytest.fixture(scope="session")
def silver_quality(project_root: Path) -> dict[str, Any]:
    return read_json(
        project_root
        / "data"
        / "silver"
        / "metadata"
        / "loans_clean_quality.json"
    )


@pytest.fixture(scope="session")
def gold_manifest(project_root: Path) -> dict[str, Any]:
    return read_json(
        project_root
        / "data"
        / "gold"
        / "metadata"
        / "gold_manifest.json"
    )


@pytest.fixture(scope="session")
def ai_metrics(project_root: Path) -> dict[str, Any]:
    return read_json(
        project_root
        / "data"
        / "gold"
        / "ai_validation_metrics.json"
    )


@pytest.fixture(scope="session")
def db() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")

    yield connection

    connection.close()
