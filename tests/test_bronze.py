from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "id",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "annual_inc",
    "issue_d",
    "loan_status",
    "purpose",
    "dti",
    "addr_state",
    "emp_length",
    "home_ownership",
    "verification_status",
}


def test_bronze_manifest_internal_consistency(
    bronze_manifest: dict[str, Any],
) -> None:
    assert bronze_manifest["layer"] == "bronze"
    assert bronze_manifest["row_count"] == 2_260_701
    assert bronze_manifest["column_count"] == 151

    status_distribution = bronze_manifest[
        "loan_status_distribution"
    ]

    assert sum(status_distribution.values()) == bronze_manifest[
        "row_count"
    ]

    assert len(status_distribution) == bronze_manifest[
        "distinct_loan_statuses"
    ]

    first_date = date.fromisoformat(
        bronze_manifest["first_issue_date"]
    )
    last_date = date.fromisoformat(
        bronze_manifest["last_issue_date"]
    )

    assert first_date <= last_date


def test_bronze_source_file_matches_manifest(
    project_root: Path,
    bronze_manifest: dict[str, Any],
) -> None:
    source_file = (
        project_root
        / bronze_manifest["source_path"]
    )

    assert source_file.exists()
    assert source_file.is_file()

    assert source_file.name == bronze_manifest["source_file"]

    assert (
        source_file.stat().st_size
        == bronze_manifest["file_size_bytes"]
    )

    assert len(bronze_manifest["sha256"]) == 64


def test_bronze_contains_required_columns(
    bronze_manifest: dict[str, Any],
) -> None:
    available_columns = set(bronze_manifest["columns"])

    missing_columns = REQUIRED_COLUMNS - available_columns

    assert not missing_columns, (
        f"Missing required Bronze columns: "
        f"{sorted(missing_columns)}"
    )
