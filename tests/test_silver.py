from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def test_bronze_to_silver_row_reconciliation(
    bronze_manifest: dict[str, Any],
    silver_quality: dict[str, Any],
) -> None:
    removed_rows = (
        bronze_manifest["row_count"]
        - silver_quality["row_count"]
    )

    null_status_rows = bronze_manifest[
        "loan_status_distribution"
    ].get("__NULL__", 0)

    assert removed_rows == 33
    assert removed_rows == null_status_rows


def test_silver_primary_key_and_domain_rules(
    project_root: Path,
    silver_quality: dict[str, Any],
    db: duckdb.DuckDBPyConnection,
) -> None:
    silver_file = (
        project_root
        / "data"
        / "silver"
        / "loans_clean.parquet"
    )

    assert silver_file.exists()

    path = sql_path(silver_file)

    result = db.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,

            COUNT(DISTINCT loan_id)
                AS distinct_loan_ids,

            COUNT(*)
                FILTER (WHERE loan_id IS NULL)
                AS missing_loan_ids,

            COUNT(*)
                FILTER (
                    WHERE loan_amount IS NULL
                       OR loan_amount <= 0
                ) AS invalid_loan_amounts,

            COUNT(*)
                FILTER (
                    WHERE interest_rate_pct IS NULL
                       OR interest_rate_pct < 0
                       OR interest_rate_pct > 100
                ) AS invalid_interest_rates,

            COUNT(*)
                FILTER (
                    WHERE term_months NOT IN (36, 60)
                       OR term_months IS NULL
                ) AS invalid_terms,

            COUNT(*)
                FILTER (
                    WHERE grade NOT IN (
                        'A', 'B', 'C', 'D',
                        'E', 'F', 'G'
                    )
                    OR grade IS NULL
                ) AS invalid_grades,

            COUNT(*)
                FILTER (
                    WHERE status_group = 'unknown'
                ) AS unknown_statuses

        FROM read_parquet('{path}')
        """
    ).fetchone()

    assert result[0] == silver_quality["row_count"]
    assert result[1] == result[0]
    assert result[2] == 0
    assert result[3] == 0
    assert result[4] == 0
    assert result[5] == 0
    assert result[6] == 0
    assert result[7] == 0


def test_silver_status_distribution(
    project_root: Path,
    silver_quality: dict[str, Any],
    db: duckdb.DuckDBPyConnection,
) -> None:
    silver_file = (
        project_root
        / "data"
        / "silver"
        / "loans_clean.parquet"
    )

    path = sql_path(silver_file)

    rows = db.execute(
        f"""
        SELECT
            status_group,
            COUNT(*) AS loans
        FROM read_parquet('{path}')
        GROUP BY status_group
        """
    ).fetchall()

    observed_distribution = {
        status: int(count)
        for status, count in rows
    }

    expected_distribution = silver_quality[
        "status_group_distribution"
    ]

    assert observed_distribution == expected_distribution

    assert (
        sum(observed_distribution.values())
        == silver_quality["row_count"]
    )
