from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def test_gold_manifest_files_exist(
    project_root: Path,
    gold_manifest: dict[str, Any],
) -> None:
    for file_metadata in gold_manifest["files"].values():
        file_path = (
            project_root
            / file_metadata["relative_path"]
        )

        assert file_path.exists(), (
            f"Gold output not found: {file_path}"
        )

        assert (
            file_path.stat().st_size
            == file_metadata["size_bytes"]
        )

        assert len(file_metadata["sha256"]) == 64


def test_gold_analysis_cohort_rules(
    project_root: Path,
    gold_manifest: dict[str, Any],
    db: duckdb.DuckDBPyConnection,
) -> None:
    analysis_file = (
        project_root
        / "data"
        / "gold"
        / "analysis_cohort.parquet"
    )

    path = sql_path(analysis_file)

    result = db.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,

            COUNT(DISTINCT loan_id)
                AS distinct_loan_ids,

            MIN(issue_date) AS first_issue_date,
            MAX(issue_date) AS last_issue_date,

            COUNT(*)
                FILTER (
                    WHERE term_months <> 36
                       OR term_months IS NULL
                ) AS invalid_terms,

            COUNT(*)
                FILTER (
                    WHERE issue_date < DATE '2014-01-01'
                       OR issue_date >= DATE '2016-01-01'
                ) AS invalid_dates,

            COUNT(*)
                FILTER (
                    WHERE default_flag IS NULL
                       OR default_flag NOT IN (0, 1)
                ) AS invalid_default_flags,

            COUNT(*)
                FILTER (
                    WHERE (
                        default_flag = 1
                        AND outcome <> 'Defaulted'
                    )
                    OR (
                        default_flag = 0
                        AND outcome <> 'Fully paid'
                    )
                ) AS inconsistent_outcomes

        FROM read_parquet('{path}')
        """
    ).fetchone()

    assert result[0] == 445_596
    assert result[0] == gold_manifest["row_count"]
    assert result[1] == result[0]

    assert result[2].isoformat() == "2014-01-01"
    assert result[3].isoformat() == "2015-12-01"

    assert result[4] == 0
    assert result[5] == 0
    assert result[6] == 0
    assert result[7] == 0


def test_gold_aggregate_tables_reconcile(
    project_root: Path,
    gold_manifest: dict[str, Any],
    db: duckdb.DuckDBPyConnection,
) -> None:
    expected_total = gold_manifest["row_count"]

    aggregate_files = (
        "quarterly_risk_metrics.csv",
        "risk_by_grade.csv",
        "risk_by_purpose.csv",
    )

    for filename in aggregate_files:
        file_path = (
            project_root
            / "data"
            / "gold"
            / filename
        )

        path = sql_path(file_path)

        total = db.execute(
            f"""
            SELECT
                CAST(SUM(total_loans) AS BIGINT)
            FROM read_csv_auto(
                '{path}',
                header = true
            )
            """
        ).fetchone()[0]

        assert total == expected_total, (
            f"{filename} reconciles to {total:,}, "
            f"expected {expected_total:,}"
        )
