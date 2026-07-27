from __future__ import annotations

import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "raw"
    / "accepted_2007_to_2018Q4.csv.gz"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "metadata"
    / "accepted_loans_manifest.json"
)


def calculate_sha256(path: Path) -> str:
    """Calculate a SHA-256 checksum without loading the full file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_csv_header(path: Path) -> list[str]:
    """Read the compressed CSV header."""
    with gzip.open(
        path,
        mode="rt",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file)
        return next(reader)


def sql_safe_path(path: Path) -> str:
    """Escape a filesystem path for use inside a DuckDB SQL string."""
    return str(path.resolve()).replace("'", "''")


def build_manifest() -> dict:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Bronze source file not found: {SOURCE_FILE}"
        )

    columns = read_csv_header(SOURCE_FILE)
    source_path = sql_safe_path(SOURCE_FILE)

    connection = duckdb.connect()
    connection.execute("SET threads = 4")

    connection.execute(
        f"""
        CREATE TEMP TABLE bronze_profile AS
        SELECT
            TRY_STRPTIME(issue_d, '%b-%Y') AS issue_date,
            COALESCE(
                NULLIF(TRIM(loan_status), ''),
                '__NULL__'
            ) AS loan_status
        FROM read_csv_auto(
            '{source_path}',
            header = true,
            compression = 'gzip',
            all_varchar = true,
            ignore_errors = true,
            sample_size = 200000
        )
        """
    )

    profile = connection.execute(
        """
        SELECT
            COUNT(*) AS row_count,
            MIN(issue_date) AS first_issue_date,
            MAX(issue_date) AS last_issue_date,
            COUNT(DISTINCT loan_status) AS distinct_loan_statuses
        FROM bronze_profile
        """
    ).fetchone()

    status_rows = connection.execute(
        """
        SELECT
            loan_status,
            COUNT(*) AS loans
        FROM bronze_profile
        GROUP BY loan_status
        ORDER BY loans DESC
        """
    ).fetchall()

    connection.close()

    first_issue_date = (
        profile[1].date().isoformat()
        if profile[1] is not None
        else None
    )

    last_issue_date = (
        profile[2].date().isoformat()
        if profile[2] is not None
        else None
    )

    return {
        "dataset": "Lending Club Loan Data",
        "layer": "bronze",
        "source_file": SOURCE_FILE.name,
        "source_path": str(SOURCE_FILE.relative_to(PROJECT_ROOT)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": SOURCE_FILE.stat().st_size,
        "sha256": calculate_sha256(SOURCE_FILE),
        "row_count": int(profile[0]),
        "column_count": len(columns),
        "first_issue_date": first_issue_date,
        "last_issue_date": last_issue_date,
        "distinct_loan_statuses": int(profile[3]),
        "loan_status_distribution": {
            status: int(count)
            for status, count in status_rows
        },
        "columns": columns,
    }


def main() -> None:
    manifest = build_manifest()

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Bronze layer validated successfully.")
    print(f"Source: {manifest['source_path']}")
    print(f"Rows: {manifest['row_count']:,}")
    print(f"Columns: {manifest['column_count']}")
    print(
        "Period: "
        f"{manifest['first_issue_date']} "
        f"to {manifest['last_issue_date']}"
    )
    print(f"Manifest: {MANIFEST_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
