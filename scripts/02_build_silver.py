from __future__ import annotations

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

SILVER_FILE = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "loans_clean.parquet"
)

QUALITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "metadata"
    / "loans_clean_quality.json"
)


def sql_safe_path(path: Path) -> str:
    """Escape a filesystem path for use inside DuckDB SQL."""
    return str(path.resolve()).replace("'", "''")


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 without loading the full file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_silver() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Bronze source file not found: {SOURCE_FILE}"
        )

    SILVER_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_FILE.parent.mkdir(parents=True, exist_ok=True)

    source_path = sql_safe_path(SOURCE_FILE)
    silver_path = sql_safe_path(SILVER_FILE)

    connection = duckdb.connect()
    connection.execute("SET threads = 4")

    connection.execute(
        f"""
        COPY (
            WITH raw AS (
                SELECT *
                FROM read_csv_auto(
                    '{source_path}',
                    header = true,
                    compression = 'gzip',
                    all_varchar = true,
                    sample_size = 200000
                )
            ),

            typed AS (
                SELECT
                    NULLIF(TRIM(id), '') AS loan_id,

                    TRY_CAST(
                        NULLIF(TRIM(loan_amnt), '')
                        AS DOUBLE
                    ) AS loan_amount,

                    TRY_CAST(
                        NULLIF(TRIM(funded_amnt), '')
                        AS DOUBLE
                    ) AS funded_amount,

                    TRY_CAST(
                        REGEXP_EXTRACT(
                            TRIM(term),
                            '([0-9]+)',
                            1
                        )
                        AS SMALLINT
                    ) AS term_months,

                    TRY_CAST(
                        REPLACE(
                            NULLIF(TRIM(int_rate), ''),
                            '%',
                            ''
                        )
                        AS DOUBLE
                    ) AS interest_rate_pct,

                    TRY_CAST(
                        NULLIF(TRIM(installment), '')
                        AS DOUBLE
                    ) AS installment,

                    NULLIF(TRIM(grade), '') AS grade,
                    NULLIF(TRIM(sub_grade), '') AS sub_grade,

                    NULLIF(
                        TRIM(emp_length),
                        ''
                    ) AS employment_length_raw,

                    CASE
                        WHEN TRIM(emp_length) = '< 1 year'
                            THEN 0
                        WHEN TRIM(emp_length) = '10+ years'
                            THEN 10
                        ELSE TRY_CAST(
                            REGEXP_EXTRACT(
                                TRIM(emp_length),
                                '([0-9]+)',
                                1
                            )
                            AS SMALLINT
                        )
                    END AS employment_length_years_lower_bound,

                    NULLIF(
                        TRIM(home_ownership),
                        ''
                    ) AS home_ownership,

                    TRY_CAST(
                        NULLIF(TRIM(annual_inc), '')
                        AS DOUBLE
                    ) AS annual_income,

                    NULLIF(
                        TRIM(verification_status),
                        ''
                    ) AS verification_status,

                    TRY_STRPTIME(
                        issue_d,
                        '%b-%Y'
                    )::DATE AS issue_date,

                    NULLIF(
                        TRIM(loan_status),
                        ''
                    ) AS loan_status,

                    NULLIF(TRIM(purpose), '') AS purpose,
                    NULLIF(TRIM(addr_state), '') AS state,

                    TRY_CAST(
                        NULLIF(TRIM(dti), '')
                        AS DOUBLE
                    ) AS debt_to_income_ratio,

                    TRY_CAST(
                        NULLIF(TRIM(delinq_2yrs), '')
                        AS INTEGER
                    ) AS delinquencies_2y,

                    TRY_STRPTIME(
                        earliest_cr_line,
                        '%b-%Y'
                    )::DATE AS earliest_credit_line_date,

                    TRY_CAST(
                        NULLIF(TRIM(open_acc), '')
                        AS INTEGER
                    ) AS open_accounts,

                    TRY_CAST(
                        NULLIF(TRIM(pub_rec), '')
                        AS INTEGER
                    ) AS public_records,

                    TRY_CAST(
                        NULLIF(TRIM(revol_bal), '')
                        AS DOUBLE
                    ) AS revolving_balance,

                    TRY_CAST(
                        REPLACE(
                            NULLIF(TRIM(revol_util), ''),
                            '%',
                            ''
                        )
                        AS DOUBLE
                    ) AS revolving_utilization_pct,

                    TRY_CAST(
                        NULLIF(TRIM(total_acc), '')
                        AS INTEGER
                    ) AS total_accounts,

                    NULLIF(
                        TRIM(initial_list_status),
                        ''
                    ) AS initial_list_status,

                    NULLIF(
                        TRIM(application_type),
                        ''
                    ) AS application_type,

                    TRY_CAST(
                        NULLIF(TRIM(mort_acc), '')
                        AS INTEGER
                    ) AS mortgage_accounts,

                    TRY_CAST(
                        NULLIF(
                            TRIM(pub_rec_bankruptcies),
                            ''
                        )
                        AS INTEGER
                    ) AS public_record_bankruptcies,

                    TRY_CAST(
                        NULLIF(TRIM(fico_range_low), '')
                        AS DOUBLE
                    ) AS fico_range_low,

                    TRY_CAST(
                        NULLIF(TRIM(fico_range_high), '')
                        AS DOUBLE
                    ) AS fico_range_high

                FROM raw

                WHERE TRY_STRPTIME(
                    issue_d,
                    '%b-%Y'
                ) IS NOT NULL

                  AND NULLIF(
                    TRIM(loan_status),
                    ''
                  ) IS NOT NULL
            )

            SELECT
                loan_id,

                MD5(
                    CONCAT_WS(
                        '|',
                        COALESCE(loan_id, ''),
                        CAST(issue_date AS VARCHAR),
                        COALESCE(
                            CAST(loan_amount AS VARCHAR),
                            ''
                        ),
                        COALESCE(grade, ''),
                        COALESCE(sub_grade, ''),
                        COALESCE(loan_status, '')
                    )
                ) AS record_hash,

                loan_amount,
                funded_amount,
                term_months,
                interest_rate_pct,
                installment,
                grade,
                sub_grade,

                CASE
                    WHEN grade IN ('D', 'E')
                        THEN TRUE
                    ELSE FALSE
                END AS is_grade_de,

                employment_length_raw,
                employment_length_years_lower_bound,
                home_ownership,
                annual_income,
                verification_status,

                issue_date,
                EXTRACT(YEAR FROM issue_date)::INTEGER
                    AS issue_year,

                STRFTIME(issue_date, '%Y')
                    || '-Q'
                    || CAST(
                        QUARTER(issue_date)
                        AS VARCHAR
                    ) AS issue_quarter,

                loan_status,

                CASE
                    WHEN loan_status IN (
                        'Fully Paid',
                        'Does not meet the credit policy. Status:Fully Paid'
                    )
                        THEN 'resolved_paid'

                    WHEN loan_status IN (
                        'Charged Off',
                        'Default',
                        'Does not meet the credit policy. Status:Charged Off'
                    )
                        THEN 'resolved_default'

                    WHEN loan_status = 'Current'
                        THEN 'active_current'

                    WHEN loan_status IN (
                        'Late (16-30 days)',
                        'Late (31-120 days)',
                        'In Grace Period'
                    )
                        THEN 'active_delinquent'

                    ELSE 'unknown'
                END AS status_group,

                CASE
                    WHEN loan_status IN (
                        'Charged Off',
                        'Default',
                        'Does not meet the credit policy. Status:Charged Off'
                    )
                        THEN 1

                    WHEN loan_status IN (
                        'Fully Paid',
                        'Does not meet the credit policy. Status:Fully Paid'
                    )
                        THEN 0

                    ELSE NULL
                END AS default_flag,

                loan_status IN (
                    'Fully Paid',
                    'Charged Off',
                    'Default',
                    'Does not meet the credit policy. Status:Fully Paid',
                    'Does not meet the credit policy. Status:Charged Off'
                ) AS is_resolved,

                loan_status = 'Current'
                    AS is_current,

                loan_status IN (
                    'Late (16-30 days)',
                    'Late (31-120 days)',
                    'In Grace Period'
                ) AS is_delinquent,

                purpose,
                state,
                debt_to_income_ratio,
                delinquencies_2y,
                earliest_credit_line_date,

                DATE_DIFF(
                    'year',
                    earliest_credit_line_date,
                    issue_date
                ) AS credit_history_years,

                open_accounts,
                public_records,
                revolving_balance,
                revolving_utilization_pct,
                total_accounts,
                initial_list_status,
                application_type,
                mortgage_accounts,
                public_record_bankruptcies,
                fico_range_low,
                fico_range_high,

                (
                    fico_range_low
                    + fico_range_high
                ) / 2.0 AS fico_score_avg

            FROM typed
        )
        TO '{silver_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    )

    connection.close()


def build_quality_report() -> dict:
    silver_path = sql_safe_path(SILVER_FILE)

    connection = duckdb.connect()
    connection.execute("SET threads = 4")

    profile = connection.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT loan_id)
                FILTER (WHERE loan_id IS NOT NULL)
                AS distinct_non_null_loan_ids,

            COUNT(*)
                FILTER (WHERE loan_id IS NULL)
                AS missing_loan_ids,

            MIN(issue_date) AS first_issue_date,
            MAX(issue_date) AS last_issue_date,

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
                ) AS unknown_statuses,

            COUNT(*)
                FILTER (
                    WHERE default_flag IS NOT NULL
                ) AS resolved_rows,

            COUNT(*)
                FILTER (
                    WHERE default_flag = 1
                ) AS defaulted_rows,

            COUNT(*)
                FILTER (
                    WHERE annual_income IS NULL
                ) AS missing_annual_income,

            COUNT(*)
                FILTER (
                    WHERE debt_to_income_ratio IS NULL
                ) AS missing_dti,

            COUNT(*)
                FILTER (
                    WHERE fico_score_avg IS NULL
                ) AS missing_fico

        FROM read_parquet('{silver_path}')
        """
    ).fetchone()

    statuses = connection.execute(
        f"""
        SELECT
            status_group,
            COUNT(*) AS loans
        FROM read_parquet('{silver_path}')
        GROUP BY status_group
        ORDER BY loans DESC
        """
    ).fetchall()

    schema_rows = connection.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{silver_path}')
        """
    ).fetchall()

    connection.close()

    first_issue_date = (
        profile[3].isoformat()
        if profile[3] is not None
        else None
    )

    last_issue_date = (
        profile[4].isoformat()
        if profile[4] is not None
        else None
    )

    return {
        "dataset": "Lending Club accepted loans",
        "layer": "silver",
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_file": str(
            SOURCE_FILE.relative_to(PROJECT_ROOT)
        ),
        "output_file": str(
            SILVER_FILE.relative_to(PROJECT_ROOT)
        ),
        "output_file_size_bytes": SILVER_FILE.stat().st_size,
        "output_sha256": calculate_sha256(SILVER_FILE),
        "row_count": int(profile[0]),
        "distinct_non_null_loan_ids": int(profile[1]),
        "missing_loan_ids": int(profile[2]),
        "first_issue_date": first_issue_date,
        "last_issue_date": last_issue_date,
        "quality_checks": {
            "invalid_loan_amounts": int(profile[5]),
            "invalid_interest_rates": int(profile[6]),
            "invalid_terms": int(profile[7]),
            "invalid_grades": int(profile[8]),
            "unknown_statuses": int(profile[9]),
            "resolved_rows": int(profile[10]),
            "defaulted_rows": int(profile[11]),
            "missing_annual_income": int(profile[12]),
            "missing_dti": int(profile[13]),
            "missing_fico": int(profile[14]),
        },
        "status_group_distribution": {
            status: int(count)
            for status, count in statuses
        },
        "columns": [
            {
                "name": row[0],
                "type": row[1],
                "nullable": row[2],
            }
            for row in schema_rows
        ],
    }


def main() -> None:
    build_silver()
    quality_report = build_quality_report()

    QUALITY_FILE.write_text(
        json.dumps(
            quality_report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    checks = quality_report["quality_checks"]

    print("Silver layer built successfully.")
    print(
        f"Rows: {quality_report['row_count']:,}"
    )
    print(
        "Period: "
        f"{quality_report['first_issue_date']} "
        f"to {quality_report['last_issue_date']}"
    )
    print(
        "Resolved loans: "
        f"{checks['resolved_rows']:,}"
    )
    print(
        "Defaulted loans: "
        f"{checks['defaulted_rows']:,}"
    )
    print(
        "Unknown statuses: "
        f"{checks['unknown_statuses']:,}"
    )
    print(
        "Output: "
        f"{quality_report['output_file']}"
    )
    print(
        "Quality report: "
        f"{QUALITY_FILE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
