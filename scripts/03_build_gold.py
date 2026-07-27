from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_FILE = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "loans_clean.parquet"
)

GOLD_DIR = PROJECT_ROOT / "data" / "gold"

ANALYSIS_FILE = GOLD_DIR / "analysis_cohort.parquet"
DASHBOARD_FILE = GOLD_DIR / "dashboard_loans.csv"
QUARTERLY_FILE = GOLD_DIR / "quarterly_risk_metrics.csv"
GRADE_FILE = GOLD_DIR / "risk_by_grade.csv"
PURPOSE_FILE = GOLD_DIR / "risk_by_purpose.csv"

AI_METRICS_FILE = (
    GOLD_DIR
    / "ai_validation_metrics.json"
)

MANIFEST_FILE = (
    GOLD_DIR
    / "metadata"
    / "gold_manifest.json"
)


def sql_safe_path(path: Path) -> str:
    """Escape a path for use inside a DuckDB SQL string."""
    return str(path.resolve()).replace("'", "''")


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 without loading the full file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def remove_existing_outputs() -> None:
    """Allow the Gold build to be safely rerun."""
    for path in (
        ANALYSIS_FILE,
        DASHBOARD_FILE,
        QUARTERLY_FILE,
        GRADE_FILE,
        PURPOSE_FILE,
        AI_METRICS_FILE,
        MANIFEST_FILE,
    ):
        path.unlink(missing_ok=True)


def build_analysis_cohort(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    silver_path = sql_safe_path(SILVER_FILE)
    output_path = sql_safe_path(ANALYSIS_FILE)

    connection.execute(
        f"""
        COPY (
            SELECT
                loan_id,
                record_hash,
                issue_date,
                issue_year,
                issue_quarter,

                loan_amount,
                funded_amount,
                term_months,
                interest_rate_pct,
                installment,

                grade,
                sub_grade,
                is_grade_de,

                CASE
                    WHEN is_grade_de
                        THEN 'Grades D/E'
                    ELSE 'Other grades'
                END AS grade_group,

                annual_income,

                CASE
                    WHEN annual_income IS NULL
                        THEN 'Unknown'
                    WHEN annual_income < 40000
                        THEN '< $40k'
                    WHEN annual_income < 60000
                        THEN '$40k–$60k'
                    WHEN annual_income < 80000
                        THEN '$60k–$80k'
                    WHEN annual_income < 120000
                        THEN '$80k–$120k'
                    ELSE '$120k+'
                END AS income_band,

                debt_to_income_ratio,

                CASE
                    WHEN debt_to_income_ratio IS NULL
                        THEN 'Unknown'
                    WHEN debt_to_income_ratio < 10
                        THEN '< 10'
                    WHEN debt_to_income_ratio < 20
                        THEN '10–20'
                    WHEN debt_to_income_ratio < 30
                        THEN '20–30'
                    ELSE '30+'
                END AS dti_band,

                CASE
                    WHEN loan_amount < 5000
                        THEN '< $5k'
                    WHEN loan_amount < 10000
                        THEN '$5k–$10k'
                    WHEN loan_amount < 20000
                        THEN '$10k–$20k'
                    WHEN loan_amount < 30000
                        THEN '$20k–$30k'
                    ELSE '$30k+'
                END AS loan_amount_band,

                purpose,
                state,
                home_ownership,
                verification_status,
                employment_length_years_lower_bound,

                fico_score_avg,
                credit_history_years,
                revolving_utilization_pct,
                delinquencies_2y,
                open_accounts,
                total_accounts,
                mortgage_accounts,
                public_record_bankruptcies,

                default_flag,

                CASE
                    WHEN default_flag = 1
                        THEN 'Defaulted'
                    ELSE 'Fully paid'
                END AS outcome

            FROM read_parquet('{silver_path}')

            WHERE issue_date >= DATE '2014-01-01'
              AND issue_date < DATE '2016-01-01'
              AND term_months = 36
              AND is_resolved = TRUE
              AND default_flag IS NOT NULL
        )
        TO '{output_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    )


def build_dashboard_dataset(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    analysis_path = sql_safe_path(ANALYSIS_FILE)
    output_path = sql_safe_path(DASHBOARD_FILE)

    connection.execute(
        f"""
        COPY (
            SELECT
                loan_id,
                issue_date,
                issue_year,
                issue_quarter,
                loan_amount,
                loan_amount_band,
                interest_rate_pct,
                grade,
                sub_grade,
                grade_group,
                annual_income,
                income_band,
                debt_to_income_ratio,
                dti_band,
                purpose,
                state,
                home_ownership,
                verification_status,
                fico_score_avg,
                default_flag,
                outcome
            FROM read_parquet('{analysis_path}')
            ORDER BY issue_date, loan_id
        )
        TO '{output_path}'
        (
            FORMAT CSV,
            HEADER TRUE
        )
        """
    )


def build_quarterly_metrics(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    analysis_path = sql_safe_path(ANALYSIS_FILE)
    output_path = sql_safe_path(QUARTERLY_FILE)

    connection.execute(
        f"""
        COPY (
            SELECT
                issue_quarter,
                COUNT(*) AS total_loans,
                SUM(default_flag) AS defaulted_loans,

                ROUND(
                    100.0 * AVG(default_flag),
                    2
                ) AS default_rate_pct,

                SUM(
                    CASE
                        WHEN is_grade_de THEN 1
                        ELSE 0
                    END
                ) AS grade_de_loans,

                ROUND(
                    100.0 * AVG(
                        CASE
                            WHEN is_grade_de THEN 1
                            ELSE 0
                        END
                    ),
                    2
                ) AS grade_de_share_pct,

                ROUND(
                    100.0 * AVG(default_flag)
                    FILTER (WHERE is_grade_de),
                    2
                ) AS grade_de_default_rate_pct,

                ROUND(
                    100.0 * AVG(default_flag)
                    FILTER (WHERE NOT is_grade_de),
                    2
                ) AS other_grades_default_rate_pct,

                ROUND(AVG(loan_amount), 2)
                    AS average_loan_amount_usd,

                ROUND(MEDIAN(loan_amount), 2)
                    AS median_loan_amount_usd,

                ROUND(AVG(interest_rate_pct), 2)
                    AS average_interest_rate_pct,

                ROUND(AVG(annual_income), 2)
                    AS average_annual_income_usd,

                ROUND(AVG(fico_score_avg), 2)
                    AS average_fico_score

            FROM read_parquet('{analysis_path}')
            GROUP BY issue_quarter
            ORDER BY issue_quarter
        )
        TO '{output_path}'
        (
            FORMAT CSV,
            HEADER TRUE
        )
        """
    )


def build_grade_metrics(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    analysis_path = sql_safe_path(ANALYSIS_FILE)
    output_path = sql_safe_path(GRADE_FILE)

    connection.execute(
        f"""
        COPY (
            SELECT
                grade,
                COUNT(*) AS total_loans,

                ROUND(
                    100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER (),
                    2
                ) AS portfolio_share_pct,

                SUM(default_flag) AS defaulted_loans,

                ROUND(
                    100.0 * AVG(default_flag),
                    2
                ) AS default_rate_pct,

                ROUND(AVG(loan_amount), 2)
                    AS average_loan_amount_usd,

                ROUND(MEDIAN(loan_amount), 2)
                    AS median_loan_amount_usd,

                ROUND(AVG(interest_rate_pct), 2)
                    AS average_interest_rate_pct,

                ROUND(AVG(annual_income), 2)
                    AS average_annual_income_usd,

                ROUND(AVG(debt_to_income_ratio), 2)
                    AS average_dti,

                ROUND(AVG(fico_score_avg), 2)
                    AS average_fico_score

            FROM read_parquet('{analysis_path}')
            GROUP BY grade
            ORDER BY grade
        )
        TO '{output_path}'
        (
            FORMAT CSV,
            HEADER TRUE
        )
        """
    )


def build_purpose_metrics(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    analysis_path = sql_safe_path(ANALYSIS_FILE)
    output_path = sql_safe_path(PURPOSE_FILE)

    connection.execute(
        f"""
        COPY (
            SELECT
                purpose,
                COUNT(*) AS total_loans,

                ROUND(
                    100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER (),
                    2
                ) AS portfolio_share_pct,

                SUM(default_flag) AS defaulted_loans,

                ROUND(
                    100.0 * AVG(default_flag),
                    2
                ) AS default_rate_pct,

                ROUND(AVG(loan_amount), 2)
                    AS average_loan_amount_usd,

                ROUND(MEDIAN(loan_amount), 2)
                    AS median_loan_amount_usd,

                ROUND(
                    100.0 * AVG(
                        CASE
                            WHEN grade IN ('B', 'C')
                                THEN 1
                            ELSE 0
                        END
                    ),
                    2
                ) AS grade_bc_share_pct,

                ROUND(AVG(interest_rate_pct), 2)
                    AS average_interest_rate_pct,

                ROUND(AVG(annual_income), 2)
                    AS average_annual_income_usd,

                ROUND(MEDIAN(annual_income), 2)
                    AS median_annual_income_usd,

                ROUND(AVG(debt_to_income_ratio), 2)
                    AS average_dti,

                ROUND(AVG(fico_score_avg), 2)
                    AS average_fico_score

            FROM read_parquet('{analysis_path}')
            GROUP BY purpose
            ORDER BY total_loans DESC
        )
        TO '{output_path}'
        (
            FORMAT CSV,
            HEADER TRUE
        )
        """
    )


def build_ai_metrics(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    analysis_path = sql_safe_path(ANALYSIS_FILE)

    overall = connection.execute(
        f"""
        SELECT
            COUNT(*) AS total_loans,
            SUM(default_flag) AS defaulted_loans,
            100.0 * AVG(default_flag) AS default_rate_pct,
            AVG(loan_amount) AS average_loan_amount_usd,
            AVG(interest_rate_pct)
                AS average_interest_rate_pct
        FROM read_parquet('{analysis_path}')
        """
    ).fetchone()

    segment = connection.execute(
        f"""
        SELECT
            COUNT(*) AS total_loans,
            SUM(default_flag) AS defaulted_loans,
            100.0 * AVG(default_flag)
                AS default_rate_pct,
            AVG(loan_amount)
                AS average_loan_amount_usd,
            MEDIAN(loan_amount)
                AS median_loan_amount_usd,
            100.0 * AVG(
                CASE
                    WHEN grade IN ('B', 'C')
                        THEN 1
                    ELSE 0
                END
            ) AS grade_bc_share_pct,
            AVG(interest_rate_pct)
                AS average_interest_rate_pct,
            AVG(annual_income)
                AS average_annual_income_usd,
            MEDIAN(annual_income)
                AS median_annual_income_usd,
            AVG(debt_to_income_ratio)
                AS average_dti,
            AVG(fico_score_avg)
                AS average_fico_score
        FROM read_parquet('{analysis_path}')
        WHERE purpose = 'debt_consolidation'
        """
    ).fetchone()

    portfolio_share = 100.0 * segment[0] / overall[0]

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "metric_scope": {
            "origination_period": "2014-01-01 to 2015-12-31",
            "term_months": 36,
            "population": "loans with a known final outcome",
            "unit_of_analysis": "loan contract",
            "currency": "USD",
        },
        "definitions": {
            "default_rate": (
                "Defaulted loans divided by loans with a known "
                "final outcome."
            ),
            "portfolio_share": (
                "Segment loan contracts divided by all contracts "
                "in the analytical cohort."
            ),
            "average_ticket": (
                "Arithmetic mean of the original requested "
                "loan amount."
            ),
        },
        "overall_portfolio": {
            "total_loans": int(overall[0]),
            "defaulted_loans": int(overall[1]),
            "default_rate_pct": round(float(overall[2]), 4),
            "average_loan_amount_usd": round(
                float(overall[3]),
                2,
            ),
            "average_interest_rate_pct": round(
                float(overall[4]),
                4,
            ),
        },
        "debt_consolidation": {
            "total_loans": int(segment[0]),
            "portfolio_share_pct": round(
                float(portfolio_share),
                4,
            ),
            "defaulted_loans": int(segment[1]),
            "default_rate_pct": round(
                float(segment[2]),
                4,
            ),
            "average_loan_amount_usd": round(
                float(segment[3]),
                2,
            ),
            "median_loan_amount_usd": round(
                float(segment[4]),
                2,
            ),
            "grade_bc_share_pct": round(
                float(segment[5]),
                4,
            ),
            "average_interest_rate_pct": round(
                float(segment[6]),
                4,
            ),
            "average_annual_income_usd": round(
                float(segment[7]),
                2,
            ),
            "median_annual_income_usd": round(
                float(segment[8]),
                2,
            ),
            "average_dti": round(
                float(segment[9]),
                4,
            ),
            "average_fico_score": round(
                float(segment[10]),
                2,
            ),
        },
    }


def build_manifest(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    analysis_path = sql_safe_path(ANALYSIS_FILE)

    profile = connection.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT loan_id)
                AS distinct_loan_ids,
            MIN(issue_date) AS first_issue_date,
            MAX(issue_date) AS last_issue_date,
            COUNT(*)
                FILTER (WHERE default_flag IS NULL)
                AS missing_default_flags,
            COUNT(*)
                FILTER (
                    WHERE default_flag NOT IN (0, 1)
                ) AS invalid_default_flags,
            COUNT(*)
                FILTER (WHERE loan_amount IS NULL)
                AS missing_loan_amounts,
            COUNT(*)
                FILTER (WHERE grade IS NULL)
                AS missing_grades,
            COUNT(*)
                FILTER (WHERE purpose IS NULL)
                AS missing_purposes
        FROM read_parquet('{analysis_path}')
        """
    ).fetchone()

    files = {}

    for path in (
        ANALYSIS_FILE,
        DASHBOARD_FILE,
        QUARTERLY_FILE,
        GRADE_FILE,
        PURPOSE_FILE,
        AI_METRICS_FILE,
    ):
        files[path.name] = {
            "relative_path": str(
                path.relative_to(PROJECT_ROOT)
            ),
            "size_bytes": path.stat().st_size,
            "sha256": calculate_sha256(path),
        }

    return {
        "dataset": "FinLend business-ready credit datasets",
        "layer": "gold",
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_file": str(
            SILVER_FILE.relative_to(PROJECT_ROOT)
        ),
        "cohort_definition": {
            "issue_date_start": "2014-01-01",
            "issue_date_end_exclusive": "2016-01-01",
            "term_months": 36,
            "resolved_only": True,
        },
        "row_count": int(profile[0]),
        "distinct_loan_ids": int(profile[1]),
        "first_issue_date": profile[2].isoformat(),
        "last_issue_date": profile[3].isoformat(),
        "quality_checks": {
            "missing_default_flags": int(profile[4]),
            "invalid_default_flags": int(profile[5]),
            "missing_loan_amounts": int(profile[6]),
            "missing_grades": int(profile[7]),
            "missing_purposes": int(profile[8]),
        },
        "files": files,
    }


def main() -> None:
    if not SILVER_FILE.exists():
        raise FileNotFoundError(
            f"Silver dataset not found: {SILVER_FILE}"
        )

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    remove_existing_outputs()

    connection = duckdb.connect()
    connection.execute("SET threads = 4")

    build_analysis_cohort(connection)
    build_dashboard_dataset(connection)
    build_quarterly_metrics(connection)
    build_grade_metrics(connection)
    build_purpose_metrics(connection)

    ai_metrics = build_ai_metrics(connection)

    AI_METRICS_FILE.write_text(
        json.dumps(
            ai_metrics,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_manifest(connection)

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    connection.close()

    print("Gold layer built successfully.")
    print(f"Rows: {manifest['row_count']:,}")
    print(
        "Period: "
        f"{manifest['first_issue_date']} "
        f"to {manifest['last_issue_date']}"
    )
    print(
        "Distinct loans: "
        f"{manifest['distinct_loan_ids']:,}"
    )
    print(
        "Dashboard dataset: "
        f"{DASHBOARD_FILE.relative_to(PROJECT_ROOT)}"
    )
    print(
        "AI validation metrics: "
        f"{AI_METRICS_FILE.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Manifest: "
        f"{MANIFEST_FILE.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
