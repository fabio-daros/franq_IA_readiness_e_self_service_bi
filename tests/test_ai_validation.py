from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest


PERCENTAGE_TOLERANCE_PP = 0.25
CURRENCY_RELATIVE_TOLERANCE = 0.01


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def is_validated(
    claimed_value: float,
    actual_value: float,
    tolerance: float,
) -> bool:
    return abs(claimed_value - actual_value) <= tolerance


def is_validated_currency(
    claimed_value: float,
    actual_value: float,
    relative_tolerance: float = CURRENCY_RELATIVE_TOLERANCE,
) -> bool:
    if actual_value == 0:
        return claimed_value == 0
    relative_error = abs(actual_value - claimed_value) / abs(
        actual_value
    )
    return relative_error <= relative_tolerance


def test_ai_metrics_match_gold_source(
    project_root: Path,
    ai_metrics: dict[str, Any],
    db: duckdb.DuckDBPyConnection,
) -> None:
    analysis_file = (
        project_root
        / "data"
        / "gold"
        / "analysis_cohort.parquet"
    )

    path = sql_path(analysis_file)

    overall = db.execute(
        f"""
        SELECT
            COUNT(*) AS total_loans,
            SUM(default_flag) AS defaulted_loans,
            100.0 * AVG(default_flag)
                AS default_rate_pct,
            AVG(loan_amount)
                AS average_loan_amount_usd,
            AVG(interest_rate_pct)
                AS average_interest_rate_pct
        FROM read_parquet('{path}')
        """
    ).fetchone()

    segment = db.execute(
        f"""
        SELECT
            COUNT(*) AS total_loans,
            SUM(default_flag) AS defaulted_loans,
            100.0 * AVG(default_flag)
                AS default_rate_pct,
            AVG(loan_amount)
                AS average_loan_amount_usd,
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
                AS average_annual_income_usd
        FROM read_parquet('{path}')
        WHERE purpose = 'debt_consolidation'
        """
    ).fetchone()

    overall_metrics = ai_metrics["overall_portfolio"]
    segment_metrics = ai_metrics["debt_consolidation"]

    assert overall_metrics["total_loans"] == overall[0]
    assert overall_metrics["defaulted_loans"] == overall[1]

    assert overall_metrics["default_rate_pct"] == pytest.approx(
        overall[2],
        abs=0.0001,
    )

    assert overall_metrics[
        "average_loan_amount_usd"
    ] == pytest.approx(
        overall[3],
        abs=0.01,
    )

    assert segment_metrics["total_loans"] == segment[0]
    assert segment_metrics["defaulted_loans"] == segment[1]

    assert segment_metrics["default_rate_pct"] == pytest.approx(
        segment[2],
        abs=0.0001,
    )

    assert segment_metrics[
        "average_loan_amount_usd"
    ] == pytest.approx(
        segment[3],
        abs=0.01,
    )

    assert segment_metrics[
        "grade_bc_share_pct"
    ] == pytest.approx(
        segment[4],
        abs=0.0001,
    )

    assert segment_metrics[
        "average_annual_income_usd"
    ] == pytest.approx(
        segment[6],
        abs=0.01,
    )

    expected_portfolio_share = (
        100.0 * segment[0] / overall[0]
    )

    assert segment_metrics[
        "portfolio_share_pct"
    ] == pytest.approx(
        expected_portfolio_share,
        abs=0.0001,
    )


def test_case_llm_claim_classification(
    ai_metrics: dict[str, Any],
) -> None:
    overall = ai_metrics["overall_portfolio"]
    segment = ai_metrics["debt_consolidation"]

    # Claim: debt consolidation represents 48%.
    assert not is_validated(
        claimed_value=48.0,
        actual_value=segment["portfolio_share_pct"],
        tolerance=PERCENTAGE_TOLERANCE_PP,
    )

    # Claim: segment default rate is 12.3%.
    assert not is_validated(
        claimed_value=12.3,
        actual_value=segment["default_rate_pct"],
        tolerance=PERCENTAGE_TOLERANCE_PP,
    )

    # Claim: overall default rate is 14.1%.
    assert not is_validated(
        claimed_value=14.1,
        actual_value=overall["default_rate_pct"],
        tolerance=PERCENTAGE_TOLERANCE_PP,
    )

    # Claim: average ticket is USD 15,200.
    assert not is_validated_currency(
        claimed_value=15_200.0,
        actual_value=segment["average_loan_amount_usd"],
    )

    # Claim: average annual income is USD 72,000.
    assert is_validated_currency(
        claimed_value=72_000.0,
        actual_value=segment["average_annual_income_usd"],
    )

    # Claim: 62% are Grades B and C.
    assert is_validated(
        claimed_value=62.0,
        actual_value=segment["grade_bc_share_pct"],
        tolerance=PERCENTAGE_TOLERANCE_PP,
    )

    # Claim: average interest rate is 13.8%.
    assert not is_validated(
        claimed_value=13.8,
        actual_value=segment[
            "average_interest_rate_pct"
        ],
        tolerance=PERCENTAGE_TOLERANCE_PP,
    )

    # Directional claim: segment default is "below" the portfolio.
    # Numerically close rates must not hide an inverted conclusion.
    segment_default_rate = segment["default_rate_pct"]
    portfolio_default_rate = overall["default_rate_pct"]

    assert segment_default_rate > portfolio_default_rate

    claimed_direction = "below"
    actual_direction = (
        "below"
        if segment_default_rate < portfolio_default_rate
        else "above"
    )
    assert actual_direction != claimed_direction
