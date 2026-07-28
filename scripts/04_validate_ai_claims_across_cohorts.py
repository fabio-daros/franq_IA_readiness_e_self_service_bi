"""Validate debt-consolidation LLM claims across multiple Silver scopes.

Preserves Gold metric definitions:
- default_rate: defaulted loans / loans with a known final outcome
- portfolio_share: segment contracts / all contracts in the scope
- average_ticket: mean of original requested loan amount (loan_amount)
- grade B/C share: share of grades B and C within the segment
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_FILE = (
    PROJECT_ROOT / "data" / "silver" / "loans_clean.parquet"
)
GOLD_AI_METRICS_FILE = (
    PROJECT_ROOT / "data" / "gold" / "ai_validation_metrics.json"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "ai_claims_cross_cohort_validation.csv"
)

PERCENTAGE_TOLERANCE_PP = 0.25
CURRENCY_RELATIVE_TOLERANCE = 0.01
RECONCILE_ABS_TOLERANCE = 1e-6
GOLD_RECONCILE_ABS_PP = 0.01
GOLD_RECONCILE_ABS_USD = 0.05

LLM_CLAIMS: dict[str, float | str] = {
    "portfolio_share_pct": 48.0,
    "segment_default_rate_pct": 12.3,
    "overall_default_rate_pct": 14.1,
    "segment_vs_overall_direction": "below",
    "average_loan_amount_usd": 15_200.0,
    "average_annual_income_usd": 72_000.0,
    "grade_bc_share_pct": 62.0,
    "average_interest_rate_pct": 13.8,
}


@dataclass(frozen=True)
class ScopeDefinition:
    scope_id: str
    scope_label: str
    where_sql: str
    resolved_only: bool
    interpretation: str


SCOPES: tuple[ScopeDefinition, ...] = (
    ScopeDefinition(
        scope_id="primary_gold",
        scope_label=(
            "2014–2015 originations, 36-month term, "
            "resolved loans only"
        ),
        where_sql=(
            "issue_date >= DATE '2014-01-01' "
            "AND issue_date < DATE '2016-01-01' "
            "AND term_months = 36 "
            "AND is_resolved = TRUE "
            "AND default_flag IS NOT NULL"
        ),
        resolved_only=True,
        interpretation=(
            "Official analytical cohort aligned with Gold. "
            "Default rates are final observed outcomes."
        ),
    ),
    ScopeDefinition(
        scope_id="mature_36m_2007_2015",
        scope_label=(
            "2007–2015 originations, 36-month term, "
            "resolved loans only"
        ),
        where_sql=(
            "issue_date >= DATE '2007-01-01' "
            "AND issue_date < DATE '2016-01-01' "
            "AND term_months = 36 "
            "AND is_resolved = TRUE "
            "AND default_flag IS NOT NULL"
        ),
        resolved_only=True,
        interpretation=(
            "Broader mature 36-month window. "
            "Default rates are final observed outcomes."
        ),
    ),
    ScopeDefinition(
        scope_id="mature_60m_2007_2013",
        scope_label=(
            "2007–2013 originations, 60-month term, "
            "resolved loans only"
        ),
        where_sql=(
            "issue_date >= DATE '2007-01-01' "
            "AND issue_date < DATE '2014-01-01' "
            "AND term_months = 60 "
            "AND is_resolved = TRUE "
            "AND default_flag IS NOT NULL"
        ),
        resolved_only=True,
        interpretation=(
            "Mature 60-month window restricted to earlier "
            "originations so outcomes are largely known. "
            "Default rates are final observed outcomes."
        ),
    ),
    ScopeDefinition(
        scope_id="full_snapshot_descriptive",
        scope_label=(
            "Full Silver snapshot, all terms and statuses "
            "(descriptive only)"
        ),
        where_sql="TRUE",
        resolved_only=False,
        interpretation=(
            "DESCRIPTIVE ONLY. Includes unresolved/current loans. "
            "Reported default percentages are the default rate among "
            "contracts with a known final outcome inside the full "
            "snapshot and must NOT be interpreted as final credit risk "
            "over all 2.26M contracts."
        ),
    ),
)


def sql_safe_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def is_validated_absolute(
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
        return claimed_value == 0.0
    relative_error = abs(actual_value - claimed_value) / abs(
        actual_value
    )
    return relative_error <= relative_tolerance


def classify_percentage(
    claimed_value: float,
    actual_value: float,
) -> str:
    return (
        "PASS"
        if is_validated_absolute(
            claimed_value,
            actual_value,
            PERCENTAGE_TOLERANCE_PP,
        )
        else "FAIL"
    )


def classify_currency(
    claimed_value: float,
    actual_value: float,
) -> str:
    return (
        "PASS"
        if is_validated_currency(claimed_value, actual_value)
        else "FAIL"
    )


def compute_scope_metrics(
    connection: duckdb.DuckDBPyConnection,
    silver_path: str,
    scope: ScopeDefinition,
) -> dict[str, Any]:
    """Compute metrics for one analytical scope.

    Default rates always use AVG(default_flag), which excludes NULL
    outcomes and therefore matches the Gold definition (known final
    outcome only). For the full snapshot this remains descriptive.
    """
    query = f"""
        WITH scoped AS (
            SELECT
                loan_id,
                purpose,
                grade,
                loan_amount,
                annual_income,
                interest_rate_pct,
                default_flag,
                is_resolved,
                is_current
            FROM read_parquet('{silver_path}')
            WHERE {scope.where_sql}
        ),
        overall AS (
            SELECT
                COUNT(*) AS total_loans,
                COUNT(*) FILTER (
                    WHERE NOT is_resolved
                ) AS unresolved_loans,
                COUNT(*) FILTER (
                    WHERE is_current
                ) AS current_loans,
                COUNT(*) FILTER (
                    WHERE default_flag IS NOT NULL
                ) AS known_outcome_loans,
                AVG(default_flag) AS overall_default_rate
            FROM scoped
        ),
        segment AS (
            SELECT
                COUNT(*) AS segment_loans,
                AVG(default_flag) AS segment_default_rate,
                AVG(loan_amount) AS average_loan_amount_usd,
                AVG(annual_income)
                    AS average_annual_income_usd,
                AVG(
                    CASE
                        WHEN grade IN ('B', 'C') THEN 1.0
                        ELSE 0.0
                    END
                ) AS grade_bc_share,
                AVG(interest_rate_pct)
                    AS average_interest_rate_pct
            FROM scoped
            WHERE purpose = 'debt_consolidation'
        )
        SELECT
            overall.total_loans,
            overall.unresolved_loans,
            overall.current_loans,
            overall.known_outcome_loans,
            overall.overall_default_rate,
            segment.segment_loans,
            segment.segment_default_rate,
            segment.average_loan_amount_usd,
            segment.average_annual_income_usd,
            segment.grade_bc_share,
            segment.average_interest_rate_pct
        FROM overall
        CROSS JOIN segment
    """

    row = connection.execute(query).fetchone()
    if row is None:
        raise RuntimeError(
            f"No metrics returned for scope {scope.scope_id}"
        )

    (
        total_loans,
        unresolved_loans,
        current_loans,
        known_outcome_loans,
        overall_default_rate,
        segment_loans,
        segment_default_rate,
        average_loan_amount_usd,
        average_annual_income_usd,
        grade_bc_share,
        average_interest_rate_pct,
    ) = row

    if total_loans == 0:
        raise RuntimeError(
            f"Scope {scope.scope_id} returned zero loans"
        )
    if segment_loans == 0:
        raise RuntimeError(
            f"Scope {scope.scope_id} has no debt_consolidation loans"
        )
    if known_outcome_loans == 0:
        raise RuntimeError(
            f"Scope {scope.scope_id} has no known outcomes"
        )

    portfolio_share_pct = 100.0 * segment_loans / total_loans
    overall_default_rate_pct = 100.0 * float(overall_default_rate)
    segment_default_rate_pct = 100.0 * float(segment_default_rate)
    default_rate_gap_pp = (
        segment_default_rate_pct - overall_default_rate_pct
    )
    unresolved_share_pct = (
        100.0 * unresolved_loans / total_loans
    )
    current_share_pct = 100.0 * current_loans / total_loans
    grade_bc_share_pct = 100.0 * float(grade_bc_share)

    actual_direction = (
        "below"
        if segment_default_rate_pct < overall_default_rate_pct
        else (
            "above"
            if segment_default_rate_pct > overall_default_rate_pct
            else "equal"
        )
    )
    claimed_direction = str(
        LLM_CLAIMS["segment_vs_overall_direction"]
    )

    return {
        "scope_id": scope.scope_id,
        "scope_label": scope.scope_label,
        "resolved_only": scope.resolved_only,
        "interpretation": scope.interpretation,
        "default_rate_basis": "known_final_outcome_only",
        "total_loans": int(total_loans),
        "debt_consolidation_loans": int(segment_loans),
        "known_outcome_loans": int(known_outcome_loans),
        "unresolved_loans": int(unresolved_loans),
        "current_loans": int(current_loans),
        "portfolio_share_pct": round(portfolio_share_pct, 4),
        "segment_default_rate_pct": round(
            segment_default_rate_pct,
            4,
        ),
        "overall_default_rate_pct": round(
            overall_default_rate_pct,
            4,
        ),
        "segment_minus_overall_pp": round(default_rate_gap_pp, 4),
        "average_loan_amount_usd": round(
            float(average_loan_amount_usd),
            2,
        ),
        "average_annual_income_usd": round(
            float(average_annual_income_usd),
            2,
        ),
        "grade_bc_share_pct": round(grade_bc_share_pct, 4),
        "average_interest_rate_pct": round(
            float(average_interest_rate_pct),
            4,
        ),
        "unresolved_share_pct": round(
            unresolved_share_pct,
            4,
        ),
        "current_share_pct": round(current_share_pct, 4),
        "claimed_portfolio_share_pct": LLM_CLAIMS[
            "portfolio_share_pct"
        ],
        "claim_portfolio_share_status": classify_percentage(
            float(LLM_CLAIMS["portfolio_share_pct"]),
            portfolio_share_pct,
        ),
        "claimed_segment_default_rate_pct": LLM_CLAIMS[
            "segment_default_rate_pct"
        ],
        "claim_segment_default_status": classify_percentage(
            float(LLM_CLAIMS["segment_default_rate_pct"]),
            segment_default_rate_pct,
        ),
        "claimed_overall_default_rate_pct": LLM_CLAIMS[
            "overall_default_rate_pct"
        ],
        "claim_overall_default_status": classify_percentage(
            float(LLM_CLAIMS["overall_default_rate_pct"]),
            overall_default_rate_pct,
        ),
        "claimed_segment_vs_overall_direction": claimed_direction,
        "actual_segment_vs_overall_direction": actual_direction,
        "claim_direction_status": (
            "PASS"
            if actual_direction == claimed_direction
            else "FAIL"
        ),
        "claimed_average_loan_amount_usd": LLM_CLAIMS[
            "average_loan_amount_usd"
        ],
        "claim_ticket_status": classify_currency(
            float(LLM_CLAIMS["average_loan_amount_usd"]),
            float(average_loan_amount_usd),
        ),
        "claimed_average_annual_income_usd": LLM_CLAIMS[
            "average_annual_income_usd"
        ],
        "claim_income_status": classify_currency(
            float(LLM_CLAIMS["average_annual_income_usd"]),
            float(average_annual_income_usd),
        ),
        "claimed_grade_bc_share_pct": LLM_CLAIMS[
            "grade_bc_share_pct"
        ],
        "claim_grade_bc_status": classify_percentage(
            float(LLM_CLAIMS["grade_bc_share_pct"]),
            grade_bc_share_pct,
        ),
        "claimed_average_interest_rate_pct": LLM_CLAIMS[
            "average_interest_rate_pct"
        ],
        "claim_interest_rate_status": classify_percentage(
            float(LLM_CLAIMS["average_interest_rate_pct"]),
            float(average_interest_rate_pct),
        ),
    }


def run_reconciliation_checks(
    rows: list[dict[str, Any]],
    gold_metrics: dict[str, Any],
) -> None:
    """Fail loudly if internal consistency or Gold alignment breaks."""
    by_id = {row["scope_id"]: row for row in rows}

    for row in rows:
        scope_id = row["scope_id"]

        if row["debt_consolidation_loans"] > row["total_loans"]:
            raise AssertionError(
                f"{scope_id}: segment loans exceed total loans"
            )

        expected_share = (
            100.0
            * row["debt_consolidation_loans"]
            / row["total_loans"]
        )
        if abs(expected_share - row["portfolio_share_pct"]) > 0.01:
            raise AssertionError(
                f"{scope_id}: portfolio share does not reconcile"
            )

        expected_gap = (
            row["segment_default_rate_pct"]
            - row["overall_default_rate_pct"]
        )
        if abs(expected_gap - row["segment_minus_overall_pp"]) > (
            RECONCILE_ABS_TOLERANCE + 0.0001
        ):
            raise AssertionError(
                f"{scope_id}: default-rate gap does not reconcile"
            )

        for share_col in (
            "portfolio_share_pct",
            "segment_default_rate_pct",
            "overall_default_rate_pct",
            "grade_bc_share_pct",
            "unresolved_share_pct",
            "current_share_pct",
        ):
            value = row[share_col]
            if not (0.0 <= value <= 100.0):
                raise AssertionError(
                    f"{scope_id}: {share_col}={value} out of [0, 100]"
                )

        if row["resolved_only"] and row[
            "unresolved_share_pct"
        ] > 0.0:
            raise AssertionError(
                f"{scope_id}: resolved-only scope has unresolved loans"
            )

        if (
            not row["resolved_only"]
            and row["unresolved_share_pct"] <= 0.0
        ):
            raise AssertionError(
                f"{scope_id}: full snapshot unexpectedly has no "
                "unresolved loans"
            )

        expected_direction = row[
            "actual_segment_vs_overall_direction"
        ]
        if expected_direction == "above":
            assert row["segment_minus_overall_pp"] > 0
        elif expected_direction == "below":
            assert row["segment_minus_overall_pp"] < 0
        else:
            assert abs(row["segment_minus_overall_pp"]) < 1e-9

    primary = by_id["primary_gold"]
    gold_overall = gold_metrics["overall_portfolio"]
    gold_segment = gold_metrics["debt_consolidation"]

    checks = [
        (
            "total_loans",
            primary["total_loans"],
            gold_overall["total_loans"],
            0,
        ),
        (
            "debt_consolidation_loans",
            primary["debt_consolidation_loans"],
            gold_segment["total_loans"],
            0,
        ),
        (
            "portfolio_share_pct",
            primary["portfolio_share_pct"],
            gold_segment["portfolio_share_pct"],
            GOLD_RECONCILE_ABS_PP,
        ),
        (
            "segment_default_rate_pct",
            primary["segment_default_rate_pct"],
            gold_segment["default_rate_pct"],
            GOLD_RECONCILE_ABS_PP,
        ),
        (
            "overall_default_rate_pct",
            primary["overall_default_rate_pct"],
            gold_overall["default_rate_pct"],
            GOLD_RECONCILE_ABS_PP,
        ),
        (
            "average_loan_amount_usd",
            primary["average_loan_amount_usd"],
            gold_segment["average_loan_amount_usd"],
            GOLD_RECONCILE_ABS_USD,
        ),
        (
            "average_annual_income_usd",
            primary["average_annual_income_usd"],
            gold_segment["average_annual_income_usd"],
            GOLD_RECONCILE_ABS_USD,
        ),
        (
            "grade_bc_share_pct",
            primary["grade_bc_share_pct"],
            gold_segment["grade_bc_share_pct"],
            GOLD_RECONCILE_ABS_PP,
        ),
        (
            "average_interest_rate_pct",
            primary["average_interest_rate_pct"],
            gold_segment["average_interest_rate_pct"],
            GOLD_RECONCILE_ABS_PP,
        ),
    ]

    for name, actual, expected, tolerance in checks:
        if abs(actual - expected) > tolerance:
            raise AssertionError(
                "Primary scope does not reconcile to Gold "
                f"ai_validation_metrics.json for {name}: "
                f"script={actual}, gold={expected}"
            )


def main() -> None:
    if not SILVER_FILE.exists():
        raise FileNotFoundError(
            f"Silver dataset not found: {SILVER_FILE}"
        )
    if not GOLD_AI_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Gold AI metrics not found: {GOLD_AI_METRICS_FILE}"
        )

    gold_metrics = json.loads(
        GOLD_AI_METRICS_FILE.read_text(encoding="utf-8")
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    silver_path = sql_safe_path(SILVER_FILE)

    rows = [
        compute_scope_metrics(connection, silver_path, scope)
        for scope in SCOPES
    ]
    connection.close()

    run_reconciliation_checks(rows, gold_metrics)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_FILE, index=False)

    print("AI claims cross-cohort validation complete.")
    print(f"Output: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print()
    print(
        frame[
            [
                "scope_id",
                "total_loans",
                "portfolio_share_pct",
                "segment_default_rate_pct",
                "overall_default_rate_pct",
                "segment_minus_overall_pp",
                "average_loan_amount_usd",
                "average_annual_income_usd",
                "claim_direction_status",
                "claim_ticket_status",
                "claim_income_status",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Note: full_snapshot_descriptive default rates are "
        "descriptive only and must not be read as final credit risk."
    )


if __name__ == "__main__":
    main()
