# FinLend Credit Intelligence

Technical case for a Data Analyst role focused on Self-Service BI, credit risk
analysis and AI Readiness.

## Deliverables

### Part 1 — Self-Service BI

[Open the Looker Studio dashboard](COLE_O_LINK)

### Part 2 — Statistical Analysis

[`notebooks/finlend_default_hypothesis_analysis.ipynb`](notebooks/finlend_default_hypothesis_analysis.ipynb)

### Part 3 — Generative AI Validation

[`docs/ai_validation_report.md`](docs/ai_validation_report.md)

---

## Data architecture

The project follows a lightweight Medallion architecture:

- **Bronze:** immutable source files and ingestion metadata
- **Silver:** cleaned loan-level dataset with standardized types and outcomes
- **Gold:** governed analytical cohort and metrics used by the dashboard,
  statistical analysis and AI validation

Automated tests reconcile the data across the three layers.

The primary analytical cohort is:

- originations from **2014–2015**
- contractual term of **36 months**
- loans with a **known final outcome**

This cohort is the shared source of truth for the Looker dashboard, the
statistical notebook and the AI validation report.

---

## Repository structure

```text
finlend-credit-intelligence/
├── README.md
├── requirements.txt
├── pytest.ini
├── notebooks/
│   └── finlend_default_hypothesis_analysis.ipynb
├── docs/
│   ├── ai_validation_report.md
│   └── methodology.md
├── scripts/
│   ├── 01_build_bronze.py
│   ├── 02_build_silver.py
│   ├── 03_build_gold.py
│   └── 04_validate_ai_claims_across_cohorts.py
├── tests/
├── outputs/
│   ├── figures/
│   └── tables/
└── data/
    ├── bronze/
    ├── silver/
    └── gold/
```

Large raw files and regenerated datasets are intentionally excluded from Git.

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Register the Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name finlend --display-name "Python 3.12 — FinLend"
```

---

## Data source

Use the public Lending Club dataset from Kaggle:

[Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Place the accepted loans file at:

```text
data/bronze/raw/accepted_2007_to_2018Q4.csv.gz
```

The original Kaggle archive may also be kept locally as
`data/bronze/raw/archive.zip`, but it is not required by the pipeline and is
not versioned.

---

## Rebuild the Medallion layers

From the project root, with the virtual environment activated:

```bash
python scripts/01_build_bronze.py
python scripts/02_build_silver.py
python scripts/03_build_gold.py
```

Expected Gold outputs include:

- `data/gold/analysis_cohort.parquet` — notebook / statistical cohort
- `data/gold/dashboard_loans.csv` — Looker Studio source
- `data/gold/quarterly_risk_metrics.csv`
- `data/gold/risk_by_grade.csv`
- `data/gold/risk_by_purpose.csv`
- `data/gold/ai_validation_metrics.json` — official metrics for Part 3

Small governed Gold aggregates and manifests are tracked in Git for auditability.
Loan-level Parquet/CSV extracts remain local because of size.

---

## Validate AI claims and run tests

```bash
python scripts/04_validate_ai_claims_across_cohorts.py
python -m pytest -q
```

The cross-cohort validation writes:

```text
outputs/tables/ai_claims_cross_cohort_validation.csv
```

---

## Notes

- Part 1 consumes `data/gold/dashboard_loans.csv` in Looker Studio.
- Part 2 reads `data/gold/analysis_cohort.parquet`.
- Part 3 validates the LLM response against Gold metrics and checks robustness
  across alternative mature cohorts on Silver.
