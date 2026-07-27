# Analytical Methodology

## Data source

Lending Club accepted loans dataset covering originations from June 2007
through December 2018.

## Loan outcome definitions

Resolved loans:

- Fully Paid
- Charged Off
- Default
- Does not meet the credit policy. Status:Fully Paid
- Does not meet the credit policy. Status:Charged Off

Defaulted loans:

- Charged Off
- Default
- Does not meet the credit policy. Status:Charged Off

Active and delinquent loans are not classified as successfully paid loans.

## Maturity assessment

Loan outcome maturity was assessed by origination year and contractual term.

For 36-month loans:

- 2014 loans were 100% resolved;
- 2015 loans were 99.95% resolved;
- 2016 loans were only 71.83% resolved.

Loans with 60-month terms showed substantially lower resolution rates during
the candidate analysis period.

## Primary analytical cohort

The primary statistical analysis uses:

- originations from January 2014 through December 2015;
- 36-month contractual term;
- loans with a known final outcome;
- quarterly origination cohorts.

This produces approximately 445,596 resolved loan records.

## Preliminary hypothesis assessment

The business hypothesis states that default increased because the company
approved more Grade D and E loans.

Between 2014-Q1 and 2015-Q4:

- overall default rate increased from 12.87% to 14.82%;
- Grade D/E portfolio share decreased from 14.93% to 12.27%;
- Grade D/E default rate increased from 22.26% to 29.16%;
- default rate among other grades increased from 11.22% to 12.81%.

The preliminary evidence supports that Grades D and E are higher-risk
segments, but does not support the claim that increased D/E portfolio share
caused the rise in overall default.

The results instead indicate deterioration within risk groups. Additional
statistical tests and multivariable analysis are required before attributing
the change to specific borrower or loan characteristics.

## Sensitivity analyses

Planned robustness checks include:

- including 2013 originations;
- evaluating grades separately rather than grouping D and E;
- comparing subgrades;
- evaluating 60-month loans separately;
- controlling for borrower and contract characteristics.
