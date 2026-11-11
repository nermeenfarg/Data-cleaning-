Retail Performance & Econometric Audit Engine
Overview
A modular Python data pipeline designed to clean transactional retail data, audit return rates across product categories, and estimate revenue elasticity using Ordinary Least Squares (OLS) linear regression.
Project Architecture
data_cleaning.py: Handles missing values using category-level mean imputation, validates date schemas, and computes net revenue accounting for returned orders.
analysis_model.py: Generates aggregated category KPIs (Gross Sales, Net Profit, Return Rate) and fits a simple linear regression model (Y = a + bX) using NumPy.
main.py: The entry-point script that orchestrates the cleaning and modeling modules and exports the final audit report (final_category_audit_report.csv).
Key Engineering Decisions
Category-Level Imputation: Missing unit prices are imputed using the category mean rather than dropping rows, preserving transaction volume metrics across smaller sample sizes.
Outlier Resilience: Missing shipping costs are imputed using the overall median to prevent skewed cost deductions.
Installation & Execution
