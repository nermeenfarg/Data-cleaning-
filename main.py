from data_cleaning import DataCleaner
from analysis_model import EconomicAnalyzer

def main():
    print("=" * 65)
    print("Starting Retail Performance & Econometric Audit Engine")
    print("=" * 65)

    # Step 1: Initialize data cleaning pipeline
    cleaner = DataCleaner("retail_data.csv")
    cleaned_data = cleaner.load_and_clean()
    print(f"[+] Data cleaned successfully. Total valid records: {len(cleaned_data)}\n")

    # Step 2: Generate category-level performance summary
    analyzer = EconomicAnalyzer(cleaned_data)
    summary_table = analyzer.get_category_summary()
    
    print("--- 1. Category Financial Performance & Return Rates ---")
    print(summary_table.to_string(index=False))
    print("\n" + "-" * 65)

    # Step 3: Run OLS linear regression model
    regression_results = analyzer.run_simple_regression()
    print("--- 2. Ordinary Least Squares (OLS) Regression Results ---")
    print(f"* Correlation Coefficient (R): {regression_results['correlation_r']}")
    print(f"* Linear Equation: Y = {regression_results['intercept_a']} + ({regression_results['slope_b']} * X)")
    print(f"* Predicted Net Profit (20 Units): ${regression_results['predicted_profit_for_20_units']}")
    print("=" * 65)

    # Export final audit report to CSV
    output_file = "final_category_audit_report.csv"
    summary_table.to_csv(output_file, index=False)
    print(f"[+] Final audit report exported to: {output_file}")

if __name__ == "__main__":
    main()
