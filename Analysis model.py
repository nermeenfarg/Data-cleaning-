import pandas as pd
import numpy as np

class EconomicAnalyzer:
    """
    Computes category-level key performance indicators (KPIs) and runs a simple
    Ordinary Least Squares (OLS) linear regression model.
    """
    def __init__(self, cleaned_df: pd.DataFrame):
        self.df = cleaned_df

    def get_category_summary(self):
        """Aggregates metrics by product category to evaluate sales and return rates."""
        summary = self.df.groupby("category").agg(
            total_orders=("transaction_id", "count"),
            total_units=("quantity", "sum"),
            gross_sales=("gross_revenue", "sum"),
            net_profit=("net_revenue", "sum"),
            return_rate=("return_flag", lambda x: np.round(x.mean() * 100, 2))
        ).reset_index()

        return summary.sort_values(by="net_profit", ascending=False)

    def run_simple_regression(self):
        """
        Fits a simple linear regression model (Y = a + bX) using NumPy
        to estimate the impact of order quantity (X) on net revenue (Y).
        """
        valid_sales = self.df[self.df["return_flag"] == 0]
        x = valid_sales["quantity"].values
        y = valid_sales["net_revenue"].values

        # Compute slope (b) and intercept (a)
        slope, intercept = np.polyfit(x, y, 1)
        correlation = np.corrcoef(x, y)[0, 1]

        return {
            "slope_b": round(float(slope), 2),
            "intercept_a": round(float(intercept), 2),
            "correlation_r": round(float(correlation), 4),
            "predicted_profit_for_20_units": round(float(intercept + slope * 20), 2)
        }
