import pandas as pd
import numpy as np
import os

class DataCleaner:
    """
    Handles synthetic dataset generation (if no source file exists) and performs
    data cleaning, missing value imputation, and feature engineering.
    """
    def __init__(self, file_path="retail_data.csv"):
        self.file_path = file_path

    def generate_sample_data_if_needed(self):
        """Generates a sample retail dataset if the CSV file does not exist."""
        if not os.path.exists(self.file_path):
            np.random.seed(42)
            dates = pd.date_range(start="2025-01-01", periods=120, freq="D")
            categories = ["Electronics", "Groceries", "Home Appliances", "Medical Supplies"]
            
            data = {
                "transaction_id": range(1001, 1121),
                "date": dates,
                "category": np.random.choice(categories, size=120),
                "unit_price": np.round(np.random.uniform(50.0, 1200.0, size=120), 2),
                "quantity": np.random.randint(1, 30, size=120),
                "return_flag": np.random.choice([0, 1], size=120, p=[0.88, 0.12]),
                "shipping_cost": np.round(np.random.uniform(10.0, 100.0, size=120), 2)
            }
            
            df = pd.DataFrame(data)
            # Inject missing values intentionally to demonstrate the imputation pipeline
            df.loc[5:8, "unit_price"] = np.nan
            df.loc[15:17, "shipping_cost"] = np.nan
            df.to_csv(self.file_path, index=False)

    def load_and_clean(self):
        """Loads the dataset, imputes missing records, and computes net revenue."""
        try:
            self.generate_sample_data_if_needed()
            df = pd.read_csv(self.file_path)

            # 1. Standardize date column format
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # 2. Impute missing unit prices using category-level mean
            df["unit_price"] = df.groupby("category")["unit_price"].transform(
                lambda x: x.fillna(x.mean())
            )

            # 3. Impute missing shipping costs using the overall median
            df["shipping_cost"] = df["shipping_cost"].fillna(df["shipping_cost"].median())

            # 4. Filter out invalid records with zero or negative quantities
            df = df[df["quantity"] > 0].copy()

            # 5. Feature engineering: Calculate Gross Revenue and Net Revenue
            df["gross_revenue"] = np.round(df["unit_price"] * df["quantity"], 2)
            df["net_revenue"] = np.where(
                df["return_flag"] == 1, 
                -df["shipping_cost"],  # Returned orders incur shipping cost loss
                df["gross_revenue"] - df["shipping_cost"]
            )

            return df

        except Exception as e:
            raise RuntimeError(f"Error during data cleaning pipeline: {e}")
