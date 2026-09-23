import pandas as pd
import numpy as np

df = pd.read_csv("ecommerce_data_cleaned.csv")

print("BEFORE TRANSFORMATION")
print(df.head())

df["Gross_Sales"] = df["Price"] * df["Quantity"]

df["Discount_Percentage"] = np.where(
    df["Gross_Sales"] >= 5000, 15,
    np.where(df["Gross_Sales"] >= 2500, 10, 5)
)

df["Discount_Amount"] = (
    df["Gross_Sales"] * df["Discount_Percentage"] / 100
)

df["Net_Sales"] = df["Gross_Sales"] - df["Discount_Amount"]

df["Profit"] = df["Net_Sales"] * np.where(
    df["Category"] == "Electronics", 0.20,
    np.where(df["Category"] == "Clothing", 0.25, 0.15)
)

df["Profit_Margin"] = np.where(
    df["Net_Sales"] > 0,
    (df["Profit"] / df["Net_Sales"]) * 100,
    0
)

df["Sales_Level"] = np.where(
    df["Net_Sales"] >= 5000,
    "High",
    np.where(df["Net_Sales"] >= 2500, "Medium", "Low")
)

print("\nTRANSFORMED DATA")
print(df.head(10))

print("\nVALIDATION")

print("Missing values:")
print(df.isnull().sum())

print("\nNegative Net Sales:")
print((df["Net_Sales"] < 0).sum())

print("\nNegative Discount Amount:")
print((df["Discount_Amount"] < 0).sum())

print("\nProfit Margin Range:")
print(df["Profit_Margin"].min(), "to", df["Profit_Margin"].max())

print("\nSales Level Counts:")
print(df["Sales_Level"].value_counts())

print("\nData Types:")
print(df.dtypes)

df.to_csv("ecommerce_data_transformed.csv", index=False)

print("\nTransformed file saved successfully.")