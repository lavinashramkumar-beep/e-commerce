import pandas as pd
import numpy as np

df = pd.read_csv("ecommerce_data_transformed.csv")

print("E-COMMERCE DATA ANALYSIS")
print("=" * 40)

total_sales = df["Net_Sales"].sum()
total_profit = df["Profit"].sum()
total_quantity = df["Quantity"].sum()
average_order_value = df["Net_Sales"].mean()
average_profit = df["Profit"].mean()
total_orders = df["Order_ID"].nunique()

print("\nOVERALL METRICS")
print("Total Sales:", total_sales)
print("Total Profit:", total_profit)
print("Total Quantity Sold:", total_quantity)
print("Total Orders:", total_orders)
print("Average Order Value:", average_order_value)
print("Average Profit per Order:", average_profit)

print("\nSALES BY CATEGORY")
category_sales = df.groupby("Category")["Net_Sales"].sum().sort_values(ascending=False)
print(category_sales)

print("\nPROFIT BY CATEGORY")
category_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
print(category_profit)

print("\nQUANTITY BY CATEGORY")
category_quantity = df.groupby("Category")["Quantity"].sum().sort_values(ascending=False)
print(category_quantity)

print("\nSALES BY CITY")
city_sales = df.groupby("City")["Net_Sales"].sum().sort_values(ascending=False)
print(city_sales)

print("\nPROFIT BY CITY")
city_profit = df.groupby("City")["Profit"].sum().sort_values(ascending=False)
print(city_profit)

print("\nTOP 5 PRODUCTS BY SALES")
top_products = (
    df.groupby("Product")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top_products)

print("\nTOP 5 PRODUCTS BY QUANTITY")
top_products_quantity = (
    df.groupby("Product")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top_products_quantity)

print("\nTOP 5 CUSTOMERS BY SALES")
top_customers = (
    df.groupby("Customer_Name")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top_customers)

print("\nTOP 5 CUSTOMERS BY PROFIT")
top_customers_profit = (
    df.groupby("Customer_Name")["Profit"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top_customers_profit)

print("\nPAYMENT METHOD ANALYSIS")
payment_analysis = df.groupby("Payment_Method").agg(
    Total_Sales=("Net_Sales", "sum"),
    Total_Orders=("Order_ID", "count"),
    Average_Sales=("Net_Sales", "mean")
).sort_values("Total_Sales", ascending=False)

print(payment_analysis)

print("\nORDER STATUS ANALYSIS")
status_analysis = df.groupby("Order_Status").agg(
    Total_Sales=("Net_Sales", "sum"),
    Total_Orders=("Order_ID", "count"),
    Total_Quantity=("Quantity", "sum")
)

print(status_analysis)

print("\nDISCOUNT ANALYSIS")
print("Total Discount:", df["Discount_Amount"].sum())
print("Average Discount:", df["Discount_Amount"].mean())

print("\nPROFIT MARGIN")
print("Average Profit Margin:", df["Profit_Margin"].mean())

print("\nHIGH VALUE ORDERS")
high_value_orders = df[df["Net_Sales"] >= 5000]
print(high_value_orders[["Order_ID", "Customer_Name", "Product", "Net_Sales"]])

print("\nSALES STATISTICS")
print("Maximum Sale:", df["Net_Sales"].max())
print("Minimum Sale:", df["Net_Sales"].min())
print("Median Sale:", df["Net_Sales"].median())
print("Standard Deviation:", np.std(df["Net_Sales"]))

print("\nCATEGORY SUMMARY")
category_summary = df.groupby("Category").agg(
    Total_Sales=("Net_Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Average_Sales=("Net_Sales", "mean"),
    Average_Profit=("Profit", "mean")
).sort_values("Total_Sales", ascending=False)

print(category_summary)

category_summary.to_csv("category_analysis.csv")
city_sales.to_csv("city_sales_analysis.csv")
top_products.to_csv("top_products_analysis.csv")
top_customers.to_csv("top_customers_analysis.csv")

print("\nAnalysis files saved successfully.")