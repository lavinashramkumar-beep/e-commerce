import pandas as pd
import numpy as np

df = pd.read_csv("ecommerce_data_150_rows.csv")

print("BEFORE CLEANING")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df["Customer_Name"] = df["Customer_Name"].str.strip().str.title()
df["Product"] = df["Product"].str.strip().str.title()
df["Category"] = df["Category"].str.strip().str.title()
df["City"] = df["City"].str.strip().str.title()
df["Payment_Method"] = df["Payment_Method"].str.strip().str.title()

df["Customer_Name"] = df["Customer_Name"].fillna("Unknown")
df["Product"] = df["Product"].fillna("Unknown Product")
df["Category"] = df["Category"].fillna("Unknown")
df["City"] = df["City"].fillna("Unknown")
df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Customer_Rating"] = pd.to_numeric(df["Customer_Rating"], errors="coerce")
df["Total_Sales"] = pd.to_numeric(df["Total_Sales"], errors="coerce")

df["Price"] = df["Price"].fillna(df["Price"].median())
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Customer_Rating"] = df["Customer_Rating"].fillna(df["Customer_Rating"].median())
df["Total_Sales"] = df["Total_Sales"].fillna(df["Price"] * df["Quantity"])

df["Category"] = df["Category"].replace({
    "Electronics ": "Electronics",
    "Electronic": "Electronics",
    "Cloth": "Clothing",
    "Home ": "Home"
})

df["City"] = df["City"].replace({
    "Chennai ": "Chennai",
    "Bengaluru": "Bangalore",
    "Bombay": "Mumbai"
})

df["Payment_Method"] = df["Payment_Method"].replace({
    "Upi": "UPI",
    "Creditcard": "Credit Card",
    "Debitcard": "Debit Card",
    "Netbanking": "Net Banking"
})

df["Age"] = df["Age"].clip(18, 100)
df["Quantity"] = df["Quantity"].clip(lower=1)
df["Price"] = df["Price"].clip(lower=0)
df["Customer_Rating"] = df["Customer_Rating"].clip(1, 5)

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Order_Date"] = df["Order_Date"].fillna(pd.Timestamp("2026-01-01"))

df = df.drop_duplicates()

df["Order_ID"] = pd.to_numeric(df["Order_ID"], errors="coerce").astype("Int64")
df["Quantity"] = df["Quantity"].round().astype("Int64")
df["Age"] = df["Age"].round().astype("Int64")

df["Total_Sales"] = df["Price"] * df["Quantity"]

print("\nAFTER CLEANING")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nDATA TYPES")
print(df.dtypes)

print("\nCLEANED DATA")
print(df)

df.to_csv("ecommerce_data_cleaned.csv", index=False)

print("\nCleaned file saved successfully.")