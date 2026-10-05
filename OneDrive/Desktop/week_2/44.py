
import csv
import os

python_path = r"C:\Users\RAHUL\AppData\Local\Programs\Python\Python312\python.exe"

os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum,
    avg,
    min,
    max,
    round,
    when,
    coalesce,
    expr,
    unix_timestamp,
    desc,
    trim,
    upper
)

spark = SparkSession.builder \
    .appName("Transportation ETL Pipeline") \
    .master("local[2]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

input_file = "transportation_dataset.csv"
output_file = "transportation_final_output.csv"

print("========== TRANSPORTATION ETL PIPELINE ==========")

print("\n========== DATA LOADING ==========")

with open(input_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    data = list(reader)

df = spark.createDataFrame(data)

original_count = df.count()
original_columns = len(df.columns)

print("Dataset loaded successfully.")
print("Original Rows:", original_count)
print("Original Columns:", original_columns)

print("\n========== DATA EXPLORATION ==========")

df.show(10, truncate=False)

df.printSchema()

df.select(
    "VendorID",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "total_amount"
).show(10)

df.describe(
    "trip_distance",
    "fare_amount",
    "total_amount",
    "passenger_count"
).show()

print("\n========== ORIGINAL DATA QUALITY ==========")

original_duplicates = original_count - df.dropDuplicates().count()

print("Original Duplicate Rows:", original_duplicates)

missing_expressions = []

for column in df.columns:
    missing_expressions.append(
        sum(
            when(
                col(column).isNull() | (trim(col(column)) == ""),
                1
            ).otherwise(0)
        ).alias(column)
    )

missing_result = df.agg(*missing_expressions).collect()[0]

print("Missing Values:")

for column in df.columns:
    missing_count = missing_result[column]

    if missing_count > 0:
        print(column, ":", missing_count)

print("\n========== DATA CLEANING ==========")

df = df.replace("", None)

if "ehail_fee" in df.columns:
    df = df.drop("ehail_fee")

if "store_and_fwd_flag" in df.columns:
    df = df.withColumn(
        "store_and_fwd_flag",
        upper(trim(col("store_and_fwd_flag")))
    )

numeric_columns = {
    "fare_amount": "double",
    "total_amount": "double",
    "trip_distance": "double",
    "passenger_count": "int",
    "VendorID": "int",
    "PULocationID": "int",
    "DOLocationID": "int",
    "RatecodeID": "int",
    "payment_type": "int",
    "trip_type": "int"
}

for column, data_type in numeric_columns.items():
    if column in df.columns:
        df = df.withColumn(
            column,
            expr(f"try_cast(`{column}` AS {data_type})")
        )

df = df.withColumn(
    "pickup_time",
    coalesce(
        expr("try_to_timestamp(lpep_pickup_datetime, 'MM-dd-yyyy HH:mm')"),
        expr("try_to_timestamp(lpep_pickup_datetime, 'dd-MM-yyyy HH:mm')"),
        expr("try_to_timestamp(lpep_pickup_datetime, 'yyyy-MM-dd HH:mm:ss')")
    )
)

df = df.withColumn(
    "dropoff_time",
    coalesce(
        expr("try_to_timestamp(lpep_dropoff_datetime, 'MM-dd-yyyy HH:mm')"),
        expr("try_to_timestamp(lpep_dropoff_datetime, 'dd-MM-yyyy HH:mm')"),
        expr("try_to_timestamp(lpep_dropoff_datetime, 'yyyy-MM-dd HH:mm:ss')")
    )
)

df = df.dropDuplicates()

before_filter_count = df.count()

df = df.filter(
    (col("fare_amount") >= 0) &
    (col("trip_distance") >= 0) &
    (col("total_amount") >= 0)
)

after_cleaning_count = df.count()

print("Duplicate Rows Removed:", original_count - before_filter_count)
print("Invalid Numeric Rows Removed:", before_filter_count - after_cleaning_count)
print("Rows Before Cleaning:", original_count)
print("Rows After Cleaning:", after_cleaning_count)
print("Total Rows Removed:", original_count - after_cleaning_count)

print("\n========== DATA TRANSFORMATION ==========")

df = df.withColumn(
    "Trip_Duration_Min",
    round(
        (
            unix_timestamp("dropoff_time") -
            unix_timestamp("pickup_time")
        ) / 60,
        2
    )
)

df = df.withColumn(
    "Fare_Per_KM",
    round(
        when(
            col("trip_distance") > 0,
            col("fare_amount") / col("trip_distance")
        ),
        2
    )
)

df = df.withColumn(
    "Trip_Category",
    when(col("trip_distance").isNull(), "Unknown")
    .when(col("trip_distance") <= 2, "Short")
    .when(col("trip_distance") <= 5, "Medium")
    .otherwise("Long")
)

df = df.withColumn(
    "Fare_Category",
    when(col("fare_amount").isNull(), "Unknown")
    .when(col("fare_amount") < 10, "Low")
    .when(col("fare_amount") <= 20, "Medium")
    .otherwise("High")
)

df = df.withColumn(
    "Passenger_Load",
    when(col("passenger_count").isNull(), "Unknown")
    .when(col("passenger_count") <= 1, "Low")
    .when(col("passenger_count") <= 3, "Medium")
    .otherwise("High")
)

df = df.withColumn(
    "Payment_Category",
    when(col("payment_type") == 1, "Credit Card")
    .when(col("payment_type") == 2, "Cash")
    .when(col("payment_type") == 3, "No Charge")
    .when(col("payment_type") == 4, "Dispute")
    .otherwise("Unknown")
)

print("Derived Columns Created Successfully.")

df.select(
    "trip_distance",
    "fare_amount",
    "total_amount",
    "passenger_count",
    "Trip_Duration_Min",
    "Fare_Per_KM",
    "Trip_Category",
    "Fare_Category",
    "Passenger_Load",
    "Payment_Category"
).show(20, truncate=False)

print("\n========== DATA VALIDATION ==========")

final_count = df.count()

print("Final Rows:", final_count)
print("Final Columns:", len(df.columns))

print(
    "Remaining Duplicate Rows:",
    final_count - df.dropDuplicates().count()
)

print(
    "Missing Fare:",
    df.filter(col("fare_amount").isNull()).count()
)

print(
    "Missing Distance:",
    df.filter(col("trip_distance").isNull()).count()
)

print(
    "Missing Passenger Count:",
    df.filter(col("passenger_count").isNull()).count()
)

print(
    "Invalid Pickup Dates:",
    df.filter(col("pickup_time").isNull()).count()
)

print(
    "Invalid Dropoff Dates:",
    df.filter(col("dropoff_time").isNull()).count()
)

print("\n========== OVERALL STATISTICS ==========")

df.select(
    count("*").alias("Total_Trips"),
    round(sum("fare_amount"), 2).alias("Total_Fare"),
    round(sum("total_amount"), 2).alias("Total_Revenue"),
    round(avg("fare_amount"), 2).alias("Average_Fare"),
    round(avg("trip_distance"), 2).alias("Average_Distance"),
    round(avg("Trip_Duration_Min"), 2).alias("Average_Duration"),
    round(min("fare_amount"), 2).alias("Minimum_Fare"),
    round(max("fare_amount"), 2).alias("Maximum_Fare"),
    round(min("trip_distance"), 2).alias("Minimum_Distance"),
    round(max("trip_distance"), 2).alias("Maximum_Distance")
).show(truncate=False)

print("\n========== TRIP CATEGORY ANALYSIS ==========")

df.groupBy("Trip_Category") \
    .agg(
        count("*").alias("Total_Trips"),
        round(sum("total_amount"), 2).alias("Total_Revenue"),
        round(avg("fare_amount"), 2).alias("Average_Fare"),
        round(avg("trip_distance"), 2).alias("Average_Distance"),
        min("fare_amount").alias("Minimum_Fare"),
        max("fare_amount").alias("Maximum_Fare")
    ) \
    .orderBy(desc("Total_Trips")) \
    .show()

print("\n========== PAYMENT ANALYSIS ==========")

df.groupBy("Payment_Category") \
    .agg(
        count("*").alias("Total_Trips"),
        round(sum("total_amount"), 2).alias("Total_Revenue"),
        round(avg("total_amount"), 2).alias("Average_Transaction")
    ) \
    .orderBy(desc("Total_Revenue")) \
    .show()

print("\n========== PASSENGER ANALYSIS ==========")

df.groupBy("passenger_count") \
    .agg(
        count("*").alias("Total_Trips"),
        round(avg("fare_amount"), 2).alias("Average_Fare"),
        round(sum("total_amount"), 2).alias("Total_Revenue"),
        min("trip_distance").alias("Minimum_Distance"),
        max("trip_distance").alias("Maximum_Distance")
    ) \
    .orderBy("passenger_count") \
    .show()

print("\n========== PICKUP LOCATION ANALYSIS ==========")

df.groupBy("PULocationID") \
    .agg(
        count("*").alias("Total_Trips"),
        round(sum("total_amount"), 2).alias("Total_Revenue"),
        round(avg("fare_amount"), 2).alias("Average_Fare")
    ) \
    .orderBy(desc("Total_Revenue")) \
    .show(10)

print("\n========== DROPOFF LOCATION ANALYSIS ==========")

df.groupBy("DOLocationID") \
    .agg(
        count("*").alias("Total_Trips"),
        round(sum("total_amount"), 2).alias("Total_Revenue")
    ) \
    .orderBy(desc("Total_Trips")) \
    .show(10)

print("\n========== HIGH FARE TRIPS ==========")

df.filter(
    col("fare_amount") > 50
).select(
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "total_amount"
).orderBy(desc("fare_amount")) \
 .show(10, truncate=False)

print("\n========== LONG DISTANCE TRIPS ==========")

df.filter(
    col("trip_distance") > 10
).select(
    "trip_distance",
    "fare_amount",
    "total_amount",
    "Trip_Duration_Min"
).orderBy(desc("trip_distance")) \
 .show(10, truncate=False)

print("\n========== FARE CATEGORY ANALYSIS ==========")

df.groupBy("Fare_Category") \
    .agg(
        count("*").alias("Total_Trips"),
        round(sum("total_amount"), 2).alias("Total_Revenue"),
        round(avg("fare_amount"), 2).alias("Average_Fare")
    ) \
    .orderBy(desc("Total_Revenue")) \
    .show()

print("\n========== FILTERED TRIP ANALYSIS ==========")

print(
    "Trips with fare above 50:",
    df.filter(col("fare_amount") > 50).count()
)

print(
    "Trips above 10 distance units:",
    df.filter(col("trip_distance") > 10).count()
)

print(
    "Trips with duration above 60 minutes:",
    df.filter(col("Trip_Duration_Min") > 60).count()
)

print("\n========== SAVING PROCESSED DATA ==========")

rows = df.collect()

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(df.columns)

    for row in rows:
        writer.writerow([
            value.isoformat(sep=" ") if hasattr(value, "isoformat") else value
            for value in row
        ])

print("Processed data saved successfully.")
print("Output File:", os.path.abspath(output_file))
print("Saved Rows:", len(rows))

print("\n========== PIPELINE SUMMARY ==========")

print("Original Rows:", original_count)
print("Processed Rows:", final_count)
print("Original Columns:", original_columns)
print("Processed Columns:", len(df.columns))
print("Rows Removed:", original_count - final_count)
print("Output Format: CSV")

print("\n========== BUSINESS INSIGHTS ==========")

print("Trip category analysis identifies common trip distances.")
print("Payment analysis compares transaction volume and revenue.")
print("Passenger analysis compares passenger patterns and fares.")
print("Pickup location analysis identifies high-revenue locations.")
print("Fare analysis identifies expensive trips.")
print("Data quality comparison measures the effect of cleaning.")

print("\n========== ETL PIPELINE COMPLETED ==========")

spark.stop()