import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "Data/global_traffic_accidents.csv"
df = pd.read_csv(file_path)

# Display sample rows of the dataset
print(df.head())
print('\n')
print(df.tail())

# Display the data types of each column
print(df.dtypes)

# Display the number of missing values in each column
print(df.isnull().sum())

# Display boxplot vehicle involved and casualties that show outliers
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
df.boxplot(column=["Vehicles Involved"])
plt.subplot(1, 2, 2)
df.boxplot(column=["Casualties"])
plt.show()

# Convert Date and Time to Datetime
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
df.drop(columns=["Date", "Time"], inplace=True)

# Check for missing values
missing_values = df.isin(["", "Unknown", "-", "NA", "N/A"]).sum()

# Check for invalid latitude and longitude
valid_lat_long = (df["Latitude"].between(-90, 90)) & (df["Longitude"].between(-180, 180))
df = df[valid_lat_long]

# Identify outliers using IQR
numerical_columns = ["Vehicles Involved", "Casualties"]
for col in numerical_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Remove duplicates
df.drop_duplicates(inplace=True)

# split data 20% test and 80% train
x = df[['Weather Condition']]
y = df['Casualties']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

