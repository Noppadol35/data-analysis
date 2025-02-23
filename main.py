import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "Data/NVDA.csv"
df = pd.read_csv(file_path)

# Display sample rows of the dataset
print(df.sample(5))  # Show 5 random rows instead of head and tail

# Display the data types of each column
print(df.dtypes)

# Display the number of missing values in each column
print(df.isnull().sum())

# Convert Date to Datetime
if "Date" in df.columns:
    df["Datetime"] = pd.to_datetime(df["Date"], errors="coerce")
    df.drop(columns=["Date"], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Define feature and target variable
x = df[["Volume"]]  # Adjust feature selection as needed
y = df["Close"]  # Adjust target variable if needed

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train SVM model
svr_model = SVR(kernel='rbf', epsilon=0.1)
svr_model.fit(x_train, y_train)

# Predict on test set using SVM
y_pred_svr = svr_model.predict(x_test)

# Evaluate SVM model using Mean Absolute Error (MAE)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
print(f"SVM Mean Absolute Error: {mae_svr}")

# Train RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Predict on test set using RandomForest
y_pred_rf = rf_model.predict(x_test)

# Evaluate RandomForest model using Mean Absolute Error (MAE)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"RandomForest Mean Absolute Error: {mae_rf}")

# Compare Two Models with Scatter Plots
plt.figure(figsize=(12, 6))

# SVM Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_svr, alpha=0.5, color='blue', label='SVM Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('True Volume')
plt.ylabel('Predicted Volume')
plt.title('SVM: True vs Predicted Volume')
plt.legend()
plt.grid()

# RandomForest Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='green', label='RF Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('True Volume')
plt.ylabel('Predicted Volume')
plt.title('RandomForest: True vs Predicted Volume')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
