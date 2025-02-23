import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
file_path = "Data/NVDA.csv"
df = pd.read_csv(file_path)

# Convert Date to Datetime
if "Date" in df.columns:
    df["Datetime"] = pd.to_datetime(df["Date"], errors="coerce")
    df.drop(columns=["Date"], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Define feature and target variable
x = df[["Volume"]]
y = df["Close"]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train SVM model
svr_model = SVR(kernel='rbf', epsilon=0.1)
svr_model.fit(x_train, y_train)
y_pred_svr = svr_model.predict(x_test)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
print(f"SVM Mean Absolute Error: {mae_svr}")

# Train RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"RandomForest Mean Absolute Error: {mae_rf}")

# Plot SVR Predictions vs True Values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_svr, alpha=0.5, color='blue', label='SVR Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('True Close Price')
plt.ylabel('Predicted Close Price')
plt.title('SVR: True vs Predicted Close Price')
plt.legend()
plt.grid()

# Plot RandomForest Predictions vs True Values
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='green', label='RF Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('True Close Price')
plt.ylabel('Predicted Close Price')
plt.title('RandomForest: True vs Predicted Close Price')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Plot Prediction Results Over Time
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(y_test.values, label="True Values", color="black")
plt.plot(y_pred_svr, label="SVR Predictions", color="blue", linestyle="dashed")
plt.xlabel("Data Index")
plt.ylabel("Close Price")
plt.title("SVR Prediction Results Over Time")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(y_test.values, label="True Values", color="black")
plt.plot(y_pred_rf, label="RF Predictions", color="green", linestyle="dashed")
plt.xlabel("Data Index")
plt.ylabel("Close Price")
plt.title("RandomForest Prediction Results Over Time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
