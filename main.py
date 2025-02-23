import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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

# Train models
svr_model = SVR(kernel='rbf', epsilon=0.1)
svr_model.fit(x_train, y_train)
y_pred_svr = svr_model.predict(x_test)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Streamlit UI
st.title("Stock Price Prediction (NVDA)")
st.write("This app predicts the closing price of NVDA stock using Machine Learning models.")

# Model Performance
st.subheader("Model Performance")
st.write(f"SVM Mean Absolute Error: {mae_svr}")
st.write(f"RandomForest Mean Absolute Error: {mae_rf}")

# Interactive scatter plots
if st.button("Show Prediction Scatter Plot"):
    fig1 = px.scatter(x=y_test, y=y_pred_svr, labels={'x': 'True Close Price', 'y': 'Predicted Close Price'},
                    title="SVR: True vs Predicted Close Price")
    fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig1)
    
    fig2 = px.scatter(x=y_test, y=y_pred_rf, labels={'x': 'True Close Price', 'y': 'Predicted Close Price'},
                    title="RandomForest: True vs Predicted Close Price")
    fig2.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig2)

# Interactive line plots over time
if st.button("Show Prediction Results Over Time"):
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=y_test.values, mode='lines', name='True Values', line=dict(color='black')))
    fig3.add_trace(go.Scatter(y=y_pred_svr, mode='lines', name='SVR Predictions', line=dict(color='blue', dash='dash')))
    fig3.update_layout(title="SVR Prediction Results Over Time", xaxis_title="Volume", yaxis_title="Close Price")
    st.plotly_chart(fig3)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=y_test.values, mode='lines', name='True Values', line=dict(color='black')))
    fig4.add_trace(go.Scatter(y=y_pred_rf, mode='lines', name='RF Predictions', line=dict(color='green', dash='dash')))
    fig4.update_layout(title="RandomForest Prediction Results Over Time", xaxis_title="Volume", yaxis_title="Close Price")
    st.plotly_chart(fig4)
