import streamlit as st
import pandas as pd
import plotly.express as px
from MLRF import train_rf_model
from MLSVM import train_svm_model

# Load data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    price_columns = ["Close/Last", "Open", "High", "Low"]
    for col in price_columns:
        df[col] = df[col].replace({'\\$': ''}, regex=True).astype(float)
    return df

# Load data
file_path = "Data/NVDA.csv"
df = load_data(file_path)

# Remove outliers
Q1, Q3 = df["Volume"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df_cleaned = df[(df["Volume"] >= lower_bound) & (df["Volume"] <= upper_bound)]

# Sidebar for page selection
page = st.sidebar.radio("Select Page:", ["📊 Data", "📈 Stock Forecasting"])

# ----------------------------- Page 1: Data -----------------------------
if page == "📊 Data":
    st.title("📊 NVIDIA Stock Data")
    st.write("This page displays NVIDIA stock data and prepares data for forecasting")

    if st.checkbox("🔍 Show Raw Data"):
        st.write(df.head())

    st.write("✅ **Outliers removed from Volume**")

    st.subheader("🔎 Data Distribution")
    fig = px.box(df_cleaned, y=["Close/Last", "Open", "High", "Low", "Volume"], title="Boxplot of Data with Outliers Removed")
    st.plotly_chart(fig)

# ----------------------------- Page 2: Stock Forecasting -----------------------------
elif page == "📈 Stock Forecasting":
    st.title("📈 NVIDIA Stock Price Forecasting")
    st.write("This page uses **SVM** and **Random Forest** to forecast stock prices")

    st.write("✅ **Data split into Train/Test sets** (80% Train, 20% Test)")

    # Train models
    y_test_svm, y_pred_svm, mae_svm = train_svm_model(df_cleaned)
    y_test_rf, y_pred_rf, mae_rf = train_rf_model(df_cleaned)

    st.write(f"✅ **SVM Mean Absolute Error:** {mae_svm:.8f}")
    st.write(f"✅ **Random Forest Mean Absolute Error:** {mae_rf:.8f}")

    # Compare model results
    st.subheader("📉 Model Prediction Comparison")

    tab1, tab2 = st.tabs(["🔴 SVM Predictions", "🟢 Random Forest Predictions"])

    with tab1:
        fig_svm = px.scatter(
            x=y_test_svm, y=y_pred_svm, 
            labels={"x": "True Volume", "y": "Predicted Volume"},
            title="SVM: Predicted vs True Volume",
            color_discrete_sequence=["red"]
        )
        fig_svm.update_traces(marker=dict(size=6, opacity=0.7))
        fig_svm.update_layout(
            xaxis_title="True Volume",
            yaxis_title="Predicted Volume",
            template="plotly_dark"
        )
        st.plotly_chart(fig_svm)

    with tab2:
        fig_rf = px.scatter(
            x=y_test_rf, y=y_pred_rf, 
            labels={"x": "True Volume", "y": "Predicted Volume"},
            title="Random Forest: Predicted vs True Volume",
            color_discrete_sequence=["green"]
        )
        fig_rf.update_traces(marker=dict(size=6, opacity=0.7))
        fig_rf.update_layout(
            xaxis_title="True Volume",
            yaxis_title="Predicted Volume",
            template="plotly_dark"
        )
        st.plotly_chart(fig_rf)