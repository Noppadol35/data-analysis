import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from NL import model, X_test, cause_encoder, history, processed_df  # Import from NL.py
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
file_path = "NVDA.csv"
df = load_data(file_path)

# Remove outliers
Q1, Q3 = df["Volume"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df_cleaned = df[(df["Volume"] >= lower_bound) & (df["Volume"] <= upper_bound)]

# Sidebar for page selection
page = st.sidebar.radio("Select Page:", ["📊 Data", "📈 Stock Forecasting", "🤖 Neural Network"])

# ----------------------------- Page 1: Data -----------------------------
if page == "📊 Data":
    st.title("📊 NVIDIA Stock Data")
    if st.checkbox("🔍 Show Raw Data"):
        st.write(df.head())
    
    st.subheader("🔎 Data Distribution")
    fig = px.box(df_cleaned, y=["Close/Last", "Open", "High", "Low", "Volume"], title="Boxplot of Data with Outliers Removed")
    st.plotly_chart(fig)

# ----------------------------- Page 2: Stock Forecasting -----------------------------
elif page == "📈 Stock Forecasting":
    st.title("📈 NVIDIA Stock Price Forecasting")
    y_test_svm, y_pred_svm, mae_svm = train_svm_model(df_cleaned)
    y_test_rf, y_pred_rf, mae_rf = train_rf_model(df_cleaned)
    st.write(f"✅ **SVM Mean Absolute Error:** {mae_svm:.8f}")
    st.write(f"✅ **Random Forest Mean Absolute Error:** {mae_rf:.8f}")

    st.subheader("📉 Model Prediction Comparison")
    tab1, tab2 = st.tabs(["🔴 SVM Predictions", "🟢 Random Forest Predictions"])
    
    with tab1:
        fig_svm = px.scatter(x=y_test_svm, y=y_pred_svm, labels={"x": "True Volume", "y": "Predicted Volume"}, title="SVM: Predicted vs True Volume", color_discrete_sequence=["red"])
        st.plotly_chart(fig_svm)
    with tab2:
        fig_rf = px.scatter(x=y_test_rf, y=y_pred_rf, labels={"x": "True Volume", "y": "Predicted Volume"}, title="Random Forest: Predicted vs True Volume", color_discrete_sequence=["green"])
        st.plotly_chart(fig_rf)

# ----------------------------- Page 3: Neural Network -----------------------------
elif page == "🤖 Neural Network":
    st.title("🤖 Neural Network Model for Accident Prediction")
    
    st.subheader("📊 Training History")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Total Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("📈 Cause Prediction Accuracy")
    fig, ax = plt.subplots()
    ax.plot(history.history['cause_output_accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_cause_output_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("🔥 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(processed_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.subheader("🧠 Sample Predictions")
    sample_data = np.array(X_test.iloc[:5])
    predicted_causes, predicted_casualties = model.predict(sample_data)
    predicted_causes = np.argmax(predicted_causes, axis=1)
    predicted_causes = cause_encoder.inverse_transform(predicted_causes)
    predicted_casualties = predicted_casualties.flatten()
    
    predictions_df = pd.DataFrame({"Predicted Cause": predicted_causes, "Predicted Casualties": predicted_casualties})
    st.write(predictions_df)
