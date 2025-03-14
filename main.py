import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from NL import raw_data, model, X_test, cause_encoder, history, processed_df  # Import raw_data from NL.py
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

    # ทำนายผลด้วยโมเดล SVM และ RF
    y_test_svm, y_pred_svm, mae_svm = train_svm_model(df_cleaned)
    y_test_rf, y_pred_rf, mae_rf = train_rf_model(df_cleaned)

    # แสดงค่า MAE
    st.write(f"✅ **SVM Mean Absolute Error:** {mae_svm:.8f}")
    st.write(f"✅ **Random Forest Mean Absolute Error:** {mae_rf:.8f}")

    # คำนวณค่าสูงสุด (Max), ค่าเฉลี่ย (Avg), และค่าต่ำสุด (Min)
    svm_max, svm_avg, svm_min = np.max(y_pred_svm), np.mean(y_pred_svm), np.min(y_pred_svm)
    rf_max, rf_avg, rf_min = np.max(y_pred_rf), np.mean(y_pred_rf), np.min(y_pred_rf)

    # แสดงผลลัพธ์จากโมเดล SVM และ RF พร้อมกราฟ
    st.subheader("📉 Model Prediction Comparison")
    tab1, tab2 = st.tabs(["🔴 SVM Predictions", "🟢 Random Forest Predictions"])
    
    with tab1:
        st.markdown("### 🔴 SVM Model")
        st.write(f"📌 **Max Price:** ${svm_max:.2f}")
        st.write(f"📌 **Avg Price:** ${svm_avg:.2f}")
        st.write(f"📌 **Min Price:** ${svm_min:.2f}")

        fig_svm = px.scatter(x=y_test_svm, y=y_pred_svm, labels={"x": "True Price", "y": "Predicted Price"},
                                title="SVM: Predicted vs True Price", color_discrete_sequence=["red"])
        st.plotly_chart(fig_svm)

    with tab2:
        st.markdown("### 🟢 Random Forest Model")
        st.write(f"📌 **Max Price:** ${rf_max:.2f}")
        st.write(f"📌 **Avg Price:** ${rf_avg:.2f}")
        st.write(f"📌 **Min Price:** ${rf_min:.2f}")

        fig_rf = px.scatter(x=y_test_rf, y=y_pred_rf, labels={"x": "True Price", "y": "Predicted Price"},
                            title="Random Forest: Predicted vs True Price", color_discrete_sequence=["green"])
        st.plotly_chart(fig_rf)

# ----------------------------- Page 3: Neural Network -----------------------------
elif page == "🤖 Neural Network":
    st.title("🤖 Accident Prediction")
    
    # แสดงข้อมูลดิบก่อนการ encode
    if st.checkbox("🔍 Show Raw Data"):
        st.subheader("📊 Raw Data")
        st.write(raw_data.head())  # แสดงข้อมูลดิบที่ยังไม่ได้ทำการ encode
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Training History", "📈 Cause Prediction Accuracy", "🔥 Feature Correlation", "🧠 Sample Predictions"])
    
    with tab1:
        st.subheader("📊 Training History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines+markers', name='Total Loss'))
        fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines+markers', name='Validation Loss'))
        fig.update_layout(xaxis_title='Epochs', yaxis_title='Loss', template='plotly_dark')
        st.plotly_chart(fig)
    
    with tab2:
        st.subheader("📈 Cause Prediction Accuracy")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['cause_output_accuracy'], mode='lines+markers', name='Train Accuracy'))
        fig.add_trace(go.Scatter(y=history.history['val_cause_output_accuracy'], mode='lines+markers', name='Validation Accuracy'))
        fig.update_layout(xaxis_title='Epochs', yaxis_title='Accuracy', template='plotly_dark')
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader("🔥 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(processed_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)
    
    with tab4:
        st.subheader("🧠 Sample Predictions")
        sample_data = np.array(X_test.iloc[:5])
        predicted_causes, predicted_casualties = model.predict(sample_data)
        predicted_causes = np.argmax(predicted_causes, axis=1)
        predicted_causes = cause_encoder.inverse_transform(predicted_causes)
        predicted_casualties = predicted_casualties.flatten()
        
        predictions_df = pd.DataFrame({
            "Predicted Cause": predicted_causes,
            "Predicted Casualties": predicted_casualties
        })
        st.dataframe(predictions_df.style.format({"Predicted Casualties": "{:.0f}"}).set_properties(**{'text-align': 'center'}))