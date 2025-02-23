import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# โหลดข้อมูล
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    price_columns = ["Close/Last", "Open", "High", "Low"]
    for col in price_columns:
        df[col] = df[col].replace({'\\$': ''}, regex=True).astype(float)
    return df

# โหลดข้อมูล
file_path = "Data/NVDA.csv"
df = load_data(file_path)

# ลบ Outliers ล่วงหน้า เพื่อใช้ได้ทุกหน้า
Q1, Q3 = df["Volume"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df_cleaned = df[(df["Volume"] >= lower_bound) & (df["Volume"] <= upper_bound)]

# Sidebar สำหรับเลือกหน้า
page = st.sidebar.radio("เลือกหน้า:", ["📊 ข้อมูล", "📈 การพยากรณ์หุ้น"])

# ----------------------------- หน้า 1: ข้อมูล -----------------------------
if page == "📊 ข้อมูล":
    st.title("📊 ข้อมูลหุ้น NVIDIA")
    st.write("หน้านี้แสดงข้อมูลหุ้นของ NVIDIA และการเตรียมข้อมูลสำหรับการพยากรณ์")

    if st.checkbox("🔍 แสดงข้อมูลดิบ (Raw Data)"):
        st.write(df.head())

    st.write("✅ **ลบค่า Outliers จากปริมาณการซื้อขาย (Volume) แล้ว**")

    st.subheader("🔎 การแจกแจงข้อมูล (Data Distribution)")
    fig = px.box(df_cleaned, y=["Close/Last", "Open", "High", "Low", "Volume"], title="Boxplot แสดงค่า Outliers")
    st.plotly_chart(fig)

# ----------------------------- หน้า 2: การพยากรณ์ -----------------------------
elif page == "📈 การพยากรณ์หุ้น":
    st.title("📈 การพยากรณ์ราคาหุ้น NVIDIA")
    st.write("หน้านี้ใช้ **SVM** และ **Random Forest** เพื่อพยากรณ์ราคาหุ้น")

    # เตรียมข้อมูล
    X = df_cleaned[["Open", "High", "Low"]]
    y = df_cleaned["Close/Last"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("✅ **แบ่งข้อมูลเป็นชุด Train/Test แล้ว** (80% Train, 20% Test)")

    # ฝึกโมเดล
    st.subheader("⚙️ ฝึกโมเดล Machine Learning")
    
    svm_model = SVR(kernel='rbf', C=100, epsilon=0.1)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    mae_svm = mean_absolute_error(y_test, y_pred_svm)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    st.write(f"✅ **SVM Mean Absolute Error:** {mae_svm:.8f}")
    st.write(f"✅ **Random Forest Mean Absolute Error:** {mae_rf:.8f}")

    # เปรียบเทียบผลลัพธ์ของโมเดล
    st.subheader("📉 เปรียบเทียบผลพยากรณ์")

    tab1, tab2 = st.tabs(["🔴 SVM Predictions", "🟢 Random Forest Predictions"])

    with tab1:
        fig_svm = px.scatter(
            x=y_test, y=y_pred_svm, 
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
            x=y_test, y=y_pred_rf, 
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
