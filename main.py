import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ISNeuralNetwork import raw_data, model, X_test, cause_encoder, history, processed_df
from MLRF import train_rf_model
from MLSVM import train_svm_model

import sys
sys.path.append('D:\Work\Code\data-analysis\MLSVM.py')
sys.path.append('D:\Work\Code\data-analysis\MLRF.py')
sys.path.append('D:\Work\Code\data-analysis\ISNeuralNetwork')


st.set_page_config(
    page_title="Intelligent System Project",
    page_icon="📊" ,
)

# Load data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # กำหนดคอลัมน์ที่ต้องการแปลงเป็นตัวเลข
    numeric_cols = ["Close/Last", "Open", "High", "Low", "Volume"]

    for col in numeric_cols:
        # ลบสัญลักษณ์พิเศษ เช่น $ และช่องว่าง
        df[col] = df[col].replace({'\\$': '', ' ': ''}, regex=True)
        
        # แปลงเป็น float โดยกำหนด errors='coerce' เพื่อให้แปลงไม่ได้เป็น NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ลบแถวที่มีค่า NaN (ถ้ามี)
    df.dropna(inplace=True)

    return df
# Sidebar for page selection
page = st.sidebar.radio("Select Page:", [ "📊 Summarize ML", "📈 Demo Stock Forecasting", "⚛️ Summarize NL","🤖 Demo Neural Network"])

# ----------------------------- Page 1: Data -----------------------------
if page == "📊 Summarize ML":
    st.title("📊 Dataset US Stock From NASDAQ")
    st.markdown("🔗 [Download Dataset](https://www.nasdaq.com/market-activity/stocks/msft/historical?page=1&rows_per_page=10&timeline=y1)")
    
    # อัปโหลดไฟล์หรือใช้ไฟล์ตัวอย่าง
    uploaded_file = st.file_uploader("Upload your CSV file (Optional)", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        file_path = "NVDA.csv"
        df = load_data(file_path)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summarize Data", "📈 Data Cleaning Demo", "🌠 SVM", "🌲 RF", "Ref."])
    
    with tab1:
        st.write("### 🔹 Features in Dataset")
        
        feature_descriptions = {
            "Date": "📅 วันที่ที่บันทึกข้อมูล (MM/DD/YYYY)",
            "Close/Last": "💰 ราคาปิดของหุ้นในวันนั้น",
            "Volume": "📊 ปริมาณการซื้อขายหุ้นในวันนั้น",
            "Open": "🚀 ราคาเปิดของหุ้นในวันนั้น",
            "High": "📈 ราคาสูงสุดของหุ้นในวันนั้น",
            "Low": "📉 ราคาต่ำสุดของหุ้นในวันนั้น"
        }
        
        st.table(df.head())
        
        for col in df.columns:
            if col in feature_descriptions:
                st.write(f"**{col}** - {feature_descriptions[col]}")

        st.write("### 📊 Summary Statistics")
        st.write(df.describe())

        st.write("### 🛠 Data Cleaning Process")
        st.write("1. ลบแถวที่มีค่า Missing Values")
        st.code("df.dropna(inplace=True)")
        st.write("2. แปลงข้อมูลให้อยู่ในรูปแบบที่ถูกต้อง")
        st.code("df[col] = pd.to_numeric(df[col], errors='coerce')")
        st.write("3. ลบข้อมูลที่ซ้ำซ้อน")
        st.code("df.drop_duplicates(inplace=True)")
        st.write("4. ลบค่า Outliers ด้วย IQR Method")
        st.code("Q1, Q3 = df['Volume'].quantile([0.25, 0.75])")
        st.code("IQR = Q3 - Q1")
        st.code("lower_bound = Q1 - 1.5 * IQR")
        st.code("upper_bound = Q3 + 1.5 * IQR")
        st.code("df = df[(df['Volume'] >= lower_bound) & (df['Volume'] <= upper_bound)]")
        st.write("5. แสดงผลลัพธ์ของข้อมูล")
        

        # กำหนดคอลัมน์ที่ต้องแปลงเป็นตัวเลข
        numeric_cols = ["Close/Last", "Volume", "Open", "High", "Low"]
        
        # ลบสัญลักษณ์พิเศษ เช่น "$" และช่องว่าง แล้วแปลงเป็น float อย่างปลอดภัย
        for col in numeric_cols:
            df[col] = df[col].replace({'\$': '', ',': '', ' ': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')  # ใช้ pd.to_numeric() เพื่อหลีกเลี่ยงข้อผิดพลาด


        # ลบแถวที่มีค่า NaN
        df_cleaned = df.dropna()



        # คำนวณเปอร์เซ็นต์การเปลี่ยนแปลงของราคา
        df_cleaned["Price Change (%)"] = ((df_cleaned["Close/Last"] - df_cleaned["Open"]) / df_cleaned["Open"]) * 100

        st.write("### 📉 Daily Price Change Percentage (หลังทำความสะอาด)")
        st.write(df_cleaned[["Date", "Open", "Close/Last", "Price Change (%)"]])

        # แสดงกราฟข้อมูลหลังทำความสะอาด
        fig = px.line(df_cleaned, x="Date", y="Price Change (%)", title="📉 Daily Price Change (%) Over Time (หลังทำความสะอาด)")
        st.plotly_chart(fig)
        
        st.write("## Why using Boxplot manage outlier ?")
        st.write("📊 ใช้ Boxplot เพื่อแสดงการกระจายตัวของข้อมูล และช่วยในการตรวจสอบค่า Outliers ในข้อมูล")
        st.image("asset/Boxplot.png", use_container_width=True)
        st.write("🔍 จากกราฟ Boxplot จะเห็นได้ว่ามีค่า Outliers ที่มี Upper = Q3 + 1.5 * IQR และ Lower = Q1 - 1.5 * IQR")
        
        



    with tab2:
        st.write("### 📌 Raw Data (Before Cleaning)")
        st.write(df.head())
    
        # ตัวเลือกการทำ Data Cleaning
        st.subheader("🛠️ Choose Data Cleaning Method")
        remove_duplicates = st.checkbox("🗑️ Remove Duplicate Rows")
        remove_outliers = st.checkbox("📉 Remove Outliers (IQR Method)")
        
        # ทำ Data Cleaning
        df_cleaned = df.copy()
        
        # ลบข้อมูลซ้ำซ้อน
        if remove_duplicates:
            df_cleaned = df_cleaned.drop_duplicates()
        
        
        # ลบค่า Outliers
        if remove_outliers:
            Q1, Q3 = df_cleaned["Volume"].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned["Volume"] >= lower_bound) & (df_cleaned["Volume"] <= upper_bound)]
        
        # แสดงผลหลังทำ Data Cleaning
        st.write("### ✅ Cleaned Data (After Cleaning)")
        st.write(df_cleaned.head())
        
        # เปรียบเทียบจำนวนแถวก่อนและหลังทำความสะอาด
        st.write(f"📊 Data Before Cleaning: {df.shape[0]} rows")
        st.write(f"📊 Data After Cleaning: {df_cleaned.shape[0]} rows")
        
        # แสดงกราฟเปรียบเทียบการกระจายตัวของข้อมูล
        st.subheader("🔎 Data Distribution Before vs After Cleaning")
        fig_before = px.box(df, y=["Close/Last", "Open", "High", "Low", "Volume"], title="Before Cleaning")
        fig_after = px.box(df_cleaned, y=["Close/Last", "Open", "High", "Low", "Volume"], title="After Cleaning")
        st.plotly_chart(fig_before)
        st.plotly_chart(fig_after)
        
        st.write("✅ **Data Cleaning Completed!**")
    
    with tab3:
        st.write("### SVM Algorithm")
        st.write("คือ เป็นหนึ่งในอัลกอริธึมการเรียนรู้ของเครื่อง (Machine Learning) ที่ใช้สำหรับ Classification และ Regression โดยเฉพาะอย่างยิ่งในงานที่มีขนาดข้อมูลไม่ใหญ่มากและต้องการความแม่นยำสูง")
        st.write("กำหนด Open , High , Low เป็น Feture X และผมได้กำหนด Close/Last เป็น label แทนลงใน y หลังจากนั้นก็ทำการ split ข้อมูลแบ่งเป็น train 80% กับ test 20% โดยให้ random state = 42")
        st.image("asset/SVM1.png", use_container_width=True)
        
        st.write("**⚡ Kernel Trick คืออะไร?**")
        st.write("Kernel Trick เป็นเทคนิคที่ใช้ในการแปลงข้อมูลที่ไม่สามารถแบ่งได้ในมิติปัจจุบันให้ไปอยู่ใน มิติที่สูงขึ้น (Higher Dimensional Space) เพื่อให้สามารถแบ่งด้วยเส้นตรงได้")
        #แบ่งเป็น Bullet Points
        st.write("🔹 **Linear Kernel** ➝ ใช้สำหรับข้อมูลที่แบ่งได้ด้วยเส้นตรง")
        st.write("🔹 **Polynomial Kernel** ➝ ใช้สำหรับข้อมูลที่ต้องการเส้นแบ่งแบบ Polynomial (วงกลม)")
        st.write("🔹 **RBF Kernel** ➝ ใช้สำหรับข้อมูลที่มีความซับซ้อนและไม่เป็นเชิงเส้น ✅")
        st.write(" **📌 สรุป**")
        st.write("SVM เป็นอัลกอริธึมที่ทรงพลัง ใช้งานได้ทั้ง Classification และ Regression แต่โดยทั่วไปนิยมใช้ใน Classification มากกว่า ใช้ Hyperplane และ Margin ในการแยกข้อมูลออกจากกัน และสามารถใช้ Kernel Trick เพื่อทำให้แยกข้อมูลที่ซับซ้อนได้ดียิ่งขึ้น 🚀")
        st.image("asset/SVM2.png", use_container_width=True)
        st.write("การหยิบ model Support Vector Machine เลือกใช้แบบ Regression ในการประมาณค่าของข้อมูล จะให้ค่าคลาดเคลื่อนอยู่ในช่วงที่กำหนด โดยถ้า จุดข้อมูลที่อยู่ในระยะ ε ถือว่าไม่มี error แต่ถ้า จุดที่อยู่ นอกช่วง ε จะถูก penalized ด้วย loss function หลังจากนั้นก็ประเมิน model ด้วย ค่า mean absolute error")
        
        y_test_svm, y_pred_svm, mae_svm = train_svm_model(df_cleaned)
        
        st.write(f"✅ **SVM Mean Absolute Error:** {mae_svm:.8f}")
        st.write(f"✅ **SVM Predictions**")
        st.write(pd.DataFrame({"True Price": y_test_svm, "Predicted Price": y_pred_svm}).head())
        
    with tab4:
        st.write("### 🌲 Random Forest Algorithm")
        st.write("เป็นอัลกอริธึม Machine Learning แบบ Ensemble Learning ที่สร้างโมเดลจาก หลายๆ Decision Trees และใช้การโหวตหรือค่าเฉลี่ยจากต้นไม้ทั้งหมดเพื่อให้ได้ผลลัพธ์ที่แม่นยำขึ้น ")
        st.write("กำหนด Open , High , Low เป็น Feture X และผมได้กำหนด Close/Last เป็น label แทนลงใน y หลังจากนั้นก็ทำการ split ข้อมูลแบ่งเป็น train 80% กับ test 20% โดยให้ random state = 42")
        st.image("asset/RF1.png", use_container_width=True)
        st.write("#### 🎯 หลักการทำงานของ Random Forest")
        st.write("🏗️ 1. การสร้างหลายต้นไม้ (Multiple Decision Trees)")
        st.write("🔹 Random Forest จะสร้าง หลายๆ Decision Trees โดยใช้ข้อมูลที่แตกต่างกันเล็กน้อย")
        st.write("🔹 แต่ละต้นไม้จะถูกเทรนด้วย ชุดข้อมูลสุ่มบางส่วน (Bootstrap Sampling)")
        st.write("🔹 แต่ละต้นไม้จะเลือก Feature แบบสุ่ม ในแต่ละจุดแตกแขนง (Split)")
        st.write("🎲 2. การรวมผลลัพธ์")
        st.write("🔹 สำหรับ **Classification** → ใช้ Majority Vote (เลือกผลที่มีเสียงโหวตมากที่สุด)")
        st.write("🔹 สำหรับ **Regression** → ใช้ ค่าเฉลี่ย ของค่าที่ทำนายจากทุกต้นไม้ ✅")
        st.image("asset/RF2.png", use_container_width=True)
        st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
        st.write("#### 🔍 อธิบายโค้ด")
        st.write("1. ใช้ RandomForestRegressor(n_estimators=100) → สร้าง 100 ต้นไม้")
        st.write("2. แบ่งข้อมูลเป็น Train/Test ด้วย train_test_split()")
        st.write("3. เทรนโมเดล rf_model.fit(X_train, y_train)")
        st.write("4. ทำนาย y_pred_rf = rf_model.predict(X_test)")
        st.write("5. วัดค่าความผิดพลาด Mean Absolute Error (MAE)")
        st.write("การรวมกันของหลายๆ Decision Trees โดย สร้างหลายๆ Decision Trees แต่ละต้นไม้เรียนรู้ข้อมูลที่สุ่มมา ทำให้มีความแตกต่างกัน จากนั้น แต่ละต้นไม้เลือกใช้ บางฟีเจอร์เท่านั้น ทำให้ต้นไม้แต่ละต้นไม่เหมือนกัน สุดท้าย รวมผลลัพธ์ของทุกต้นไม้ โดยใช้ ใช้ค่าเฉลี่ยของทุกต้นไม้เป็นค่าทำนายสุดท้าย หลังจากนั้นผมก็ประเมิน model ด้วย ค่า mean absolute error")
        y_test_rf, y_pred_rf, mae_rf = train_rf_model(df_cleaned)
        
        st.write(f"✅ **Random Forest Mean Absolute Error:** {mae_rf:.8f}")
        st.write(f"✅ **Random Forest Predictions**")
        st.write(pd.DataFrame({"True Price": y_test_rf, "Predicted Price": y_pred_rf}).head())
    
    with tab5:
        st.link_button("🔗 Random Forest Algorithm", "https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/")
        st.link_button("🔗 Support Vector Machine Algorithm", "https://www.geeksforgeeks.org/support-vector-machine-algorithm/")
        st.link_button("🔗 Data Cleaning", "https://1stcraft.com/what-is-data-cleansing/")
        st.link_button("🔗 Stock NASDAQ Data-set", "https://www.nasdaq.com/market-activity/stocks/msft/historical?page=1&rows_per_page=10&timeline=y1")
        

# ----------------------------- Page 2: Stock Forecasting -----------------------------
elif page == "📈 Demo Stock Forecasting":
    st.title("📈 NVIDIA Stock Price Forecasting")

    # Choose to upload a file or use the demo file
    uploaded_file = st.file_uploader("Upload your CSV file (Optional)", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        # Default demo file
        file_path = "NVDA.csv"
        df = load_data(file_path)

    # Remove outliers
    Q1, Q3 = df["Volume"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_cleaned = df[(df["Volume"] >= lower_bound) & (df["Volume"] <= upper_bound)]

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
    
    st.line_chart(df_cleaned["Close/Last"])
    
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
        
# ----------------------------- Page 3: Summarize NL -----------------------------

elif page == "⚛️ Summarize NL":
    st.title("⚛️ Natural Language Processing")
    
    # แสดงข้อมูลดิบก่อนการ encode
    if st.checkbox("🔍 Show Raw Data"):
        st.subheader("📊 Raw Data")
        st.write(raw_data.head())  # แสดงข้อมูลดิบที่ยังไม่ได้ทำการ encode
    
    tab1, tab2, tab3= st.tabs(["📊 Summarize Data", "🌲 Neural Network", "Ref."])
    
    with tab1:
        st.write("### 🔹 Features in Dataset")
        
        feature_descriptions = {
            "Accident ID": "🔢 รหัสอุบัติเหตุ",
            "Date": "📅 วันที่เกิดอุบัติเหตุ",
            "Time": "🕒 เวลาเกิดอุบัติเหตุ",
            "Location": "📍 สถานที่เกิดอุบัติเหตุ",
            "Weather": "🌦️ สภาพอากาศ",
            "Cause": "🚗 สาเหตุของอุบัติเหตุ",
            "Casualties": "🤕 จำนวนผู้บาดเจ็บ",
            "Road Condition": "🛣️ สภาพถนน",
        }
        st.table(raw_data.head())
        
        for col in raw_data.columns:
            if col in feature_descriptions:
                st.write(f"**{col}** - {feature_descriptions[col]}")
                
        st.write("### 📊 Summary Statistics")
        st.write(raw_data.describe())
        
        st.write("### 🛠 Data Cleaning Process")
        st.write("1. ลบแถวที่มีค่า Missing Values")
        st.code("df.dropna(inplace=True)")
        st.write("2. แปลงข้อมูลให้อยู่ในรูปแบบที่ถูกต้อง")
        st.code("df[col] = pd.to_numeric(df[col], errors='coerce')")
        st.write("3. ลบข้อมูลที่ซ้ำซ้อน")
        st.code("df.drop_duplicates(inplace=True)")
        st.write("4. ลบค่า Outliers ด้วย IQR Method")
        st.code("Q1, Q3 = df['Volume'].quantile([0.25, 0.75])")
        st.code("IQR = Q3 - Q1")
        st.code("lower_bound = Q1 - 1.5 * IQR")
        st.code("upper_bound = Q3 + 1.5 * IQR")
        st.code("df = df[(df['Volume'] >= lower_bound) & (df['Volume'] <= upper_bound)]")
        st.write("5. แสดงผลลัพธ์ของข้อมูล")
        
        st.write("ลบแถวที่มีค่า Missing Values")
        st.code("df_cleaned = df.dropna()")
        
    with tab2:
        st.write("### 🌲 Neural Network Algorithm")
        st.write("การพัฒนาระบบการทำนายสาเหตุและจำนวนผู้บาดเจ็บจากอุบัติเหตุทางถนน โดยใช้ Neural Network แบบ Multi-Output เป็นการใช้เทคนิค Machine Learning เพื่อทำการทำนาย 2 ประเภทของข้อมูลในเวลาเดียวกัน ได้แก่ สาเหตุของอุบัติเหตุ (Classification) และ จำนวนผู้บาดเจ็บ (Regression)")
        
        st.write("## ใช้ One-Hot Encoding สำหรับข้อมูลหมวดหมู่")
        st.write("""
        การ **One-Hot Encoding** เป็นการแปลงข้อมูลหมวดหมู่ให้เป็นตัวเลขโดยการสร้างคอลัมน์ใหม่สำหรับแต่ละประเภทของข้อมูล โดยคอลัมน์นั้นจะมีค่าเป็น `1` หากแถวข้อมูลนั้นมีหมวดหมู่นั้น และ `0` หากไม่มี
        ตัวอย่างเช่น ในกรณีที่เรามีข้อมูลเกี่ยวกับสภาพอากาศ เช่น `Sunny`, `Rainy`, และ `Cloudy` ข้อมูลที่ได้จาก One-Hot Encoding จะมี 3 คอลัมน์ใหม่:
        - `Weather_Condition_Sunny`
        - `Weather_Condition_Rainy`
        - `Weather_Condition_Cloudy`
        
        ข้อมูลที่ได้จะมีลักษณะดังนี้:

        | Weather Condition | Weather_Condition_Sunny | Weather_Condition_Rainy | Weather_Condition_Cloudy |
        |-------------------|-------------------------|-------------------------|--------------------------|
        | Sunny             | 1                       | 0                       | 0                        |
        | Rainy             | 0                       | 1                       | 0                        |
        | Cloudy            | 0                       | 0                       | 1                        |
        | Sunny             | 1                       | 0                       | 0                        |

        การใช้ **One-Hot Encoding** เหมาะสำหรับข้อมูลที่ไม่มีลำดับที่ชัดเจน เช่น `สี`, `สภาพอากาศ` หรือข้อมูลที่เป็นประเภทต่าง ๆ ที่ไม่มีความสัมพันธ์ทางลำดับ
        """)
        
        st.write("## ทฤษฎีของอัลกอริธึม (Algorithm Theory)")
        st.write("ในการพัฒนาโมเดลนี้ใช้ Neural Network แบบ Multi-Output ซึ่งสามารถทำนายทั้งข้อมูลเชิงพาณิชย์ (Classification) และข้อมูลเชิงคณิตศาสตร์ (Regression) พร้อมกัน")
        st.write("🔹 Classification: ทำนาย สาเหตุของอุบัติเหตุ โดยใช้การสุ่มตัวเลือกหลายประเภทที่เป็นหมวดหมู่ (Softmax Activation).")
        st.write("🔹 Regression: ทำนาย จำนวนผู้บาดเจ็บ โดยใช้ค่าต่อเนื่อง (ReLU Activation).")
        st.write("โมเดล Neural Network นี้จะมีชั้นซ่อนหลายชั้น (Hidden Layers) ซึ่งทำให้สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนระหว่างข้อมูลอินพุต (Features) และผลลัพธ์ (Outputs)")

        st.write("## การพัฒนาโมเดล (Model Development)")
        st.write("""
        ในการพัฒนาโมเดลนี้เราใช้ **Neural Network แบบ Multi-Output** เพื่อทำนายทั้งสาเหตุของอุบัติเหตุ (Classification) และจำนวนผู้บาดเจ็บ (Regression) พร้อมกันในโมเดลเดียวกัน
        โมเดลนี้ประกอบด้วยขั้นตอนหลัก ๆ ดังนี้:
        
        1. **การเตรียมข้อมูล**:
            - ใช้ **One-Hot Encoding** เพื่อแปลงข้อมูลหมวดหมู่ เช่น `Weather Condition` และ `Road Condition` ให้เป็นตัวเลข
            - ใช้ **MinMaxScaler** เพื่อทำการปรับสเกลข้อมูลตัวเลข เช่น `Vehicles Involved`
        
        2. **การออกแบบโมเดล**:
            - โมเดลประกอบด้วยชั้นซ่อนหลายชั้น (Hidden Layers) โดยเริ่มจากชั้น **Dense** ที่มีจำนวน 256 หน่วย ใช้ฟังก์ชัน **ReLU** เพื่อให้โมเดลสามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนระหว่างข้อมูล
            - ใช้ **BatchNormalization** เพื่อปรับให้ค่าต่าง ๆ ของข้อมูลอยู่ในช่วงที่เหมาะสม
            - ใช้ **Dropout** เพื่อป้องกันการเกิด Overfitting
            - โมเดลมี **2 Output Layers**: 
                - **Cause Output**: สำหรับทำนายสาเหตุของอุบัติเหตุ (ใช้ **Softmax Activation**)
                - **Casualties Output**: สำหรับทำนายจำนวนผู้บาดเจ็บ (ใช้ **ReLU Activation**)
        
        3. **การฝึกสอนโมเดล**:
            - ใช้ **Adam Optimizer** สำหรับการคำนวณค่า Gradient Descent
            - เลือก **sparse_categorical_crossentropy** เป็น Loss Function สำหรับการทำนายสาเหตุ (Classification)
            - เลือก **Mean Squared Error (MSE)** เป็น Loss Function สำหรับการทำนายจำนวนผู้บาดเจ็บ (Regression)
            - ใช้ **Early Stopping** เพื่อหยุดการฝึกหากโมเดลไม่พัฒนาในช่วงหลาย ๆ epoch
        
        4. **การประเมินผล**:
            - ใช้ **Accuracy** ในการประเมินผลของการทำนายสาเหตุของอุบัติเหตุ
            - ใช้ **Mean Absolute Error (MAE)** ในการประเมินผลของการทำนายจำนวนผู้บาดเจ็บ
        """)
        st.write("## Code Implementation")
        st.code("""
                # ฝึกโมเดล
                history = model.fit(X_train, {"cause_output": y_cause_train, "casualties_output": y_casualties_train},
                    validation_data=(X_test, {"cause_output": y_cause_test, "casualties_output": y_casualties_test}),
                    epochs=200, batch_size=64, callbacks=[early_stopping])
                    """)
        st.code(""" 
                # ประเมินผลโมเดล
                loss, cause_loss, casualties_loss, cause_acc, casualties_mae = model.evaluate(X_test, {"cause_output": y_cause_test, "casualties_output": y_casualties_test})
                """)
        st.write("## การทดสอบกับข้อมูลใหม่ (Model Prediction)")
        st.code(""" 
                # ทดสอบการพยากรณ์กับข้อมูลใหม่
                sample_data = np.array(X_test.iloc[:5])
                predicted_causes, predicted_casualties = model.predict(sample_data)
                """)
    with tab3:
        st.write("### 📚 References")
        st.link_button("🔗 One-Hot Encoding","https://www.geeksforgeeks.org/ml-one-hot-encoding/")
        st.link_button("🔗 Neural Network", "https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/")
        st.link_button("🔗 Neural Network multi-output", "https://medium.com/@kevinnjagi83/building-deep-learning-models-with-multi-output-architectures-61d1c3c81d40")
        st.link_button("🔗 Softmax Activation", "https://medium.com/super-ai-engineer/softmax-function-%E0%B8%84%E0%B8%B7%E0%B8%AD%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3-eae1f1bbef63")
        st.link_button("🔗 ReLU Activation", "https://www.geeksforgeeks.org/relu-activation-function-in-deep-learning/")
        st.link_button("🔗 Adam Optimizer", "https://www.geeksforgeeks.org/adam-optimizer/")
        st.link_button("🔗 Sheet in classroom", "https://classroom.google.com/u/1/w/NzMyNDg3MzcyNzEy/t/all")
        st.link_button("🔗 Data set", "https://www.kaggle.com/datasets/adilshamim8/global-traffic-accidents-dataset")
# ----------------------------- Page 4: Neural Network -----------------------------
elif page == "🤖 Demo Neural Network":
    st.title("🤖 Accident Prediction")
    
    # แสดงข้อมูลดิบก่อนการ encode
    if st.checkbox("🔍 Show Raw Data"):
        st.subheader("📊 Raw Data")
        st.write(raw_data.head())  # แสดงข้อมูลดิบที่ยังไม่ได้ทำการ encode
    st.write(f"🔹 **MAE: {history.history['casualties_output_mae'][-1]}**")
    st.write(f"🔹 **Cause Prediction Accuracy: {history.history['cause_output_accuracy'][-1]}**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Training History", "📈 Cause Prediction Accuracy", "🔥 Cause Accident", "🧠 Sample Predictions"])
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
        st.subheader("📊 Cause Comparison: 2023 vs 2024")
        causes = ["Reckless Driving", "Drunk Driving", "Weather Conditions", "Speeding", "Mechanical Failure", "Distracted Driving"]
        data_2023 = [100, 150, 80, 200, 50, 120]  # Example data for 2023
        data_2024 = [120, 160, 90, 220, 55, 130]  # Example data for 2024
        df_comparison = pd.DataFrame({
            "Cause": causes,
            "2023": data_2023,
            "2024": data_2024
        })
        fig = px.bar(df_comparison, x="Cause", y=["2023", "2024"], barmode="group", title="Cause Comparison: 2023 vs 2024")
        fig.update_layout(xaxis_title="Cause", yaxis_title="Total Count", template="plotly_dark")
        st.plotly_chart(fig)
    
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
