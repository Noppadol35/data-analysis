import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# โหลดข้อมูล
df = pd.read_csv("global_traffic_accidents.csv")

# ลบแถวที่มีค่า Missing Values
df.dropna(inplace=True)

# เลือกคอลัมน์ที่ต้องใช้
categorical_cols = ["Weather Condition", "Road Condition"]
numerical_cols = ["Vehicles Involved"]

# ใช้ One-Hot Encoding สำหรับข้อมูลหมวดหมู่
encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_categorical = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))

# จัดลำดับสาเหตุอุบัติเหตุโดยเรียงตามจำนวนผู้บาดเจ็บ
cause_order = df.groupby("Cause")["Casualties"].mean().sort_values().index
cause_encoder = LabelEncoder()
cause_encoder.fit(cause_order)
df["Cause_Encoded"] = cause_encoder.transform(df["Cause"])

# รวมข้อมูลตัวเลขและข้อมูลที่ถูกเข้ารหัส
processed_df = pd.concat([df[numerical_cols], encoded_df, df[["Cause_Encoded"]]], axis=1)

# เพิ่ม target (Cause และ Casualties)
processed_df["Cause"] = df["Cause_Encoded"]
processed_df["Casualties"] = df["Casualties"]

# ใช้ MinMaxScaler เพื่อปรับสเกลข้อมูล
feature_scaler = MinMaxScaler()
processed_df[numerical_cols] = feature_scaler.fit_transform(processed_df[numerical_cols])

# แยก features และ targets
X = processed_df.drop(columns=["Cause", "Casualties"])
y_cause = processed_df["Cause"]
y_casualties = processed_df["Casualties"]

# แบ่ง Train/Test (80/20)
X_train, X_test, y_cause_train, y_cause_test, y_casualties_train, y_casualties_test = train_test_split(
    X, y_cause, y_casualties, test_size=0.2, random_state=42)

# สร้าง Neural Network Multi-Output Model
input_layer = layers.Input(shape=(X_train.shape[1],))
hidden = layers.Dense(256, activation="relu")(input_layer)
hidden = layers.BatchNormalization()(hidden)
hidden = layers.Dropout(0.3)(hidden)
hidden = layers.Dense(128, activation="relu")(hidden)
hidden = layers.BatchNormalization()(hidden)
hidden = layers.Dropout(0.3)(hidden)
hidden = layers.Dense(64, activation="relu")(hidden)

# Output Layer สำหรับทำนายสาเหตุอุบัติเหตุ (Classification)
cause_output = layers.Dense(len(cause_order), activation="softmax", name="cause_output")(hidden)

# Output Layer สำหรับทำนายจำนวนผู้บาดเจ็บ (Regression)
casualties_output = layers.Dense(1, activation="relu", name="casualties_output")(hidden)

# รวมโมเดล
model = keras.Model(inputs=input_layer, outputs=[cause_output, casualties_output])

# คอมไพล์โมเดล
model.compile(optimizer="adam", 
                loss={"cause_output": "sparse_categorical_crossentropy", "casualties_output": "mse"},
                metrics={"cause_output": "accuracy", "casualties_output": "mae"})

# ใช้ Early Stopping
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# ฝึกโมเดล
history = model.fit(X_train, {"cause_output": y_cause_train, "casualties_output": y_casualties_train},
                    validation_data=(X_test, {"cause_output": y_cause_test, "casualties_output": y_casualties_test}),
                    epochs=200, batch_size=64, callbacks=[early_stopping])

# ประเมินผลโมเดล
loss, cause_loss, casualties_loss, cause_acc, casualties_mae = model.evaluate(X_test, {"cause_output": y_cause_test, "casualties_output": y_casualties_test})
print(f"Cause Prediction Accuracy: {cause_acc}")
print(f"Casualties Prediction MAE: {casualties_mae}")

# วาดกราฟแสดงผล Training & Validation Loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Total Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')
plt.show()

# วาดกราฟแสดง Accuracy ของการพยากรณ์สาเหตุอุบัติเหตุ
plt.figure(figsize=(12, 5))
plt.plot(history.history['cause_output_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_cause_output_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Cause Prediction Accuracy')
plt.show()

# วาด Heatmap แสดงความสัมพันธ์ระหว่างตัวแปร
plt.figure(figsize=(10, 6))
sns.heatmap(processed_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# ทดสอบการพยากรณ์กับข้อมูลใหม่
sample_data = np.array(X_test.iloc[:5])
predicted_causes, predicted_casualties = model.predict(sample_data)
predicted_causes = np.argmax(predicted_causes, axis=1)
predicted_causes = cause_encoder.inverse_transform(predicted_causes)
predicted_casualties = np.round(predicted_casualties.flatten()).astype(int)

print("Predicted Causes:", predicted_causes)
print("Predicted Casualties:", predicted_casualties)