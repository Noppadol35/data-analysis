import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

def train_svm_model(df_cleaned):
    # Prepare data
    X = df_cleaned[["Open", "High", "Low"]]
    y = df_cleaned["Close/Last"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM model
    svm_model = SVR(kernel='rbf', C=100, epsilon=0.1)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    mae_svm = mean_absolute_error(y_test, y_pred_svm)

    return y_test, y_pred_svm, mae_svm