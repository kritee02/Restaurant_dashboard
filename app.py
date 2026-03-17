import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("🍽️ Restaurant Sales Forecasting Dashboard")
st.markdown("This dashboard presents a complete machine learning pipeline for restaurant sales prediction.")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/restaurant_sales.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Navigation")
option = st.sidebar.radio("Go to", [
    "Dataset Overview",
    "EDA (Exploration)",
    "Modeling",
    "Model Comparison",
    "Feature Importance",
    "Forecast"
])

# =========================
# FEATURE ENGINEERING
# =========================
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)
df['lag_14'] = df['sales'].shift(14)

df['rolling_7'] = df['sales'].rolling(7).mean()

df = df.dropna()

features = ['day_of_week','month','is_weekend','lag_1','lag_7','lag_14','rolling_7']

X = df[features]
y = df['sales']

# =========================
# TRAIN TEST SPLIT
# =========================
split = int(len(df)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================
# MODELS
# =========================
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

# =========================
# METRICS
# =========================
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

mae_lr, rmse_lr = evaluate(y_test,y_pred_lr)
mae_rf, rmse_rf = evaluate(y_test,y_pred_rf)

# =========================
# 1. DATASET OVERVIEW
# =========================
if option == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(df.describe())

# =========================
# 2. EDA
# =========================
elif option == "EDA (Exploration)":

    st.subheader("Sales Trend Over Time")

    fig, ax = plt.subplots()
    ax.plot(df['date'], df['sales'])
    ax.set_title("Sales Over Time")
    st.pyplot(fig)

    st.subheader("Average Sales by Day of Week")

    fig2, ax2 = plt.subplots()
    df.groupby('day_of_week')['sales'].mean().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

# =========================
# 3. MODELING
# =========================
elif option == "Modeling":

    st.subheader("Linear Regression Results")

    st.write("MAE:", mae_lr)
    st.write("RMSE:", rmse_lr)

    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual")
    ax.plot(y_pred_lr, label="Predicted (LR)")
    ax.legend()
    ax.set_title("Linear Regression Prediction")
    st.pyplot(fig)

    st.subheader("Random Forest Results")

    st.write("MAE:", mae_rf)
    st.write("RMSE:", rmse_rf)

    fig2, ax2 = plt.subplots()
    ax2.plot(y_test.values, label="Actual")
    ax2.plot(y_pred_rf, label="Predicted (RF)")
    ax2.legend()
    ax2.set_title("Random Forest Prediction")
    st.pyplot(fig2)

# =========================
# 4. MODEL COMPARISON
# =========================
elif option == "Model Comparison":

    st.subheader("Model Performance Comparison")

    comparison = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "MAE": [mae_lr, mae_rf],
        "RMSE": [rmse_lr, rmse_rf]
    })

    st.table(comparison)

# =========================
# 5. FEATURE IMPORTANCE
# =========================
elif option == "Feature Importance":

    st.subheader("Feature Importance (Random Forest)")

    importance = pd.Series(rf.feature_importances_, index=features)

    fig, ax = plt.subplots()
    importance.sort_values().plot(kind='barh', ax=ax)
    st.pyplot(fig)

# =========================
# 6. FORECAST
# =========================
elif option == "Forecast":

    st.subheader("Actual vs Predicted (Random Forest)")

    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual")
    ax.plot(y_pred_rf, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Next Day Prediction")

    latest = df.iloc[-1]

    input_data = pd.DataFrame({
        'day_of_week': [latest['day_of_week']],
        'month': [latest['month']],
        'is_weekend': [latest['is_weekend']],
        'lag_1': [latest['sales']],
        'lag_7': [df.iloc[-7]['sales']],
        'lag_14': [df.iloc[-14]['sales']],
        'rolling_7': [df['sales'].tail(7).mean()]
    })

    prediction = rf.predict(input_data)

    st.success(f"Predicted Sales for Next Day: {int(prediction[0])}")