import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from models import prepare_data, train_models, evaluate_models, get_best_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("🍽️ Restaurant Sales Forecasting Dashboard")
st.markdown("This dashboard presents a complete machine learning pipeline for restaurant sales prediction.")
st.caption("Compare multiple forecasting models, inspect feature importance, and generate next-day sales predictions.")

# =========================
# LOAD DATA
# =========================
df_raw = pd.read_csv("data/restaurant_sales.csv")

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
# PREPARE DATA
# =========================
df, X, y, features = prepare_data(df_raw)

# =========================
# TRAIN TEST SPLIT
# =========================
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================
# TRAIN MODELS
# =========================
models = train_models(X_train, y_train)
results = evaluate_models(models, X_test, y_test)

best_model_name, best_model_info = get_best_model(results)
best_model = best_model_info["model"]
best_model_pred = best_model_info["pred"]

# =========================
# 1. DATASET OVERVIEW
# =========================
if option == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Summary Statistics")
    st.write(df.describe())

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# =========================
# 2. EDA
# =========================
elif option == "EDA (Exploration)":

    st.subheader("Sales Trend Over Time")

    selected_days = st.slider(
        "Select number of recent days to display:",
        min_value=30,
        max_value=len(df),
        value=min(120, len(df)),
        step=10
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['date'].tail(selected_days), df['sales'].tail(selected_days))
    ax.set_title("Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    st.subheader("Average Sales by Day of Week")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df.groupby('day_of_week')['sales'].mean().plot(kind='bar', ax=ax2)
    ax2.set_title("Average Sales by Day of Week")
    ax2.set_xlabel("Day of Week (0 = Monday)")
    ax2.set_ylabel("Average Sales")
    st.pyplot(fig2)

# =========================
# 3. MODELING
# =========================
elif option == "Modeling":

    st.subheader("Interactive Model Results")

    selected_model = st.selectbox(
        "Select a model to view:",
        list(results.keys())
    )

    show_points = st.slider(
        "Select number of test points to display:",
        min_value=20,
        max_value=len(y_test),
        value=min(60, len(y_test)),
        step=10
    )

    selected_pred = results[selected_model]["pred"]
    selected_mae = results[selected_model]["mae"]
    selected_rmse = results[selected_model]["rmse"]

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{selected_mae:.2f}")
    col2.metric("RMSE", f"{selected_rmse:.2f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values[:show_points], label="Actual")
    ax.plot(selected_pred[:show_points], label=f"Predicted ({selected_model})")
    ax.legend()
    ax.set_title(f"{selected_model} Prediction")
    ax.set_xlabel("Test Observations")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    st.info(f"Best model based on lowest RMSE: {best_model_name}")

# =========================
# 4. MODEL COMPARISON
# =========================
elif option == "Model Comparison":

    st.subheader("Model Performance Comparison")

    comparison = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE": [results[m]["mae"] for m in results],
        "RMSE": [results[m]["rmse"] for m in results]
    })

    st.dataframe(comparison, use_container_width=True)

    csv = comparison.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download comparison table as CSV",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

    st.success(f"Best model: {best_model_name}")

    compare_points = st.slider(
        "Select number of points for model comparison chart:",
        min_value=20,
        max_value=len(y_test),
        value=min(60, len(y_test)),
        step=10,
        key="compare_slider"
    )

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(y_test.values[:compare_points], label="Actual Sales")
    ax5.plot(results["Random Forest"]["pred"][:compare_points], label="Random Forest")
    ax5.plot(results["Gradient Boosting"]["pred"][:compare_points], label="Gradient Boosting")
    ax5.plot(results["XGBoost"]["pred"][:compare_points], label="XGBoost")
    ax5.legend()
    ax5.set_title("Actual vs Predicted Sales Comparison")
    ax5.set_xlabel("Test Observations")
    ax5.set_ylabel("Sales")
    st.pyplot(fig5)

# =========================
# 5. FEATURE IMPORTANCE
# =========================
elif option == "Feature Importance":

    st.subheader("Interactive Feature Importance")

    feature_model = st.selectbox(
        "Select model for feature importance:",
        ["Random Forest", "XGBoost"]
    )

    if feature_model == "Random Forest":
        importance = pd.Series(models["Random Forest"].feature_importances_, index=features)
    else:
        importance = pd.Series(models["XGBoost"].feature_importances_, index=features)

    fig, ax = plt.subplots(figsize=(10, 5))
    importance.sort_values().plot(kind='barh', ax=ax)
    ax.set_title(f"Feature Importance - {feature_model}")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

# =========================
# 6. FORECAST
# =========================
elif option == "Forecast":

    st.subheader("Forecast Viewer")

    forecast_model_choice = st.radio(
        "Choose forecasting model:",
        ["Best Model Automatically"] + list(results.keys())
    )

    if forecast_model_choice == "Best Model Automatically":
        chosen_model_name = best_model_name
    else:
        chosen_model_name = forecast_model_choice

    chosen_model = results[chosen_model_name]["model"]
    chosen_pred = results[chosen_model_name]["pred"]

    forecast_points = st.slider(
        "Select number of test points to display:",
        min_value=20,
        max_value=len(y_test),
        value=min(60, len(y_test)),
        step=10,
        key="forecast_slider"
    )

    st.subheader(f"Actual vs Predicted ({chosen_model_name})")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values[:forecast_points], label="Actual")
    ax.plot(chosen_pred[:forecast_points], label=f"Predicted ({chosen_model_name})")
    ax.legend()
    ax.set_title(f"Forecast using {chosen_model_name}")
    ax.set_xlabel("Test Observations")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    st.write("Selected Model MAE:", round(results[chosen_model_name]["mae"], 2))
    st.write("Selected Model RMSE:", round(results[chosen_model_name]["rmse"], 2))

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

    prediction = chosen_model.predict(input_data)

    st.success(f"Predicted Sales for Next Day using {chosen_model_name}: {int(prediction[0])}")