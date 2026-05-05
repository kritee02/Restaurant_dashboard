import streamlit as st
from styles import apply_styles  
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from models import (
    prepare_data,
    chronological_split,
    baseline_models,
    train_models,
    evaluate_models,
    get_best_model,
    forecast_future
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

apply_styles()

st.title("🍽️ Restaurant Sales Forecasting Dashboard")

st.markdown(
    "This dashboard presents a leakage-safe machine learning pipeline for restaurant sales forecasting."
)

st.caption(
    "The system compares multiple models, evaluates performance, explains feature importance, "
    "and forecasts future sales for one year."
)

st.success(
    "📌This dashboard is developed for academic and study purposes only. "
    "The results are based on a synthetic dataset and may not reflect real-world performance."
)

# =========================
# MODEL COLORS
# =========================
COLOR_MAP = {
    "Actual Sales": "blue",

    # Machine Learning
    "Linear Regression": "orange",
    "Random Forest": "green",
    "Gradient Boosting": "red",
    "XGBoost": "purple",

    # Baselines (NEW)
    "Naïve": "#555555",              # dark grey
    "Seasonal Naïve": "#ba78d6",     
    "Moving Average": "#97691F"      
}

# =========================
# LOAD DATA
# =========================
try:
    df_raw = pd.read_csv("data/restaurant_sales.csv")
except FileNotFoundError:
    st.error("Dataset not found. Make sure restaurant_sales.csv exists inside the data folder.")
    st.stop()

if df_raw.empty:
    st.error("Dataset is empty.")
    st.stop()

required_columns = ["date", "sales"]
missing_columns = [col for col in required_columns if col not in df_raw.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Navigation")

option = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "EDA (Exploration)",
        "Modeling",
        "Model Comparison",
        "Feature Importance",
        "Future Forecast"
    ]
)

# =========================
# PREPARE DATA
# =========================
df, X, y, features = prepare_data(df_raw)

# =========================
# SPLIT DATA (LEAKAGE-SAFE)
# =========================
X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2)

# =========================
# TRAIN MODELS
# =========================
models = train_models(X_train, y_train)

# =========================
# BASELINE MODELS
# =========================
baseline_preds = baseline_models(df, test_size=0.2)

# =========================
# EVALUATE ALL MODELS
# =========================
results = evaluate_models(
    models=models,
    X_test=X_test,
    y_test=y_test,
    baseline_preds=baseline_preds
)

# =========================
# GET BEST MODEL (ONLY ML MODELS)
# =========================
best_model_name, best_model = get_best_model(
    results,
    metric="rmse",
    only_ml=True
)
# =========================
# 1. DATASET OVERVIEW
# =========================
if option == "Dataset Overview":

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    start = df['date'].min()
    end = df['date'].max()

    col3.markdown(f"""
    <div style="
    background-color: white;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.08);
    ">
    <div style="font-size:14px; color:#6c757d;">Date Range</div>
    <div style="font-size:20px; font-weight:600;">
        {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}
       </div>
      </div>
      """, unsafe_allow_html=True)
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# =========================
# 2. EDA
# =========================
elif option == "EDA (Exploration)":

    st.subheader("Sales Trend Over Time")

    selected_days = st.slider(
        "Select number of recent days to display:",
        min_value=30,
        max_value=len(df),
        value=min(180, len(df)),
        step=10
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df["date"].tail(selected_days),
        df["sales"].tail(selected_days),
        color=COLOR_MAP["Actual Sales"]
    )
    ax.set_title("Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Average Sales by Day of Week")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    temp = df.copy()
    temp["day_name"] = temp["date"].dt.day_name()

    avg_day = temp.groupby("day_name")["sales"].mean().reindex(day_order)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    avg_day.plot(kind="bar", ax=ax2)
    ax2.set_title("Average Sales by Day of Week")
    ax2.set_xlabel("Day of Week")
    ax2.set_ylabel("Average Sales")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("Average Sales by Month")

    avg_month = df.groupby("month")["sales"].mean()

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    avg_month.plot(kind="bar", ax=ax3)
    ax3.set_title("Average Sales by Month")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Average Sales")
    st.pyplot(fig3)

# =========================
# 3. MODELING
# =========================
elif option == "Modeling":

    st.subheader("Model Evaluation")

    selected_model = st.selectbox(
        "Select a model to view:",
        list(results.keys())
    )

    selected_pred = results[selected_model]["pred"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{results[selected_model]['mae']:.2f}")
    col2.metric("RMSE", f"{results[selected_model]['rmse']:.2f}")
    col3.metric("MAPE", f"{results[selected_model]['mape']:.2f}%")
    col4.metric("R²", f"{results[selected_model]['r2']:.3f}")

    show_points = st.slider(
        "Select number of test points to display:",
        min_value=20,
        max_value=len(y_test),
        value=min(80, len(y_test)),
        step=10
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        y_test.values[:show_points],
        label="Actual Sales",
        color=COLOR_MAP["Actual Sales"]
    )
    ax.plot(
        selected_pred[:show_points],
        label=selected_model,
        color=COLOR_MAP.get(selected_model, "black")
    )
    ax.legend()
    ax.set_title(f"Actual vs Predicted Sales - {selected_model}")
    ax.set_xlabel("Test Observations")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    st.info(f"Best model based on lowest RMSE: {best_model_name}")
# =========================
# 4. MODEL COMPARISON
# =========================
elif option == "Model Comparison":

    st.subheader("📊 Model Performance Comparison")

    comparison = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE": [results[m]["mae"] for m in results],
        "RMSE": [results[m]["rmse"] for m in results],
        "MAPE (%)": [results[m]["mape"] for m in results],
        "R²": [results[m]["r2"] for m in results]
    })

    comparison = comparison.sort_values("RMSE").reset_index(drop=True)

    st.dataframe(comparison, use_container_width=True)

    st.info(f"📌 Best machine learning model based on RMSE: {best_model_name}")

    # =========================
    # CSV DOWNLOAD
    # =========================
    csv = comparison.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download comparison table as CSV",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv",
        type="secondary"
    )

    # =========================
    # SLIDER
    # =========================
    compare_points = st.slider(
        "Select number of test observations to display:",
        min_value=20,
        max_value=len(y_test),
        value=min(80, len(y_test)),
        step=10
    )

    # =========================
    # PLOT
    # =========================
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        y_test.values[:compare_points],
        label="Actual Sales",
        color=COLOR_MAP["Actual Sales"],
        linewidth=2
    )

    for model_name, model_info in results.items():

        linestyle = "--" if model_info["model"] is None else "-"

        ax.plot(
            model_info["pred"][:compare_points],
            label=model_name,
            color=COLOR_MAP.get(model_name, "black"),
            linestyle=linestyle
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4
    )

    ax.set_title("Actual vs Predicted Sales Comparison")
    ax.set_xlabel("Test Observations")
    ax.set_ylabel("Sales")

    fig.tight_layout()

    st.pyplot(fig)
# =========================
# 5. FEATURE IMPORTANCE
# =========================
elif option == "Feature Importance":

    st.subheader("Feature Importance")

    feature_models = [
        model_name for model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]
        if model_name in models
    ]

    feature_model = st.selectbox(
        "Select model for feature importance:",
        feature_models
    )

    importance = pd.Series(
        models[feature_model].feature_importances_,
        index=features
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    importance.plot(
        kind="barh",
        ax=ax,
        color=COLOR_MAP.get(feature_model, "black")
    )
    ax.set_title(f"Feature Importance - {feature_model}")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

# =========================
# 6. FUTURE FORECAST
# =========================
elif option == "Future Forecast":

    st.subheader("Future Sales Forecast")

    ml_model_names = [
    name for name, info in results.items()
    if info["model"] is not None
]

    forecast_model_choice = st.radio(
    "Choose forecasting model:",
    ["Best Model Automatically"] + ml_model_names
     )

    if forecast_model_choice == "Best Model Automatically":
        chosen_model_name = best_model_name
        chosen_model = best_model
    else:
        chosen_model_name = forecast_model_choice
        chosen_model = results[chosen_model_name]["model"]

    st.info(f"Selected forecasting model: {chosen_model_name}")

    forecast_days = st.slider(
        "Forecast horizon:",
        min_value=30,
        max_value=365,
        value=365,
        step=30
    )

    forecast_df = forecast_future(
        model=chosen_model,
        df=df,
        features=features,
        days=forecast_days
    )

    st.write(f"Forecast starts from: {forecast_df['date'].min().date()}")
    st.write(f"Forecast ends on: {forecast_df['date'].max().date()}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Forecasted Sales", f"{forecast_df['predicted_sales'].mean():.2f}")
    col2.metric("Highest Forecasted Sales", f"{forecast_df['predicted_sales'].max():.2f}")
    col3.metric("Lowest Forecasted Sales", f"{forecast_df['predicted_sales'].min():.2f}")

    st.subheader("Forecast Table")
    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download future forecast as CSV",
        data=csv,
        file_name="future_sales_forecast.csv",
        mime="text/csv"
    )

    st.subheader("Future Forecast Line Chart")

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        forecast_df["date"],
        forecast_df["predicted_sales"],
        label=f"{chosen_model_name} Forecast",
        color=COLOR_MAP.get(chosen_model_name, "black")
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    ax.set_title(f"Future Sales Forecast for {forecast_days} Days")
    ax.set_xlabel("Future Date")
    ax.set_ylabel("Predicted Sales")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Business Insight")

    avg_historical_sales = df["sales"].mean()
    avg_forecast_sales = forecast_df["predicted_sales"].mean()

    if avg_forecast_sales > avg_historical_sales:
        st.info(
            "The forecast suggests that future sales may be higher than the historical average. "
            "The restaurant may need to prepare additional staffing and inventory."
        )
    else:
        st.info(
            "The forecast suggests that future sales may be close to or below the historical average. "
            "The restaurant should monitor demand carefully and avoid unnecessary overstocking."
        )

    st.caption(
        "Future forecasts are generated recursively using previous predictions as lag values. "
        "This avoids using unknown future sales and supports leakage-safe forecasting."
    )