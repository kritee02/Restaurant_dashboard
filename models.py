import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# PREPROCESSING FUNCTION
# =========================
def prepare_data(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if "time_index" not in df.columns:
        df["time_index"] = np.arange(len(df))

    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)

    # Leakage-safe rolling mean:
    # uses only previous sales values, not the current day sales
    df["rolling_7"] = df["sales"].shift(1).rolling(7).mean()

    df = df.dropna().reset_index(drop=True)

    features = [
        "day_of_week",
        "month",
        "is_weekend",
        "time_index",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_7"
    ]

    X = df[features]
    y = df["sales"]

    return df, X, y, features


# =========================
# CHRONOLOGICAL SPLIT
# =========================
def chronological_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# =========================
# BASELINE MODELS
# =========================
def baseline_models(df, test_size=0.2):
    """
    Generates leakage-safe baseline predictions for:
    - Naïve Forecasting
    - Seasonal Naïve Forecasting (weekly / 7-day lag)
    - Moving Average Forecasting (7-day average)
    """

    split_index = int(len(df) * (1 - test_size))
    y = df["sales"].values

    naive_pred = []
    seasonal_pred = []
    moving_average_pred = []

    for i in range(split_index, len(y)):

        # Naïve: next value equals previous observed value
        naive_pred.append(y[i - 1])

        # Seasonal naïve: next value equals same day from previous week
        if i - 7 >= 0:
            seasonal_pred.append(y[i - 7])
        else:
            seasonal_pred.append(y[i - 1])

        # Moving average: average of previous 7 days only
        start = max(0, i - 7)
        moving_average_pred.append(np.mean(y[start:i]))

    return {
        "Naïve": np.array(naive_pred),
        "Seasonal Naïve": np.array(seasonal_pred),
        "Moving Average": np.array(moving_average_pred)
    }


# =========================
# TRAIN MODELS
# =========================
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


# =========================
# EVALUATION
# =========================
def evaluate_models(models, X_test, y_test, baseline_preds=None):
    results = {}

    # Machine learning models
    for name, model in models.items():
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
        r2 = r2_score(y_test, pred)

        results[name] = {
            "model": model,
            "pred": pred,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "type": "Machine Learning"
        }

    # Baseline models
    if baseline_preds is not None:
        for name, pred in baseline_preds.items():
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
            r2 = r2_score(y_test, pred)

            results[name] = {
                "model": None,
                "pred": pred,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2": r2,
                "type": "Baseline"
            }

    return results


# =========================
# GET BEST MODEL
# =========================
def get_best_model(results, metric="rmse", only_ml=True):
    """
    Selects the best model based on a chosen metric.

    For MAE, RMSE, and MAPE: lower is better.
    For R²: higher is better.

    only_ml=True means baseline models are excluded from future forecasting,
    because baseline entries do not contain trained model objects.
    """

    available_results = results

    if only_ml:
        available_results = {
            name: info
            for name, info in results.items()
            if info["model"] is not None
        }

    if metric in ["mae", "rmse", "mape"]:
        best_model_name = min(available_results, key=lambda k: available_results[k][metric])
    elif metric == "r2":
        best_model_name = max(available_results, key=lambda k: available_results[k][metric])
    else:
        raise ValueError("Metric must be one of: mae, rmse, mape, r2")

    best_model = available_results[best_model_name]["model"]

    return best_model_name, best_model


# =========================
# FUTURE FORECASTING
# =========================
def forecast_future(model, df, features, days=365):
    if model is None:
        raise ValueError("Future forecasting requires a trained machine learning model, not a baseline model.")

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "time_index" not in df.columns:
        df["time_index"] = np.arange(len(df))

    sales_history = list(df["sales"].values)

    last_date = df["date"].max()
    last_time_index = df["time_index"].max()

    future_rows = []

    for i in range(1, days + 1):
        future_date = last_date + pd.Timedelta(days=i)

        day_of_week = future_date.dayofweek
        month = future_date.month
        is_weekend = 1 if day_of_week in [5, 6] else 0
        time_index = last_time_index + i

        lag_1 = sales_history[-1]
        lag_7 = sales_history[-7]
        lag_14 = sales_history[-14]
        rolling_7 = np.mean(sales_history[-7:])

        X_future = pd.DataFrame([{
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "time_index": time_index,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "rolling_7": rolling_7
        }])

        X_future = X_future[features]

        predicted_sales = model.predict(X_future)[0]
        predicted_sales = max(0, round(predicted_sales, 2))

        sales_history.append(predicted_sales)

        future_rows.append({
            "date": future_date,
            "predicted_sales": predicted_sales,
            "day_of_week": future_date.day_name(),
            "month": month,
            "is_weekend": is_weekend
        })

    forecast_df = pd.DataFrame(future_rows)

    return forecast_df