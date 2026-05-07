import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# MODEL FACTORY
# =========================
def get_model_objects():
    return {
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

    # Leakage-safe rolling means: shifted so current-day sales are not included
    df["rolling_7"] = df["sales"].shift(1).rolling(7).mean()
    df["rolling_14"] = df["sales"].shift(1).rolling(14).mean()

    df = df.dropna().reset_index(drop=True)

    features = [
        "day_of_week",
        "month",
        "is_weekend",
        "time_index",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_7",
        "rolling_14"
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
    split_index = int(len(df) * (1 - test_size))
    y = df["sales"].values

    naive_pred = []
    seasonal_pred = []
    moving_average_pred = []

    for i in range(split_index, len(y)):
        naive_pred.append(y[i - 1])
        seasonal_pred.append(y[i - 7] if i - 7 >= 0 else y[i - 1])

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
    models = get_model_objects()

    for model in models.values():
        model.fit(X_train, y_train)

    return models


# =========================
# EVALUATION
# =========================
def evaluate_models(models, X_test, y_test, baseline_preds=None):
    results = {}

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
        raise ValueError("Future forecasting requires a trained machine learning model.")

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
        rolling_14 = np.mean(sales_history[-14:])

        X_future = pd.DataFrame([{
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "time_index": time_index,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "rolling_7": rolling_7,
            "rolling_14": rolling_14
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

    return pd.DataFrame(future_rows)


# =========================
# WALK-FORWARD VALIDATION
# =========================
def walk_forward_validation(df, features, target="sales", initial_train_size=0.7):
    models = get_model_objects()
    start = int(len(df) * initial_train_size)

    output = []

    for model_name, base_model in models.items():
        maes = []
        rmses = []

        for i in range(start, len(df) - 1):
            train = df.iloc[:i]
            test = df.iloc[i:i + 1]

            X_train = train[features]
            y_train = train[target]

            X_test = test[features]
            y_test = test[target]

            model = get_model_objects()[model_name]
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            maes.append(mean_absolute_error(y_test, pred))
            rmses.append(np.sqrt(mean_squared_error(y_test, pred)))

        output.append({
            "Model": model_name,
            "MAE Mean": round(np.mean(maes), 4),
            "MAE Std": round(np.std(maes), 4),
            "RMSE Mean": round(np.mean(rmses), 4),
            "RMSE Std": round(np.std(rmses), 4)
        })

    return pd.DataFrame(output)


# =========================
# ABLATION STUDY
# =========================
def ablation_study(df, target="sales"):
    feature_sets = {
        "Temporal features only": [
            "day_of_week", "month", "is_weekend", "time_index"
        ],
        "Temporal + lag features": [
            "day_of_week", "month", "is_weekend", "time_index",
            "lag_1", "lag_7", "lag_14"
        ],
        "Temporal + lag + rolling features": [
            "day_of_week", "month", "is_weekend", "time_index",
            "lag_1", "lag_7", "lag_14", "rolling_7", "rolling_14"
        ]
    }

    split_index = int(len(df) * 0.8)
    results = []

    for config_name, selected_features in feature_sets.items():
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]

        X_train = train[selected_features]
        y_train = train[target]

        X_test = test[selected_features]
        y_test = test[target]

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "Feature Configuration": config_name,
            "RMSE": round(rmse, 4)
        })

    return pd.DataFrame(results)


# =========================
# MULTI-STEP FORECASTING EVALUATION
# =========================
def multi_step_forecast_evaluation(model, df, features, horizons=[7, 30, 90, 365]):
    """
    Evaluates recursive forecasting using available ground truth.
    The model is trained on the first 80% and recursively predicts inside the test period.
    """

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    max_horizon = min(max(horizons), len(test_df))

    history = train_df.copy()
    predictions = []

    for i in range(max_horizon):
        current_date = test_df.iloc[i]["date"]

        sales_history = list(history["sales"].values)

        input_row = pd.DataFrame([{
            "day_of_week": current_date.dayofweek,
            "month": current_date.month,
            "is_weekend": 1 if current_date.dayofweek in [5, 6] else 0,
            "time_index": len(history),
            "lag_1": sales_history[-1],
            "lag_7": sales_history[-7],
            "lag_14": sales_history[-14],
            "rolling_7": np.mean(sales_history[-7:]),
            "rolling_14": np.mean(sales_history[-14:])
        }])

        input_row = input_row[features]

        pred = model.predict(input_row)[0]
        predictions.append(pred)

        new_row = test_df.iloc[i].copy()
        new_row["sales"] = pred
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    results = []

    for horizon in horizons:
        if horizon <= len(predictions):
            actual = test_df["sales"].values[:horizon]
            predicted = np.array(predictions[:horizon])

            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)

            results.append({
                "Forecast Horizon": f"{horizon} days",
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4)
            })

    return pd.DataFrame(results)