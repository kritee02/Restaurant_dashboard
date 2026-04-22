import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# PREPROCESSING FUNCTION
# =========================
def prepare_data(df):

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['lag_14'] = df['sales'].shift(14)

    df['rolling_7'] = df['sales'].rolling(7).mean()

    df = df.dropna()

    features = ['day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'lag_14', 'rolling_7']

    X = df[features]
    y = df['sales']

    return df, X, y, features


# =========================
# TRAIN MODELS
# =========================
def train_models(X_train, y_train):

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gbr.fit(X_train, y_train)

    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        objective='reg:squarederror'
    )
    xgb.fit(X_train, y_train)

    return {
        "Linear Regression": lr,
        "Random Forest": rf,
        "Gradient Boosting": gbr,
        "XGBoost": xgb
    }


# =========================
# EVALUATION
# =========================
def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {
            "model": model,
            "pred": preds,
            "mae": mae,
            "rmse": rmse
        }

    return results


# =========================
# GET BEST MODEL
# =========================
def get_best_model(results):

    best_model_name = min(results, key=lambda k: results[k]["rmse"])
    return best_model_name, results[best_model_name]