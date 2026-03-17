import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Title
st.title("Restaurant Sales Forecasting Dashboard")

# Load data
df = pd.read_csv("data/restaurant_sales.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

st.subheader("Dataset Preview")
st.write(df.head())

# Feature Engineering
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

# Train Test Split
split = int(len(df)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Train model
model = RandomForestRegressor()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Show metrics
mae = mean_absolute_error(y_test,y_pred)

st.subheader("Model Performance")
st.write("MAE:", mae)

# Actual vs Predicted plot
fig, ax = plt.subplots()

ax.plot(y_test.values,label="Actual")
ax.plot(y_pred,label="Predicted")

ax.set_title("Actual vs Predicted Sales")

ax.legend()

st.pyplot(fig)

# Feature Importance
importance = pd.Series(model.feature_importances_,index=features)

st.subheader("Feature Importance")

fig2, ax2 = plt.subplots()

importance.sort_values().plot(kind='barh',ax=ax2)

st.pyplot(fig2)