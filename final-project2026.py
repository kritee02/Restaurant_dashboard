#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# # FINAL YEAR PROJECT 2026

# ## Data Loading
# 
# The restaurant sales dataset is loaded using the Pandas library.  
# This dataset contains daily sales information which will be used to train the forecasting model.
# 
# The first few rows are displayed to confirm that the dataset has been imported correctly.

# In[2]:


import pandas as pd

df = pd.read_csv("/kaggle/input/datasets/kriteethapa/restaurant-sales/restaurant_sales.csv")

df.head()

# ## Checking  Structure
# Data Inspection
# The structure of the dataset is examined using df.info() and df.describe().
# 
# This step helps understand:
# - the number of observations
# - data types of each column
# - statistical properties of the sales variable
# 
# Understanding the dataset structure is important before performing preprocessing and modelling.

# In[3]:


df.info()
df.describe()

# ## Date Conversion
# 
# The date column is converted into datetime format and sorted chronologically.
# 
# This is required for time-series forecasting because the model must learn patterns from past observations to predict future sales.

# In[4]:


df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ## Duplicate Check
# 
# Duplicate records are checked using df.duplicated().
# 
# Removing duplicates ensures that each observation is unique and prevents biased model training.

# In[5]:


df.duplicated().sum()

# ## Missing Values
# 
# The dataset is checked for missing values using df.isnull().
# 
# Handling missing data is necessary because machine learning models require complete numerical input.

# In[6]:


df.isnull().sum()

# In[7]:


df.describe()

# ### Checking aggregation
# 

# In[8]:


print("Unique dates:", df['date'].nunique())
print("Total rows:", len(df))

# ## Time Feature Engineering
# 
# Time-based features are extracted from the date column.
# 
# These include:
# - day_of_week
# - month
# - weekend indicator
# 
# These variables help capture temporal patterns in restaurant demand.

# In[9]:


df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

df.head(15)

# ## Lag Features
# 
# Lag variables represent past sales values.
# 
# Examples:
# - lag_1 → sales from the previous day
# - lag_7 → sales from the previous week
# - lag_14 → sales from two weeks earlier
# 
# Lag features allow the model to learn historical demand patterns.

# In[10]:


df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)
df['lag_14'] = df['sales'].shift(14)

df.head(20)

# ## Rolling Average Feature
# 
# A 7-day rolling average of sales is created.
# 
# This feature helps capture the overall weekly trend and reduces noise in daily sales data.

# In[11]:


df['rolling_7'] = df['sales'].rolling(7).mean()

df.head(10)

# In[12]:


df[['sales','lag_1','lag_7','lag_14','rolling_7']].head(15)

# ## Handling Missing Values from Feature Engineering
# 
# Lag and rolling features create missing values at the beginning of the dataset.
# 
# These rows are removed to ensure the dataset is suitable for machine learning models.

# In[13]:


df = df.dropna()

df.head()

# ## Exploratory Data Analysis (EDA)

# ### Sales trend over time

# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df['date'], df['sales'])
plt.title("Restaurant Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# ### Sales by weekday

# In[15]:


df.groupby('day_of_week')['sales'].mean().plot(kind='bar')
plt.title("Average Sales by Day of Week")
plt.show()

# ## Feature and Target Definition
# 
# The input variables (features) are separated from the target variable.
# 
# Features include temporal and lag-based variables that describe historical demand patterns.
# 
# The target variable is daily sales, which the model aims to predict.

# In[16]:


features = ['day_of_week','month','is_weekend','lag_1','lag_7','lag_14','rolling_7']
X = df[features]

y = df['sales']

# ## Chronological Train-Test Split
# 
# The dataset is divided into training and testing sets using chronological order.
# 
# This prevents data leakage by ensuring the model is trained on past data and evaluated on future observations.

# In[17]:


split = int(len(df)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ## Linear Regression Model
# 
# A Linear Regression model is trained as a baseline forecasting model.
# 
# The model learns relationships between historical sales features
# (day_of_week, lag variables, rolling averages) and future sales values.

# In[18]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

# In[19]:


y_pred_lr = model.predict(X_test)

# ## Model Evaluation
# 
# The forecasting model is evaluated using Mean Absolute Error (MAE)
# and Root Mean Squared Error (RMSE).
# 
# These metrics measure how closely predicted sales match actual sales values.
# Lower values indicate better model performance.

# In[20]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_lr = mean_absolute_error(y_test,y_pred_lr)
rmse_lr = mean_squared_error(y_test,y_pred_lr)**0.5

print("Linear Regression MAE:",mae_lr)
print("Linear Regression RMSE:",rmse_lr)

# In[21]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(y_test.values, label="Actual Sales")
plt.plot(y_pred_lr, label="Predicted Sales")

plt.legend()
plt.title("Actual vs Predicted Sales (Linear Regression)")

plt.show()

# ## Random Forest Model
# 
# A Random Forest regression model is trained to capture non-linear
# relationships in the data.
# 
# Random Forest combines multiple decision trees to improve prediction
# accuracy and robustness compared to simple linear models.

# In[22]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)

# In[23]:


plt.figure(figsize=(10,5))

plt.plot(y_test.values,label="Actual Sales")
plt.plot(y_pred_rf,label="Random Forest Prediction")

plt.legend()

plt.title("Actual vs Predicted Sales")

plt.show()

# ## Model Comparison
# 
# Two forecasting models were compared:
# 
# 1. Linear Regression
# 2. Random Forest
# 
# Performance metrics show that Random Forest produced lower error values,
# indicating improved forecasting accuracy.

# In[24]:


mae_rf = mean_absolute_error(y_test,y_pred_rf)
rmse_rf = mean_squared_error(y_test,y_pred_rf)**0.5

print("Linear Regression MAE:",mae_lr)
print("Random Forest MAE:",mae_rf)

print("Linear Regression RMSE:",rmse_lr)
print("Random Forest RMSE:",rmse_rf)

# ## Feature Importance Analysis
# 
# Feature importance analysis was performed using the Random Forest model.
# 
# The results indicate which variables have the strongest influence
# on sales predictions. Lag-based features, particularly lag_14,
# were the most important predictors, suggesting strong weekly
# seasonality in restaurant demand.

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(rf.feature_importances_, index=features)

importance.sort_values().plot(kind='barh')

plt.title("Feature Importance for Sales Prediction")

plt.show()

# ## Forecast Visualization
# 
# Actual and predicted sales values are plotted to visually assess forecasting performance.
# 
# This allows observation of how well the model captures sales trends.

# In[26]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test.values,label="Actual Sales")
plt.plot(y_pred_rf,label="Predicted Sales")
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.show()
