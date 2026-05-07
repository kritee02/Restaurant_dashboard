# Restaurant Sales Forecasting Dashboard

## Leakage-Safe Forecasting and Interpretability Analysis of Restaurant Sales Using Machine Learning

This project is a Streamlit-based dashboard for leakage-safe restaurant sales forecasting using machine learning techniques. The system uses a synthetic two-year daily restaurant sales dataset and compares baseline forecasting approaches with multiple machine learning models.

The project focuses on:
- Leakage-safe time-series forecasting
- Interpretability analysis
- Baseline comparison
- Recursive multi-step forecasting
- Interactive dashboard visualisation

---

# Project Overview

The system forecasts future restaurant sales using historical daily revenue data and presents results through an interactive Streamlit dashboard.

The dashboard includes:

- Dataset overview and summary statistics
- Exploratory data analysis
- Leakage-safe feature engineering
- Chronological train-test split
- Baseline forecasting methods
- Machine learning forecasting models
- Model comparison using MAE, RMSE, MAPE, and R²
- Actual vs predicted sales visualisation
- Feature importance analysis
- Recursive future sales forecasting
- Forecast download functionality

This project was developed for academic purposes as part of a final-year undergraduate dissertation project.

---

# Models Implemented

## Baseline Forecasting Models
- Naïve Forecasting
- Seasonal Naïve Forecasting
- Moving Average Forecasting

## Machine Learning Models
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

# Project Structure

```text
Restaurant_dashboard/
│
├── app.py
├── models.py
├── generate_data.py
├── styles.py
├── README.md
│
└── data/
    └── restaurant_sales.csv
```

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

---

# Dataset

The dataset is synthetically generated to simulate realistic restaurant sales behaviour over two years (2024–2025).

The dataset includes:
- Daily sales revenue
- Day-of-week patterns
- Weekend effects
- Monthly seasonality
- Long-term trend
- Random daily variability

The forecasting target variable is:

```text
Sales = Total Daily Revenue
```

---

# Leakage-Safe Forecasting

The system is designed to prevent data leakage in time-series forecasting.

Leakage prevention methods include:

- Chronological data splitting
- No random shuffling
- Lag features created using historical values only
- Rolling averages calculated using shifted historical observations
- Test data excluded during training
- Recursive future forecasting using previous predictions only

---

# Feature Engineering

The system generates the following features:

## Temporal Features
- day_of_week
- month
- is_weekend
- time_index

## Lag Features
- lag_1
- lag_7
- lag_14

## Rolling Features
- rolling_7
- rolling_14

These features help the models capture:
- Weekly seasonality
- Short-term dependencies
- Local sales trends

---

# Evaluation Metrics

Model performance is evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

The system also includes:
- Walk-forward validation
- Ablation analysis
- Multi-step forecasting evaluation

---

# Dashboard Features

The Streamlit dashboard allows users to:

- View dataset statistics
- Compare forecasting models
- Visualise prediction performance
- Analyse feature importance
- Generate future forecasts
- Download forecasting outputs

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/restaurant-sales-forecasting-dashboard.git
cd restaurant-sales-forecasting-dashboard
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

## 3. Install Required Packages

```bash
pip install streamlit pandas numpy matplotlib scikit-learn xgboost
```

---

# Generate the Dataset

Run:

```bash
python generate_data.py
```

This creates:

```text
data/restaurant_sales.csv
```

---

# Run the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open at:

```text
http://localhost:8501
```

---

# Main Files

## app.py
Contains the Streamlit dashboard interface, charts, forecasting outputs, visualisations, and user interaction components.

## models.py
Contains:
- Data preprocessing
- Leakage-safe feature engineering
- Chronological splitting
- Baseline forecasting
- Machine learning model training
- Evaluation metrics
- Recursive forecasting
- Feature importance analysis

## generate_data.py
Generates the synthetic two-year restaurant sales dataset using:
- Trend
- Seasonality
- Weekend effects
- Random noise

## styles.py
Contains custom CSS styling used in the dashboard interface.

---

# Results Summary

The experimental results showed that Linear Regression achieved the best overall forecasting performance on the synthetic dataset.

The findings demonstrate that:
- Simpler interpretable models can outperform more complex ensemble methods in structured datasets
- Leakage-safe evaluation is critical for realistic forecasting
- Lag-based features significantly improve forecasting accuracy
- Recursive forecasting error increases as prediction horizons become longer

---

# Future Improvements

Possible future improvements include:

- Using real-world restaurant transaction data
- Adding weather, holiday, and promotion variables
- Implementing ARIMA, ETS, or Prophet forecasting models
- Applying SHAP explainability techniques
- Comparing recursive and direct forecasting approaches
- Deploying the dashboard online
- Integrating real-time forecasting pipelines

---

# Author

Kritee Thapa  
BSc (Hons) Computer Science  
De Montfort University  
Module: CTEC3451