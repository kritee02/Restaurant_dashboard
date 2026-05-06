# Restaurant Sales Forecasting Dashboard

This project is a Streamlit-based dashboard for leakage-safe restaurant sales forecasting using machine learning. It uses a synthetic two-year daily restaurant sales dataset and compares baseline forecasting methods with machine learning models.

## Project Overview

The system forecasts daily restaurant sales and presents the results in an interactive dashboard. It includes:

- Dataset preview and summary statistics
- Exploratory data analysis
- Leakage-safe feature engineering
- Chronological train-test split
- Baseline forecasting models
- Machine learning models
- Model comparison using MAE, RMSE, MAPE, and R²
- Feature importance visualisation
- Recursive future sales forecasting
- CSV download options

This project is developed for academic and study purposes only. The dataset is synthetic and may not reflect real-world restaurant performance.

## Models Used

The dashboard compares the following models:

- Naïve forecasting
- Seasonal naïve forecasting
- Moving average forecasting
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Project Structure

```text
Restaurant_dashboard/
│
├── app.py                  # Main Streamlit dashboard
├── models.py               # Data preparation, model training, evaluation, and forecasting
├── generate_data.py        # Synthetic restaurant sales dataset generator
├── styles.py               # Custom CSS styling for the dashboard
│
├── data/
│   └── restaurant_sales.csv
│
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/restaurant-sales-forecasting-dashboard.git
cd restaurant-sales-forecasting-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

For Mac/Linux:

```bash
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install streamlit pandas numpy matplotlib scikit-learn xgboost
```

## Generate the Dataset

Run the data generation script:

```bash
python generate_data.py
```

This creates:

```text
data/restaurant_sales.csv
```

## Run the Dashboard

Start the Streamlit app:

```bash
streamlit run app.py
```

The dashboard will open in the browser at:

```text
http://localhost:8501
```

## Main Files

### app.py

Contains the Streamlit user interface, navigation menu, dashboard sections, charts, metrics, and CSV download buttons.

### models.py

Contains the core machine learning pipeline, including preprocessing, leakage-safe feature engineering, chronological splitting, baseline models, model training, evaluation, best model selection, and recursive forecasting.

### generate_data.py

Generates the synthetic two-year daily restaurant sales dataset using weekend effects, monthly seasonality, trend, and random noise.

### styles.py

Contains custom CSS used to improve dashboard appearance.

## Evaluation Metrics

The models are evaluated using:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- R²: Coefficient of determination

## Leakage-Safe Forecasting

The system prevents data leakage by:

- Sorting data chronologically
- Creating lag features using past values only
- Creating rolling averages using shifted historical values
- Splitting data chronologically instead of randomly
- Evaluating models only on unseen future test data

## Future Improvements

Possible improvements include:

- Replacing synthetic data with real restaurant transaction data
- Adding external variables such as weather, holidays, promotions, and local events
- Implementing walk-forward validation
- Comparing recursive forecasting with direct multi-step forecasting
- Adding SHAP or permutation importance for deeper interpretability
- Deploying the dashboard online

## Author

Kritee Thapa  
BSc (Hons) Computing  
De Montfort University
