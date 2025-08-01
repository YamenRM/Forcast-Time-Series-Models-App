# ⏳ Time Series Forecasting App

A user-friendly Streamlit web application that allows you to forecast time series data using **ARIMA**, **Prophet**, and **Random Forest Regression**. The app visualizes and compares the forecasts with actual data, providing key performance metrics like RMSE and R².

## 📌 Features

- Upload your own `.csv` time series dataset
- Automatic preprocessing and cleaning
- Interactive parameter tuning (ARIMA order, Random Forest estimators)
- Visual forecast comparison for:
  - ARIMA
  - Facebook Prophet
  - Random Forest Regressor
- Evaluation Metrics: RMSE and R² Score

## 🧠 Models Used

### 🔸 ARIMA
A classic statistical model for univariate time series forecasting, best for stable and linear patterns.

### 🔸 Prophet
A robust model developed by Facebook for trend-seasonality forecasting, especially suitable for business data.

### 🔸 Random Forest Regressor
A machine learning model that uses lag features and statistical features (moving average, standard deviation) for regression-based forecasting.

## 📁 Example Dataset Format

Your dataset must contain at least:

| date       | target       |
|------------|--------------|
| 2023-01-01 | 112.0        |
| 2023-01-02 | 114.5        |
| ...        | ...          |

- `date` must be convertible to datetime.
- There must be a date coulman (its not cass-sensetive because there is a method to transfer it to lower-case).
- The target variable should be numerical or convertible to numeric.

## 📝 Notebook

The full modeling process is documented in Model.ipynb for a ([TESLA stock dataset](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021) ), which includes:

 - Data exploration

 - Cleaning and transformations

 - ARIMA parameter selection

 - Prophet tuning and visualizations

 - Feature engineering for Random Forest

 - Performance comparison

## 🚀 Deployment

This app can be deployed via:

 Streamlit Cloud > https://forcast-time-series-models-app-fbrmpjhrpxepphcwbh3m8j.streamlit.app/

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/YamenRM/Forcast-Time-Series-Models-App.git
cd Forcast-Time-Series-Models-App

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📌 Future Improvements

 - Add LSTM and XGBoost forecasting models

 - Allow automatic ARIMA and Prophet parameter tuning

 - Enable multivariate time series support

 - Interactive plots using Plotly

### 👨‍💻 Author

  **YamenRM**
  
💡 AI/ML Engineering Student | UP

📍 PALESTINE | Gaza Strip

💪 Stay strong!

#### ⭐ Star this repo if you find it helpful!

#### 📬 Feel free to open an issue or pull request.
