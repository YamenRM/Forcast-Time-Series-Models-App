import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error , r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


# set the title of the app
st.title('Time Series Forecasting App')
st.markdown('This app allows you to forecast time series data using various models such as ARIMA, Prophet, and Random Forest Regression.')

# Load the dataset
st.markdown('### Upload your time series data')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.error("Please upload a CSV file.")
    st.stop()

# Display the dataset
st.markdown('### Dataset Preview')
st.write(data.head())

# Convert column names to lowercase
data.columns = data.columns.str.lower()

# Check if the dataset has a datetime index
if 'date'  in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
else:
    st.error("The dataset must contain a 'date' column.")
    st.stop()

# Select the target variable for forecasting
st.markdown('### Select the target variable for forecasting')
target_variable = st.selectbox('Select target variable', data.columns)
if target_variable is None:
    st.error("Please select a target variable.")
    st.stop()

# convert the target variable to numeric
data[target_variable] = pd.to_numeric(data[target_variable], errors='coerce')

# handel missing values in the target variable
data[target_variable].fillna(method='ffill', inplace=True)

# Check if the target variable has enough data points
if data[target_variable].isnull().all():
    st.error(f"The target variable '{target_variable}' does not have enough data points. Please check your dataset.")
    st.stop()

# Split the data into training and testing sets for the last 60 days
train_size = int(len(data) - 60)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:] 
if train_data.empty or test_data.empty:
    st.error("The training or testing dataset is empty. Please ensure your dataset has enough data points.")
    st.stop()

# ARIMA Model
st.markdown('### ARIMA Model ')
arima_order = st.text_input('Enter ARIMA order (p,d,q) as comma-separated values', '1,1,1')
if arima_order:
    p, d, q = map(int, arima_order.split(','))
    arima_model = ARIMA(train_data[target_variable], order=(p, d, q))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=60)
else:
    st.error("Please enter a valid ARIMA order in the format p,d,q.")

# Prophet Model
st.markdown('### Prophet Model ')
prophet_data = train_data.reset_index().rename(columns={'date': 'ds', target_variable: 'y'})
if not prophet_data.empty:
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=60)
    prophet_forecast = prophet_model.predict(future)
else:
    st.error("Prophet model requires a non-empty training dataset with 'ds' and 'y' columns.")


# Random Forest Regression Model
st.markdown('### Random Forest Regression Model ')

# preprocess the data for Random Forest
data_rf = data.copy()
data_rf['lag_1'] = data_rf[target_variable].shift(1)
data_rf['lag_2'] = data_rf[target_variable].shift(2)
data_rf['MA_7'] = data_rf[target_variable].rolling(window=7).mean()
data_rf['STD_7'] = data_rf[target_variable].rolling(window=7).std()
data_rf.dropna(inplace=True)

# scaling the features
X = ['lag_1', 'lag_2', 'MA_7', 'STD_7']
scaler = MinMaxScaler()
data_rf[X] = scaler.fit_transform(data_rf[X])

# split the data into training and testing sets
X_train = data_rf.iloc[:train_size][X]
y_train = data_rf.iloc[:train_size][target_variable]
X_test = data_rf.iloc[train_size:][X]
y_test = data_rf.iloc[train_size:][target_variable]

# fit the Random Forest model and selcting n_estimators
n_estimators = st.slider('Select number of estimators for Random Forest', min_value=10, max_value=200, value=100, step=10)
rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
rf_model.fit(X_train, y_train)
rf_forecast = rf_model.predict(X_test)
# Check if the forecasts are empty
if rf_forecast.size == 0:
    st.error("Random Forest model did not produce any forecasts. Please check your data and model parameters.")
    st.stop()



# Display the results
st.markdown('### Forecasting Results')
st.write("ARIMA RMSE:", root_mean_squared_error(test_data[target_variable][:60], arima_forecast))
st.write("Prophet RMSE:", root_mean_squared_error(test_data[target_variable][:60], prophet_forecast['yhat'][-60:]))
st.write("Random Forest RMSE:", root_mean_squared_error(y_test[:60], rf_forecast))
st.write("ARIMA R2 Score:", r2_score(test_data[target_variable][:60], arima_forecast))
st.write("Prophet R2 Score:", r2_score(test_data[target_variable][:60], prophet_forecast['yhat'][-60:]))
st.write("Random Forest R2 Score:", r2_score(y_test[:60], rf_forecast))

# visualize the results
st.markdown('### Visualization of Forecasts')
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[:60], test_data[target_variable][:60], label='Actual', color='blue')
plt.plot(test_data.index[:60], arima_forecast, label='ARIMA Forecast', color='orange')
plt.plot(test_data.index[:60], prophet_forecast['yhat'][-60:], label='Prophet Forecast', color='green')
plt.plot(y_test.index[:60], rf_forecast, label='Random Forest Forecast', color='red')
plt.title('Forecast Comparison')
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.legend()
st.pyplot(plt)

# conclusion
st.markdown('### Conclusion')
st.write("This app provides a simple interface to forecast time series data using ARIMA, Prophet, and Random Forest Regression models. You can upload your own dataset and visualize the forecasts.")
st.write("Feel free to explore different parameters for each model to improve the forecasting accuracy.")
