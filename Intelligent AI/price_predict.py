import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def price_predict(ticker):
    # Load Data
    yf.pdr_override()
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2021, 1, 1)

    data = pdr.DataReader(ticker, start, end)

    # Preparing Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 30

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predicts closing price of the next day

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=36)

    ''' Test Model Accuracy on Existing Data '''

    # Load Test Data
    test_start = dt.datetime(2021, 1, 1)
    end_year = dt.datetime.today().year
    end_month = dt.datetime.today().month
    end_day = dt.datetime.today().day - 1
    test_end = dt.datetime(end_year, end_month, end_day)

    test_data = pdr.get_data_yahoo(ticker, test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Predictions on Real Data
    real_data = []

    for x in range(prediction_days, len(model_inputs)):
        real_data.append(model_inputs[x - prediction_days:x, 0])

    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    predicted_prices = model.predict(real_data)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Predict Next Day
    real_data = [model_inputs[len(model_inputs + 1) - prediction_days: len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    print(f"Next Day Prediction: {prediction} ")

    # Plot the Test Predictions
    plt.plot(actual_prices, color="black", label=f'Actual {ticker} Price')
    plt.plot(predicted_prices, color='Red', label=f'Predicted {ticker} Price')
    plt.title(f"{ticker} Share Price ")
    plt.xlabel('Time')
    plt.ylabel(f"{ticker} Share Price ")
    plt.legend()
    plt.show()

    '''
    time_year = dt.datetime.today().year
    time_month = dt.datetime.today().month
    time_day = dt.datetime.today().day
    start_price = dt.datetime(time_year, time_month, time_day)
    end_price = dt.datetime(time_year, time_month, time_day)
    data_price = pdr.get_data_yahoo(ticker, start_price, end_price)
    price = data_price['Adj Close'].values
    accuracy = abs(((abs((prediction - price)) / price) * 100) - 100)
    time_price = dt.datetime(time_year, time_month, time_day, 4)
    now_time = dt.datetime.now()
    if now_time > time_price:
        print(f'This prediction is: {accuracy} % accurate')
    else:
        pass
    '''


if __name__ == "__main__":
    company = str(input("Enter the company's ticker symbol: ").upper())
    price_predict(company)
