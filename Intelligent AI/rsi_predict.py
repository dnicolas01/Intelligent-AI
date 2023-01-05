import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from price_predict import price_predict


def RSI():
    while True:
        ticker = input("Enter ticker symbol of the company you want to analyze: ").upper()
        yf.pdr_override()
        start = dt.datetime(2020, 1, 1)
        end = dt.datetime.now()

        data = pdr.get_data_yahoo(ticker, start, end)

        delta = data['Adj Close'].diff(1)
        delta.dropna(inplace=True)

        positive = delta.copy()
        negative = delta.copy()

        positive[positive < 0] = 0
        negative[negative > 0] = 0

        days = 14
        average_gain = positive.rolling(window=days).mean()
        average_loss = abs(negative.rolling(window=days).mean())

        relative_strength = average_gain / average_loss
        RSI = 100 - (100 / (1.0 + relative_strength))

        combined = pd.DataFrame()
        combined['Adj Close'] = data['Adj Close']
        combined['RSI'] = RSI

        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(211)
        ax1.plot(combined.index, combined['Adj Close'], color='lightgrey')
        ax1.set_title(f'Adjusted Close Price of {ticker}', color='white')

        ax1.grid(True, color='#555555')
        ax1.set_axisbelow(True)
        ax1.set_facecolor('black')
        ax1.figure.set_facecolor('#121212')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(combined.index, combined['RSI'], color='lightgrey')
        ax2.axhline(0, linestyle='--', color='#ff0000')
        ax2.axhline(10, linestyle='--', color='#ffaa00')
        ax2.axhline(20, linestyle='--', color='#00ff00')
        ax2.axhline(30, linestyle='--', color='#cccccc')
        ax2.axhline(70, linestyle='--', color='#cccccc')
        ax2.axhline(80, linestyle='--', color='#00ff00')
        ax2.axhline(90, linestyle='--', color='#ffaa00')
        ax2.axhline(100, linestyle='--', color='#ff0000')

        ax2.set_title('RSI Value', color='white')
        ax2.set_axisbelow(False)
        ax2.set_facecolor('black')
        ax2.figure.set_facecolor('#121212')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        plt.show()

        price = str(input(f"Is {ticker} overbought or oversold? (y/n): ").lower())
        if price == 'y':
            answer = input("Do you want to check another company? (y/n): ").lower()
            if answer == 'y':
                continue
            else:
                break
        else:
            print("Here is your End of Day Stock Prediction")
            price_predict(ticker)
            break


if __name__ == '__main__':
    RSI()
