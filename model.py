import numpy as np
import pandas as pd
# import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
import tensorflow as tf

@dataclass
class FinanceBro:
    '''A prediction model to maximize profit and minimize volatility.'''
    
    def get_tickers(self):
        '''Fetches tickers of the S&P 500 companies.'''
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        sp500_tickers = [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
        return sp500_tickers[:20]

    def get_data(self, ticker):
        '''Fetches stock data for a given ticker.'''
        try:
            data = pd.read_csv(f'data/{ticker}.csv', index_col='Date', parse_dates=True)
            print(f"Loaded data for {ticker}")
        except FileNotFoundError:
            data = yf.download(ticker, start='2020-01-01', end='2024-11-09')[['Close']]
            data.rename(columns={'Close': ticker}, inplace=True)
            print(f"Downloaded data for {ticker}")
        return data

    def get_combined_data(self, tickers):
        '''Fetches and combines data for all tickers into a single DataFrame.'''
        try:
            combined_data = pd.read_csv('combined_data.csv', skiprows=2, index_col='Date', parse_dates=True)
            print("Loaded combined data.")
            return combined_data
        except (FileNotFoundError, ValueError):
            combined_data = pd.DataFrame()
            for ticker in tickers:
                ticker_data = self.get_data(ticker)
                combined_data = pd.concat([combined_data, ticker_data], axis=1)
            combined_data.dropna(inplace=True)
            combined_data.to_csv('combined_data.csv')
            print("Saved combined data for all tickers.")
            return combined_data

    def cnn_model(self, input_shape):
        '''Creates a Convolutional Neural Network model.'''
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(50, return_sequences=False),
            Dense(20)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def predict(self, model, data, timesteps):
        '''Predicts stock prices using the model.'''
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        X_train, X_test, y_train, y_test = train_test_split(scaled_data, scaled_data, test_size=0.2, random_state=42)
        
        # Reshape data
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_train = y_train
        y_test = y_test

        # Train model
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

        # Predict
        prediction = model.predict(X_test)
        return prediction, history, X_test, y_test, X_train, y_train 
    
    def plot_history(self, prediction, y_test):
        '''Plots prediction vs actual stock prices. make aplot for every column feature'''
        y_test = y_test  # Convert y_test to a NumPy array
        for i in range(y_test.shape[1]):
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[:, i], label='Actual')
            plt.plot(prediction[:, i], label='Predicted')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    finance_bro = FinanceBro()
    tickers = finance_bro.get_tickers()
    data = finance_bro.get_combined_data(tickers)
    input_shape = data.shape[1]
    model = finance_bro.cnn_model(input_shape)
    prediction, history, X_test, y_test, X_train, y_train = finance_bro.predict(model, data, timesteps=None)
    finance_bro.plot_history(prediction, y_test)
    print(f'Available metrics: {history.history.keys()}')
    print(f'loss: {history.history["loss"][-1] if "loss" in history.history else "N/A"}, r2_score: {r2_score(y_test, prediction)}')