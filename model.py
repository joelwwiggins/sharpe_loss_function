import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
        return sp500_tickers

    def get_data(self, ticker):
        '''Fetches stock data for a given ticker.'''
        try:
            data = pd.read_csv(f'data/{ticker}.csv', index_col='Date', parse_dates=True)
            print(f"Loaded data for {ticker}")
        except FileNotFoundError:
            data = yf.download(ticker, start='2022-01-01', end='2024-11-09')[['Close']]
            data.rename(columns={'Close': ticker}, inplace=True)
            print(f"Downloaded data for {ticker}")
            time.sleep(1)
        return data

    def get_combined_data(self, tickers):
        '''Fetches and combines data for all tickers into a single DataFrame.'''
        try:
            combined_data = pd.read_csv('combined_data.csv', skiprows=2, index_col='Date', parse_dates=True)
            print("Loaded combined data.")
            return combined_data
        except FileNotFoundError:
            combined_data = pd.DataFrame()
            for ticker in tickers:
                ticker_data = self.get_data(ticker)
                combined_data = pd.concat([combined_data, ticker_data], axis=1)

            combined_data.dropna(inplace=True)
            combined_data.to_csv('combined_data.csv')
            print("Saved combined data for all tickers.")
            return combined_data
        except ValueError as e:
            print(f"Error reading combined_data.csv: {e}")
            print("Creating combined data from scratch.")
            combined_data = pd.DataFrame()
            for ticker in tickers:
                ticker_data = self.get_data(ticker)
                combined_data = pd.concat([combined_data, ticker_data], axis=1)

            combined_data.dropna(inplace=True)
            combined_data.to_csv('combined_data.csv')
            print("Saved combined data for all tickers.")
            return combined_data

    def add_profit_volatility(self, data):
        '''Adds profit and volatility columns to the dataset.'''
        data['Profit'] = data.pct_change().mean(axis=1)
        data['Volatility'] = data.pct_change().std(axis=1)
        data.dropna(inplace=True)
        return data

    def split_data(self, data):
        '''splits the data into training and testing sets.'''
        X_train, X_test,Y_train, Y_test = train_test_split(data, data, test_size=0.2, random_state=42, shuffle=False)
        return X_train, X_test, Y_train, Y_test


        



    def train_xgboostreg_model(self, data):
        '''Trains an XGBoost regression model on the data.'''
        X_train, X_test, Y_train, Y_test = self.split_data(data)
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, Y_train)
        return model
    
    def eval_xgb_model(self, model, X_test, Y_test):
        '''Evaluates the XGBoost model on the test data.'''
        labels = model.predict(X_test)
        mse = mean_squared_error(Y_test, labels)
        r2 = r2_score(Y_test, labels)
        print(f'Mean Squared Error: {mse}')
        print(f'R2 Score: {r2}')
        return labels
    
    def plot_xgb(self, model, X_test, Y_test):
        '''Plots the predictions of the XGBoost model against the actual data. Plotting the first stock in the dataset.'''
        labels = model.predict(X_test)
        plt.figure(figsize=(12, 6))
        plt.plot(Y_test.index, Y_test.iloc[:, 1], label='Actual')
        plt.plot(Y_test.index, labels[:, 1], label='Predicted')

        plt.legend()    

    #using tensorflow

    def tf_portfolio_loss(y_true, y_pred, position_limit=0.1):
        positions = tf.clip_by_value(y_pred, -position_limit, position_limit)
        returns = positions * y_true
        return -tf.reduce_mean(returns)  # Negative for maximization
    
    def train_tf_model(self, data):
        '''Trains a TensorFlow model on the data.'''
        X_train, X_test, Y_train, Y_test = self.split_data(data)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(500)  # Output layer
        ])
        model.compile(optimizer='adam', loss=tf_portfolio_loss)
        model.fit(X_train, Y_train, epochs=10)
        return model
    
    def eval_tf_model(self, model, X_test, Y_test):
        '''Evaluates the TensorFlow model on the test data.'''
        labels = model.predict(X_test)
        loss = tf_portfolio_loss(Y_test, labels)
        print(f'Loss: {loss}')
        return labels
    
    def plot_tf(self, model, X_test, Y_test):
        '''Plots the predictions of the TensorFlow model against the actual data. Plotting the first stock in the dataset.'''
        labels = model.predict(X_test)
        plt.figure(figsize=(12, 6))
        plt.plot(Y_test.index, Y_test.iloc[:, 1], label='Actual')
        plt.plot(Y_test.index, labels[:, 1], label='Predicted')

        plt.legend()

if __name__ == '__main__':
    fb = FinanceBro()
    
    tickers = fb.get_tickers()
    data = fb.get_combined_data(tickers)
    data = fb.add_profit_volatility(data)
    # model = fb.train_xgboostreg_model(data)
    # labels = fb.eval_xgb_model(model, data, data)
    # fb.plot_xgb(model, data, data) 
    model = fb.train_tf_model(data)
    labels = fb.eval_tf_model(model, data, data)
    fb.plot_tf(model, data, data)
    print(labels)

    
