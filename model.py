import numpy as np
import pandas as pd
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
import xgboost as xgb
from sklearn.kernel_approximation import RBFSampler

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



    def xgboost_model(self, X_train, y_train):
        '''Creates and trains an XGBoost model.'''
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.01,reg_alpha=1)
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_test):
        '''Predicts stock prices using the model.'''
        prediction = model.predict(X_test)
        return prediction

    def plot_history(self, prediction, y_test):
        '''Plots prediction vs actual stock prices.'''
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual Profit')
        plt.plot(prediction, label='Predicted Profit')
        plt.legend()
        plt.show()


    def add_profit(self, data):
        total_profit = data.diff().sum(axis=1)
        data['Total_Profit'] = total_profit
        # data['month'] = data.index.month
        # data['day'] = data.index.dayofyear
        # data['week'] = data.index.isocalendar().week

        # # Drop rows with NaN values
        # data.dropna(inplace=True)

        # # Add RBF features for month and week
        # rbf_sampler = RBFSampler(gamma=1.0, n_components=10, random_state=42)
        # month_rbf_features = rbf_sampler.fit_transform(data[['month']])
        # week_rbf_features = rbf_sampler.fit_transform(data[['week']])
        # day_rbf_features = rbf_sampler.fit_transform(data[['day']])

        # # Add RBF features to the dataframe
        # for i in range(month_rbf_features.shape[1]):
        #     data[f'month_rbf_{i}'] = month_rbf_features[:, i]
        # for i in range(week_rbf_features.shape[1]):
        #     data[f'week_rbf_{i}'] = week_rbf_features[:, i]
        # for i in range(day_rbf_features.shape[1]):
        #     data[f'day_rbf_{i}'] = day_rbf_features[:, i]

        return data


if __name__ == '__main__':
    finance_bro = FinanceBro()
    tickers = finance_bro.get_tickers()
    data = finance_bro.get_combined_data(tickers)

    # Preprocess data
    data = finance_bro.add_profit(data)

    # Define target variables
    y_data = data[['Total_Profit']].values

    # Exclude target variables from features
    X_data = data.drop(columns=['Total_Profit']).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_data, test_size=0.2, random_state=42, shuffle=False
    )

    # Train the model
    model = finance_bro.xgboost_model(X_train, y_train)

    # Predict
    prediction = finance_bro.predict(model, X_test)

    # Plotting
    finance_bro.plot_history(prediction, y_test)