import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt


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
            time.sleep(1)  # Pause to avoid rate-limiting
        return data

    def get_combined_data(self, tickers):
        '''Import csv data. if none Fetches and combines data for all tickers into a single DataFrame.'''

        try:
            combined_data = pd.read_csv('combined_data.csv',index_col='Date', parse_dates=True)
            print("Loaded combined data.")
            return combined_data
        except FileNotFoundError:
            pass
            combined_data = pd.DataFrame()
            for ticker in tickers:
                ticker_data = self.get_data(ticker)
                combined_data = pd.concat([combined_data, ticker_data], axis=1)
                
            #only keep one column index
            combined_data.columns = combined_data.columns.get_level_values(0)
            #add column for profit and volatility
            combined_data['Profit'] = combined_data.iloc[:, 0].pct_change()
            combined_data['Volatility'] = combined_data.iloc[:, 0].rolling(30).std()
            #drop na values
            combined_data.dropna(inplace=True)
            combined_data.to_csv('combined_data.csv')  # Save combined data once at the end
            print("Saved combined data for all tickers.")
            return combined_data
        

    def split_data(self, data):
        '''separate profit and volatility column then Splits the data into features and target for training and testing.'''
        X_train, X_test, y_train, y_test = train_test_split(data,data test_size=0.2, random_state=42, shuffle=False)
        return X_train, X_test, y_train, y_test,

    def train_model(self, X_train, y_train):

        '''Trains the model on the training data. Use custom objective function to maximize profit and minimize volatility using grad and hess.'''

        profit = y_train['Profit']
        volatility = y_train['Volatility']
        loss = profit - volatility
        model = xgb.XGBRegressor(
        model.fit(X_train, y_train, sample_weight=loss)
        return model

    def evaluate_model(self, model, X_test, y_test):
        '''Evaluates the model on the test data.'''
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Mean Squared Error: {mse}, R2 Score: {r2}")
        return mse, r2
    
    def plot_predictions(self, model, X_test, y_test):
        '''Plots the model's predictions against the actual values.'''
        predictions = model.predict(X_test)
        
        # Convert to 1D if necessary
        y_test = y_test.values.flatten() if y_test.ndim > 1 else y_test
        predictions = predictions.flatten() if predictions.ndim > 1 else predictions

        # Ensure lengths match before plotting
        if len(predictions) != len(y_test):
            raise ValueError("Mismatch between predictions and actual values lengths.")
        
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        results_df.plot()
        plt.show()

if __name__ == '__main__':
    fb = FinanceBro()
    tickers = fb.get_tickers()[:10]
    
    # Aggregate all stock data
    data = fb.get_combined_data(tickers)

    # Split data
    X_train, X_test, y_train, y_test = fb.split_data(data)

    # # Train model
    # model = fb.train_model(X_train, y_train)

    # # Evaluate model
    # mse, r2 = fb.evaluate_model(model, X_test, y_test)

    # # Plot predictions
    # fb.plot_predictions(model, X_test[:1], y_test[:1])
