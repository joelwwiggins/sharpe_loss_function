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
            data = yf.download(ticker, start='2020-01-01', end='2024-11-09')[['Close']]
            data.rename(columns={'Close': ticker}, inplace=True)
            print(f"Downloaded data for {ticker}")
        return data

    def get_combined_data(self, tickers, batch_size=5, delay=2):
        '''Fetches and combines data for all tickers into a single DataFrame in batches.'''
        try:
            combined_data = pd.read_csv('combined_data.csv', skiprows=2, index_col='Date', parse_dates=True)
            print("Loaded combined data.")
            return combined_data
        except (FileNotFoundError, ValueError):
            combined_data = pd.DataFrame()
            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i+batch_size]
                batch_data = yf.download(batch_tickers, start='2020-01-01', end='2024-11-09')['Close']
                combined_data = pd.concat([combined_data, batch_data], axis=1)
                print(f"Downloaded batch: {batch_tickers}")
                time.sleep(delay)  # Sleep between batches to avoid rate limiting
            combined_data.to_csv('combined_data.csv')
            return combined_data
    
    

    

if __name__ == '__main__':
    finance_bro = FinanceBro()
    tickers = finance_bro.get_tickers()
    data = finance_bro.get_combined_data(tickers)

    print(data.head())

    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = {}

    for ticker in tickers:
        data[ticker] = yf.download(ticker, start='2020-01-01', end='2024-11-09')[['Close']]
        time.sleep(2)  # Sleep for 2 seconds between requests