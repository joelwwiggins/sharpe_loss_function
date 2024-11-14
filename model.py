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

@dataclass
class FinanceBro:
    '''A prediction model to maximize profit and minimize volatility.'''
    risk_penalty: float = 0.2
    max_position: float = 0.4
    n_assets: int = None

    def _softmax(self, x):
        '''Apply softmax to ensure weights sum to 1'''
        if x.ndim == 1:
            x = x.reshape(1, -1)
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def custom_objective(self, predt: np.ndarray, dtrain) -> tuple:
        '''Custom objective function for portfolio optimization'''
        returns = dtrain.get_label().reshape(-1, self.n_assets)
        n_samples = returns.shape[0]
        
        weights = self._softmax(predt.reshape(n_samples, -1))
        portfolio_returns = np.sum(weights * returns, axis=1)
        
        expected_return = np.mean(portfolio_returns)
        portfolio_risk = np.std(portfolio_returns)
        
        position_penalties = np.maximum(0, weights - self.max_position)
        position_penalty = np.sum(position_penalties)
        
        loss = -(expected_return - self.risk_penalty * portfolio_risk - position_penalty)
        
        grad_returns = -returns
        risk_grad = 2 * self.risk_penalty * (portfolio_returns[:, None] - np.mean(portfolio_returns)) / (n_samples * portfolio_risk)
        grad_penalty = 2 * position_penalties
        
        grad = grad_returns + risk_grad * returns + grad_penalty
        hess = np.ones_like(grad) * 2
        
        return grad.flatten(), hess.flatten()

    def train_model(self, returns_data):
        '''Trains the model using XGBoost'''
        self.n_assets = returns_data.shape[1]
        
        X = np.arange(len(returns_data)).reshape(-1, 1)
        y = returns_data.values
        
        dtrain = xgb.DMatrix(X, label=y)
        
        params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror'
        }
        
        num_rounds = 100
        model = xgb.train(
            params,
            dtrain,
            num_rounds,
            obj=self.custom_objective
        )
        
        return model

    def get_optimal_weights(self, model, returns_data):
        '''Get optimized portfolio weights'''
        X = np.array([[len(returns_data)]])
        dtest = xgb.DMatrix(X)
        
        raw_weights = model.predict(dtest)
        weights = self._softmax(raw_weights.reshape(1, -1))
        
        return pd.Series(weights[0], index=returns_data.columns)

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

    def prepare_data(self, data):
        '''Prepares data for portfolio optimization'''
        stock_columns = [col for col in data.columns if col not in ['Profit', 'Volatility']]
        returns = data[stock_columns].pct_change()
        returns.dropna(inplace=True)
        return returns, data['Profit'].loc[returns.index], data['Volatility'].loc[returns.index]

    def evaluate_portfolio(self, weights, returns_data, original_profit, original_volatility):
        '''Evaluates portfolio performance'''
        portfolio_returns = (weights * returns_data).sum(axis=1)
        
        performance = {
            'Portfolio Return': portfolio_returns.mean() * 252,
            'Portfolio Volatility': portfolio_returns.std() * np.sqrt(252),
            'Sharpe Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'Original Profit Mean': original_profit.mean(),
            'Original Volatility Mean': original_volatility.mean(),
            'Max Weight': weights.max(),
            'Min Weight': weights.min()
        }
        
        return performance

    def plot_results(self, weights, returns_data, original_profit):
        '''Plots portfolio performance and weight distribution'''
        portfolio_returns = (weights * returns_data).sum(axis=1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        cumulative_original = (1 + original_profit).cumprod()
        
        ax1.plot(cumulative_portfolio.index, cumulative_portfolio, label='Optimized Portfolio')
        ax1.plot(cumulative_original.index, cumulative_original, label='Original')
        ax1.set_title('Cumulative Returns Comparison')
        ax1.legend()
        ax1.grid(True)
        
        ax2.bar(weights.index, weights.values)
        ax2.set_title('Portfolio Weights Distribution')
        plt.xticks(rotation=45)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    fb = FinanceBro(risk_penalty=0.4, max_position=0.4)
    
    print("Loading data...")
    raw_data = fb.get_combined_data(fb.get_tickers()[:10])
    raw_data = fb.add_profit_volatility(raw_data)
    
    print("Preparing data...")
    returns_data, original_profit, original_volatility = fb.prepare_data(raw_data)
    
    print("Training model...")
    model = fb.train_model(returns_data)
    
    print("Getting optimal weights...")
    optimal_weights = fb.get_optimal_weights(model, returns_data)
    
    print("Evaluating performance...")
    performance = fb.evaluate_portfolio(optimal_weights, returns_data, original_profit, original_volatility)
    
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in optimal_weights.items():
        print(f"{ticker}: {weight:.4f}")
    
    print("\nPortfolio Performance:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPlotting results...")
    fb.plot_results(optimal_weights, returns_data, original_profit)