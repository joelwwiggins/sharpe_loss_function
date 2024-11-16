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
import tensorflow as tf
from sklearn.preprocessing import StandardScaler




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

    @staticmethod
    def tf_portfolio_loss(y_true, y_pred, position_limit=0.5):
        positions = tf.clip_by_value(y_pred, -position_limit, position_limit)
        returns = positions * y_true
        return -tf.reduce_mean(returns)  # Negative for maximization
    
    def train_tf_model(self, data):
        '''Trains a TensorFlow model on the data.'''
        X_train, X_test, Y_train, Y_test = self.split_data(data)
        
        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

            # Reshape data for LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(50),
                tf.keras.layers.Dense(10)  # Output layer for 10 stocks
            ])
            
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        
        history = model.fit(X_train, Y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        return model, X_test, Y_test, history
    
    def eval_tf_model(self, model, X_test, Y_test):
        '''Evaluates the TensorFlow model on the test data.'''
        labels = model.predict(X_test)
        loss = self.tf_portfolio_loss(Y_test, labels)
        print(f'Loss: {loss}')
        return labels
    
    def plot_tf(self, model, X_test, Y_test):
        '''Plots the predictions of the TensorFlow model against the actual data. Plotting the first stock in the dataset.'''
        labels = model.predict(X_test)
        plt.figure(figsize=(12, 6))
        plt.plot(Y_test.index, Y_test.iloc[:, 1], label='Actual')
        plt.plot(Y_test.index, labels[:, 1], label='Predicted')

        plt.legend()
    
    def plot_loss(self, history):
        '''Plots the loss of the TensorFlow model during training.'''
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Loss')
        plt.legend()

    #plot the portfolio change over time vs the actual data
    def plot_portfolio(self, labels, X_test, original_data):
        '''Plots the portfolio value over time and shows the improvement compared to an equal-weighted portfolio.'''
        # Predicted portfolio value
        portfolio = labels.sum(axis=1)

        portfolio_df = pd.DataFrame(portfolio, index=original_data.index[-len(portfolio):], columns=['Predicted Portfolio Value'])
        
        # Reshape X_test back to 2D
        X_test_2d = X_test.reshape((X_test.shape[0], X_test.shape[2]))

        # Equal-weighted portfolio value
        equal_weighted_portfolio = X_test_2d.mean(axis=1)
        portfolio_df['Equal-Weighted Portfolio Value'] = equal_weighted_portfolio
        
        # Calculate percentage change for both portfolios
        portfolio_df['Predicted Portfolio Change (%)'] = portfolio_df['Predicted Portfolio Value'].pct_change().fillna(0)
        portfolio_df['Equal-Weighted Portfolio Change (%)'] = portfolio_df['Equal-Weighted Portfolio Value'].pct_change().fillna(0)
        
        plt.figure(figsize=(12, 12))
        
        # Plot portfolio values
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_df.index, portfolio_df['Predicted Portfolio Value'], label='Predicted Portfolio Value')
        plt.plot(portfolio_df.index, portfolio_df['Equal-Weighted Portfolio Value'], label='Equal-Weighted Portfolio Value', linestyle='--')
        plt.legend()
        plt.title('Portfolio Value Over Time')
        
        # Plot portfolio percentage changes
        plt.subplot(2, 1, 2)
        plt.plot(portfolio_df.index, portfolio_df['Predicted Portfolio Change (%)'], label='Predicted Portfolio Change (%)', color='orange')
        plt.plot(portfolio_df.index, portfolio_df['Equal-Weighted Portfolio Change (%)'], label='Equal-Weighted Portfolio Change (%)', color='green', linestyle='--')
        plt.legend()
        plt.title('Portfolio Percentage Change Over Time')
        
        plt.tight_layout()
        plt.show()
    

if __name__ == '__main__':
    fb = FinanceBro()
    
    tickers = fb.get_tickers()
    data = fb.get_combined_data(tickers)
    # data = fb.add_profit_volatility(data)
    
    # Train TensorFlow model
    model, X_test, Y_test, history = fb.train_tf_model(data)
    
    # Evaluate TensorFlow model
    labels = fb.eval_tf_model(model, X_test, Y_test)
    
    # Plot TensorFlow model predictions
    fb.plot_tf(model, X_test, Y_test)

    #plot loss
    fb.plot_loss(history)

    # Plot portfolio value
    fb.plot_portfolio(labels, X_test, data)
    
    #print weights vs actual data
    print(model.get_weights())