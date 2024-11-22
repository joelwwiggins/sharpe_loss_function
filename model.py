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
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, Huber, LogCosh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling1D, Concatenate
from tensorflow.keras.layers import Cropping1D
from tensorflow.keras.layers import GlobalAveragePooling1D




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
        
    def custom_loss(self, y_true, y_pred):
        '''Custom loss function combining MSE and profit calculation.'''
        mse = MeanSquaredError()(y_true, y_pred)
        profit = tf.reduce_sum(y_pred[:, 1:] - y_pred[:, :-1])
        volatility = tf.math.reduce_std(y_pred)
        profit_volatility_ratio = profit / (volatility + 1e-6)
        return mse - profit_volatility_ratio
    
    def self_attention_model(self, input_shape, output_shape):
        '''Creates a Self-Attention model.'''
        inputs = tf.keras.Input(shape=(input_shape, 1))
        
        # Initial Convolutional Layer
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Self-Attention Layer
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        attn_output = tf.keras.layers.Add()([x, attn_output])  # Residual connection
        attn_output = tf.keras.layers.LayerNormalization()(attn_output)
        
        # Flatten and Dense Layers
        x = Flatten()(attn_output)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(output_shape)(x)
        
        # Build Model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def predict(self, model, data, timesteps):
        '''Predicts stock prices using the model.'''
        '''Add profit and volatility columns to the data.'''
        data = self.add_profit_and_volatility(data)
        '''Scale the data using StandardScaler.profit and voloatility as input and stock prices as output'''

        y_data = data[['Total_Profit', 'Total_Volatility']].values
        X_data = data.drop(columns=['Total_Profit', 'Total_Volatility']).values

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_data)

        X_train, X_test, y_train, y_test = train_test_split(scaled_data, y_data, test_size=0.2, random_state=42)
        
        # Reshape data
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_train = y_train
        y_test = y_test

        # Train model
        history = model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1)

        # Predict
        prediction = model.predict(X_test)
        return prediction, history, X_test, y_test, X_train, y_train 
    
    def plot_history(self, prediction, y_test):
        '''Plots prediction vs actual stock prices. make aplot for every column feature'''
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(prediction, label='Predicted')
        plt.legend()
        plt.show()

    def plot_loss_history(self, history):
        '''Plots the training loss history.'''
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.show()
    
    def plot_pca_clusters(self,data, title):
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)
        plt.figure(figsize=(10, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(title)
        plt.colorbar()
        plt.show()


    
    def print_output_layer_weights(self,model):
        '''Prints the activation of the output layer in the model.'''
        output_layer = model.layers[-1]  # Get the last layer of the model
        weights = output_layer.get_weights()  # Returns a list of numpy arrays
        print(f"Layer: {output_layer.name}")
        for i, weight in enumerate(weights):
            print(f"  Weight {i}: {weight.shape}")
            print(weight)
    
    def get_stock_allocation(self,model):
        '''Gets stock allocation from the weights of the last dense layer.'''
        # Get the last dense layer
        last_dense_layer = model.layers[-1]
        
        # Extract weights (assuming weights are in the form [weights, biases])
        weights = last_dense_layer.get_weights()[0]
        
        # Sum the absolute values of the weights for each stock
        allocation = np.sum(np.abs(weights), axis=0)
        
        # Normalize the allocation to sum to 1
        allocation /= np.sum(allocation)
        
        # Convert to percentage
        allocation *= 100
        
        return allocation
    
    def add_profit_and_volatility(self,data):
        
        '''Adds columns for total profit and total volatility to the data.'''
        total_profit = data.diff().sum(axis=1)
        total_volatility = data.diff().rolling(30).std().sum(axis=1)
        data['Total_Profit'] = total_profit
        data['Total_Volatility'] = total_volatility
        # Drop rows with NaN values
        data.dropna(inplace=True)
        return data 

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    finance_bro = FinanceBro()
    tickers = finance_bro.get_tickers()
    data = finance_bro.get_combined_data(tickers)
    input_shape = data.shape[1]
    output_shape = 2
    model = finance_bro.self_attention_model(input_shape,output_shape)

    
    
    
    # Train and visualize the Conv1D model
    conv_model = finance_bro.self_attention_model(input_shape, output_shape)
    prediction_conv, history_conv, X_test_conv, y_test_conv, X_train_conv, y_train_conv = finance_bro.predict(conv_model, data, timesteps=None)
    finance_bro.plot_pca_clusters(prediction_conv, 'PCA of Conv1D Model Predictions')
    finance_bro.plot_history(prediction_conv, y_test_conv)
    finance_bro.plot_loss_history(history_conv)
