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
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




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

    def add_profit_volatility(self, data):
        '''Adds profit and volatility columns to the dataset.'''
        daily_returns = data.pct_change()
        data['Profit'] = daily_returns.mean(axis=1)
        data['Volatility'] = daily_returns.std(axis=1)
        data.dropna(inplace=True)
        return data

    def train_cnn_model(self, data):
        '''Trains a 2D CNN using PyTorch to predict Profit.'''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare dataset
        dataset = FinanceDataset(data)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        # Define model
        num_features = data.shape[1] - 2  # Exclude 'Profit' and 'Volatility'
        model = CNNModel(num_features=num_features).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        print(f'Test MSE: {mse:.4f}, R2 Score: {r2:.4f}')

        return model, actuals, predictions
    
    def plot_actual_vs_predicted(self, actuals, predictions):
        '''Plots actual vs. predicted values.'''
        plt.figure(figsize=(10, 5))
        plt.plot(actuals, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        plt.show()

class FinanceDataset(Dataset):
    def __init__(self, data):
        self.X = data.drop(['Profit', 'Volatility'], axis=1).values
        self.y = data['Profit'].values
        # Normalize features
        self.X = StandardScaler().fit_transform(self.X)
        # Reshape for CNN input
        self.X = self.X.reshape(-1, 1, self.X.shape[1], 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class CNNModel(nn.Module):
    def __init__(self, num_features):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
        )
        conv_output_size = 16 * ((num_features - 2) // 2)
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    


if __name__ == '__main__':
    fb = FinanceBro()
    tickers = fb.get_tickers()
    data = fb.get_combined_data(tickers)
    data = fb.add_profit_volatility(data)
    model, actuals, predictions = fb.train_cnn_model(data)
    fb.plot_actual_vs_predicted(actuals, predictions)
    
    


