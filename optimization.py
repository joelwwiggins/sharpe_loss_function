import dataclasses as dc
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from model import FinanceBro
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt

@dc.dataclass
class Optimizer:
    '''This class uses Gaussian Mixture Model to create inputs for the XGBoost model in model.py.'''

    def generate_gmm_inputs(self, data, n_components=10, n_samples=10000):
        '''Generates inputs using Gaussian Mixture Model.'''
        features = data.drop(columns=['Total_Profit'])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)

        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(scaled_data)

        generated_data = gmm.sample(n_samples)[0]
        generated_data = scaler.inverse_transform(generated_data)
        generated_data = pd.DataFrame(generated_data, columns=features.columns)

        return generated_data

    def optimize_portfolio_xgboost(self, data, n_components=5, n_samples=10000):
        '''Optimizes the portfolio weights using the generated inputs and XGBoost model.'''
        finance_bro = FinanceBro()

        # Generate GMM inputs
        gmm_inputs = self.generate_gmm_inputs(data, n_components, n_samples)

        # Prepare training data
        y_data = data['Total_Profit'].values
        X_data = data.drop(columns=['Total_Profit']).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_data, test_size=0.2, random_state=42, shuffle=False
        )

        # Train the model
        model = finance_bro.xgboost_model(X_train, y_train)

        # Predict using the generated GMM inputs
        gmm_inputs_scaled = scaler.transform(gmm_inputs.values)
        predicted_profits = model.predict(gmm_inputs_scaled)

        # Define features for optimization
        features = gmm_inputs
        returns = features.mean().values
        cov_matrix = features.cov().values
        num_assets = features.shape[1]

        def negative_sharpe_ratio(weights):
            '''Calculates the negative Sharpe ratio.'''
            portfolio_returns = np.dot(features.values, weights)
            target_return = 0  # Set your target return or risk-free rate
            portfolio_std = np.std(portfolio_returns)

            # Avoid division by zero
            if portfolio_std == 0:
                return np.inf

            expected_portfolio_return = np.mean(portfolio_returns)
            sharpe_ratio = (expected_portfolio_return - target_return) / portfolio_std

            # Optionally, add L1 penalty to promote sparsity
            l1_penalty = 0.0 * np.sum(np.abs(weights))
            return -sharpe_ratio + l1_penalty

        # Constraints: sum of weights is 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess: equal weights
        initial_guess = np.array([1 / num_assets] * num_assets)

        # Optimize the portfolio
        result = minimize(
            negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 1000}
        )

        optimal_weights = result.x

        # Recalculate performance with optimal weights
        optimal_returns = np.dot(features.values, optimal_weights)
        optimal_return = np.mean(optimal_returns)
        optimal_volatility = np.std(optimal_returns)

        return optimal_weights, optimal_return, optimal_volatility

    def optimize_portfolio_covariance(self, data):
        '''Optimizes the portfolio weights using the covariance matrix.'''
        features = data.drop(columns=['Total_Profit'])
        returns = features.mean().values
        cov_matrix = features.cov().values
        num_assets = features.shape[1]

        def negative_sharpe_ratio(weights):
            '''Calculates the negative Sharpe ratio.'''
            portfolio_return = np.dot(weights, returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Avoid division by zero
            if portfolio_volatility == 0:
                return np.inf

            sharpe_ratio = (portfolio_return - 0) / portfolio_volatility  # target return is 0
            return -sharpe_ratio

        # Constraints: sum of weights is 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess: equal weights
        initial_guess = np.array([1 / num_assets] * num_assets)

        # Optimize the portfolio
        result = minimize(
            negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 1000}
        )

        optimal_weights = result.x

        # Recalculate performance with optimal weights
        portfolio_return = np.dot(optimal_weights, returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

        return optimal_weights, portfolio_return, portfolio_volatility

    def plot_profit_comparison(self, data, optimal_weights_xgboost, optimal_weights_cov):
        '''Plots the profit comparison between XGBoost and Covariance optimization.'''
        # Implement your plotting code here
        pass

if __name__ == '__main__':
    optimizer = Optimizer()
    finance_bro = FinanceBro()
    tickers = finance_bro.get_tickers()
    data = finance_bro.get_combined_data(tickers)

    # Preprocess data
    data = finance_bro.add_profit(data)

    # Optimize portfolio using XGBoost
    optimal_weights_xgboost, optimal_return_xgboost, optimal_volatility_xgboost = optimizer.optimize_portfolio_xgboost(data)
    print("XGBoost Optimization:")
    print(f"Optimal Weights: {optimal_weights_xgboost}")
    print(f"Expected Return: {optimal_return_xgboost}")
    print(f"Volatility: {optimal_volatility_xgboost}")

    # Optimize portfolio using Covariance Matrix
    optimal_weights_cov, optimal_return_cov, optimal_volatility_cov = optimizer.optimize_portfolio_covariance(data)
    print("\nCovariance Matrix Optimization:")
    print(f"Optimal Weights: {optimal_weights_cov}")
    print(f"Expected Return: {optimal_return_cov}")
    print(f"Volatility: {optimal_volatility_cov}")

    # Plot the comparison
    optimizer.plot_profit_comparison(data, optimal_weights_xgboost, optimal_weights_cov)