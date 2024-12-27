from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from get_data import FinanceBro
import matplotlib.pyplot as plt

@dataclass
class ConvolutionalAutoencoder:
    time_steps: int = 32        # Adjusted to 32 for clean up/down sampling
    num_features: int = 8       # Adjusted to 8 for cleaner reconstruction
    channels: int = 1
    batch_size: int = 64
    epochs: int = 1000

    def __post_init__(self):
        self.model = None
        self.X_train = None
        self.X_val = None
        self.train_size = None
        self.val_size = None

    def load_data(self):
        finance_bro = FinanceBro()
        tickers = finance_bro.get_tickers()
        combined_data = finance_bro.get_combined_data(tickers)
        print("Loaded combined data.")
        print(f"Combined data shape: {combined_data.shape}")

        # Drop columns with 90% or more missing values
        data = combined_data.dropna(thresh=0.9 * len(combined_data), axis=1)
        print(f"Data shape after dropna: {data.shape}")

        data = data.pct_change().dropna()

        # Ensure data length is divisible by time_steps
        data = data[:-(len(data) % self.time_steps)]
        print(f"Data shape after trimming: {data.shape}")

        # Convert DataFrame to NumPy array
        dates = data.index.to_numpy()  # Convert DatetimeIndex to NumPy array
        data = data.values
        print(f"Data shape after converting to NumPy array: {data.shape}")

        # Check that we have at least num_features
        if data.shape[1] < self.num_features:
            raise ValueError("Not enough features to match num_features setting.")

        # Select the first num_features columns
        data = data[:, :self.num_features]

        # Reshape data into (samples, time_steps, num_features)
        samples = data.shape[0] // self.time_steps
        data = data.reshape(samples, self.time_steps, self.num_features)
        dates = dates[:samples * self.time_steps].reshape(samples, self.time_steps)
        print(f"Data shape after reshaping: {data.shape}")

        # Split into train and validation sets
        self.train_size = int(0.8 * samples)
        self.val_size = samples - self.train_size

        X_train = data[:self.train_size]
        X_val = data[self.train_size:self.train_size + self.val_size]
        self.train_dates = dates[:self.train_size, 0]  # One date per sample (e.g. the first date)
        self.val_dates = dates[self.train_size:self.train_size + self.val_size, 0]
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")

        # Compute min and max from X_train for normalization
        X_min = X_train.min(axis=(0, 1), keepdims=True)
        X_max = X_train.max(axis=(0, 1), keepdims=True)

        # Avoid division by zero
        denominator = X_max - X_min
        denominator[denominator == 0] = 1

        # Normalize data
        X_train_norm = (X_train - X_min) / denominator
        X_val_norm = (X_val - X_min) / denominator

        # Handle NaNs or Infs
        if np.isnan(X_train_norm).any() or np.isinf(X_train_norm).any():
            print("Found NaNs or Infs in X_train after normalization.")
            X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(X_val_norm).any() or np.isinf(X_val_norm).any():
            print("Found NaNs or Infs in X_val after normalization.")
            X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Add channel dimension
        self.X_train = X_train_norm.reshape((-1, self.time_steps, self.num_features, self.channels))
        self.X_val = X_val_norm.reshape((-1, self.time_steps, self.num_features, self.channels))

        print("Final X_train shape:", self.X_train.shape)
        print("Final X_val shape:", self.X_val.shape)

    def build_model(self):
        input_layer = tf.keras.Input(shape=(self.time_steps, self.num_features, self.channels))

        # Encoder
        x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(input_layer)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)  # from (32,8) to (16,4)
        x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)  # from (16,4) to (8,2)
        encoded = x

        # Decoder
        x = layers.UpSampling2D((2,2))(encoded) # (8,2) -> (16,4)
        x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = layers.UpSampling2D((2,2))(x)       # (16,4) -> (32,8)
        x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)

        # Final layer
        decoded = layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')(x)

        self.model = models.Model(inputs=input_layer, outputs=decoded)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

    def train(self):
        self.model.fit(
            self.X_train, self.X_train,
            validation_data=(self.X_val, self.X_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False
        )
        print("Training completed.")

    def evaluate(self):
        # Combine training and validation sets
        X_combined = np.concatenate([self.X_train, self.X_val])
        dates_combined = np.concatenate([self.train_dates, self.val_dates])

        # Predict reconstructed values for the combined dataset
        X_combined_reconstructed = self.model.predict(X_combined)

        # Compute reconstruction errors for the combined dataset
        reconstruction_errors = np.mean(np.square(X_combined - X_combined_reconstructed), axis=(1,2,3))

        print("Reconstruction error stats:")
        print(f"Mean: {reconstruction_errors.mean():.4f}")
        print(f"Std:  {reconstruction_errors.std():.4f}")
        print(f"Max:  {reconstruction_errors.max():.4f}")

        threshold = reconstruction_errors.mean() + 3 * reconstruction_errors.std()
        anomalies = reconstruction_errors > threshold

        print(f"Anomaly threshold: {threshold:.4f}")
        print(f"Number of anomalies in combined set: {np.sum(anomalies)}")

        # Plot reconstruction errors using aggregated values
        self.plot_reconstruction_errors(dates_combined, reconstruction_errors, threshold, X_combined, X_combined_reconstructed)
        print("Evaluation completed.")

    def plot_reconstruction_errors(self, dates, reconstruction_errors, threshold, X_combined, X_combined_reconstructed):
        plt.figure(figsize=(18, 12))

        # Plot reconstruction errors
        plt.subplot(3, 1, 1)
        plt.plot(dates, reconstruction_errors, label='Reconstruction Errors')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
        plt.legend()
        plt.title('Reconstruction Errors')
        plt.xlabel('Date')
        plt.ylabel('Reconstruction Error')

        # Aggregate the original and reconstructed data per sample
        X_combined_mean = X_combined.mean(axis=(1, 2, 3))
        X_combined_reconstructed_mean = X_combined_reconstructed.mean(axis=(1, 2, 3))

        # Plot original data (mean per sample)
        plt.subplot(3, 1, 2)
        plt.plot(dates, X_combined_mean, label='Original Data (Mean per Sample)')
        plt.legend()
        plt.title('Original Data')
        plt.xlabel('Date')
        plt.ylabel('Mean Value')

        # Plot reconstructed data (mean per sample)
        plt.subplot(3, 1, 3)
        plt.plot(dates, X_combined_reconstructed_mean, label='Reconstructed Data (Mean per Sample)')
        plt.legend()
        plt.title('Reconstructed Data')
        plt.xlabel('Date')
        plt.ylabel('Mean Value')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    autoencoder = ConvolutionalAutoencoder()
    autoencoder.load_data()
    autoencoder.build_model()
    autoencoder.train()
    autoencoder.evaluate()
    print("Script finished successfully.")
