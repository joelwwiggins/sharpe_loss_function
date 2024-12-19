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
    epochs: int = 10

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
        print(f"Combined data shape: {combined_data.shape}")

        # Drop columns with 90% or more missing values
        data = combined_data.dropna(thresh=0.1 * len(combined_data), axis=1)
        print(f"Data shape after dropna: {data.shape}")

        # Ensure we have enough data and convert to a shape divisible by time_steps
        # For simplicity, just trim the data
        data = data.iloc[:-(len(data) % self.time_steps)]
        print(f"Data shape after trimming: {data.shape}")

        # Convert to numpy array
        data = data.values
        print(f"Data shape after converting to NumPy array: {data.shape}")

        # Here we assume we have enough features (num_features=8). 
        # If we don't have 8 features, you'd need to select or engineer features.
        # For demonstration, let's just pick the first 8 columns if available:
        if data.shape[1] < self.num_features:
            raise ValueError("Not enough features to match num_features setting.")
        data = data[:, :self.num_features]

        # Reshape the data: 
        # The total number of samples after reshaping is (num_samples * time_steps)
        samples = data.shape[0] // self.time_steps
        data = data.reshape(samples, self.time_steps, self.num_features)
        print(f"Data shape after reshaping: {data.shape}")

        self.train_size = int(0.8 * len(data))
        self.val_size = len(data) - self.train_size

        X_train = data[:self.train_size]
        X_val = data[self.train_size:self.train_size + self.val_size]
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")

        # Normalize data
        X_min = X_train.min()
        X_max = X_train.max()
        self.X_train = (X_train - X_min) / (X_max - X_min)
        self.X_val = (X_val - X_min) / (X_max - X_min)

        # Add channels dimension
        self.X_train = self.X_train.reshape((-1, self.time_steps, self.num_features, self.channels))
        self.X_val = self.X_val.reshape((-1, self.time_steps, self.num_features, self.channels))

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

        # Decoder using UpSampling2D for better shape control
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
            batch_size=self.batch_size
        )

    def evaluate(self):
        # Predict reconstructed values for both training and validation sets
        X_train_reconstructed = self.model.predict(self.X_train)
        X_val_reconstructed = self.model.predict(self.X_val)

        # Compute reconstruction errors for both sets
        train_reconstruction_errors = np.mean(np.square(self.X_train - X_train_reconstructed), axis=(1,2,3))
        val_reconstruction_errors = np.mean(np.square(self.X_val - X_val_reconstructed), axis=(1,2,3))

        # Combine reconstruction errors
        reconstruction_errors = np.concatenate([train_reconstruction_errors, val_reconstruction_errors])

        print("Reconstruction error stats:")
        print(f"Mean: {reconstruction_errors.mean():.4f}")
        print(f"Std:  {reconstruction_errors.std():.4f}")
        print(f"Max:  {reconstruction_errors.max():.4f}")

        threshold = reconstruction_errors.mean() + 3 * reconstruction_errors.std()
        anomalies = reconstruction_errors > threshold

        print(f"Anomaly threshold: {threshold:.4f}")
        print(f"Number of anomalies in combined set: {np.sum(anomalies)}")

        

    def plot_reconstruction_errors(self, reconstruction_errors, threshold):
            # Plot reconstruction errors
            plt.figure(figsize=(12, 6))
            plt.plot(reconstruction_errors, label='Reconstruction Errors')
            plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
            plt.legend()
            plt.title('Reconstruction Errors')
            plt.xlabel('Sample Index')
            plt.ylabel('Reconstruction Error')
            plt.show()



if __name__ == '__main__':
    autoencoder = ConvolutionalAutoencoder()
    autoencoder.load_data()
    autoencoder.build_model()
    autoencoder.train()
    autoencoder.evaluate()
    print("Script finished successfully.")