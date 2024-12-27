from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import numpy as np
from model import ConvolutionalAutoencoder
import gym

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Initialize and build the autoencoder model
        self.autoencoder = ConvolutionalAutoencoder()
        self.autoencoder.load_data()
        self.autoencoder.build_model()
        
        # Example: Simple feed-forward
        # The observation space shape = (3,) from the environment above
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Use the autoencoder to extract features
        observations_np = observations.detach().cpu().numpy()
        features = self.autoencoder.model.predict(observations_np)
        features_th = th.tensor(features, dtype=th.float32).to(observations.device)
        return self.net(features_th)

# Then create a custom policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128)
        )

# Use the custom policy
env = gym.make('CartPole-v1')  # Replace with your environment here
model = PPO(
    CustomPolicy, 
    env, 
    verbose=1
)
model.learn(total_timesteps=10_000)