from stable_baselines3 import DQN
from trading_env import TradingEnv
import pandas as pd

# Load the market data
df = pd.read_csv('your_data.csv')

# Create the environment
env = TradingEnv(df)

# Initialize the DQN agent
model = DQN('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("trading_bot")

# To continue training, load the model and continue learning
# model = DQN.load("trading_bot", env=env)
