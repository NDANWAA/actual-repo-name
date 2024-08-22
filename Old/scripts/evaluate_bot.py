from stable_baselines3 import DQN
from trading_env import TradingEnv
import pandas as pd

# Load the market data
df = pd.read_csv('your_data.csv')

# Create the environment
env = TradingEnv(df)

# Load the trained model
model = DQN.load("trading_bot")

obs = env.reset()
for _ in range(len(df) - env.window_size):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
