import logging
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Configuration
STAKE = 10
CURRENCY = "USD"
RSI_PERIOD = 14
WINDOW_SIZE = 300  # 5-minute window at 1 tick per second

# Setup logging
logger = logging.getLogger("trading_bot_logger")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

# Trading environment
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = None  # Can be 'long', 'short', or None
        self.position_value = 0
        
        # Define observation space (candlestick data and account status)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns) + 2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

    def reset(self, **kwargs):  # Modify to accept any additional arguments
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.position_value = 0
        return self._next_observation()

    def _next_observation(self):
        obs = np.array(self.data.iloc[self.current_step].tolist())
        # Include balance and position value in the observation
        return np.append(obs, [self.balance, self.position_value])

    def step(self, action):
        # Execute the trade
        current_price = self.data['Close'].iloc[self.current_step]
        if action == 1:  # Buy
            if self.position is None:
                self.position = 'long'
                self.position_value = self.balance
                self.balance = 0
        elif action == 2:  # Sell
            if self.position == 'long':
                self.balance += self.position_value * (current_price / self.data['Close'].iloc[self.current_step - 1])
                self.position = None
                self.position_value = 0

        # Calculate reward (change in balance)
        reward = self.balance + self.position_value - self.initial_balance

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        # Optional: visualize the current state (e.g., plot the current candlestick chart)
        pass

def create_env():
    # Load the simulated market data
    data = pd.read_csv('D:\\trading_bot_env\\data\\processed_ohlc.csv', parse_dates=True, index_col='Timestamp')
    return TradingEnv(data)

def main():
    env = DummyVecEnv([create_env for _ in range(1)])  # Use DummyVecEnv for simplicity
    dqn_agent = DQN("MlpPolicy", env, verbose=1)

    # Train the bot
    dqn_agent.learn(total_timesteps=10000)

    # Save the model
    dqn_agent.save("trading_bot_model")

if __name__ == "__main__":
    main()
