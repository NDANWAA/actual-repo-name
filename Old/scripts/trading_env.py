import gym
import numpy as np
import pandas as pd
import time
import os
from gym import spaces
from datetime import datetime

# Define file paths
LIVE_DATA_PATH = r"D:\trading_bot_env\data\live_data.csv"
PROCESSED_OHLC_PATH = r"D:\trading_bot_env\data\processed_ohlc.csv"

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        print("Script started")
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = None  # Can be 'long', 'short', or None
        self.position_value = 0
        
        # Define observation space (candlestick data and account status)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns) + 2,), dtype=np.float32)

        # Define action space (0: Hold, 1: Buy, 2: Sell)
        self.action_space = spaces.Discrete(3)

    def reset(self):
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

def convert_to_ohlc(df, interval='5Min'):
    print(f"Converting to OHLC with DataFrame columns: {df.columns}")
    df.set_index('Timestamp', inplace=True)
    ohlc_df = df['Quote'].resample(interval).ohlc()
    
    if not ohlc_df.empty:
        ohlc_df.to_csv(PROCESSED_OHLC_PATH, mode='a', header=not os.path.exists(PROCESSED_OHLC_PATH))
    
    return ohlc_df

def process_and_visualize():
    last_processed_time = None

    while True:
        if os.path.exists(LIVE_DATA_PATH):
            df = pd.read_csv(LIVE_DATA_PATH, header=0, names=["Timestamp", "Quote", "Symbol"])

            if 'Timestamp' not in df.columns:
                print("Error: 'Timestamp' column is missing in the DataFrame.")
                continue

            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
            except Exception as e:
                print(f"Error converting 'Timestamp' to datetime: {e}")
                continue
            
            if not df.empty:
                if last_processed_time:
                    df = df[df['Timestamp'] > last_processed_time]
                last_processed_time = df['Timestamp'].max()
                
                try:
                    ohlc_df = convert_to_ohlc(df)
                    print(ohlc_df)
                except Exception as e:
                    print(f"Error converting to OHLC: {e}")
                
                # Reset index without dropping the 'Timestamp' column
                df.reset_index(inplace=True)
                
                if 'Timestamp' in df.columns:
                    df = df[df['Timestamp'] > last_processed_time]
                else:
                    print("Error: 'Timestamp' column is missing after reset_index.")
                    continue

                df.to_csv(LIVE_DATA_PATH, mode='w', header=False, index=False)
                print(f"Processed data saved, remaining data: {len(df)} rows.")
                
        time.sleep(30)

if __name__ == "__main__":
    # Start data processing
    process_and_visualize()

    # Load processed OHLC data into the environment
    if os.path.exists(PROCESSED_OHLC_PATH):
        data = pd.read_csv(PROCESSED_OHLC_PATH)
        env = TradingEnv(data)
        obs = env.reset()

        for _ in range(len(data)):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            if done:
                break
