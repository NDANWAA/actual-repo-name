import time
import json
import logging
import websocket
import pandas as pd
import numpy as np
from collections import deque
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv  # Parallel environments
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import threading
from datetime import datetime
from flask import Flask, jsonify, request
import os
import signal
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration
API_TOKEN = "os.getenv("API_DERIVDEMO")"
APP_ID = "os.getenv("APP_IDDEMO")"
SYMBOL = "1HZ10V"
STAKE = 10
CURRENCY = "USD"
WINDOW_SIZE = 300  # 5-minute window at 1 tick per second
BUFFER_SIZE = 100000

# Setup Flask app
flask_app = Flask(__name__)

@flask_app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"status": "running"})

@flask_app.route('/update_symbol', methods=['POST'])
def update_symbol():
    data = request.get_json()
    symbol = data.get('symbol')
    return jsonify({"message": f"Symbol updated to {symbol}."})

# Setup logging
logger = logging.getLogger("trading_bot_logger")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

# Global variables
tick_data = deque(maxlen=WINDOW_SIZE)
ohlc_data_m1 = pd.DataFrame(columns=["close"])  # Reduced observation space to only 'close' prices
ohlc_data_m5 = pd.DataFrame(columns=["close"])
ohlc_data_m15 = pd.DataFrame(columns=["close"])
tick_counter = 0
trades = []
performance_data = []
concept_dict = {}  # Concept dictionary to store definitions
lock = threading.Lock()

# TensorBoard callback for monitoring
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose=verbose)

    def _on_step(self) -> bool:
        self.logger.record('epsilon', self.model.exploration_rate)  # Log epsilon value
        return True

def query_rasa_for_concept(concept):
    response = requests.post("http://localhost:5005/webhooks/rest/webhook", json={
        "sender": "trading_bot",
        "message": f"Explain {concept}"
    })
    if response.ok:
        return response.json()[0]['text']
    return "No definition found."

def get_concept_definition(concept):
    if concept not in concept_dict:
        definition = query_rasa_for_concept(concept)
        concept_dict[concept] = definition
    return concept_dict[concept]

def process_tick_data_multi_timeframes(tick_data):
    with lock:
        df = pd.DataFrame(list(tick_data), columns=["price"])
    df['datetime'] = pd.to_datetime(df.index, unit='s')
    df.set_index('datetime', inplace=True)
    ohlc_m1 = df['price'].resample('1Min').ohlc()['close']
    ohlc_m5 = df['price'].resample('5Min').ohlc()['close']
    ohlc_m15 = df['price'].resample('15Min').ohlc()['close']
    return ohlc_m1.dropna(), ohlc_m5.dropna(), ohlc_m15.dropna()

def on_message(ws, message):
    global tick_data, ohlc_data_m1, ohlc_data_m5, ohlc_data_m15, tick_counter, trades, performance_data

    try:
        data = json.loads(message)
        if 'tick' in data:
            price = data['tick']['quote']
            with lock:
                tick_data.append(price)
                tick_counter += 1
            logger.info(f"Received tick: {price}")

            if tick_counter >= 10:
                ohlc_m1, ohlc_m5, ohlc_m15 = process_tick_data_multi_timeframes(tick_data)
                with lock:
                    tick_data.clear()
                    tick_counter = 0
                logger.info("OHLC data updated for M1, M5, M15 time frames")

                action = env.get_attr('step')[0]
                logger.info(f"DQN action: {action}")

                if action != 0:
                    trade_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    balance = env.get_attr('balance')[0]
                    with lock:
                        trades.append({'time': trade_time, 'action': action})
                        performance_data.append({'time': trade_time, 'balance': balance, 'action': action})
                    logger.info(f"Trade executed: {action} at {trade_time}")

    except (KeyError, ValueError) as e:
        logger.error(f"Error processing response: {e}")

def on_open(ws):
    logger.info("WebSocket connection opened")
    subscribe_request = json.dumps({
        "ticks": SYMBOL,
        "subscribe": 1
    })
    ws.send(subscribe_request)
    logger.info(f"Sent subscription request: {subscribe_request}")

def generate_trade_report():
    with lock:
        if trades:
            df_trades = pd.DataFrame(trades)
            report_filename = f"trade_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(report_filename, index=False)
            logger.info(f"Trade report saved: {report_filename}")
        else:
            logger.info("No trades to report.")

def monte_carlo_simulation(current_price, num_simulations=1000, time_horizon=10, mu=0.001, sigma=0.02):
    simulations = []
    for _ in range(num_simulations):
        prices = [current_price]
        for _ in range(time_horizon):
            price = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * np.random.normal())
            prices.append(price)
        simulations.append(prices)
    return np.array(simulations)

def add_monte_carlo_to_observation(env, ohlc_data):
    current_price = ohlc_data['close'].iloc[-1]
    monte_carlo_predictions = monte_carlo_simulation(current_price)
    expected_price = np.mean(monte_carlo_predictions[:, -1])  # Mean price at the end of time horizon
    env.monte_carlo_feature = expected_price

class SyntheticMarketEnv(gym.Env):
    def __init__(self, ohlc_data_m1, ohlc_data_m5, ohlc_data_m15):
        super(SyntheticMarketEnv, self).__init__()
        self.ohlc_data_m1 = ohlc_data_m1
        self.ohlc_data_m5 = ohlc_data_m5
        self.ohlc_data_m15 = ohlc_data_m15
        self.current_step = 0
        self.balance = 10000
        self.monte_carlo_feature = 0
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy CALL, 2: Buy PUT
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)  # Reduced to 2 features

    def step(self, action):
        reward = 0
        done = False
        self.current_step += 1

        # Simulate the impact of the action (buy/sell) on balance
        if action == 1:  # Buy CALL
            reward = random.gauss(1, 0.1)  # RNG, can be replaced with Monte Carlo sim
            self.balance += reward
        elif action == 2:  # Buy PUT
            reward = random.gauss(1, 0.1)
            self.balance += reward

        if self.current_step >= len(self.ohlc_data_m1):
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def _next_observation(self):
        add_monte_carlo_to_observation(self, self.ohlc_data_m1)
        return np.array([
            self.ohlc_data_m1.iloc[self.current_step]['close'],
            self.monte_carlo_feature
        ])

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        return self._next_observation()

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}")

    def get_attr(self, name):
        if name == 'balance':
            return [self.balance]
        if name == 'step':
            return [self.current_step]
        return None

def run_bot():
    global ohlc_data_m1, ohlc_data_m5, ohlc_data_m15, env, dqn_agent

    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message)
    
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    
    logger.info("Waiting for sufficient OHLC data...")

    while ohlc_data_m1.empty or ohlc_data_m5.empty or ohlc_data_m15.empty:
        time.sleep(1)

    env = SubprocVecEnv([lambda: SyntheticMarketEnv(ohlc_data_m1, ohlc_data_m5, ohlc_data_m15)] * 4)  # 4 parallel environments
    dqn_agent = DQN(
        "MlpPolicy", env, 
        learning_rate=0.001,  # Increased learning rate
        exploration_fraction=0.1,  # Reduced exploration phase
        verbose=1, tensorboard_log="./dqn_tensorboard/",
        device="cuda"  # Use GPU if available
    )

    logger.info("Starting training...")
    dqn_agent.learn(total_timesteps=10000, callback=TensorboardCallback())

    dqn_agent.save("trading_bot_model")
    logger.info("Model saved")

    report_interval = 3600
    while True:
        time.sleep(report_interval)
        generate_trade_report()

def graceful_shutdown(signum, frame):
    logger.info("Received shutdown signal. Closing bot gracefully...")
    os._exit(0)

def main():
    logger.info("Starting main function")

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    bot_thread.join()  # Wait for bot thread to finish

if __name__ == "__main__":
    main()
