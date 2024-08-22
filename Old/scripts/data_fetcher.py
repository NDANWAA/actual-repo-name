import time
import csv
import os
import json
import websocket
import pandas as pd

API_TOKEN = "bkuO5aZaGMMuAmA"
APP_ID = "63268"
CSV_FILE_PATH = r"D:\trading_bot_env\data\live_data.csv"

def on_message(ws, message):
    try:
        data = json.loads(message)
        print(data)  # Print the entire response for debugging

        if 'tick' in data:
            quote = data['tick']['quote']
            timestamp = data['tick']['epoch']
            symbol = data['tick']['symbol']
            print(f"Saving tick data: {quote} at {timestamp} for {symbol}")  # Debug print for saving process
            save_to_csv(timestamp, quote, symbol)
        else:
            print("Error: 'tick' not found in response.")
    except (KeyError, ValueError) as e:
        print(f"Error processing response: {e}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    # Sending subscription request for ticks
    subscribe_request = json.dumps({
        "ticks": "1HZ10V",
        "subscribe": 1
    })
    ws.send(subscribe_request)

def save_to_csv(timestamp, quote, symbol):
    try:
        file_exists = os.path.isfile(CSV_FILE_PATH)
        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Quote", "Symbol"])
            writer.writerow([timestamp, quote, symbol])
            file.flush()  # Ensure data is written immediately
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main():
    websocket_url = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    ws = websocket.WebSocketApp(websocket_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

if __name__ == "__main__":
    main()
