import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv
import os


def get_market_data(symbol):
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, "1 Jan, 2017")
    if len(klines) > 0:
        trades = pd.DataFrame(klines)
        trades = trades.iloc[:, :6]
        trades.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        trades["Date"] = pd.to_datetime(trades["Date"], unit="ms")
        trades = trades.set_index("Date")
        for col in trades.columns[:]:
            trades[col] = pd.to_numeric(trades[col])
        trades['Asset_name'] = symbol
        return trades
    else:
        return None



if __name__ == '__main__' :

    load_dotenv()  # take environment variables from .env.
    api_secret_key = os.getenv('BINANCE_API_SECRET_KEY')
    api_public_key = os.getenv('BINANCE_API_KEY')

    client = Client(api_public_key, api_secret_key)

    coinList = client.get_all_tickers()
    assets = [coin["symbol"] for coin in coinList if coin["symbol"].endswith("USDT")]




    for i in range(len(assets)):
        print(f"Downloading {assets[i]} data {round((i / len(assets)) * 10000) / 100}% done")
        data = get_market_data(assets[i])
        if type(data) == pd.DataFrame:
            print('OK')
            data.to_csv(f'data/raw_data_4_hour/{assets[i]}.csv')

