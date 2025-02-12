import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv
import os
from constants import COINS_LIST


def get_market_data(symbol, timeframe=Client.KLINE_INTERVAL_4HOUR):
    klines = client.get_historical_klines(symbol, timeframe, "1 Jan, 2017")
    if len(klines) > 0:
        trades = pd.DataFrame(klines)
        trades = trades.iloc[:, :6]
        trades.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        trades["Date"] = pd.to_datetime(trades["Date"], unit="ms")
        trades = trades.set_index("Date")
        for col in trades.columns[:]:
            trades[col] = pd.to_numeric(trades[col])
        trades['Asset_name'] = symbol
        trades_micro = trades.copy()
        trades_micro['Asset_name'] = 'µ' + symbol
        trades_micro = trades_micro[["Open", "High", "Low", "Close"]].apply(lambda x: x*10^(-6))
        return trades, trades_micro
    else:
        return None

if __name__ == '__main__' :

    load_dotenv()  # take environment variables from .env.
    api_secret_key = os.getenv('BINANCE_API_SECRET_KEY')
    api_public_key = os.getenv('BINANCE_API_KEY')

    client = Client(api_public_key, api_secret_key)
    # coinList = client.get_all_tickers()
    # assets = [coin["symbol"] for coin in COINS_LIST if coin["symbol"].endswith("USDT")]

    for i in range(len(COINS_LIST)):
        print(f"Downloading {COINS_LIST[i]} data {round((i / len(COINS_LIST)) * 10000) / 100}% done")
        data, micro_data = get_market_data(COINS_LIST[i])
        if type(data) == pd.DataFrame and type(micro_data) == pd.DataFrame:
            print('OK')
            data.to_csv(f'data/raw_data_4_hour/{COINS_LIST[i]}.csv')
            data.to_csv(f'data/raw_data_4_hour/µ{COINS_LIST[i]}.csv')