import pandas as pd
import numpy as np
import datetime as dt

from supervised_ml_btc_trading_strategy.preprocessing import create_train_test_data

def backtest(raw_data, initial_balance=10000, trade_buy_fraction=0.1, trade_sold_fraction=0.5,trading_fee=0.001, **params):

    data, signal, _, _ = create_train_test_data(raw_data, **params)
    balance = initial_balance
    position = 0
    trade_log = []

    for i in range(1, len(data)):
        price = data["Close"].iloc[i]
        trade_size = trade_buy_fraction*balance
        if signal["signal"].iloc[i] == 1 and balance >= trade_size:  # Buy
            quantity_bought = (trade_size / price) * (1 - trading_fee)
            position += quantity_bought
            balance -= quantity_bought * price
            trade_log.append(("BUY", data["Date"].iloc[i], price, quantity_bought))

        elif signal["signal"].iloc[i] == -1 and position > 0:  # Sell
            trade_size = trade_sold_fraction*position
            quantity_sold = (trade_size / price) * (1 - trading_fee)
            balance += quantity_sold * price
            position -= quantity_sold
            trade_log.append(("SELL", data["Date"].iloc[i], price, quantity_sold))

    final_balance = balance + (position * data["Close"].iloc[-1])
    return final_balance, trade_log

