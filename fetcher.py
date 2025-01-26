import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import logging
from typing import List
from data_utils import prepare_data
from config import exchange_config

async def fetch_markets(exchange):
    return await exchange.load_markets()

async def fetch_ticker_volume(exchange, symbol):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        return symbol, ticker['quoteVolume']
    except Exception as e:
        logging.error(f"Error fetching ticker volume for {symbol}: {e}")
        return symbol, None

async def get_top_symbols(exchange, symbols, top_n=120):
    tasks = [fetch_ticker_volume(exchange, symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    symbol_volumes = [(symbol, volume) for symbol, volume in results
                      if volume is not None]
    symbol_volumes.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in symbol_volumes[:top_n]]

async def fetch_min_amounts(exchange, top_symbols, markets):
    min_amounts = {}
    for symbol in top_symbols:
        market = markets.get(symbol)
        if market and 'limits' in market and 'amount' in market['limits'] and 'min' in market['limits']['amount']:
            min_amounts[symbol] = market['limits']['amount']['min']
        else:
            min_amounts[symbol] = 1
    return min_amounts

async def get_data_async(exchange, symbol, timeframe='15m', limit=500):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol,
                                           timeframe=timeframe,
                                           limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None
