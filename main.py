import sys
import asyncio
import ccxt.async_support as ccxt_async
import logging
import time
from config import exchange_config
from logging_config import *
from fetcher import fetch_markets, get_top_symbols, fetch_min_amounts, get_data_async
from model_loader import load_lstm_model_func, load_random_forest_model_func
from trainer import train_lstm_model, train_random_forest_model_wrapper
from predictor import predict_signal_ensemble
from trade_manager import get_real_balance, manage_position
from collections import deque

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    async_exchange = ccxt_async.bybit(exchange_config)
    try:
        markets = await fetch_markets(async_exchange)
        excluded_symbols = ['UNFIUSDT', 'TIAUSDT']
        all_symbols = [
            market['symbol'] for market in markets.values()
            if market.get('quote') == 'USDT' and market.get('active') and market.get('type') == 'swap' and market['symbol'] not in excluded_symbols
        ]
        top_symbols = await get_top_symbols(async_exchange, all_symbols)
        top_symbols = [
            symbol for symbol in top_symbols if symbol not in excluded_symbols
        ]
        min_amounts = await fetch_min_amounts(async_exchange, top_symbols, markets)
        lstm_model, lstm_scaler = await asyncio.to_thread(load_lstm_model_func)
        rf_model, rf_scaler = await asyncio.to_thread(load_random_forest_model_func)
        if not lstm_model or not rf_model:
            lstm_model, lstm_scaler = await train_lstm_model(async_exchange, top_symbols)
            if lstm_model and lstm_scaler:
                rf_model, rf_scaler = await train_random_forest_model_wrapper(top_symbols, async_exchange)
            else:
                logging.critical("Failed to load or train models. Exiting program.")
                return
        trades_deque = deque(maxlen=1000)

        async def trade_signals():
            while True:
                usdt_balance = await get_real_balance(async_exchange)
                if usdt_balance is None:
                    logging.warning("Failed to get USDT balance. Retrying in 5 seconds.")
                    await asyncio.sleep(5)
                    continue
                for symbol in top_symbols:
                    try:
                        df = await get_data_async(async_exchange, symbol, timeframe='15m')
                        if df is not None:
                            signal = predict_signal_ensemble(df, lstm_model, lstm_scaler,
                                                             rf_model, rf_scaler)
                            if signal is not None:
                                await manage_position(async_exchange, symbol, signal,
                                                      usdt_balance, min_amounts, lstm_model,
                                                      lstm_scaler, rf_model, rf_scaler)
                    except Exception as e:
                        logging.error(f"Error processing signal for {symbol}: {e}")
                await asyncio.sleep(60)

        await asyncio.gather(trade_signals())
    except KeyboardInterrupt:
        logging.info("Interrupt signal received. Shutting down...")
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        await async_exchange.close()
        logging.info("Program terminated")

if __name__ == "__main__":
    asyncio.run(main())
