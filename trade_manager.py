import asyncio
import json
import logging
import time
import aiofiles
from fetcher import get_data_async
from predictor import predict_signal_ensemble

last_trade_time = {}
lock = asyncio.Lock()

async def get_real_balance(exchange):
    try:
        balance = await exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        if usdt_balance == 0:
            logging.warning("USDT balance is zero or not found.")
        return usdt_balance
    except Exception as e:
        logging.error(f"Error fetching real balance: {e}")
        return None

async def calculate_position_size(exchange,
                                  symbol,
                                  usdt_balance,
                                  risk_percentage=0.3,
                                  min_amount=0):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker.get('last', None)
        if current_price is None or not isinstance(current_price, (int, float)):
            logging.error(f"Current price for {symbol} is missing")
            return None
        risk_amount = usdt_balance * risk_percentage
        position_size = risk_amount / current_price
        position_size = float(exchange.amount_to_precision(symbol, position_size))
        logging.info(
            f"Calculated position size for {symbol}: {position_size} contracts (Risk Amount: {risk_amount} USDT)"
        )
        if position_size < min_amount:
            logging.warning(
                f"Position size {position_size} < min_amount {min_amount} for {symbol}. Setting to min_amount."
            )
            position_size = min_amount
        return position_size
    except Exception as e:
        logging.error(f"Error calculating position size for {symbol}: {e}")
        return None

async def log_trade(trade):
    try:
        async with aiofiles.open('trades_log.json', 'a') as f:
            await f.write(json.dumps(trade) + '\n')
    except Exception as e:
        logging.error(f"Error logging trade: {e}")

async def manage_position(exchange, symbol, signal, usdt_balance, min_amounts,
                         lstm_model, lstm_scaler, rf_model, rf_scaler):
    TRADE_COOLDOWN = 60
    current_time = time.time()
    async with lock:
        last_time = last_trade_time.get(symbol, 0)
        if current_time - last_time < TRADE_COOLDOWN:
            return
    try:
        position_size = await calculate_position_size(exchange,
                                                       symbol,
                                                       usdt_balance,
                                                       min_amount=min_amounts.get(
                                                           symbol, 0.1))
        if position_size is None or position_size < min_amounts.get(symbol, 0.1):
            return
        ticker = await exchange.fetch_ticker(symbol)
        price = ticker.get('last', None)
        if price is None or not isinstance(price, (int, float)):
            logging.error(f"Current price for {symbol} is missing")
            return
        if usdt_balance < (position_size * price):
            logging.warning(
                f"Insufficient USDT balance for {symbol}. Required: {position_size * price}, Available: {usdt_balance}"
            )
            return
        df = await get_data_async(exchange, symbol, timeframe='15m', limit=100)
        if df is None:
            return
        signal_pred = signal
        if signal_pred == 1:
            try:
                order = await exchange.create_market_buy_order(symbol, position_size)
                if order.get('status') != 'closed' or order.get('average') is None:
                    logging.error(
                        f"Order not closed for {symbol}. Order details: {order}")
                    return
                entry_price = order.get('average', None)
                if entry_price is None:
                    entry_price = await fetch_average_price(exchange, symbol)
                if entry_price is not None:
                    trade = {
                        'symbol': symbol,
                        'action': 'buy',
                        'amount': position_size,
                        'price': entry_price,
                        'timestamp': current_time
                    }
                    await log_trade(trade)
                    logging.info(
                        f"Opened long position for {symbol} at price {entry_price}")
            except Exception as e:
                logging.error(f"Error opening long position for {symbol}: {e}")
        elif signal_pred == 0:
            try:
                order = await exchange.create_market_sell_order(symbol, position_size)
                if order.get('status') != 'closed' or order.get('average') is None:
                    logging.error(
                        f"Order not closed for {symbol}. Order details: {order}")
                    return
                entry_price = order.get('average', None)
                if entry_price is None:
                    entry_price = await fetch_average_price(exchange, symbol)
                if entry_price is not None:
                    trade = {
                        'symbol': symbol,
                        'action': 'sell',
                        'amount': position_size,
                        'price': entry_price,
                        'timestamp': current_time
                    }
                    await log_trade(trade)
                    logging.info(
                        f"Opened short position for {symbol} at price {entry_price}")
            except Exception as e:
                logging.error(f"Error opening short position for {symbol}: {e}")
        async with lock:
            last_trade_time[symbol] = current_time
    except Exception as e:
        logging.error(f"Error managing position for {symbol}: {e}")

async def fetch_average_price(exchange, symbol):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        average_price = ticker.get('average', None)
        if average_price is not None and isinstance(average_price, (int, float)):
            return average_price
        last_price = ticker.get('last', None)
        return last_price if isinstance(last_price, (int, float)) else None
    except Exception as e:
        logging.error(f"Error fetching average price for {symbol}: {e}")
        return None
