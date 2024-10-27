import sys
import asyncio
import matplotlib.pyplot as plt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
import logging
import os
import json
import joblib
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from ta import momentum, trend, volatility
from ta.trend import IchimokuIndicator
from ta.volume import VolumeWeightedAveragePrice
from keras_tuner import HyperModel, RandomSearch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import aiofiles
from collections import deque

if sys.platform.startswith('win'):
  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('trading_bot_derivatives.log'),
                        logging.StreamHandler()
                    ])

API_KEY = "IiF****55m735****G"
API_SECRET = "nV****hR65TTKh71L****6dZWyU7YjWxdXlb"

exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
        'recvWindow': 10000
    },
    'timeout': 30000
}


class FocalLoss(Loss):

  def __init__(self, gamma=2., alpha=None, **kwargs):
    super(FocalLoss, self).__init__(**kwargs)
    self.gamma = gamma
    self.alpha = alpha

  def call(self, y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    alpha = self.alpha if self.alpha is not None else 0.25
    weight = alpha * y_true * K.pow((1 - y_pred), self.gamma)
    loss = weight * cross_entropy
    return K.mean(loss)


def add_technical_indicators(df):
  df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi()
  df['ema20'] = trend.EMAIndicator(df['close'], window=20).ema_indicator()
  macd = trend.MACD(df['close'])
  df['macd'] = macd.macd()
  df['macd_signal'] = macd.macd_signal()
  bollinger = volatility.BollingerBands(df['close'], window=20, window_dev=2)
  df['bollinger_hband'] = bollinger.bollinger_hband()
  df['bollinger_lband'] = bollinger.bollinger_lband()
  df['stoch'] = momentum.StochasticOscillator(df['high'],
                                              df['low'],
                                              df['close'],
                                              window=14).stoch()
  vwap = VolumeWeightedAveragePrice(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=14)
  df['vwap'] = vwap.volume_weighted_average_price()
  df['atr'] = volatility.AverageTrueRange(high=df['high'],
                                          low=df['low'],
                                          close=df['close'],
                                          window=14).average_true_range()
  ichimoku = IchimokuIndicator(high=df['high'],
                               low=df['low'],
                               window1=9,
                               window2=26,
                               window3=52)
  df['ichimoku_a'] = ichimoku.ichimoku_a()
  df['ichimoku_b'] = ichimoku.ichimoku_b()
  df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
  df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
  df.ffill(inplace=True)
  df.bfill(inplace=True)
  return df


def prepare_data(df, time_steps=60):
  df = add_technical_indicators(df)
  data = df[[
      'open', 'high', 'low', 'close', 'volume', 'rsi', 'ema20', 'macd',
      'macd_signal', 'bollinger_hband', 'bollinger_lband', 'stoch', 'vwap',
      'atr', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base_line',
      'ichimoku_conversion_line'
  ]].values
  return data


def create_lstm_model(input_shape):
  model = Sequential()
  model.add(Input(shape=input_shape))
  model.add(Bidirectional(LSTM(100, return_sequences=True)))
  model.add(Dropout(0.3))
  model.add(Bidirectional(LSTM(100, return_sequences=False)))
  model.add(Dropout(0.3))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])
  return model


class LSTMHyperModel(HyperModel):

  def build(self, hp):
    model = Sequential()
    model.add(Input(shape=(60, 18)))
    model.add(
        Bidirectional(
            LSTM(units=hp.Int('units1', min_value=32, max_value=256, step=32),
                 return_sequences=True)))
    model.add(
        Dropout(
            rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(
        Bidirectional(
            LSTM(units=hp.Int('units2', min_value=32, max_value=256, step=32),
                 return_sequences=False)))
    model.add(
        Dropout(
            rate=hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(
        Dense(units=hp.Int('dense_units', min_value=16, max_value=128,
                           step=16),
              activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])
    return model


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
    if market and 'limits' in market and 'amount' in market[
        'limits'] and 'min' in market['limits']['amount']:
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


def load_lstm_model_func():
  if os.path.exists('lstm_trading_model.h5') and os.path.exists(
      'lstm_scaler.pkl'):
    try:
      model = load_model('lstm_trading_model.h5',
                         custom_objects={'FocalLoss': FocalLoss})
      scaler = joblib.load('lstm_scaler.pkl')
      return model, scaler
    except Exception as e:
      logging.error(f"Error loading LSTM model or scaler: {e}")
      return None, None
  logging.warning("LSTM model or scaler files not found.")
  return None, None


def load_random_forest_model_func():
  if os.path.exists('random_forest_model.pkl') and os.path.exists(
      'random_forest_scaler.pkl'):
    try:
      model = joblib.load('random_forest_model.pkl')
      scaler = joblib.load('random_forest_scaler.pkl')
      return model, scaler
    except Exception as e:
      logging.error(f"Error loading Random Forest model or scaler: {e}")
      return None, None
  logging.warning("Random Forest model or scaler files not found.")
  return None, None


async def train_lstm_model(exchange, symbols):
  X_list = []
  y_list = []
  for symbol in symbols:
    df = await get_data_async(exchange, symbol)
    if df is not None:
      data = prepare_data(df)
      if len(data) < 61:
        continue
      scaler = StandardScaler()
      data_scaled = scaler.fit_transform(data)
      X, y = [], []
      for i in range(60, len(data_scaled) - 1):
        X.append(data_scaled[i - 60:i])
        y.append(1 if df['close'].iloc[i + 1] > df['close'].iloc[i] else 0)
      if len(X) > 0:
        X_list.append(np.array(X))
        y_list.append(np.array(y))
  if not X_list:
    logging.error("Failed to collect data for LSTM training")
    return None, None
  X_all = np.concatenate(X_list)
  y_all = np.concatenate(y_list)
  split = int(0.8 * len(X_all))
  X_train, X_test = X_all[:split], X_all[split:]
  y_train, y_test = y_all[:split], y_all[split:]
  class_weights_dict = class_weight.compute_class_weight(
      'balanced', classes=np.unique(y_train), y=y_train)
  class_weights = {
      i: class_weights_dict[i]
      for i in range(len(class_weights_dict))
  }
  hypermodel = LSTMHyperModel()
  tuner = RandomSearch(hypermodel,
                       objective='val_accuracy',
                       max_trials=7,
                       executions_per_trial=2,
                       directory='lstm_tuning',
                       project_name='trading_bot')
  tuner.search(X_train,
               y_train,
               epochs=10,
               validation_data=(X_test, y_test),
               class_weight=class_weights,
               callbacks=[
                   tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    restore_best_weights=True)
               ])
  best_model = tuner.get_best_models(num_models=1)[0]
  best_model.save('lstm_trading_model.h5')
  scaler = StandardScaler()
  scaler.fit(X_train.reshape(-1, X_train.shape[2]))
  joblib.dump(scaler, 'lstm_scaler.pkl')
  logging.info("LSTM model trained and saved")
  return best_model, scaler


async def train_random_forest_model(X_train, y_train):

  def train_rf(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_res_scaled, y_res, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100,
                                class_weight='balanced',
                                random_state=42)
    rf.fit(X_train_split, y_train_split)
    y_pred = rf.predict(X_val_split)
    accuracy = accuracy_score(y_val_split, y_pred)
    logging.info(f"Random Forest Validation Accuracy: {accuracy:.2f}")
    logging.debug(classification_report(y_val_split, y_pred))
    joblib.dump(rf, 'random_forest_model.pkl')
    joblib.dump(scaler, 'random_forest_scaler.pkl')
    logging.info("Random Forest model trained and saved")
    return rf, scaler

  return await asyncio.to_thread(train_rf, X_train, y_train)


def predict_signal_ensemble(df,
                            lstm_model,
                            lstm_scaler,
                            rf_model,
                            rf_scaler,
                            time_steps=60):
  try:
    df = add_technical_indicators(df)
    data = df[[
        'open', 'high', 'low', 'close', 'volume', 'rsi', 'ema20', 'macd',
        'macd_signal', 'bollinger_hband', 'bollinger_lband', 'stoch', 'vwap',
        'atr', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base_line',
        'ichimoku_conversion_line'
    ]].values
    data_scaled = lstm_scaler.transform(data)
    if len(data_scaled) < time_steps:
      logging.warning("Insufficient data for signal prediction")
      return None
    X_input_lstm = data_scaled[-time_steps:]
    X_input_lstm = np.expand_dims(X_input_lstm, axis=0)
    lstm_pred = lstm_model.predict(X_input_lstm)[0][0]
    lstm_signal = 1 if lstm_pred > 0.5 else 0
    X_input_rf = data_scaled[-time_steps:].flatten().reshape(1, -1)
    X_input_rf_scaled = rf_scaler.transform(X_input_rf)
    rf_pred = rf_model.predict(X_input_rf_scaled)[0]
    final_signal = int(lstm_signal) + int(rf_pred)
    return 1 if final_signal >= 2 else 0
  except Exception as e:
    logging.error(f"Error in signal prediction: {e}")
    return None


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
  last_trade_time = {}
  current_time = time.time()
  async with asyncio.Lock():
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
    async with asyncio.Lock():
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


async def train_random_forest_model_wrapper(top_symbols, exchange):
  X_combined = []
  y_combined = []
  for symbol in top_symbols:
    df = await get_data_async(exchange, symbol)
    if df is not None:
      data = prepare_data(df)
      if len(data) < 61:
        continue
      scaler = StandardScaler()
      data_scaled = scaler.fit_transform(data)
      X, y = [], []
      for i in range(60, len(data_scaled) - 1):
        X.append(data_scaled[i - 60:i].flatten())
        y.append(1 if df['close'].iloc[i + 1] > df['close'].iloc[i] else 0)
      if len(X) > 0:
        X_combined.extend(X)
        y_combined.extend(y)
  if X_combined and y_combined:
    X_all = np.array(X_combined)
    y_all = np.array(y_combined)
    X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42)
    rf_model, rf_scaler = await train_random_forest_model(
        X_train_rf, y_train_rf)
    return rf_model, rf_scaler
  logging.error("Failed to collect data for Random Forest training")
  return None, None


async def main():
  async_exchange = ccxt_async.bybit(exchange_config)
  try:
    markets = await fetch_markets(async_exchange)
    excluded_symbols = ['UNFIUSDT', 'TIAUSDT']
    all_symbols = [
        market['symbol'] for market in markets.values()
        if market.get('quote') == 'USDT' and market.get('active') and market.
        get('type') == 'swap' and market['symbol'] not in excluded_symbols
    ]
    top_symbols = await get_top_symbols(async_exchange, all_symbols)
    top_symbols = [
        symbol for symbol in top_symbols if symbol not in excluded_symbols
    ]
    min_amounts = await fetch_min_amounts(async_exchange, top_symbols, markets)
    lstm_model, lstm_scaler = await asyncio.to_thread(load_lstm_model_func)
    rf_model, rf_scaler = await asyncio.to_thread(load_random_forest_model_func
                                                  )
    if not lstm_model or not rf_model:
      lstm_model, lstm_scaler = await train_lstm_model(async_exchange,
                                                       top_symbols)
      if lstm_model and lstm_scaler:
        rf_model, rf_scaler = await train_random_forest_model_wrapper(
            top_symbols, async_exchange)
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
