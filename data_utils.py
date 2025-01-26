import pandas as pd
from ta import momentum, trend, volatility
from ta.trend import IchimokuIndicator
from ta.volume import VolumeWeightedAveragePrice
from sklearn.preprocessing import StandardScaler
import numpy as np

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
