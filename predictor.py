import logging
import numpy as np
from data_utils import add_technical_indicators

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
