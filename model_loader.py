import os
import logging
from tensorflow.keras.models import load_model
import joblib
from models import FocalLoss

def load_lstm_model_func():
    if os.path.exists('lstm_trading_model.h5') and os.path.exists('lstm_scaler.pkl'):
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
    if os.path.exists('random_forest_model.pkl') and os.path.exists('random_forest_scaler.pkl'):
        try:
            model = joblib.load('random_forest_model.pkl')
            scaler = joblib.load('random_forest_scaler.pkl')
            return model, scaler
        except Exception as e:
            logging.error(f"Error loading Random Forest model or scaler: {e}")
            return None, None
    logging.warning("Random Forest model or scaler files not found.")
    return None, None
