import asyncio
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_utils import prepare_data
from models import LSTMHyperModel
import tensorflow as tf
from model_loader import load_lstm_model_func, load_random_forest_model_func
from fetcher import get_data_async

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

def train_rf_sync(X, y):
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

async def train_random_forest_model(X_train, y_train):
    return await asyncio.to_thread(train_rf_sync, X_train, y_train)

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
