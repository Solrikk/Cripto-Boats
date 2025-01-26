import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from keras_tuner import HyperModel, RandomSearch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

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
