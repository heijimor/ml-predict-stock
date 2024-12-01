import yfinance as yf
import pandas as pd
import os
from prometheus_client import start_http_server, Summary, Counter, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from keras.models import load_model as keras_load_model

class StockManager:
  def __init__(self):
    # REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
    # INFERENCE_COUNT = Counter('inference_count', 'Total number of inferences made')
    # MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy over time')
    self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

  def collect(
    self,
    ticker: str,
    start_date: str,
    end_date: str
  ) -> pd.DataFrame:

    df = yf.download(ticker, start=start_date, end=end_date)
    print(f'Yahoo Downloaded: \n ${df}')
    df.to_csv(os.path.join(self.BASE_DIR, "../../data/raw_data.csv"))
    data = df[['Close']]
    print(f'Yahoo Close: \n ${data}')
    return data
  
  def load_model(self, model_path, scaler_path):
    model = keras_load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Load the scaler parameters from JSON
    with open(scaler_path, "r") as f:
        scaler_params = json.load(f)
    
    # Reconstruct the scaler using loaded parameters
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_ = np.array(scaler_params["min"])
    scaler.scale_ = np.array(scaler_params["scale"])
    scaler.data_min_ = np.array(scaler_params["data_min"])
    scaler.data_max_ = np.array(scaler_params["data_max"])
    print(f"Scaler loaded and reconstructed from {scaler_path}")

    return model, scaler

  def predict(self, model, scaler, recent_data, seq_length):
    scaled_data = scaler.transform(recent_data[['Close']].values)
    X_recent, _ = self.sequencialize(scaled_data, seq_length)
    print(f"X_recent \n {X_recent}")
    predictions = model.predict(X_recent)
    return scaler.inverse_transform(predictions)
  
  def normalize(self, data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print(f'Scaled_data: \n ${scaled_data}')
    self.save_scaler(scaler, os.path.join(self.BASE_DIR, '../../models/scaler.json'))
    return scaled_data, scaler

  def save_scaler(self, scaler, path):
    """Save scaler parameters to a JSON file."""
    scaler_params = {
        "min": scaler.min_.tolist(),
        "scale": scaler.scale_.tolist(),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
    }
    with open(path, "w") as f:
        json.dump(scaler_params, f)
    print(f"Scaler parameters saved to {path}")

  def sequencialize(self, data, seq_length):
    X, y = [], []

    if len(data) < seq_length:
      raise ValueError("Insufficient data to create sequences.")

    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

  def prepare(self, X, y):
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f'X_train: \n ${X_train}')
    print(f'X_test: \n ${X_test}')
    print(f'y_train: \n ${y_train}')
    print(f'y_test: \n ${y_test}')
    return X_train, X_test, y_train, y_test

  def build(self, input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(f'model_build: \n ${model}')
    return model
  
  def train(self, model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Treinamento do modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )
    print(f'history: \n ${history}')
  
  def evaluate(self, model, scaler, X_test, y_test):
    # Previsões no conjunto de teste
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Desnormaliza os dados
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Cálculo das métricas
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
      
  def save(self, model):
    model.save(os.path.join(self.BASE_DIR, '../../models/lstm_model.h5'))