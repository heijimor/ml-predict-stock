import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler