import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
print("--- Loading Data ---")
df = pd.read_csv('indoor_air_quality_1000.csv')

# Split features (X) and target (y)
X = df.drop('AQ_Label', axis=1).values
y = df['AQ_Label'].values

# Scale data (Important for LSTM/GRU)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape data for LSTM/GRU (expects 3D: samples, time_steps, features)
X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ---------------------------------------------------------
# 2. RANDOM FOREST
# ---------------------------------------------------------
print("\n--- Training Random Forest ---")
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
joblib.dump(rf, 'Random_Forest.joblib')

# ---------------------------------------------------------
# 3. QUANTILE MAPPING (Quantile Regression)
# ---------------------------------------------------------
print("--- Training Quantile Mapping ---")
qr = QuantileRegressor(quantile=0.5, solver='highs')
qr.fit(X_train[:500], y_train[:500]) # Using small sample for speed
joblib.dump(qr, 'Quantile_Mapping.joblib')

# ---------------------------------------------------------
# 4. LSTM
# ---------------------------------------------------------
print("--- Training LSTM ---")
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X.shape[1])),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_3d, y_train, epochs=10, verbose=0)
lstm_model.save('LSTM_Model.h5')

# ---------------------------------------------------------
# 5. GRU
# ---------------------------------------------------------
print("--- Training GRU ---")
gru_model = Sequential([
    GRU(50, activation='relu', input_shape=(1, X.shape[1])),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mse')
gru_model.fit(X_train_3d, y_train, epochs=10, verbose=0)
gru_model.save('GRU_Model.h5')

# ---------------------------------------------------------
# 6. HYBRID LSTM (LSTM + Dense layers)
# ---------------------------------------------------------
print("--- Training Hybrid LSTM ---")
hybrid_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X.shape[1])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])
hybrid_model.compile(optimizer='adam', loss='mse')
hybrid_model.fit(X_train_3d, y_train, epochs=10, verbose=0)
hybrid_model.save('Hybrid_LSTM.h5')

print("\nSUCCESS: All 5 models trained and saved to the 56GB drive!")
