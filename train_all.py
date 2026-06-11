import os
# Force CPU to avoid the CUDA crash
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Silences FutureWarnings
warnings.filterwarnings('ignore')

print("--- Step 1: Loading Data ---")
file_list = ['indoor_air_quality_1000.csv', 'one_room_apartement.csv', 'IoT_Indoor_Air_Quality_Dataset.csv']
dataframes = [pd.read_csv(f) for f in file_list]
df = pd.concat(dataframes, ignore_index=True)

print("--- Step 2: Cleaning Data ---")
df = df.dropna()
print(f"Done! {len(df)} rows ready for training.")

X = df.drop('AQ_Label', axis=1)
y = df['AQ_Label'].values

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- CRITICAL ADDITION: Save the scaler for the Raspberry Pi ---
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved as 'scaler.joblib'")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

print("--- Step 3: Training Random Forest ---")
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
joblib.dump(rf, 'Random_Forest.joblib')
print("Random Forest Saved!")

print("--- Step 4: Training Quantile Mapping ---")
qr = QuantileRegressor(quantile=0.5, solver='highs')
qr.fit(X_train[:1000], y_train[:1000]) 
joblib.dump(qr, 'Quantile_Mapping.joblib')
print("Quantile Mapping Saved!")

print("--- Step 5: Training LSTM ---")
# Using Tanh for better stability in Air Quality trends
lstm_model = Sequential([Input(shape=(1, X.shape[1])), LSTM(50, activation='tanh'), Dense(1)])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_3d, y_train, epochs=5, batch_size=128, verbose=1) 
lstm_model.save('LSTM_Model.keras')

print("\nALL MODELS AND SCALER DONE! PHASE 1 IS READY.")
