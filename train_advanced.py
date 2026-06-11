import os
import warnings
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, Attention, Conv1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- SETTINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')

# 1. DATA PREPARATION
print("--- Step 1: Loading Multi-Dataset Data ---")
file_list = ['indoor_air_quality_1000.csv', 'one_room_apartement.csv', 'IoT_Indoor_Air_Quality_Dataset.csv']
df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True).ffill().bfill()

X = df.drop('AQ_Label', axis=1).values
y = df['AQ_Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.joblib')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

# --- 2. LIGHTGBM (High-Speed Edge Classifier) ---
print("\n--- Step 2: Training LightGBM (Decision Node) ---")
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, verbose=-1)
lgb_model.fit(X_train, y_train)
joblib.dump(lgb_model, 'LightGBM_Model.joblib')

# --- 3. XGBOOST (High-Precision Numerical Monitor) ---
print("--- Step 3: Training XGBoost (Regression Monitor) ---")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'XGBoost_Model.joblib')

# --- 4. ATTENTION BI-LSTM (Temporal Forecaster) ---
def build_attention_bilstm(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    query_value_attention_seq = Attention()([lstm_out, lstm_out])
    x = GlobalAveragePooling1D()(query_value_attention_seq)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

print("--- Step 4: Training Attention Bi-LSTM ---")
attn_model = build_attention_bilstm((1, X.shape[1]))
attn_model.fit(X_train_3d, y_train, epochs=10, verbose=0)
attn_model.save('Attention_BiLSTM_Model.h5')

# --- 5. TCN-LSTM HYBRID (Spatial-Temporal Specialist) ---
def build_tcn_hybrid(input_shape):
    inputs = Input(shape=input_shape)
    # TCN Element: Causal Convolution to capture local patterns
    x = Conv1D(filters=64, kernel_size=1, padding='causal', activation='relu')(inputs)
    x = Dropout(0.2)(x)
    # LSTM Element: Temporal dependency
    x = LSTM(50, activation='tanh')(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

print("--- Step 5: Training TCN-LSTM Hybrid ---")
tcn_model = build_tcn_hybrid((1, X.shape[1]))
tcn_model.fit(X_train_3d, y_train, epochs=10, verbose=0)
tcn_model.save('TCN_LSTM_Model.h5')

print("\n✅ SUCCESS: All 4 Advanced Intelligence Assets Saved to 56GB Drive!")
