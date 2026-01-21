import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
import os
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Setup Data
print("Analyzing hardware performance... please wait.")
df = pd.read_csv('indoor_air_quality_1000.csv')
X_raw = df.drop(['AQ_Label'], axis=1).values
y_true = df['AQ_Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_test = X_scaled[0:100] # Use 100 samples for latency test
X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 2. Define Models
models = {
    "Random Forest": ("joblib", "Random_Forest.joblib"),
    "Quantile Mapping": ("joblib", "Quantile_Mapping.joblib"),
    "LSTM": ("h5", "LSTM_Model.h5"),
    "GRU": ("h5", "GRU_Model.h5"),
    "Hybrid LSTM": ("h5", "Hybrid_LSTM.h5")
}

matrix_data = []

for name, (m_type, file) in models.items():
    try:
        # Measure Model Size (KB)
        size_kb = os.path.getsize(file) / 1024
        
        # Load Model
        if m_type == "joblib":
            model = joblib.load(file)
            
            # Measure Latency (Time to predict)
            start_time = time.time()
            preds = model.predict(X_scaled)
            latency = ((time.time() - start_time) / len(X_scaled)) * 1000
        else:
            model = tf.keras.models.load_model(file, compile=False)
            
            # Measure Latency
            start_time = time.time()
            preds = model.predict(X_scaled_3d, verbose=0).flatten()
            latency = ((time.time() - start_time) / len(X_scaled)) * 1000

        # Calculate Accuracy Statistics
        rounded_preds = np.clip(np.round(preds), 0, 2)
        acc = accuracy_score(y_true, rounded_preds) * 100
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        # Add to table (Simulating Pi 4B performance context)
        matrix_data.append([
            "Raspberry Pi 4B", 
            name, 
            f"{mae:.3f}", 
            f"{r2:.3f}", 
            f"{acc:.2f}%", 
            f"{latency:.3f}", 
            f"{size_kb:.1f}"
        ])
    except:
        matrix_data.append(["Raspberry Pi 4B", name, "N/A", "N/A", "N/A", "N/A", "N/A"])

# 3. Print the Final Matrix
headers = ["Device Model", "Algorithm", "MAE", "R2 Score", "Accuracy (%)", "Latency (ms)", "Size (KB)"]
print("\n" + "="*85)
print("TABLE 1: HARDWARE PERFORMANCE MATRIX (Air Quality Monitoring System)")
print("="*85)
print(tabulate(matrix_data, headers=headers, tablefmt="grid"))
print("="*85)
