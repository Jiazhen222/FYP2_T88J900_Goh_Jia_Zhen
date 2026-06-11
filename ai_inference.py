import numpy as np
import tensorflow as tf
from tcn import TCN
import joblib
import time

# ==========================================
# 1. INITIALIZATION (Model & Scaler)
# ==========================================
print("[INFO] Booting Edge AI Core...")

try:
    # compile=False bypasses Keras version conflicts
    model = tf.keras.models.load_model(
        'TCN_LSTM_Model.h5', 
        custom_objects={'TCN': TCN},
        compile=False
    )
    print("[SUCCESS] TCN-LSTM Hybrid Model loaded.")
except Exception as e:
    print(f"[FATAL ERROR] Could not load model: {e}")
    exit()

try:
    # Load the mathematical scaler used during original training
    scaler = joblib.load('scaler.joblib')
    print("[SUCCESS] Data Scaler loaded.")
except Exception as e:
    print(f"[FATAL ERROR] Could not load scaler: {e}")
    exit()

# ==========================================
# 2. DATA PREPARATION (Scale & Shape)
# ==========================================
def prepare_sensor_data(temp, hum, gas, active_scaler):
    # 1. Create the dummy 5-timestep window
    # 2. Pad with 5 trailing zeros to satisfy the 8-feature requirement
    raw_data = [
        [temp, hum, gas, 0, 0, 0, 0, 0],
        [temp+0.1, hum, gas+5, 0, 0, 0, 0, 0],
        [temp, hum-0.5, gas, 0, 0, 0, 0, 0],
        [temp-0.1, hum, gas-2, 0, 0, 0, 0, 0],
        [temp, hum, gas, 0, 0, 0, 0, 0]
    ]
    
    # 3. Shrink the raw numbers down to the AI's preferred decimal range
    scaled_data = active_scaler.transform(raw_data)
    
    # 4. Reshape to 3D tensor: (1 sample, 5 timesteps, 8 features)
    return scaled_data.reshape(1, 5, 8)

# ==========================================
# 3. EXECUTE INFERENCE
# ==========================================
print("[INFO] Simulating sensor input...")

# Test values
current_temp = 26.5
current_hum = 60.0
current_gas = 450

# Format the data
ai_input = prepare_sensor_data(current_temp, current_hum, current_gas, scaler)

# Warm-up run (burns off the initial CPU lag)
_ = model.predict(ai_input, verbose=0)

# Real measured run
start_time = time.time()
prediction = model.predict(ai_input, verbose=0)
latency = (time.time() - start_time) * 1000

print("\n--- INFERENCE RESULTS ---")
print(f"Predicted Air Quality Level : {prediction[0][0]:.2f} (Target validation: ~2.0)")
print(f"Inference Latency           : {latency:.2f} ms")
print("-------------------------")
