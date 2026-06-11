import serial
import time
import numpy as np
import tensorflow as tf
from tcn import TCN
import joblib
from collections import deque

# ==========================================
# 1. CONFIGURATION & INITIALIZATION
# ==========================================
SERIAL_PORT = '/dev/ttyUSB0'  
BAUD_RATE = 115200            
MODEL_PATH = 'TCN_LSTM_Model.h5'
SCALER_PATH = 'scaler.joblib'
TIME_STEPS = 5                

print("==========================================================")
print("🚀 INITIALIZING LIVE AI INFERENCE BRIDGE (HIL TESTING)")
print("==========================================================")

print("[SYSTEM] Loading AI Architecture into memory...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'TCN': TCN}, compile=False)
    print("[SUCCESS] TCN-LSTM Hybrid Model loaded.")
except Exception as e:
    print(f"[FATAL ERROR] Model failure: {e}")
    exit()

try:
    scaler = joblib.load(SCALER_PATH)
    print("[SUCCESS] Data Scaler loaded.")
except Exception as e:
    print(f"[FATAL ERROR] Scaler failure: {e}")
    exit()

# Deque automatically drops the oldest reading to keep exactly 5 items
sensor_window = deque(maxlen=TIME_STEPS)

print(f"[SYSTEM] Attempting hardware link on {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    time.sleep(2) 
    ser.flushInput() 
    print("[SUCCESS] Hardware linked successfully.")
except Exception as e:
    print(f"[FATAL ERROR] Could not connect to ESP32: {e}")
    exit()

print("\n--- COMMENCING REAL-TIME SENSE-THINK-ACT CYCLE ---")
print("Awaiting continuous sensor stream...\n")

# Warm-up the model to prevent initial lag spike
dummy_data = np.zeros((1, TIME_STEPS, 8))
_ = model.predict(dummy_data, verbose=0)

# ==========================================
# 2. THE CONTINUOUS INFERENCE LOOP
# ==========================================
while True:
    try:
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8').strip()
            
            if not raw_line or "ERROR" in raw_line:
                continue
                
            parts = raw_line.split(',')
            if len(parts) == 3:
                temp = float(parts[0])
                hum = float(parts[1])
                gas = float(parts[2])
                
                print(f"[DATA] Temp: {temp:.1f}°C | Hum: {hum:.1f}% | Gas: {gas:.0f}", end=" -> ")
                
                # Append data and pad with 5 zeros to match the 8-feature requirement
                sensor_window.append([temp, hum, gas, 0, 0, 0, 0, 0])
                
                if len(sensor_window) == TIME_STEPS:
                    # Scale the entire 5-reading window
                    scaled_window = scaler.transform(sensor_window)
                    
                    # Reshape for the AI
                    ai_input = scaled_window.reshape(1, TIME_STEPS, 8)
                    
                    # Execute prediction
                    start_time = time.time()
                    prediction = model.predict(ai_input, verbose=0)
                    latency = (time.time() - start_time) * 1000
                    
                    print(f"[AI PREDICTION] AQI Level: {prediction[0][0]:.2f} (Latency: {latency:.1f}ms)")
                else:
                    print(f"[BUFFERING] Collecting data... ({len(sensor_window)}/{TIME_STEPS})")
            else:
                print(f"[WARNING] Malformed string: {raw_line}")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Administrator terminated the bridge. Closing ports safely.")
        ser.close()
        break
    except Exception as e:
        print(f"\n[ERROR] Pipeline failure: {e}")
        time.sleep(1)
