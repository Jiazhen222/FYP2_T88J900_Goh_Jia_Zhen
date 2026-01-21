import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 1. Load the data
df = pd.read_csv('indoor_air_quality_1000.csv')
X_raw = df.drop(['AQ_Label'], axis=1).values
y_labels = df['AQ_Label'].values

# 2. We must scale the data just like we did in training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Take the first row for the test
test_data_raw = X_raw[0:1]
test_data_scaled = X_scaled[0:1]
true_answer = y_labels[0]

print(f"--- Real Data from Sensors ---")
print(test_data_raw)
print(f"True Air Quality Label: {true_answer}")

# 3. Test Random Forest Output (Works fine with raw data)
print("\n--- Testing Random Forest Output ---")
rf = joblib.load('Random_Forest.joblib')
rf_prediction = rf.predict(test_data_raw)
print(f"Random Forest Predicted: {rf_prediction[0]:.2f}")

# 4. Test LSTM Output (Requires SCALED data)
print("\n--- Testing LSTM (.h5) Output ---")
lstm = tf.keras.models.load_model('LSTM_Model.h5', compile=False)

# Reshape the SCALED data to 3D for LSTM
input_data = test_data_scaled.reshape(1, 1, test_data_scaled.shape[1])
lstm_prediction = lstm.predict(input_data, verbose=0)
print(f"LSTM Predicted: {lstm_prediction[0][0]:.2f}")

print("\nConclusion: Now both should be very close to 2.0!")
