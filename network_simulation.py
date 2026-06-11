import os
import pandas as pd
import numpy as np
from tabulate import tabulate

# --- CONFIGURATION: Network Speeds (Mbps) & Latency (ms) ---
# Realistic real-world averages for each protocol
NETWORKS = {
    "Ethernet LAN": {"speed": 1000, "latency": 2},
    "4G LTE":       {"speed": 50,   "latency": 45},
    "5G NSA":       {"speed": 450,  "latency": 15},
    "6G (Simulated)":{"speed": 8000, "latency": 1},
    "WiFi 6 (ax)":  {"speed": 600,  "latency": 8},
    "WiFi 7 (be)":  {"speed": 1400, "latency": 4}
}

ALGOS = ["LSTM", "GRU", "Hybrid LSTM", "Random Forest", "Quantile Mapping"]

def get_model_info(name):
    """Retrieves actual file size from your directory."""
    m_file = "Random_Forest.joblib" if name == "Random Forest" else (
             "Quantile_Mapping.joblib" if name == "Quantile Mapping" else (
             "Hybrid_LSTM.h5" if name == "Hybrid LSTM" else (
             "LSTM_Model.h5" if name == "LSTM" else "GRU_Model.h5")))
    
    if os.path.exists(m_file):
        size_mb = os.path.getsize(m_file) / (1024 * 1024) # Convert bytes to MB
    else:
        # Fallback values if files are missing
        sizes = {"LSTM": 1.2, "GRU": 0.8, "Hybrid LSTM": 2.5, "Random Forest": 4.2, "Quantile Mapping": 0.1}
        size_mb = sizes.get(name, 1.0)
    return size_mb

def run_network_train():
    print("\n" + "="*100)
    print("📊 TABLE 2: NETWORK TRANSMISSION MATRIX (4G, 5G, 6G, WiFi, Ethernet)")
    print("="*100)
    print("Simulating model synchronization across heterogeneous network layers...\n")

    results = []
    idx = 0

    for net_name, specs in NETWORKS.items():
        for algo in ALGOS:
            size_mb = get_model_info(algo)
            
            # Calculate Transfer Delay (ms)
            # Formula: (Size in bits / Speed in bits per second) * 1000 + Network Latency
            transfer_time = ((size_mb * 8) / specs['speed']) * 1000
            total_delay = transfer_time + specs['latency']
            
            # Determine Bandwidth Status
            if total_delay < 150: status = "High"
            elif total_delay < 500: status = "Stable"
            else: status = "Critical"

            results.append([
                idx, net_name, algo, 
                f"{size_mb:.2f}", f"{total_delay:.2f}", status
            ])
            idx += 1

    headers = ["#", "Network Mode", "Algorithm", "Model Size (MB)", "Transfer Delay (ms)", "Bandwidth Status"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Save for Thesis Appendix
    pd.DataFrame(results, columns=headers).to_csv('Network_Transmission_Results.csv', index=False)
    print("\n✅ SUCCESS: Network Matrix saved to 'Network_Transmission_Results.csv'")

if __name__ == "__main__":
    run_network_train()
