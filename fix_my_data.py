import pandas as pd

def fix_dataset(filename):
    print(f"--- Fixing {filename} ---")
    try:
        # Load the file
        df = pd.read_csv(filename)
        
        # 1. CLEAN THE HEADERS (Remove spaces and weird symbols)
        # This turns "CO2 (ppm)" into just "CO2"
        df.columns = df.columns.str.replace(r'\(.*\)', '', regex=True).str.replace('?', '').str.strip()
        
        # 2. RENAME SPECIFIC COLUMNS to match our AI Brain
        name_map = {
            'Occupancy Count': 'Occupancy',
            'Ventilation Status': 'Ventilation_Status'
        }
        df = df.rename(columns=name_map)

        # 3. CREATE THE MISSING AQ_LABEL
        # Logic: CO2 < 800 is 0 (Good), 800-1200 is 1 (Moderate), >1200 is 2 (Poor)
        if 'AQ_Label' not in df.columns:
            print("Generating AQ_Label based on CO2...")
            df['AQ_Label'] = df['CO2'].apply(lambda x: 0 if x < 800 else (1 if x <= 1200 else 2))

        # 4. ADD DUMMY COLUMNS FOR MISSING SENSORS
        if 'Room_ID' not in df.columns: df['Room_ID'] = 1
        if 'DayType' not in df.columns: df['DayType'] = 0
        if 'PM2.5' not in df.columns: df['PM2.5'] = 15.0 # Default value

        # 5. KEEP ONLY THE 9 NECESSARY COLUMNS IN ORDER
        required_cols = ['Room_ID', 'DayType', 'Occupancy', 'Ventilation_Status', 'CO2', 'PM2.5', 'Humidity', 'Temperature', 'AQ_Label']
        
        # Ensure all columns exist before selecting
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
                
        df = df[required_cols]

        # 6. Convert any text to numbers (Handle "Open/Closed" if present)
        text_map = {'open': 1, 'closed': 0, 'off': 0, 'low': 1, 'high': 2}
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.lower().map(text_map).fillna(0)

        # Save the file
        df.to_csv(filename, index=False)
        print(f"✅ SUCCESS: {filename} is ready!")

    except Exception as e:
        print(f"❌ Error: {e}")

# Run it on your IoT file
fix_dataset('IoT_Indoor_Air_Quality_Dataset.csv')
