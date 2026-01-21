import tensorflow as tf

def convert_model(model_name):
    print(f"Converting {model_name}...")
    # 1. Load the .h5 model
    model = tf.keras.models.load_model(f'{model_name}.h5', compile=False)
    
    # 2. Setup the converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3. Optimization (makes it even smaller for the Pi)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Important for LSTM: allow flexible sizes
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    # 4. Convert and Save
    tflite_model = converter.convert()
    with open(f'{model_name}.tflite', 'wb') as f:
        f.write(tflite_model)
    print(f"Success! Created {model_name}.tflite")

# Run for all three
models_to_convert = ['LSTM_Model', 'GRU_Model', 'Hybrid_LSTM']

for m in models_to_convert:
    try:
        convert_model(m)
    except Exception as e:
        print(f"Error converting {m}: {e}")

print("\nAll conversions finished. You now have the .tflite files for your Raspberry Pi!")
