# debug_arima_models.py
import os
import joblib
import pandas as pd

def debug_arima_models():
    model_folder = 'arima_models'
    
    print("🔍 Debugging ARIMA models...")
    print(f"Model folder: {os.path.abspath(model_folder)}")
    
    if not os.path.exists(model_folder):
        print("❌ Model folder doesn't exist!")
        return
    
    files = os.listdir(model_folder)
    print(f"📁 Files in model folder: {files}")
    
    # Check each expected model
    expected_models = [
        'bsba_financial_management_model.pkl',
        'bsba_marketing_management_model.pkl', 
        'bsit_model.pkl',
        'bscs_model.pkl'
    ]
    
    for model_file in expected_models:
        model_path = os.path.join(model_folder, model_file)
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        
        print(f"\n🔍 Checking {model_file}:")
        print(f"   Model exists: {os.path.exists(model_path)}")
        print(f"   Metadata exists: {os.path.exists(metadata_path)}")
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print(f"   ✅ Model loaded successfully")
                print(f"   Model type: {type(model)}")
                
                # Try to check if it has forecast method
                if hasattr(model, 'forecast'):
                    print(f"   ✅ Has forecast method")
                else:
                    print(f"   ❌ No forecast method!")
                    
            except Exception as e:
                print(f"   ❌ Error loading model: {e}")
        
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                print(f"   ✅ Metadata loaded successfully")
                print(f"   Metadata keys: {list(metadata.keys())}")
            except Exception as e:
                print(f"   ❌ Error loading metadata: {e}")

if __name__ == "__main__":
    debug_arima_models()