from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
import joblib
import json
import re
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thesis_new_data'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_new_data_models')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Dataset management
CURRENT_DATASET_PATH = os.path.join(app.config['UPLOAD_FOLDER'], "current_dataset.csv")

# Department configurations
DEPARTMENTS = {
    'BED': ['BSBA-FINANCIAL_MANAGEMENT', 'BSBA-MARKETING_MANAGEMENT'],
    'CED': ['BSIT', 'BSCS']
}

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global variables
current_dataset = None
ml_models = {}

def create_custom_adam_optimizer():
    """Create Adam optimizer for compatibility"""
    return Adam(learning_rate=0.001)  # Match your training learning rate

def load_trained_models_compatible():
    """Compatible loading method that rebuilds the model architecture"""
    global ml_models
    
    try:
        print(f"DEBUG: Loading from {app.config['MODEL_FOLDER']}")
        
        # Check what files we have
        if not os.path.exists(app.config['MODEL_FOLDER']):
            print("❌ Model folder not found")
            return False
            
        files = os.listdir(app.config['MODEL_FOLDER'])
        print(f"DEBUG: Available files: {files}")
        
        models_loaded = {}
        
        # Load preprocessing components
        if 'feature_scaler (1).pkl' in files:
            try:
                models_loaded['feature_scaler'] = joblib.load(os.path.join(app.config['MODEL_FOLDER'], 'feature_scaler (1).pkl'))
                print("✅ Loaded feature scaler")
            except Exception as e:
                print(f"❌ Error loading feature scaler: {e}")
                
        if 'target_scaler (1).pkl' in files:
            try:
                models_loaded['target_scaler'] = joblib.load(os.path.join(app.config['MODEL_FOLDER'], 'target_scaler (1).pkl'))
                print("✅ Loaded target scaler")
            except Exception as e:
                print(f"❌ Error loading target scaler: {e}")
                
        if 'enrollment_encoder.pkl' in files:
            try:
                models_loaded['encoder'] = joblib.load(os.path.join(app.config['MODEL_FOLDER'], 'enrollment_encoder.pkl'))
                print("✅ Loaded encoder")
            except Exception as e:
                print(f"❌ Error loading encoder: {e}")
        
        # Rebuild LSTM models with the exact architecture from your training
        lstm_models_loaded = []
        
        for fold in range(1, 5):  # Try folds 1-4
            model_path = os.path.join(app.config['MODEL_FOLDER'], f'final_lstm_model_fold_{fold}.h5')
            if os.path.exists(model_path):
                try:
                    # Rebuild the exact Sequential architecture from your training
                    model = rebuild_sequential_model()
                    
                    # Load weights only (bypasses the serialization issue)
                    model.load_weights(model_path)
                    print(f"✅ Rebuilt and loaded LSTM model fold {fold}")
                    
                    lstm_models_loaded.append(model)
                    
                except Exception as e:
                    print(f"❌ Failed to load fold {fold}: {e}")
            else:
                print(f"⚠️ LSTM model fold {fold} not found")
        
        if not lstm_models_loaded:
            print("❌ No LSTM models could be loaded")
            return False
        
        models_loaded['lstm_models'] = lstm_models_loaded
        models_loaded['lstm_model'] = lstm_models_loaded[0]  # Use first model as default
        
        models_loaded['time_steps'] = {'TIME_STEPS': 4}
        ml_models = models_loaded
        
        print(f"✅ Successfully loaded {len(lstm_models_loaded)} LSTM models")
        return True
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def rebuild_sequential_model():
    """Rebuild the exact Sequential model architecture from your training"""
    # This matches your create_lstm_model function exactly
    model = Sequential([
        # First LSTM layer with return_sequences=True
        LSTM(50, return_sequences=True, input_shape=(4, 25),  # 4 time steps, 25 features
             activation='tanh', recurrent_activation='sigmoid',
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros'),
        Dropout(0.2),

        # Second LSTM layer
        LSTM(25, return_sequences=False,
             activation='tanh', recurrent_activation='sigmoid',
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros'),
        Dropout(0.2),

        # Output layer
        Dense(1, activation='linear')
    ])

    # Compile with same optimizer
    model.compile(
        optimizer=create_custom_adam_optimizer(),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def predict_with_lstm_ensemble(course, year, semester):
    """Make prediction using ensemble of LSTM models"""
    try:
        print(f"DEBUG: Ensemble prediction for {course}, {year}, Semester {semester}")
        
        # Check if models are loaded
        if not ml_models or 'lstm_models' not in ml_models:
            print("LSTM ensemble models not loaded, using fallback")
            return predict_enrollment_fallback(course, year, semester)
        
        # Create features
        features = create_prediction_features(course, year, semester, 4)
        if features is None:
            print("DEBUG: Feature creation failed, using fallback")
            return predict_enrollment_fallback(course, year, semester)
        
        print(f"DEBUG: Features shape: {features.shape}")
        
        # Make predictions with all models and average them
        predictions = []
        for i, model in enumerate(ml_models['lstm_models']):
            try:
                prediction_scaled = model.predict(features, verbose=0)
                predictions.append(prediction_scaled[0][0])
                print(f"DEBUG: Model {i+1} prediction (scaled): {prediction_scaled[0][0]}")
            except Exception as e:
                print(f"DEBUG: Model {i+1} prediction failed: {e}")
        
        if not predictions:
            print("DEBUG: All model predictions failed, using fallback")
            return predict_enrollment_fallback(course, year, semester)
        
        # Average predictions
        avg_prediction_scaled = np.mean(predictions)
        print(f"DEBUG: Ensemble average prediction (scaled): {avg_prediction_scaled}")
        
        # Inverse transform to get actual enrollment numbers
        if 'target_scaler' in ml_models:
            prediction = ml_models['target_scaler'].inverse_transform([[avg_prediction_scaled]])[0][0]
            print(f"DEBUG: Ensemble prediction (inverse transformed): {prediction}")
        else:
            # If no target scaler, use rough scaling
            prediction = avg_prediction_scaled * 100
            print(f"DEBUG: Ensemble prediction (rough scaled): {prediction}")
        
        # Ensure reasonable prediction
        prediction = max(10, float(prediction))
        print(f"DEBUG: Final ensemble prediction: {prediction}")
        
        return round(prediction)
        
    except Exception as e:
        print(f"LSTM ensemble prediction error for {course}: {e}")
        import traceback
        traceback.print_exc()
        return predict_enrollment_fallback(course, year, semester)

# KEEP ALL YOUR EXISTING FUNCTIONS EXACTLY THE SAME FROM HERE...
# [PASTE ALL THE REMAINING FUNCTIONS FROM YOUR WORKING FLASK APP]
# is_future_prediction, validate_future_year, load_current_dataset, analyze_dataset,
# format_file_size, validate_dataset, predict_enrollment_fallback, debug_feature_creation,
# create_prediction_features, debug_scaler_features, get_historical_enrollment, get_historical_trend
# AND ALL THE ROUTES...

def is_future_prediction(course, year, semester):
    """Check if the prediction request is for a future period"""
    try:
        if current_dataset is None:
            return True  # No dataset, allow prediction
        
        # Convert input to match dataset format
        if semester == "1":
            semester_dataset = "1st"
        elif semester == "2":
            semester_dataset = "2nd"
        else:
            semester_dataset = semester
        
        # Check if this exact record exists in historical data
        existing_record = current_dataset[
            (current_dataset['Course'] == course) & 
            (current_dataset['School_Year'] == year) & 
            (current_dataset['Semester'] == semester_dataset)
        ]
        
        if not existing_record.empty:
            return False  # Historical data exists
        else:
            return True   # No historical data, treat as future prediction
            
    except Exception as e:
        print(f"Error checking historical data: {e}")
        return True  # On error, allow prediction

def validate_future_year(year):
    """Validate that the prediction year is reasonable and sequential"""
    try:
        # Validate format first
        if not re.match(r'^\d{4}-\d{4}$', year):
            return False, "Invalid year format. Use YYYY-YYYY"
        
        # Extract years
        start_year = int(year.split('-')[0])
        end_year = int(year.split('-')[1])
        current_year = datetime.now().year
        
        # Validate sequential years
        if end_year - start_year != 1:
            return False, "School year must be sequential (e.g., 2026-2027)"
        
        # Allow predictions up to 3 years in the future
        max_future_year = current_year + 3
        
        if start_year > max_future_year:
            return False, f"Predictions can only be made up to {max_future_year}-{max_future_year+1}"
        
        if start_year < current_year - 1:
            return False, "Cannot predict for past years that don't have historical data"
            
        return True, "Valid year"
        
    except Exception as e:
        return False, f"Invalid year format: {str(e)}"

def load_current_dataset():
    """Load the current dataset into memory"""
    global current_dataset
    try:
        if os.path.exists(CURRENT_DATASET_PATH):
            file_size = os.path.getsize(CURRENT_DATASET_PATH)
            print(f"DEBUG: Loading dataset from: {CURRENT_DATASET_PATH}")
            print(f"DEBUG: Dataset file size: {file_size} bytes")
            
            current_dataset = pd.read_csv(CURRENT_DATASET_PATH)
            print(f"Current dataset loaded: {len(current_dataset)} records")
            return True
        
        print(f"DEBUG: Dataset file not found at: {CURRENT_DATASET_PATH}")
        current_dataset = None
        return False
    except Exception as e:
        print(f"Error loading current dataset: {e}")
        current_dataset = None
        return False

def analyze_dataset(df):
    """Analyze dataset and return statistics"""
    try:
        total_records = len(df)
        
        file_path = CURRENT_DATASET_PATH
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
        else:
            file_size = 0
        
        # Student counts by department
        student_counts = {}
        program_counts = {}
        year_spans = {}
        
        for dept, courses in DEPARTMENTS.items():
            dept_students = 0
            dept_programs = 0
            dept_years = set()
            
            for course in courses:
                if 'Course' in df.columns:
                    course_data = df[df['Course'] == course]
                    if not course_data.empty:
                        dept_programs += 1
                        
                        if 'total_enrollees' in df.columns:
                            dept_students += course_data['total_enrollees'].sum()
                        
                        if 'School_Year' in df.columns:
                            dept_years.update(course_data['School_Year'].unique())
            
            student_counts[dept] = int(dept_students)
            program_counts[dept] = dept_programs
            year_spans[dept] = len(dept_years)
        
        return {
            'filename': 'current_dataset.csv',
            'record_count': total_records,
            'file_size': format_file_size(file_size),
            'student_counts': student_counts,
            'program_counts': program_counts,
            'year_spans': year_spans,
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return {
            'filename': 'current_dataset.csv',
            'record_count': len(df) if df is not None else 0,
            'file_size': "0 MB",
            'student_counts': {'BED': 0, 'CED': 0},
            'program_counts': {'BED': 0, 'CED': 0},
            'year_spans': {'BED': 0, 'CED': 0},
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }   
    
def format_file_size(size_bytes):
    """Convert file size to human readable format"""
    try:
        if size_bytes == 0:
            return "0.0 MB"
        
        size_kb = size_bytes / 1024
        if size_kb < 1024:
            return f"{size_kb:.1f} KB"
        else:
            size_mb = size_kb / 1024
            return f"{size_mb:.2f} MB"
    except Exception as e:
        print(f"Error formatting file size: {e}")
        return "0.0 MB"

def validate_dataset(df):
    """Validate uploaded dataset structure"""
    try:
        if not isinstance(df, pd.DataFrame):
            return False, "Invalid data format"
        
        required_columns = ['School_Year', 'Semester', 'Course']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        if len(df) == 0:
            return False, "Dataset is empty"
        
        enrollment_columns = ['1st_year_enrollees', '2nd_year_enrollees', '3rd_year_enrollees', '4th_year_enrollees', 'total_enrollees']
        missing_enrollment = [col for col in enrollment_columns if col not in df.columns]
        if missing_enrollment:
            print(f"Warning: Missing enrollment columns: {missing_enrollment}")
            for col in missing_enrollment:
                df[col] = 0
        
        return True, "Dataset is valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def predict_enrollment_fallback(course, year, semester):
    """Simple prediction for future enrollment"""
    try:
        base_enrollment = 100
        
        if "BSBA" in course:
            base_enrollment = 120
        elif course in ["BSCS", "BSIT"]:
            base_enrollment = 80
        
        if semester == "1":
            base_enrollment *= 1.1
        else:
            base_enrollment *= 0.9
            
        variation = np.random.normal(0, 10)
        prediction = max(50, base_enrollment + variation)
        
        return round(prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 100

def debug_feature_creation(course, year, semester):
    """Debug function to check feature creation"""
    print(f"\n=== DEBUG FEATURE CREATION ===")
    print(f"Course: {course}, Year: {year}, Semester: {semester}")
    
    if current_dataset is None:
        print("No dataset available")
        return
    
    course_data = current_dataset[current_dataset['Course'] == course]
    print(f"Found {len(course_data)} records for {course}")
    
    if not course_data.empty:
        print("Recent records:")
        recent = course_data.tail(5)[['School_Year', 'Semester', 'total_enrollees']]
        print(recent)
    
    # Test feature creation
    features = create_prediction_features(course, year, semester, 4)
    if features is not None:
        print(f"Feature shape: {features.shape}")
        print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"Feature mean: {features.mean():.3f}")
    else:
        print("Feature creation failed")
    print("=== END DEBUG ===\n")

def create_prediction_features(course, year, semester, time_steps=4):
    """Create features for LSTM prediction matching EXACT training format"""
    try:
        if current_dataset is None:
            print("No dataset available for feature creation")
            return None
        
        # Get historical data for the course
        course_data = current_dataset[current_dataset['Course'] == course].copy()
        if course_data.empty:
            print(f"No historical data found for course: {course}")
            return None
        
        # Sort by time to get the most recent data
        course_data = course_data.sort_values(['School_Year', 'Semester'])
        
        # We need at least time_steps records to create a sequence
        if len(course_data) < time_steps:
            print(f"Not enough historical data. Need {time_steps}, have {len(course_data)}")
            return None
        
        # Take the most recent time_steps records
        recent_data = course_data.tail(time_steps).copy()
        
        # Create features for EACH record in the sequence
        sequences = []
        
        # Extract prediction context
        pred_start_year = int(year.split('-')[0])
        pred_semester_order = 1 if semester == "1" else 2
        
        for idx, (_, record) in enumerate(recent_data.iterrows()):
            # We need to create EXACTLY the features the scaler expects
            features = []
            
            # For the LAST record in the sequence, use prediction context
            # For historical records, use actual data
            if idx == time_steps - 1:  # Last record - use prediction context
                current_start_year = pred_start_year
                current_semester_order = pred_semester_order
                # Use the most recent enrollment data as base for prediction features
                base_record = recent_data.iloc[-1]
            else:
                # Historical records - use actual data
                current_start_year = int(record['School_Year'].split('-')[0])
                current_semester_order = 1 if record['Semester'] == '1st' else 2
                base_record = record
            
            # 1. Start_Year
            features.append(current_start_year)
            
            # 2. Semester_Order
            features.append(current_semester_order)
            
            # 3. Lag_1_Enrollees
            if idx > 0:
                prev_record = recent_data.iloc[idx - 1]
                lag_total = prev_record['total_enrollees']
            else:
                lag_total = base_record['total_enrollees']
            features.append(lag_total)
            
            # 4. Rolling_Mean_3
            # 5. Rolling_Std_3
            if idx >= 2:
                rolling_window = recent_data.iloc[max(0, idx-2):idx+1]['total_enrollees']
                rolling_mean = rolling_window.mean()
                rolling_std = rolling_window.std()
            else:
                rolling_mean = base_record['total_enrollees']
                rolling_std = 0
            features.extend([rolling_mean, rolling_std])
            
            # 6. Growth_Rate
            if idx > 0:
                prev_total = recent_data.iloc[idx - 1]['total_enrollees']
                current_total = base_record['total_enrollees']
                growth_rate = (current_total - prev_total) / (prev_total + 1e-10)
            else:
                growth_rate = 0
            features.append(growth_rate)
            
            # 7. Year_Semester_Interaction
            year_sem_interaction = current_start_year * current_semester_order
            features.append(year_sem_interaction)
            
            # 8. Time_Index
            time_index = idx + 1
            features.append(time_index)
            
            # 9. Enrollment_Trend
            if len(recent_data) > 1:
                x_values = list(range(len(recent_data)))
                y_values = recent_data['total_enrollees'].values
                enrollment_trend = np.polyfit(x_values, y_values, 1)[0]
            else:
                enrollment_trend = 0
            features.append(enrollment_trend)
            
            # 10. Dept_Enrollment_Ratio
            dept_ratio = base_record['total_enrollees'] / (recent_data['total_enrollees'].mean() + 1e-10)
            features.append(dept_ratio)
            
            # 11. Semester_Sin, 12. Semester_Cos
            semester_sin = np.sin(2 * np.pi * current_semester_order / 2)
            semester_cos = np.cos(2 * np.pi * current_semester_order / 2)
            features.extend([semester_sin, semester_cos])
            
            # 13. Semester_1st, 14. Semester_2nd (one-hot encoding)
            semester_1st = 1 if current_semester_order == 1 else 0
            semester_2nd = 1 if current_semester_order == 2 else 0
            features.extend([semester_1st, semester_2nd])
            
            # 15-18. Course encoding (one-hot)
            course_bsba_fin = 1 if course == 'BSBA-FINANCIAL_MANAGEMENT' else 0
            course_bsba_mkt = 1 if course == 'BSBA-MARKETING_MANAGEMENT' else 0
            course_bscs = 1 if course == 'BSCS' else 0
            course_bsit = 1 if course == 'BSIT' else 0
            features.extend([course_bsba_fin, course_bsba_mkt, course_bscs, course_bsit])
            
            # 19-22. Year_Level encoding (use proportions from base record)
            total = base_record['total_enrollees']
            if total > 0:
                year_level_1st = base_record['1st_year_enrollees'] / total
                year_level_2nd = base_record['2nd_year_enrollees'] / total
                year_level_3rd = base_record['3rd_year_enrollees'] / total
                year_level_4th = base_record['4th_year_enrollees'] / total
            else:
                year_level_1st = year_level_2nd = year_level_3rd = year_level_4th = 0.25
            features.extend([year_level_1st, year_level_2nd, year_level_3rd, year_level_4th])
            
            # 23-25. Department encoding (one-hot)
            dept_ba = 1 if 'BSBA' in course else 0
            dept_cs = 1 if course == 'BSCS' else 0
            dept_it = 1 if course == 'BSIT' else 0
            features.extend([dept_ba, dept_cs, dept_it])
            
            # Verify we have exactly 25 features
            if len(features) != 25:
                print(f"ERROR: Expected 25 features, but got {len(features)}")
                # Pad with zeros if needed
                while len(features) < 25:
                    features.append(0)
                features = features[:25]
            
            sequences.append(features)
        
        # Convert to numpy array
        X_sequence = np.array(sequences)
        
        # Debug: Check feature statistics
        print(f"DEBUG: Feature statistics - Min: {X_sequence.min():.2f}, Max: {X_sequence.max():.2f}, Mean: {X_sequence.mean():.2f}")
        print(f"DEBUG: Prediction context - Year: {pred_start_year}, Semester: {pred_semester_order}")
        
        # Ensure we have the right shape
        if X_sequence.shape != (time_steps, 25):
            print(f"Shape mismatch: {X_sequence.shape} vs expected ({time_steps}, 25)")
            return None
        
        print(f"DEBUG: Created sequence with {X_sequence.shape[1]} features")
        
        # Scale features if scaler is available
        if 'feature_scaler' in ml_models:
            # Reshape for scaling (scaler expects 2D)
            original_shape = X_sequence.shape
            X_flat = X_sequence.reshape(-1, X_sequence.shape[-1])
            
            try:
                X_scaled = ml_models['feature_scaler'].transform(X_flat)
                X_sequence = X_scaled.reshape(original_shape)
                print(f"DEBUG: Scaled features shape: {X_sequence.shape}")
                print(f"DEBUG: Scaled feature stats - Min: {X_sequence.min():.2f}, Max: {X_sequence.max():.2f}, Mean: {X_sequence.mean():.2f}")
            except Exception as scale_error:
                print(f"DEBUG: Scaling failed: {scale_error}")
                print("WARNING: Using unscaled features due to scaling error")
        
        # Reshape for LSTM: (1, time_steps, n_features)
        X_sequence = X_sequence.reshape(1, time_steps, X_sequence.shape[1])
        print(f"DEBUG: Final sequence shape: {X_sequence.shape}")
        
        return X_sequence
        
    except Exception as e:
        print(f"Error creating prediction features: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_scaler_features():
    """Debug what features the scaler expects"""
    if 'feature_scaler' in ml_models:
        scaler = ml_models['feature_scaler']
        print(f"DEBUG: Scaler expects {scaler.n_features_in_} features")
        if hasattr(scaler, 'feature_names_in_'):
            print(f"DEBUG: Scaler feature names: {scaler.feature_names_in_}")
        else:
            print("DEBUG: No feature names available in scaler")
    else:
        print("DEBUG: No feature scaler found")

def get_historical_enrollment(course, year, semester):
    """Get historical enrollment data for specific course, year, and semester"""
    try:
        if current_dataset is not None:
            # Convert form semester to dataset format
            if semester == "1":
                semester_dataset = "1st"
            elif semester == "2":
                semester_dataset = "2nd"
            else:
                semester_dataset = semester
                
            record = current_dataset[
                (current_dataset['Course'] == course) &
                (current_dataset['School_Year'] == year) &
                (current_dataset['Semester'] == semester_dataset)
            ]
            
            if not record.empty:
                result = {
                    'total_enrollment': int(record['total_enrollees'].iloc[0]),
                    'first_year': int(record['1st_year_enrollees'].iloc[0]),
                    'second_year': int(record['2nd_year_enrollees'].iloc[0]),
                    'third_year': int(record['3rd_year_enrollees'].iloc[0]),
                    'fourth_year': int(record['4th_year_enrollees'].iloc[0])
                }
                return result
        return None
    except Exception as e:  
        print(f"Error getting historical enrollment: {e}")
        return None

def get_historical_trend(course, max_records=10):
    """Get historical trend data for charts - showing year levels over time"""
    try:
        if current_dataset is not None and 'Course' in current_dataset.columns:
            course_data = current_dataset[current_dataset['Course'] == course]
            if not course_data.empty:
                course_data = course_data.sort_values(['School_Year', 'Semester'])
                recent_data = course_data.tail(max_records)
                
                labels = []
                first_year_data = []
                second_year_data = []
                third_year_data = []
                fourth_year_data = []
                
                for _, row in recent_data.iterrows():
                    sem_display = "S1" if row['Semester'] == "1st" else "S2"
                    labels.append(f"{row['School_Year']} {sem_display}")
                    
                    first_year_data.append(row['1st_year_enrollees'])
                    second_year_data.append(row['2nd_year_enrollees'])
                    third_year_data.append(row['3rd_year_enrollees'])
                    fourth_year_data.append(row['4th_year_enrollees'])
                
                return {
                    'labels': labels,
                    'first_year': first_year_data,
                    'second_year': second_year_data,
                    'third_year': third_year_data,
                    'fourth_year': fourth_year_data
                }
        return {
            'labels': [], 'first_year': [], 'second_year': [], 'third_year': [], 'fourth_year': []
        }
    except Exception as e:
        print(f"Error getting historical trend: {e}")
        return {
            'labels': [], 'first_year': [], 'second_year': [], 'third_year': [], 'fourth_year': []
        }

# Initialize the application
load_current_dataset()

# Try compatible loading method
print("Loading LSTM models with compatible method...")
if not load_trained_models_compatible():
    print("⚠️ LSTM models could not be loaded. Using fallback prediction method.")

@app.before_request
def before_request():
    """Execute before each request"""
    session.permanent = True
    app.permanent_session_lifetime = timedelta(days=1)

@app.route("/")
def index():
    """Main page"""
    dataset_loaded = current_dataset is not None
    lstm_loaded = ml_models and 'lstm_models' in ml_models
    model_count = len(ml_models.get('lstm_models', [])) if lstm_loaded else 0
    return render_template("frontface.html", 
                         dataset_loaded=dataset_loaded, 
                         lstm_loaded=lstm_loaded,
                         model_count=model_count)

@app.route("/check_dataset_status")
def check_dataset_status():
    """API endpoint to check dataset status"""
    dataset_loaded = current_dataset is not None
    dataset_info = None
    lstm_loaded = ml_models and 'lstm_models' in ml_models
    model_count = len(ml_models.get('lstm_models', [])) if lstm_loaded else 0
    
    if dataset_loaded:
        dataset_info = analyze_dataset(current_dataset)
    
    return jsonify({
        'loaded': dataset_loaded,
        'lstm_loaded': lstm_loaded,
        'model_count': model_count,
        'dataset_info': dataset_info
    })

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    """Handle dataset uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'File must be a CSV'})
        
        # Read and validate the dataset
        try:
            df = pd.read_csv(file.stream)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading CSV: {str(e)}'})
        
        # Validate dataset structure
        is_valid, validation_message = validate_dataset(df)
        if not is_valid:
            return jsonify({'success': False, 'error': validation_message})
        
        # Save the dataset
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_path = CURRENT_DATASET_PATH
        df.to_csv(file_path, index=False)
        
        # Reload dataset
        load_current_dataset()
        
        # Analyze dataset for response
        analysis_result = analyze_dataset(df)
        analysis_result['filename'] = secure_filename(file.filename)
        
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'dataset_info': analysis_result,
            'lstm_loaded': ml_models and 'lstm_models' in ml_models,
            'model_count': len(ml_models.get('lstm_models', [])) if ml_models else 0
        })
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route("/select_department", methods=["POST"])
def select_department():
    """Handle department selection"""
    department = request.form.get("department")
    if department == "BED":
        return redirect(url_for("bed_filter"))
    elif department == "CED":
        return redirect(url_for("ced_filter"))
    else:
        flash("Invalid department selected", "error")
        return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    """Handle AJAX prediction requests for future enrollment"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'})
        
        course = data.get('course', '').strip()
        year = data.get('year', '').strip()
        semester = data.get('semester', '').strip()
        
        if not all([course, year, semester]):
            return jsonify({'error': 'Missing required parameters'})
        
        # Validate year format and sequential years
        is_valid_year, year_message = validate_future_year(year)
        if not is_valid_year:
            return jsonify({'error': year_message})
        
        # Check if this is a historical period
        if not is_future_prediction(course, year, semester):
            return jsonify({
                'error': f'Historical data already exists for {course} in {year} (Semester {semester}). Please use the historical forecast feature instead.',
                'historical_data_available': True
            })
        
        # Use LSTM ensemble model for prediction if available
        if ml_models and 'lstm_models' in ml_models:
            prediction = predict_with_lstm_ensemble(course, year, semester)
            method = f"lstm_ensemble_{len(ml_models['lstm_models'])}_models"
            confidence_interval = [
                max(0, round(prediction * 0.85)),
                round(prediction * 1.15)
            ]
        else:
            prediction = predict_enrollment_fallback(course, year, semester)
            method = "fallback"
            confidence_interval = [
                max(0, round(prediction * 0.7)),
                round(prediction * 1.3)
            ]
        
        return jsonify({
            'prediction': prediction,
            'confidence_interval': confidence_interval,
            'prediction_method': method,
            'course': course,
            'year': year,
            'semester': semester
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route("/predict-enrollment", methods=["POST"])
def predict_enrollment():
    """Alternative prediction endpoint for BED page"""
    try:
        data = request.get_json()
        course = data.get('course', '').strip()
        year = data.get('year', '').strip()
        semester = data.get('semester', '').strip()
        
        if not all([course, year, semester]):
            return jsonify({'error': 'Missing required parameters'})
        
        # Validate year format and sequential years
        is_valid_year, year_message = validate_future_year(year)
        if not is_valid_year:
            return jsonify({'error': year_message})
        
        # Check if this is a historical period
        if not is_future_prediction(course, year, semester):
            return jsonify({
                'error': f'Historical data already exists for {course} in {year} (Semester {semester}). Please use the historical forecast feature instead.',
                'historical_data_available': True
            })
        
        # Use LSTM ensemble model for prediction
        if ml_models and 'lstm_models' in ml_models:
            prediction = predict_with_lstm_ensemble(course, year, semester)
            method = f"lstm_ensemble_{len(ml_models['lstm_models'])}_models"
        else:
            prediction = predict_enrollment_fallback(course, year, semester)
            method = "fallback"
        
        # Generate confidence intervals
        lower_bound = max(0, round(prediction * 0.85))
        upper_bound = round(prediction * 1.15)
        
        return jsonify({
            'predicted_enrollment': prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'course': course,
            'year': year,
            'semester': semester,
            'prediction_method': method
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/forecast", methods=["POST"])
def forecast():
    """Handle historical data visualization (requires dataset)"""
    if current_dataset is None:
        flash("Please upload a dataset to view historical data", "error")
        course = request.form.get("course", "").strip()
        if course.startswith('BSBA'):
            return redirect(url_for("bed_filter"))
        else:
            return redirect(url_for("ced_filter"))
    
    course = request.form.get("course", "").strip()
    year = request.form.get("year", "").strip()
    semester = request.form.get("semester", "").strip()

    if not course or not year or not semester:
        flash("Please complete all required fields", "error")
        if course.startswith('BSBA'):
            return redirect(url_for("bed_filter"))
        else:
            return redirect(url_for("ced_filter"))

    semester_text = "1st" if semester == "1" else "2nd"
    
    # Get historical data from dataset
    historical_enrollment = get_historical_enrollment(course, year, semester)
    
    if historical_enrollment is None:
        flash(f"No historical data found for {course} in {year} ({semester_text} Semester)", "error")
        if course.startswith('BSBA'):
            return redirect(url_for("bed_filter"))
        else:
            return redirect(url_for("ced_filter"))
    
    # Get historical trend for charts
    historical_trend = get_historical_trend(course)
    
    # Determine which template to use
    if course.startswith('BSBA'):
        template_name = "BED_result.html"
    else:
        template_name = "CED_result.html"
    
    model_count = len(ml_models.get('lstm_models', [])) if ml_models else 0
    
    return render_template(
        template_name,
        course=course,
        year=year,
        semester=semester_text,
        total_enrollment=historical_enrollment['total_enrollment'],
        year_levels={
            'first_year': historical_enrollment['first_year'],
            'second_year': historical_enrollment['second_year'],
            'third_year': historical_enrollment['third_year'],
            'fourth_year': historical_enrollment['fourth_year']
        },
        historical_trend=historical_trend,
        dataset_info=analyze_dataset(current_dataset) if current_dataset is not None else None,
        lstm_loaded=ml_models and 'lstm_models' in ml_models,
        model_count=model_count
    )

@app.route("/forecast_history", methods=['POST'])
def forecast_history():
    """Alias for forecast endpoint for CED page"""
    return forecast()

@app.route("/bed_filter")
def bed_filter():
    """BED filter page - ALWAYS ACCESSIBLE"""
    lstm_loaded = ml_models and 'lstm_models' in ml_models
    model_count = len(ml_models.get('lstm_models', [])) if lstm_loaded else 0
    return render_template("BEDfilter.html", lstm_loaded=lstm_loaded, model_count=model_count)

@app.route("/ced_filter")
def ced_filter():
    """CED filter page - ALWAYS ACCESSIBLE"""
    lstm_loaded = ml_models and 'lstm_models' in ml_models
    model_count = len(ml_models.get('lstm_models', [])) if lstm_loaded else 0
    return render_template("CEDfilter.html", lstm_loaded=lstm_loaded, model_count=model_count)

if __name__ == '__main__':
    print("=== Enrollment Forecasting System - COMPATIBLE SEQUENTIAL VERSION ===")
    print(f"Dataset loaded: {current_dataset is not None}")
    print(f"LSTM Models loaded: {ml_models and 'lstm_models' in ml_models}")
    
    if ml_models and 'lstm_models' in ml_models:
        print(f"Loaded {len(ml_models['lstm_models'])} LSTM models for ensemble")
        print("Loaded preprocessing components:", [k for k in ml_models.keys() if k != 'lstm_models'])
    
    if current_dataset is not None:
        print(f"Records: {len(current_dataset)}")
    
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)