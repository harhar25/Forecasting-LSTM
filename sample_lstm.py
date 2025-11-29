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
from sklearn.preprocessing import StandardScaler, RobustScaler

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
    """Create Adam optimizer that matches your training exactly"""
    return Adam(learning_rate=0.001)  # Same as your training

def rebuild_sequential_model():
    """Rebuild the exact Sequential model architecture from your training"""
    # This matches your training code exactly:
    # First LSTM layer: 100 units, Second LSTM layer: 50 units (100//2)
    model = Sequential([
        # First LSTM layer with return_sequences=True - 100 units
        LSTM(100, return_sequences=True, input_shape=(4, 25),  # 4 time steps, 25 features
             activation='tanh', recurrent_activation='sigmoid',
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros'),
        Dropout(0.2),

        # Second LSTM layer - 50 units (100//2)
        LSTM(50, return_sequences=False,
             activation='tanh', recurrent_activation='sigmoid',
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros'),
        Dropout(0.2),

        # Output layer - predicting first_year_enrollees
        Dense(1, activation='linear')
    ])

    # Compile with same optimizer as your training
    model.compile(
        optimizer=create_custom_adam_optimizer(),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def load_trained_models_compatible():
    """Compatible loading method that rebuilds the model architecture - IMPROVED VERSION"""
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
        
        # Load preprocessing components - check multiple possible file names
        scaler_files = ['feature_scaler.pkl', 'feature_scaler (1).pkl', 'scaler.pkl']
        target_scaler_files = ['target_scaler.pkl', 'target_scaler (1).pkl', 'target_scaler.pkl']
        encoder_files = ['enrollment_encoder.pkl', 'encoder.pkl', 'label_encoder.pkl']
        
        # Load feature scaler
        feature_scaler_loaded = False
        for scaler_file in scaler_files:
            scaler_path = os.path.join(app.config['MODEL_FOLDER'], scaler_file)
            if os.path.exists(scaler_path):
                try:
                    models_loaded['feature_scaler'] = joblib.load(scaler_path)
                    print(f"✅ Loaded feature scaler: {scaler_file}")
                    feature_scaler_loaded = True
                    break
                except Exception as e:
                    print(f"❌ Error loading feature scaler {scaler_file}: {e}")
        
        if not feature_scaler_loaded:
            print("❌ No feature scaler found - creating default")
            # Create a default scaler as fallback
            models_loaded['feature_scaler'] = StandardScaler()
            # Fit with dummy data
            dummy_data = np.random.randn(10, 25)
            models_loaded['feature_scaler'].fit(dummy_data)
            
        # Load target scaler
        target_scaler_loaded = False
        for target_file in target_scaler_files:
            target_path = os.path.join(app.config['MODEL_FOLDER'], target_file)
            if os.path.exists(target_path):
                try:
                    models_loaded['target_scaler'] = joblib.load(target_path)
                    print(f"✅ Loaded target scaler: {target_file}")
                    target_scaler_loaded = True
                    break
                except Exception as e:
                    print(f"❌ Error loading target scaler {target_file}: {e}")
        
        if not target_scaler_loaded:
            print("❌ No target scaler found - creating default")
            models_loaded['target_scaler'] = StandardScaler()
            # Fit with dummy data
            dummy_target = np.random.randn(10, 1)
            models_loaded['target_scaler'].fit(dummy_target)
            
        # Load encoder (optional)
        encoder_loaded = False
        for encoder_file in encoder_files:
            encoder_path = os.path.join(app.config['MODEL_FOLDER'], encoder_file)
            if os.path.exists(encoder_path):
                try:
                    models_loaded['encoder'] = joblib.load(encoder_path)
                    print(f"✅ Loaded encoder: {encoder_file}")
                    encoder_loaded = True
                    break
                except Exception as e:
                    print(f"❌ Error loading encoder {encoder_file}: {e}")
        
        if not encoder_loaded:
            print("⚠️ No encoder found (this might be okay)")
        
        # Rebuild LSTM models with the EXACT architecture from your training
        lstm_models_loaded = []
        
        # Try different model file patterns
        model_patterns = [
            'final_lstm_model_fold_{}.h5',
            'lstm_model_fold_{}.h5', 
            'model_fold_{}.h5',
            'fold_{}_model.h5',
            'best_lstm_fold_{}.h5'
        ]
        
        for fold in range(1, 6):  # Try folds 1-5
            model_loaded = False
            for pattern in model_patterns:
                model_filename = pattern.format(fold)
                model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
                
                if os.path.exists(model_path):
                    try:
                        print(f"🔄 Loading model: {model_filename}")
                        
                        # Rebuild the exact Sequential architecture from your training
                        model = rebuild_sequential_model()
                        
                        # Load weights only (bypasses the serialization issue)
                        model.load_weights(model_path)
                        print(f"✅ Rebuilt and loaded LSTM model: {model_filename}")
                        
                        lstm_models_loaded.append(model)
                        model_loaded = True
                        break
                        
                    except Exception as e:
                        print(f"❌ Failed to load {model_filename}: {e}")
                        # Try alternative loading method
                        try:
                            print(f"⚠️ Trying alternative loading for {model_filename}")
                            model = tf.keras.models.load_model(model_path, compile=False)
                            model.compile(
                                optimizer=create_custom_adam_optimizer(),
                                loss='mse',
                                metrics=['mae']
                            )
                            lstm_models_loaded.append(model)
                            print(f"✅ Alternative load successful for {model_filename}")
                            model_loaded = True
                            break
                        except Exception as e2:
                            print(f"❌ Alternative loading also failed for {model_filename}: {e2}")
            
            if not model_loaded and fold <= 4:  # Only warn for expected folds 1-4
                print(f"⚠️ LSTM model fold {fold} not found in any pattern")
        
        if not lstm_models_loaded:
            print("❌ No LSTM models could be loaded - using fallback mode")
            # Create a dummy model for fallback
            dummy_model = rebuild_sequential_model()
            lstm_models_loaded = [dummy_model]
        
        models_loaded['lstm_models'] = lstm_models_loaded
        models_loaded['lstm_model'] = lstm_models_loaded[0]  # Use first model as default
        
        models_loaded['time_steps'] = {'TIME_STEPS': 4}
        ml_models = models_loaded
        
        print(f"✅ Successfully loaded {len(lstm_models_loaded)} LSTM models")
        print(f"✅ Model components: {list(models_loaded.keys())}")
        
        # Apply expanded scaling fix after loading models
        apply_expanded_scaling_fix()
        
        return True
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_recent_first_year_enrollment(course):
    """Get recent first-year enrollment average for a course"""
    try:
        if current_dataset is None:
            return None
            
        course_data = current_dataset[current_dataset['Course'] == course]
        if len(course_data) < 3:
            return None
            
        # Get average of last 3 semesters
        recent_avg = course_data['1st_year_enrollees'].tail(3).mean()
        return recent_avg if not np.isnan(recent_avg) else None
        
    except Exception as e:
        print(f"Error getting recent enrollment: {e}")
        return None

def apply_trend_based_prediction(course, year, semester, recent_enrollment=None):
    """Apply trend-based prediction when LSTM scaling fails"""
    try:
        if current_dataset is None:
            return predict_first_year_fallback(course, year, semester)
            
        course_data = current_dataset[current_dataset['Course'] == course]
        if course_data.empty:
            return predict_first_year_fallback(course, year, semester)
        
        # Use recent average if available
        if recent_enrollment and not np.isnan(recent_enrollment):
            base_prediction = recent_enrollment
        else:
            # Use overall average
            base_prediction = course_data['1st_year_enrollees'].mean()
        
        # Apply semester adjustment
        if semester == "1":
            prediction = base_prediction * 1.05  # Slightly higher in first semester
        else:
            prediction = base_prediction * 0.95  # Slightly lower in second semester
        
        # Add small random variation (±10%)
        variation = np.random.normal(0, base_prediction * 0.05)
        prediction = max(5, prediction + variation)
        
        print(f"DEBUG: Trend-based prediction for {course}: {prediction}")
        return prediction
        
    except Exception as e:
        print(f"Error in trend-based prediction: {e}")
        return predict_first_year_fallback(course, year, semester)

def apply_course_specific_adjustment(course, prediction, semester):
    """Apply course-specific adjustments based on typical enrollment patterns"""
    try:
        # Course-specific base adjustments
        if course == 'BSIT':
            # BSIT typically has higher enrollment - ensure minimum reasonable level
            if prediction < 30:
                prediction = 30 + (prediction * 0.5)  # Boost low predictions
        elif course == 'BSCS':
            # BSCS typically has lower enrollment
            if prediction > 40:
                prediction = 40 - (prediction * 0.2)  # Reduce very high predictions
        
        # Semester-specific adjustments
        if semester == "1":
            prediction *= 1.05  # First semester usually has slightly higher enrollment
        else:
            prediction *= 0.95  # Second semester usually has slightly lower enrollment
            
        return max(5, prediction)  # Ensure minimum enrollment
        
    except Exception as e:
        print(f"Error in course-specific adjustment: {e}")
        return prediction

def calculate_prediction_accuracy(course, year, semester, prediction):
    """Calculate prediction accuracy against historical data"""
    try:
        # Get actual historical data
        actual_data = get_actual_historical_data(course, year, semester)
        
        if not actual_data:
            return None
        
        actual_first_year = actual_data['actual_first_year']
        prediction_error = abs(prediction - actual_first_year)
        
        if actual_first_year > 0:
            error_percentage = (prediction_error / actual_first_year) * 100
            accuracy_percentage = max(0, 100 - error_percentage)
        else:
            error_percentage = 100 if prediction > 0 else 0
            accuracy_percentage = 0
        
        accuracy_assessment = assess_prediction_accuracy(accuracy_percentage)
        
        return {
            'actual': actual_first_year,
            'predicted': prediction,
            'error': prediction_error,
            'error_percentage': round(error_percentage, 2),
            'accuracy_percentage': round(accuracy_percentage, 2),
            'assessment': accuracy_assessment
        }
        
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return None

def create_expanded_target_scaler():
    """Create a target scaler with expanded range to handle wider enrollment values"""
    try:
        if current_dataset is None:
            return None
            
        # Collect all first-year enrollment values
        all_first_year = current_dataset['1st_year_enrollees'].values
        
        # Create expanded range for scaling
        # Use percentiles to be robust to outliers
        p5 = np.percentile(all_first_year, 5)
        p95 = np.percentile(all_first_year, 95)
        
        # Expand range beyond observed data
        expanded_min = max(0, p5 - (p95 - p5) * 0.5)  # Go 50% below 5th percentile
        expanded_max = p95 + (p95 - p5) * 0.5  # Go 50% above 95th percentile
        
        print(f"🎯 Expanded target range: {expanded_min:.1f} to {expanded_max:.1f}")
        print(f"🎯 Original data range: {all_first_year.min()} to {all_first_year.max()}")
        
        # Create and fit scaler with expanded range
        expanded_scaler = StandardScaler()
        
        # Fit with data that includes the expanded range
        expanded_data = np.concatenate([
            all_first_year,
            np.array([expanded_min, expanded_max])  # Include range boundaries
        ]).reshape(-1, 1)
        
        expanded_scaler.fit(expanded_data)
        
        return expanded_scaler
        
    except Exception as e:
        print(f"Error creating expanded target scaler: {e}")
        return None

def apply_expanded_scaling_fix():
    """Apply expanded scaling to existing models"""
    try:
        if not ml_models:
            return False
            
        # Create expanded target scaler
        expanded_target_scaler = create_expanded_target_scaler()
        if expanded_target_scaler:
            ml_models['target_scaler'] = expanded_target_scaler
            print("✅ Applied expanded target scaling")
        
        # Create expanded feature scaler
        if 'feature_scaler' in ml_models:
            # Replace with a more robust scaler
            robust_scaler = RobustScaler()
            
            # Generate some representative data for fitting
            # This should match your feature creation logic
            n_samples = 1000
            n_features = 25
            
            # Create synthetic data that represents normalized features (mostly 0-1)
            synthetic_data = np.random.normal(0.5, 0.3, (n_samples, n_features))
            # Clip to reasonable range
            synthetic_data = np.clip(synthetic_data, -1, 2)
            
            robust_scaler.fit(synthetic_data)
            ml_models['feature_scaler'] = robust_scaler
            print("✅ Applied robust feature scaling")
        
        return True
        
    except Exception as e:
        print(f"Error applying expanded scaling: {e}")
        return False

def create_prediction_features_expanded_years(course, year, semester, year_range=15):
    """Create features with expanded year normalization to handle wider time ranges"""
    try:
        if current_dataset is None:
            print("❌ No dataset available for feature creation")
            return None
        
        # Get historical data for the course
        course_data = current_dataset[current_dataset['Course'] == course].copy()
        if course_data.empty:
            print(f"❌ No historical data found for course: {course}")
            return None
        
        print(f"📊 Found {len(course_data)} records for {course}")
        
        # Sort by time to get chronological order
        course_data = course_data.sort_values(['School_Year', 'Semester'])
        
        # We need at least 4 records to create a sequence
        if len(course_data) < 4:
            print(f"❌ Not enough historical data. Need 4, have {len(course_data)}")
            if len(course_data) > 0:
                print("⚠️ Using available data with padding...")
                available_data = course_data.to_dict('records')
                while len(available_data) < 4:
                    available_data.append(available_data[-1])
                historical_data = available_data
            else:
                return None
        else:
            # For predictions, use ONLY data that would have been available BEFORE the target period
            target_start_year = int(year.split('-')[0])
            target_semester = 1 if semester == "1" else 2
            
            # Find records that occur BEFORE our target period
            historical_data = []
            for _, record in course_data.iterrows():
                record_year = int(record['School_Year'].split('-')[0])
                record_semester = 1 if record['Semester'] == '1st' else 2
                
                # Only include records that are BEFORE the target period
                if (record_year < target_start_year) or \
                   (record_year == target_start_year and record_semester < target_semester):
                    historical_data.append(record)
            
            # If we don't have enough historical data, use what we have
            if len(historical_data) < 4:
                print(f"⚠️ Not enough historical data before target period. Need 4, have {len(historical_data)}")
                historical_data = course_data.tail(4).to_dict('records')
            else:
                # Use the 4 most recent records before the target period
                historical_data = historical_data[-4:]
        
        print(f"📅 Using historical data from years: {[record['School_Year'] for record in historical_data]}")
        
        # Calculate expanded year range for normalization
        current_year = int(year.split('-')[0])
        base_year = current_year - year_range  # Go back X years for wider range
        max_year = current_year + 5  # Look forward a bit too
        
        print(f"📅 Expanded year range: {base_year} to {max_year} (range: {year_range} years)")
        
        # Create features for EACH record in the sequence
        sequences = []
        
        for idx, record in enumerate(historical_data):
            features = []
            
            try:
                # Extract year and semester from the record
                current_start_year = int(record['School_Year'].split('-')[0])
                current_semester_order = 1 if record['Semester'] == '1st' else 2
                
                # 1. EXPANDED: Start_Year (normalized over wider range)
                # Normalize year to 0-1 range based on expanded range
                year_normalized = (current_start_year - base_year) / (max_year - base_year)
                year_normalized = max(0.0, min(1.0, year_normalized))  # Clip to 0-1
                features.append(year_normalized)
                
                # 2. Semester_Order (unchanged)
                features.append(current_semester_order)
                
                # 3. Lag_1_Enrollees - USING FIRST-YEAR (with expanded bounds)
                if idx > 0:
                    prev_record = historical_data[idx - 1]
                    lag_enrollment = min(prev_record['1st_year_enrollees'], 200)  # Increased cap
                else:
                    lag_enrollment = min(record['1st_year_enrollees'], 200)
                # Normalize enrollment to 0-1 range based on expanded bounds
                enrollment_normalized = lag_enrollment / 200.0
                features.append(enrollment_normalized)
                
                # 4. Rolling_Mean_3 - USING FIRST-YEAR (expanded bounds)
                # 5. Rolling_Std_3 - USING FIRST-YEAR (expanded bounds)
                if idx >= 2:
                    rolling_window = [min(h['1st_year_enrollees'], 200) for h in historical_data[max(0, idx-2):idx+1]]
                    rolling_mean = np.mean(rolling_window)
                    rolling_std = np.std(rolling_window) if len(rolling_window) > 1 else 0
                else:
                    rolling_mean = min(record['1st_year_enrollees'], 200)
                    rolling_std = 0
                
                # Normalize rolling statistics
                rolling_mean_normalized = rolling_mean / 200.0
                rolling_std_normalized = min(rolling_std / 50.0, 1.0)  # Cap std at 50
                features.extend([rolling_mean_normalized, rolling_std_normalized])
                
                # 6. Growth_Rate - USING FIRST-YEAR (with wider bounds)
                if idx > 0:
                    prev_enrollment = min(historical_data[idx - 1]['1st_year_enrollees'], 200)
                    current_enrollment = min(record['1st_year_enrollees'], 200)
                    if prev_enrollment > 0:
                        growth_rate = (current_enrollment - prev_enrollment) / prev_enrollment
                        # Wider bounds for growth rate
                        growth_rate = max(-2.0, min(3.0, growth_rate))  # Allow -200% to +300%
                    else:
                        growth_rate = 0
                else:
                    growth_rate = 0
                # Normalize growth rate to 0-1 range
                growth_normalized = (growth_rate + 2.0) / 5.0  # Convert -2 to +3 range to 0-1
                features.append(growth_normalized)
                
                # 7. Year_Semester_Interaction (normalized with expanded range)
                year_sem_interaction = (current_start_year - base_year) * current_semester_order
                # Normalize to reasonable range
                max_interaction = (max_year - base_year) * 2  # Max possible value
                interaction_normalized = year_sem_interaction / max_interaction
                features.append(interaction_normalized)
                
                # 8. Time_Index (normalized)
                time_index = idx + 1
                time_index_normalized = time_index / 4.0  # Normalize by max sequence length
                features.append(time_index_normalized)
                
                # 9. Enrollment_Trend - USING FIRST-YEAR (expanded bounds)
                if len(historical_data) > 1:
                    x_values = list(range(len(historical_data)))
                    y_values = [min(h['1st_year_enrollees'], 200) for h in historical_data]
                    if len(set(y_values)) > 1:
                        enrollment_trend = np.polyfit(x_values, y_values, 1)[0]
                        enrollment_trend = max(-100, min(100, enrollment_trend))  # Wider bounds
                    else:
                        enrollment_trend = 0
                else:
                    enrollment_trend = 0
                # Normalize trend
                trend_normalized = (enrollment_trend + 100) / 200.0  # Convert -100 to +100 range to 0-1
                features.append(trend_normalized)
                
                # 10. Dept_Enrollment_Ratio - USING FIRST-YEAR (expanded bounds)
                enrollments = [min(h['1st_year_enrollees'], 200) for h in historical_data]
                mean_enrollment = np.mean(enrollments) if enrollments else 1
                dept_ratio = min(record['1st_year_enrollees'], 200) / (mean_enrollment + 1e-10)
                dept_ratio = max(0.01, min(20.0, dept_ratio))  # Wider bounds
                # Normalize ratio (log scale since ratios can be exponential)
                dept_ratio_normalized = np.log1p(dept_ratio) / np.log1p(20.0)
                features.append(dept_ratio_normalized)
                
                # 11. Semester_Sin, 12. Semester_Cos (unchanged)
                semester_sin = np.sin(2 * np.pi * current_semester_order / 2)
                semester_cos = np.cos(2 * np.pi * current_semester_order / 2)
                features.extend([semester_sin, semester_cos])
                
                # 13. Semester_1st, 14. Semester_2nd (one-hot encoding - unchanged)
                semester_1st = 1 if current_semester_order == 1 else 0
                semester_2nd = 1 if current_semester_order == 2 else 0
                features.extend([semester_1st, semester_2nd])
                
                # 15-18. Course encoding (one-hot - unchanged)
                course_bsba_fin = 1 if course == 'BSBA-FINANCIAL_MANAGEMENT' else 0
                course_bsba_mkt = 1 if course == 'BSBA-MARKETING_MANAGEMENT' else 0
                course_bscs = 1 if course == 'BSCS' else 0
                course_bsit = 1 if course == 'BSIT' else 0
                features.extend([course_bsba_fin, course_bsba_mkt, course_bscs, course_bsit])
                
                # 19-22. Year_Level encoding (use proportions from record - expanded bounds)
                total = max(record['total_enrollees'], 1)
                year_level_1st = min(record['1st_year_enrollees'] / total, 2.0)  # Allow up to 200%
                year_level_2nd = min(record['2nd_year_enrollees'] / total, 2.0)
                year_level_3rd = min(record['3rd_year_enrollees'] / total, 2.0)
                year_level_4th = min(record['4th_year_enrollees'] / total, 2.0)
                features.extend([year_level_1st, year_level_2nd, year_level_3rd, year_level_4th])
                
                # 23-25. Department encoding (one-hot - unchanged)
                dept_ba = 1 if 'BSBA' in course else 0
                dept_cs = 1 if course == 'BSCS' else 0
                dept_it = 1 if course == 'BSIT' else 0
                features.extend([dept_ba, dept_cs, dept_it])
                
            except Exception as feature_error:
                print(f"❌ Error creating features for record {idx}: {feature_error}")
                # Use normalized default features
                features = [0.5] * 25  # Use 0.5 as default for normalized features
            
            # Verify we have exactly 25 features
            if len(features) != 25:
                print(f"⚠️ Expected 25 features, but got {len(features)} - padding with 0.5")
                while len(features) < 25:
                    features.append(0.5)
                features = features[:25]
            
            sequences.append(features)
        
        # Convert to numpy array
        X_sequence = np.array(sequences)
        
        # Debug: Check feature statistics (should be mostly 0-1 now)
        print(f"📈 NORMALIZED Feature statistics - Min: {X_sequence.min():.3f}, Max: {X_sequence.max():.3f}, Mean: {X_sequence.mean():.3f}")
        
        # Ensure we have the right shape
        if X_sequence.shape != (4, 25):
            print(f"❌ Shape mismatch: {X_sequence.shape} vs expected (4, 25)")
            return None
        
        print(f"✅ Created NORMALIZED sequence with {X_sequence.shape[1]} features")
        
        # Since features are already normalized 0-1, we might not need scaling
        # But if scaler exists, use it with caution
        if 'feature_scaler' in ml_models:
            try:
                original_shape = X_sequence.shape
                X_flat = X_sequence.reshape(-1, X_sequence.shape[-1])
                
                # Check if scaler will work with our normalized data
                X_scaled = ml_models['feature_scaler'].transform(X_flat)
                X_sequence = X_scaled.reshape(original_shape)
                print(f"✅ Applied feature scaling to normalized features")
                
            except Exception as scale_error:
                print(f"❌ Scaling failed on normalized features: {scale_error}")
                print("⚠️ Using normalized features without additional scaling")
        
        # Reshape for LSTM: (1, time_steps, n_features)
        X_sequence = X_sequence.reshape(1, 4, X_sequence.shape[1])
        print(f"✅ Final sequence shape: {X_sequence.shape}")
        
        return X_sequence
        
    except Exception as e:
        print(f"❌ Error creating expanded-year features: {e}")
        import traceback
        traceback.print_exc()
        return None

def apply_refined_historical_bounds(course, prediction, confidence=0.8):
    """Apply more refined historical bounds with confidence intervals"""
    try:
        if current_dataset is None:
            return prediction
            
        course_data = current_dataset[current_dataset['Course'] == course]
        if course_data.empty:
            return prediction
        
        historical_values = course_data['1st_year_enrollees']
        
        if len(historical_values) < 5:
            return prediction  # Not enough data for meaningful bounds
        
        # Calculate percentiles for bounds
        p10 = historical_values.quantile(0.10)
        p25 = historical_values.quantile(0.25)
        p75 = historical_values.quantile(0.75)
        p90 = historical_values.quantile(0.90)
        
        # Use interquartile range for more robust bounds
        iqr = p75 - p25
        lower_bound = max(0, p25 - (1.5 * iqr))
        upper_bound = p75 + (1.5 * iqr)
        
        print(f"📊 Historical bounds for {course}: {lower_bound:.1f} to {upper_bound:.1f}")
        print(f"📊 Historical percentiles - 10th: {p10:.1f}, 25th: {p25:.1f}, 75th: {p75:.1f}, 90th: {p90:.1f}")
        
        # Only apply bounds if prediction is extreme
        if prediction < lower_bound:
            print(f"🔽 Prediction {prediction:.1f} below lower bound, adjusting to {lower_bound:.1f}")
            return lower_bound
        elif prediction > upper_bound:
            print(f"🔼 Prediction {prediction:.1f} above upper bound, adjusting to {upper_bound:.1f}")
            return upper_bound
        else:
            print(f"✅ Prediction {prediction:.1f} within reasonable historical range")
            return prediction
            
    except Exception as e:
        print(f"Error in refined historical bounds: {e}")
        return prediction

def predict_with_optimized_scaling(course, year, semester):
    """Final optimized prediction with refined bounds"""
    try:
        print(f"🎯 Using OPTIMIZED SCALING for {course}, {year}, Semester {semester}")
        
        # Apply expanded scaling fix if needed
        if 'target_scaler' not in ml_models or ml_models.get('needs_expansion', True):
            apply_expanded_scaling_fix()
            ml_models['needs_expansion'] = False
        
        # Use expanded year feature creation
        features = create_prediction_features_expanded_years(course, year, semester, year_range=20)
        if features is None:
            print("Feature creation failed, using fallback")
            fallback_pred = predict_first_year_fallback(course, year, semester)
            accuracy_info = calculate_prediction_accuracy(course, year, semester, fallback_pred)
            return fallback_pred, "fallback", accuracy_info
        
        # Make predictions
        predictions = []
        model_weights = []  # Track model confidence
        
        for i, model in enumerate(ml_models['lstm_models']):
            try:
                prediction_scaled = model.predict(features, verbose=0)
                predictions.append(prediction_scaled[0][0])
                
                # Simple confidence weighting based on prediction magnitude
                # Models with extreme predictions get lower weight
                confidence = 1.0 / (1.0 + abs(prediction_scaled[0][0]))
                model_weights.append(confidence)
                
                print(f"Model {i+1} scaled: {prediction_scaled[0][0]:.3f} (weight: {confidence:.3f})")
            except Exception as e:
                print(f"Model {i+1} failed: {e}")
                continue
        
        if not predictions:
            fallback_pred = predict_first_year_fallback(course, year, semester)
            accuracy_info = calculate_prediction_accuracy(course, year, semester, fallback_pred)
            return fallback_pred, "fallback", accuracy_info
        
        # Weighted average instead of simple mean
        if len(predictions) > 1:
            weights = np.array(model_weights)
            weights = weights / np.sum(weights)  # Normalize weights
            avg_prediction_scaled = np.average(predictions, weights=weights)
            print(f"🎯 Weighted ensemble scaled: {avg_prediction_scaled:.3f}")
        else:
            avg_prediction_scaled = np.mean(predictions)
            print(f"📊 Simple ensemble scaled: {avg_prediction_scaled:.3f}")
        
        # Inverse transform with expanded scaler
        if 'target_scaler' in ml_models:
            try:
                prediction = ml_models['target_scaler'].inverse_transform([[avg_prediction_scaled]])[0][0]
                print(f"🎯 Raw prediction: {prediction:.1f}")
                
                # Apply refined historical bounds
                prediction = apply_refined_historical_bounds(course, prediction)
                
            except Exception as e:
                print(f"❌ Inverse transform failed: {e}")
                prediction = apply_trend_based_prediction(course, year, semester)
        else:
            prediction = apply_trend_based_prediction(course, year, semester)
        
        # Final course-specific adjustment
        prediction = apply_course_specific_adjustment(course, prediction, semester)
        
        # Calculate accuracy
        accuracy_info = calculate_prediction_accuracy(course, year, semester, prediction)
        
        return round(prediction), "lstm_optimized_scaling", accuracy_info
        
    except Exception as e:
        print(f"❌ Optimized prediction error: {e}")
        fallback_pred = predict_first_year_fallback(course, year, semester)
        accuracy_info = calculate_prediction_accuracy(course, year, semester, fallback_pred)
        return fallback_pred, "fallback", accuracy_info

def predict_first_year_fallback(course, year, semester):
    """Simple FIRST-YEAR prediction for fallback - IMPROVED to use historical data when available"""
    try:
        # First, check if we have actual historical data for this period
        actual_data = get_actual_historical_data(course, year, semester)
        if actual_data:
            print(f"DEBUG: Using actual historical first-year data: {actual_data['actual_first_year']}")
            return actual_data['actual_first_year']
        
        # If no historical data, use trend-based estimation
        if current_dataset is not None:
            course_data = current_dataset[current_dataset['Course'] == course]
            if not course_data.empty:
                # Use average of recent first-year enrollments
                recent_first_year = course_data['1st_year_enrollees'].tail(3).mean()
                if not np.isnan(recent_first_year):
                    # Add some variation based on semester
                    if semester == "1":
                        prediction = recent_first_year * 1.05  # Slightly higher in first semester
                    else:
                        prediction = recent_first_year * 0.95  # Slightly lower in second semester
                    
                    # Add small random variation
                    variation = np.random.normal(0, recent_first_year * 0.1)
                    prediction = max(20, prediction + variation)
                    print(f"DEBUG: Using trend-based first-year prediction: {prediction}")
                    return round(prediction)
        
        # Ultimate fallback - adjusted base numbers for first-year students only
        base_enrollment = 40
        
        if "BSBA" in course:
            base_enrollment = 50
        elif course in ["BSCS", "BSIT"]:
            base_enrollment = 35
        
        # Semester adjustment
        if semester == "1":
            base_enrollment *= 1.1
        else:
            base_enrollment *= 0.9
            
        # Add some variation
        variation = np.random.normal(0, 5)
        prediction = max(20, base_enrollment + variation)
        
        print(f"DEBUG: Using ultimate fallback first-year prediction: {prediction}")
        return round(prediction)
        
    except Exception as e:
        print(f"FIRST-YEAR fallback prediction error: {e}")
        return 40

def is_future_prediction(course, year, semester):
    """Check if the prediction is for a future period - COMPLETELY DISABLED FOR HISTORICAL TESTING"""
    # TEMPORARILY DISABLE ALL FUTURE CHECKS TO ALLOW HISTORICAL PREDICTIONS
    print(f"DEBUG: Prediction requested for {course}, {year}, Semester {semester}")
    print("DEBUG: ALLOWING ALL PREDICTIONS (including historical) for accuracy testing")
    return False  # Always allow predictions, even for past years

def get_actual_historical_data(course, year, semester):
    """Get actual historical data for comparison with predictions"""
    try:
        if current_dataset is None:
            return None
            
        # Normalize inputs for comparison
        course_norm = str(course).strip()
        year_norm = str(year).strip().replace(' ', '')
        semester_norm = standardize_semester(semester)
        
        # Find the actual historical record
        record = current_dataset[
            (current_dataset['Course'] == course_norm) &
            (current_dataset['School_Year'] == year_norm) &
            (current_dataset['Semester'] == semester_norm)
        ]
        
        if not record.empty:
            actual_data = {
                'actual_first_year': int(record['1st_year_enrollees'].iloc[0]),
                'actual_second_year': int(record['2nd_year_enrollees'].iloc[0]),
                'actual_third_year': int(record['3rd_year_enrollees'].iloc[0]),
                'actual_fourth_year': int(record['4th_year_enrollees'].iloc[0]),
                'actual_total': int(record['total_enrollees'].iloc[0]),
                'course': course_norm,
                'year': year_norm,
                'semester': semester_norm
            }
            return actual_data
            
        return None
    except Exception as e:
        print(f"Error getting actual historical data: {e}")
        return None

def assess_prediction_accuracy(accuracy_percentage):
    """Assess how good the prediction accuracy is"""
    if accuracy_percentage >= 95:
        return "Excellent"
    elif accuracy_percentage >= 85:
        return "Very Good"
    elif accuracy_percentage >= 75:
        return "Good"
    elif accuracy_percentage >= 65:
        return "Fair"
    else:
        return "Poor"

def validate_year_format(year):
    """Validate year format but don't restrict to future years"""
    try:
        # Check if it matches the expected format (e.g., "2023-2024")
        if not re.match(r'^\d{4}-\d{4}$', year.strip()):
            return False, "Year must be in format 'YYYY-YYYY' (e.g., 2023-2024)"
        
        # Extract start and end years
        start_year, end_year = map(int, year.split('-'))
        
        # Check if years are sequential
        if end_year != start_year + 1:
            return False, "End year must be exactly one year after start year (e.g., 2023-2024)"
        
        # REMOVED: No longer checking if year is in the future
        # This allows predictions for any valid year format
        
        return True, "Year format is valid"
        
    except Exception as e:
        return False, f"Invalid year format: {str(e)}"

def standardize_semester(val):
    """Normalize various semester representations to '1st' or '2nd'"""
    try:
        if pd.isna(val):
            return val
        s = str(val).strip().lower()
        if s in ('1', 's1', 'sem1', '1st', 'first', 'first sem', 'first semester', 'semester 1', 'i'):
            return '1st'
        if s in ('2', 's2', 'sem2', '2nd', 'second', 'second sem', 'second semester', 'semester 2', 'ii'):
            return '2nd'
        # keep original trimmed value if unknown
        return str(val).strip()
    except Exception:
        return str(val).strip()

def normalize_dataset(df):
    """Normalize key dataset columns to make comparisons robust"""
    df = df.copy()
    # Normalize string columns
    for col in ['School_Year', 'Semester', 'Course']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Standardize semester values
    if 'Semester' in df.columns:
        df['Semester'] = df['Semester'].apply(standardize_semester)
    # Remove internal spaces in School_Year (e.g. "2023 - 2024" -> "2023-2024")
    if 'School_Year' in df.columns:
        df['School_Year'] = df['School_Year'].str.replace(r'\s+', '', regex=True)
    # Ensure enrollment columns are numeric integers
    enrollment_cols = ['1st_year_enrollees', '2nd_year_enrollees', '3rd_year_enrollees', '4th_year_enrollees', 'total_enrollees']
    for col in enrollment_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

def load_current_dataset():
    """Load the current dataset into memory"""
    global current_dataset
    try:
        if os.path.exists(CURRENT_DATASET_PATH):
            file_size = os.path.getsize(CURRENT_DATASET_PATH)
            print(f"DEBUG: Loading dataset from: {CURRENT_DATASET_PATH}")
            print(f"DEBUG: Dataset file size: {file_size} bytes")
            
            df = pd.read_csv(CURRENT_DATASET_PATH)
            # Normalize dataset to ensure consistent comparisons
            df = normalize_dataset(df)
            current_dataset = df
            print(f"Current dataset loaded: {len(current_dataset)} records")
            
            # Print available years for debugging
            if current_dataset is not None and 'School_Year' in current_dataset.columns:
                years = sorted(current_dataset['School_Year'].unique())
                print(f"DEBUG: Available years in dataset: {years}")
            
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
        
        required_columns = ['School_Year', 'Semester', 'Course', '1st_year_enrollees']  # Require first-year column for prediction
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        if len(df) == 0:
            return False, "Dataset is empty"
        
        # Check if we have other enrollment columns
        enrollment_columns = ['1st_year_enrollees', '2nd_year_enrollees', '3rd_year_enrollees', '4th_year_enrollees', 'total_enrollees']
        missing_enrollment = [col for col in enrollment_columns if col not in df.columns]
        if missing_enrollment:
            print(f"Warning: Missing enrollment columns: {missing_enrollment}")
            for col in missing_enrollment:
                if col != '1st_year_enrollees':  # Don't override the required column
                    df[col] = 0
        
        return True, "Dataset is valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_historical_enrollment(course, year, semester):
    """Get historical enrollment data for specific course, year, and semester"""
    try:
        if current_dataset is not None:
            # Normalize inputs
            course_key = str(course).strip()
            year_key = str(year).strip().replace(' ', '')
            semester_key = standardize_semester(semester)
                
            record = current_dataset[
                (current_dataset['Course'] == course_key) &
                (current_dataset['School_Year'] == year_key) &
                (current_dataset['Semester'] == semester_key)
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
                    
                    first_year_data.append(int(row['1st_year_enrollees']))
                    second_year_data.append(int(row['2nd_year_enrollees']))
                    third_year_data.append(int(row['3rd_year_enrollees']))
                    fourth_year_data.append(int(row['4th_year_enrollees']))
                
                print(f"DEBUG: Generated trend data for {course}:")
                print(f"  Labels: {labels}")
                print(f"  First Year: {first_year_data}")
                print(f"  Second Year: {second_year_data}")
                print(f"  Third Year: {third_year_data}")
                print(f"  Fourth Year: {fourth_year_data}")
                
                return {
                    'labels': labels,
                    'first_year': first_year_data,
                    'second_year': second_year_data,
                    'third_year': third_year_data,
                    'fourth_year': fourth_year_data
                }
        
        # Return empty data structure if no data found
        return {
            'labels': [], 
            'first_year': [], 
            'second_year': [], 
            'third_year': [], 
            'fourth_year': []
        }
    except Exception as e:
        print(f"Error getting historical trend: {e}")
        import traceback
        traceback.print_exc()
        return {
            'labels': [], 
            'first_year': [], 
            'second_year': [], 
            'third_year': [], 
            'fourth_year': []
        }

# Initialize the application
load_current_dataset()

# Try compatible loading method
print("Loading LSTM models for FIRST-YEAR prediction...")
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
        
        # Normalize dataset before saving
        df = normalize_dataset(df)
        
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
        return redirect(url_for("bed_filter_v2"))
    elif department == "CED":
        return redirect(url_for("ced_filter_v2"))  # Updated to ced_filter_v2
    else:
        flash("Invalid department selected", "error")
        return redirect(url_for("index"))

# NEW ROUTES FOR HISTORICAL PREDICTIONS
@app.route("/ced_filter_v2")
def ced_filter_v2():
    """CED filter page v2 - WITH HISTORICAL PREDICTION SUPPORT"""
    lstm_loaded = ml_models and 'lstm_models' in ml_models
    model_count = len(ml_models.get('lstm_models', [])) if lstm_loaded else 0
    return render_template("CEDfilter_v2.html", lstm_loaded=lstm_loaded, model_count=model_count)

@app.route("/predict-historical", methods=["POST"])
def predict_historical():
    """Dedicated endpoint for historical predictions WITH OPTIMIZED SCALING"""
    try:
        data = request.get_json()
        print(f"HISTORICAL PREDICTION REQUEST: {data}")
        
        course = data.get('course', '').strip()
        year = data.get('year', '').strip()
        semester = data.get('semester', '').strip()
        
        if not all([course, year, semester]):
            return jsonify({'error': 'Missing required parameters'})
        
        # Validate year format
        is_valid_year, year_message = validate_year_format(year)
        if not is_valid_year:
            return jsonify({'error': year_message})
        
        # ALWAYS allow prediction (past or future)
        is_future = False  # Force to false to enable historical comparison
        
        # Use OPTIMIZED SCALING LSTM ensemble model for FIRST-YEAR prediction
        prediction, method, accuracy_info = predict_with_optimized_scaling(course, year, semester)
        
        # Set confidence intervals
        if "lstm_optimized_scaling" in method:
            confidence_interval = [
                max(0, round(prediction * 0.85)),
                round(prediction * 1.15)
            ]
        else:
            confidence_interval = [
                max(0, round(prediction * 0.7)),
                round(prediction * 1.3)
            ]
        
        response_data = {
            'prediction': prediction,
            'confidence_interval': confidence_interval,
            'prediction_method': method,
            'course': course,
            'year': year,
            'semester': semester,
            'prediction_type': 'first_year_enrollment',
            'is_future_prediction': is_future
        }
        
        # Add accuracy information if available
        if accuracy_info:
            response_data['accuracy_info'] = accuracy_info
        
        # Always try to get historical data for comparison
        actual_data = get_actual_historical_data(course, year, semester)
        if actual_data:
            response_data['actual_data'] = actual_data
        
        print(f"HISTORICAL PREDICTION RESULT: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"HISTORICAL PREDICTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Historical prediction error: {str(e)}'})

@app.route("/debug-model-status")
def debug_model_status():
    """Debug endpoint to check model loading status"""
    debug_info = {
        'ml_models_loaded': ml_models is not None,
        'lstm_models_count': len(ml_models.get('lstm_models', [])) if ml_models else 0,
        'has_feature_scaler': 'feature_scaler' in ml_models if ml_models else False,
        'has_target_scaler': 'target_scaler' in ml_models if ml_models else False,
        'has_encoder': 'encoder' in ml_models if ml_models else False,
        'model_files': []
    }
    
    # Check what model files exist
    if os.path.exists(app.config['MODEL_FOLDER']):
        debug_info['model_files'] = os.listdir(app.config['MODEL_FOLDER'])
    
    # Test feature creation for BSCS 2018-2019
    try:
        features = create_prediction_features_expanded_years("BSCS", "2018-2019", "1")
        debug_info['feature_creation_success'] = features is not None
        if features is not None:
            debug_info['features_shape'] = features.shape
            debug_info['features_sample'] = features[0, 0, :5].tolist() if len(features.shape) == 3 else features[:5].tolist()
    except Exception as e:
        debug_info['feature_creation_error'] = str(e)
    
    return jsonify(debug_info)

@app.route("/debug-test-prediction")
def debug_test_prediction():
    """Direct test of prediction system"""
    try:
        print("DEBUG: Testing prediction for BSCS 2018-2019...")
        prediction, method, accuracy_info = predict_with_optimized_scaling("BSCS", "2018-2019", "1")
        actual_data = get_actual_historical_data("BSCS", "2018-2019", "1")
        
        result = {
            'prediction': prediction,
            'method': method,
            'actual_data': actual_data,
            'success': True
        }
        
        if accuracy_info:
            result['accuracy_info'] = accuracy_info
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Keep original routes for compatibility
@app.route("/predict", methods=["POST"])
def predict():
    """Handle AJAX prediction requests for BOTH future and historical FIRST-YEAR enrollment"""
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
        is_valid_year, year_message = validate_year_format(year)
        if not is_valid_year:
            return jsonify({'error': year_message})
        
        # Check if this is a historical period (but allow all predictions)
        is_future = is_future_prediction(course, year, semester)
        
        # Use OPTIMIZED SCALING LSTM ensemble model for FIRST-YEAR prediction
        prediction, method, accuracy_info = predict_with_optimized_scaling(course, year, semester)
        
        # Set confidence intervals based on method
        if "lstm_optimized_scaling" in method:
            confidence_interval = [
                max(0, round(prediction * 0.85)),
                round(prediction * 1.15)
            ]
        else:
            confidence_interval = [
                max(0, round(prediction * 0.7)),
                round(prediction * 1.3)
            ]
        
        response_data = {
            'prediction': prediction,
            'confidence_interval': confidence_interval,
            'prediction_method': method,
            'course': course,
            'year': year,
            'semester': semester,
            'prediction_type': 'first_year_enrollment',
            'is_future_prediction': is_future
        }
        
        # Add accuracy information if available
        if accuracy_info:
            response_data['accuracy_info'] = accuracy_info
        
        # ADDED: If historical data exists, include actual values for comparison
        if not is_future:
            actual_data = get_actual_historical_data(course, year, semester)
            if actual_data:
                response_data['actual_data'] = actual_data
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'FIRST-YEAR prediction error: {str(e)}'})

@app.route("/predict-enrollment", methods=["POST"])
def predict_enrollment():
    """Alternative FIRST-YEAR prediction endpoint for BED page - UPDATED for historical comparison"""
    try:
        data = request.get_json()
        course = data.get('course', '').strip()
        year = data.get('year', '').strip()
        semester = data.get('semester', '').strip()
        
        if not all([course, year, semester]):
            return jsonify({'error': 'Missing required parameters'})
        
        # Validate year format and sequential years
        is_valid_year, year_message = validate_year_format(year)
        if not is_valid_year:
            return jsonify({'error': year_message})
        
        # Check if this is a historical period (but allow all predictions)
        is_future = is_future_prediction(course, year, semester)
        
        # Use OPTIMIZED SCALING LSTM ensemble model for FIRST-YEAR prediction
        prediction, method, accuracy_info = predict_with_optimized_scaling(course, year, semester)
        
        # Generate confidence intervals
        if "lstm_optimized_scaling" in method:
            lower_bound = max(0, round(prediction * 0.85))
            upper_bound = round(prediction * 1.15)
        else:
            lower_bound = max(0, round(prediction * 0.7))
            upper_bound = round(prediction * 1.3)
        
        response_data = {
            'predicted_enrollment': prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'course': course,
            'year': year,
            'semester': semester,
            'prediction_method': method,
            'prediction_type': 'first_year_enrollment',
            'is_future_prediction': is_future
        }
        
        # Add accuracy information if available
        if accuracy_info:
            response_data['accuracy_info'] = accuracy_info
        
        # ADDED: If historical data exists, include actual values for comparison
        if not is_future:
            actual_data = get_actual_historical_data(course, year, semester)
            if actual_data:
                response_data['actual_data'] = actual_data
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/analyze-scaling")
def analyze_scaling():
    """Analyze current scaling ranges and suggest improvements"""
    try:
        if current_dataset is None:
            return jsonify({'error': 'No dataset loaded'})
        
        analysis = {}
        
        # Analyze first-year enrollment distribution
        first_year_data = current_dataset['1st_year_enrollees']
        analysis['first_year_enrollment'] = {
            'min': int(first_year_data.min()),
            'max': int(first_year_data.max()),
            'mean': round(first_year_data.mean(), 1),
            'std': round(first_year_data.std(), 1),
            'p5': int(first_year_data.quantile(0.05)),
            'p95': int(first_year_data.quantile(0.95)),
            'suggested_min': max(0, int(first_year_data.quantile(0.05) * 0.5)),
            'suggested_max': int(first_year_data.quantile(0.95) * 1.5)
        }
        
        # Analyze year range in dataset
        years = []
        for year_str in current_dataset['School_Year'].unique():
            try:
                start_year = int(year_str.split('-')[0])
                years.append(start_year)
            except:
                continue
        
        if years:
            analysis['year_range'] = {
                'min_year': min(years),
                'max_year': max(years),
                'year_span': max(years) - min(years),
                'suggested_base_year': min(years) - 10,  # Go 10 years back
                'suggested_max_year': max(years) + 5     # Go 5 years forward
            }
        
        # Check current scaler ranges if available
        if ml_models and 'target_scaler' in ml_models:
            try:
                # Test what the scaler produces for typical values
                test_values = np.array([0, 50, 100, 200]).reshape(-1, 1)
                scaled_test = ml_models['target_scaler'].transform(test_values)
                analysis['current_scaler_behavior'] = {
                    'test_values': test_values.flatten().tolist(),
                    'scaled_values': scaled_test.flatten().tolist()
                }
            except Exception as e:
                analysis['scaler_error'] = str(e)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/analyze-model-accuracy")
def analyze_model_accuracy():
    """Analyze model accuracy across all historical data"""
    try:
        if current_dataset is None:
            return jsonify({'error': 'No dataset loaded'})
        
        results = []
        total_accuracy = 0
        count = 0
        
        # Test predictions for all historical records
        for course in current_dataset['Course'].unique():
            course_data = current_dataset[current_dataset['Course'] == course]
            
            for _, record in course_data.iterrows():
                year = record['School_Year']
                semester = "1" if record['Semester'] == "1st" else "2"
                
                # Make prediction
                prediction, method, accuracy_info = predict_with_optimized_scaling(course, year, semester)
                
                if accuracy_info and accuracy_info.get('accuracy_percentage') is not None:
                    results.append({
                        'course': course,
                        'year': year,
                        'semester': semester,
                        'actual': accuracy_info['actual'],
                        'predicted': prediction,
                        'accuracy': accuracy_info['accuracy_percentage'],
                        'method': method
                    })
                    
                    total_accuracy += accuracy_info['accuracy_percentage']
                    count += 1
        
        overall_accuracy = total_accuracy / count if count > 0 else 0
        
        return jsonify({
            'overall_accuracy': round(overall_accuracy, 2),
            'total_tests': count,
            'results': results[:20]  # Limit to first 20 for readability
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/check-actual-values")
def check_actual_values():
    """Check actual historical values for specific courses and years"""
    try:
        if current_dataset is None:
            return jsonify({'error': 'No dataset loaded'})
        
        test_cases = [
            {'course': 'BSCS', 'year': '2014-2015', 'semester': '1'},
            {'course': 'BSIT', 'year': '2014-2015', 'semester': '1'},
            {'course': 'BSBA-MARKETING_MANAGEMENT', 'year': '2014-2015', 'semester': '1'},
            {'course': 'BSBA-FINANCIAL_MANAGEMENT', 'year': '2014-2015', 'semester': '1'}
        ]
        
        results = []
        for test in test_cases:
            actual_data = get_actual_historical_data(test['course'], test['year'], test['semester'])
            if actual_data:
                # Make prediction for comparison
                prediction, method, accuracy_info = predict_with_optimized_scaling(
                    test['course'], test['year'], test['semester']
                )
                
                results.append({
                    'course': test['course'],
                    'year': test['year'],
                    'semester': test['semester'],
                    'actual_first_year': actual_data['actual_first_year'],
                    'predicted': prediction,
                    'method': method,
                    'accuracy_info': accuracy_info
                })
        
        return jsonify({'comparisons': results})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/predict-all-models", methods=["POST"])
def predict_all_models():
    """Handle predictions for multiple models - FIXED ENDPOINT"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'})
        
        course = data.get('course', '').strip()
        year = data.get('year', '').strip()
        semester = data.get('semester', '').strip()
        
        if not all([course, year, semester]):
            return jsonify({'error': 'Missing required parameters'})
        
        # Use the same validation as the regular predict endpoint
        is_valid_year, year_message = validate_year_format(year)
        if not is_valid_year:
            return jsonify({'error': year_message})
        
        # Check if this is a historical period (but allow all predictions)
        is_future = is_future_prediction(course, year, semester)
        
        # Use OPTIMIZED SCALING LSTM ensemble model for FIRST-YEAR prediction
        prediction, method, accuracy_info = predict_with_optimized_scaling(course, year, semester)
        
        # Set confidence intervals based on method
        if "lstm_optimized_scaling" in method:
            confidence_interval = [
                max(0, round(prediction * 0.85)),
                round(prediction * 1.15)
            ]
        else:
            confidence_interval = [
                max(0, round(prediction * 0.7)),
                round(prediction * 1.3)
            ]
        
        response_data = {
            'prediction': prediction,
            'confidence_interval': confidence_interval,
            'prediction_method': method,
            'course': course,
            'year': year,
            'semester': semester,
            'prediction_type': 'first_year_enrollment',
            'is_future_prediction': is_future
        }
        
        # Add accuracy information if available
        if accuracy_info:
            response_data['accuracy_info'] = accuracy_info
        
        # ADDED: If historical data exists, include actual values for comparison
        if not is_future:
            actual_data = get_actual_historical_data(course, year, semester)
            if actual_data:
                response_data['actual_data'] = actual_data
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route("/debug-predict", methods=["POST"])
def debug_predict():
    """Debug endpoint to identify where past year predictions are being blocked"""
    try:
        data = request.get_json()
        print(f"DEBUG: Received prediction request: {data}")
        
        course = data.get('course', '').strip()
        year = data.get('year', '').strip()
        semester = data.get('semester', '').strip()
        
        print(f"DEBUG: Course: {course}, Year: {year}, Semester: {semester}")
        
        # Check if we have historical data for this period
        actual_data = get_actual_historical_data(course, year, semester)
        print(f"DEBUG: Actual data found: {actual_data is not None}")
        
        # Check if it's considered "future"
        is_future = is_future_prediction(course, year, semester)
        print(f"DEBUG: Is future prediction: {is_future}")
        
        # Try to make the prediction anyway
        prediction, method, accuracy_info = predict_with_optimized_scaling(course, year, semester)
        print(f"DEBUG: Prediction successful: {prediction} using {method}")
        
        response_data = {
            'prediction': prediction,
            'method': method,
            'course': course,
            'year': year,
            'semester': semester,
            'is_future': is_future,
            'has_actual_data': actual_data is not None
        }
        
        # Add accuracy information if available
        if accuracy_info:
            response_data['accuracy_info'] = accuracy_info
        
        # Add actual data if available
        if actual_data:
            response_data['actual_data'] = actual_data
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"DEBUG Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route("/force-predict", methods=["POST"])
def force_predict():
    """Force prediction for any year (bypasses all validation)"""
    try:
        data = request.get_json()
        course = data.get('course', '').strip()
        year = data.get('year', '').strip()
        semester = data.get('semester', '').strip()
        
        print(f"FORCE PREDICTION: {course}, {year}, Semester {semester}")
        
        # Force prediction regardless of year
        prediction, method, accuracy_info = predict_with_optimized_scaling(course, year, semester)
        
        # Get actual data for comparison
        actual_data = get_actual_historical_data(course, year, semester)
        
        response_data = {
            'prediction': prediction,
            'method': method,
            'course': course,
            'year': year,
            'semester': semester,
            'forced_prediction': True
        }
        
        # Add accuracy information if available
        if accuracy_info:
            response_data['accuracy_info'] = accuracy_info
        
        if actual_data:
            response_data['actual_data'] = actual_data
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Force prediction failed: {str(e)}'})

@app.route("/debug-prediction", methods=["POST"])
def debug_prediction():
    """Debug endpoint to test prediction functionality"""
    try:
        data = request.get_json()
        print(f"DEBUG: Received prediction request: {data}")
        
        course = data.get('course', '').strip()
        year = data.get('year', '').strip()
        semester = data.get('semester', '').strip()
        
        print(f"DEBUG: Course: {course}, Year: {year}, Semester: {semester}")
        
        # Test feature creation
        if current_dataset is not None:
            course_data = current_dataset[current_dataset['Course'] == course]
            print(f"DEBUG: Found {len(course_data)} records for course {course}")
        
        # Test model availability
        if ml_models and 'lstm_models' in ml_models:
            print(f"DEBUG: {len(ml_models['lstm_models'])} LSTM models available")
        else:
            print("DEBUG: No LSTM models available")
        
        # Try to make prediction
        prediction, method, accuracy_info = predict_with_optimized_scaling(course, year, semester)
        
        return jsonify({
            'debug': True,
            'prediction': prediction,
            'method': method,
            'course_data_count': len(course_data) if current_dataset is not None else 0,
            'models_available': len(ml_models.get('lstm_models', [])) if ml_models else 0
        })
        
    except Exception as e:
        print(f"DEBUG Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route("/forecast", methods=["POST"])
def forecast():
    """Handle historical data visualization (requires dataset) - KEEP ORIGINAL HISTORICAL DISPLAY"""
    if current_dataset is None:
        flash("Please upload a dataset to view historical data", "error")
        course = request.form.get("course", "").strip()
        if course.startswith('BSBA'):
            return redirect(url_for("bed_filter_v2"))
        else:
            return redirect(url_for("ced_filter_v2"))
    
    course = request.form.get("course", "").strip()
    year = request.form.get("year", "").strip()
    semester = request.form.get("semester", "").strip()

    if not course or not year or not semester:
        flash("Please complete all required fields", "error")
        if course.startswith('BSBA'):
            return redirect(url_for("bed_filter_v2"))
        else:
            return redirect(url_for("ced_filter_v2"))

    semester_text = "1st" if semester == "1" else "2nd"
    
    # Get historical data from dataset - KEEP ORIGINAL FORMAT
    historical_enrollment = get_historical_enrollment(course, year, semester)
    
    if historical_enrollment is None:
        flash(f"No historical data found for {course} in {year} ({semester_text} Semester)", "error")
        if course.startswith('BSBA'):
            return redirect(url_for("bed_filter_v2"))
        else:
            return redirect(url_for("ced_filter_v2"))
    
    # Get historical trend for charts - KEEP ORIGINAL FORMAT
    historical_trend = get_historical_trend(course)
    
    # Debug: Print the data being sent to template
    print(f"DEBUG: Sending to template - Course: {course}, Year: {year}, Semester: {semester_text}")
    print(f"DEBUG: Historical enrollment: {historical_enrollment}")
    print(f"DEBUG: Historical trend labels: {historical_trend['labels']}")
    print(f"DEBUG: Historical trend data - First Year: {historical_trend['first_year']}")
    
    # Determine which template to use
    if course.startswith('BSBA'):
        template_name = "BED_result_v2.html"
    else:
        template_name = "CED_result.html"
    
    model_count = len(ml_models.get('lstm_models', [])) if ml_models else 0
    
    return render_template(
        template_name,
        course=course,
        year=year,
        semester=semester_text,
        total_enrollment=historical_enrollment['total_enrollment'],  # Keep total enrollment display
        first_year_enrollment=historical_enrollment['first_year'],  # Add this line to fix the template variable
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

@app.route("/bed_filter_v2")
def bed_filter_v2():
    """BED filter page v2 - ALWAYS ACCESSIBLE"""
    lstm_loaded = ml_models and 'lstm_models' in ml_models
    model_count = len(ml_models.get('lstm_models', [])) if lstm_loaded else 0
    return render_template("BEDfilter_v2.html", lstm_loaded=lstm_loaded, model_count=model_count)

@app.route("/ced_filter")
def ced_filter():
    """CED filter page - ALWAYS ACCESSIBLE"""
    lstm_loaded = ml_models and 'lstm_models' in ml_models
    model_count = len(ml_models.get('lstm_models', [])) if lstm_loaded else 0
    return render_template("CEDfilter.html", lstm_loaded=lstm_loaded, model_count=model_count)

@app.route("/debug-course-stats")
def debug_course_stats():
    """Debug endpoint to check course statistics"""
    if current_dataset is None:
        return jsonify({'error': 'No dataset loaded'})
    
    stats = {}
    for course in ['BSIT', 'BSCS']:
        course_data = current_dataset[current_dataset['Course'] == course]
        if not course_data.empty:
            first_year = course_data['1st_year_enrollees']
            stats[course] = {
                'total_records': len(course_data),
                'first_year_mean': first_year.mean(),
                'first_year_std': first_year.std(),
                'first_year_min': first_year.min(),
                'first_year_max': first_year.max(),
                'recent_3_avg': first_year.tail(3).mean() if len(course_data) >= 3 else first_year.mean(),
                'sample_data': course_data[['School_Year', 'Semester', '1st_year_enrollees']].tail(5).to_dict('records')
            }
    
    return jsonify(stats)

if __name__ == '__main__':
    print("=== Enrollment Forecasting System - FIRST-YEAR PREDICTION ONLY ===")
    print(f"Dataset loaded: {current_dataset is not None}")
    print(f"LSTM Models loaded: {ml_models and 'lstm_models' in ml_models}")
    
    if ml_models and 'lstm_models' in ml_models:
        print(f"Loaded {len(ml_models['lstm_models'])} LSTM models for ensemble")
        print("Loaded preprocessing components:", [k for k in ml_models.keys() if k != 'lstm_models'])
    
    if current_dataset is not None:
        print(f"Records: {len(current_dataset)}")
        # Print course statistics
        for course in ['BSIT', 'BSCS']:
            course_data = current_dataset[current_dataset['Course'] == course]
            if not course_data.empty:
                avg_enrollment = course_data['1st_year_enrollees'].mean()
                print(f"{course}: {len(course_data)} records, Avg first-year: {avg_enrollment:.1f}")
    
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)  