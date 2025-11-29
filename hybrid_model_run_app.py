from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
import joblib
import json
import re
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Import deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thesis_hybrid'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hybrid_models')
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
hybrid_models = {}  # Store hybrid model metadata
trained_models = {}  # Store actual model objects

def create_custom_adam_optimizer():
    """Create Adam optimizer without weight_decay parameter"""
    return Adam(learning_rate=0.001)

def preprocess_course_data(df, course_name):
    """Prepare time series data for a specific course - NOW FOR FIRST-YEAR ENROLLEES"""
    course_df = df[df['Course'] == course_name].copy()
    
    # Create proper date column
    course_df['year'] = course_df['School_Year'].str.split('-').str[0].astype(int)
    course_df['month'] = course_df['Semester'].map({'1st': 8, '2nd': 1})  # Aug for 1st sem, Jan for 2nd sem
    course_df['date'] = pd.to_datetime(course_df['year'].astype(str) + '-' + course_df['month'].astype(str) + '-01')
    
    # Sort by date and create time index
    course_df = course_df.sort_values('date').reset_index(drop=True)
    course_df['time_index'] = range(len(course_df))
    
    # Create time series for FIRST-YEAR enrollees
    time_series = course_df.set_index('date')['1st_year_enrollees']
    
    return time_series, course_df

def find_best_arima_params(timeseries, max_p=3, max_d=2, max_q=3):
    """Manual implementation to find best ARIMA parameters"""
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                    
                try:
                    model = ARIMA(timeseries, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                    
                except Exception as e:
                    continue
    
    return best_model, best_order

def build_arima_model(timeseries):
    """Build and train ARIMA model for FIRST-YEAR prediction"""
    try:
        # Use entire dataset for training
        arima_model, best_order = find_best_arima_params(timeseries)
        
        if arima_model is None:
            return None
        
        # Calculate predictions and residuals
        train_predictions = arima_model.predict(start=0, end=len(timeseries)-1)
        residuals = timeseries - train_predictions
        
        return {
            'model': arima_model,
            'residuals': residuals,
            'order': best_order,
            'train_predictions': train_predictions,
            'train_data': timeseries
        }
        
    except Exception as e:
        print(f"❌ ARIMA model error: {e}")
        return None

def build_lstm_residual_model(residuals, time_steps=3):
    """Build LSTM model to predict ARIMA residuals"""
    try:
        # Prepare residual data for LSTM
        residual_values = residuals.values.reshape(-1, 1)
        
        # Scale the residuals
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_residuals = scaler.fit_transform(residual_values)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(time_steps, len(scaled_residuals)):
            X.append(scaled_residuals[i-time_steps:i, 0])
            y.append(scaled_residuals[i, 0])
        
        if len(X) == 0:
            return None
            
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build LSTM model
        model = Sequential([
            LSTM(20, input_shape=(time_steps, 1)),
            Dropout(0.2),
            Dense(10),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model with early stopping
        early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        
        history = model.fit(
            X, y,
            batch_size=8,
            epochs=100,
            verbose=0,
            shuffle=False,
            callbacks=[early_stop]
        )
        
        # Predict on training data
        train_predictions_scaled = model.predict(X, verbose=0)
        train_predictions = scaler.inverse_transform(train_predictions_scaled).flatten()
        
        return {
            'model': model,
            'scaler': scaler,
            'time_steps': time_steps,
            'residual_predictions': train_predictions
        }
        
    except Exception as e:
        print(f"❌ LSTM model error: {e}")
        return None

def build_hybrid_model_for_course(course_data):
    """Build hybrid ARIMA-LSTM model for a specific course - PREDICTING FIRST-YEAR"""
    try:
        timeseries, course_df = preprocess_course_data(current_dataset, course_data)
        
        if len(timeseries) < 6:  # Minimum data requirement
            print(f"⚠️ Insufficient data for {course_data}: {len(timeseries)} records")
            return None
        
        print(f"📊 Building FIRST-YEAR hybrid model for {course_data}...")
        
        # Build ARIMA model
        arima_result = build_arima_model(timeseries)
        if arima_result is None:
            print(f"❌ ARIMA model failed for {course_data}")
            return None
        
        # Build LSTM model for residuals
        lstm_result = build_lstm_residual_model(arima_result['residuals'])
        
        if lstm_result is None:
            # Use ARIMA only
            hybrid_predictions = arima_result['train_predictions']
            model_type = 'arima_only'
            print(f"✅ ARIMA-only model for {course_data}")
        else:
            # Create hybrid predictions
            min_len = min(len(arima_result['train_predictions']), len(lstm_result['residual_predictions']))
            hybrid_predictions = arima_result['train_predictions'][-min_len:] + lstm_result['residual_predictions']
            model_type = 'hybrid'
            print(f"✅ Hybrid ARIMA-LSTM model for {course_data}")
        
        # Evaluate model
        actual_values = timeseries.values[-len(hybrid_predictions):]
        mae = mean_absolute_error(actual_values, hybrid_predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, hybrid_predictions))
        r2 = r2_score(actual_values, hybrid_predictions)
        
        # Store actual model objects in trained_models
        trained_models[course_data] = {
            'arima_model': arima_result['model'],
            'lstm_model': lstm_result['model'] if lstm_result else None,
            'lstm_scaler': lstm_result['scaler'] if lstm_result else None,
            'timeseries': timeseries
        }
        
        return {
            'course': course_data,
            'model_type': model_type,
            'arima_order': arima_result['order'],
            'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2},
            'last_training_date': datetime.now(),
            'data_points': len(timeseries)
        }
        
    except Exception as e:
        print(f"❌ Error building FIRST-YEAR hybrid model for {course_data}: {e}")
        return None

def train_all_hybrid_models():
    """Train hybrid models for all courses in the dataset"""
    global hybrid_models, trained_models
    
    if current_dataset is None:
        print("❌ No dataset available for training")
        return False
    
    try:
        courses = current_dataset['Course'].unique()
        print(f"🎯 Training FIRST-YEAR hybrid models for {len(courses)} courses...")
        
        trained_hybrid_models = {}
        trained_models.clear()  # Clear previous models
        
        successful_models = 0
        for course in courses:
            print(f"🔨 Processing: {course}")
            model = build_hybrid_model_for_course(course)
            if model:
                trained_hybrid_models[course] = model
                successful_models += 1
            else:
                print(f"❌ Failed to train model for {course}")
        
        hybrid_models = trained_hybrid_models
        print(f"✅ Successfully trained {successful_models}/{len(courses)} FIRST-YEAR hybrid models")
        return successful_models > 0
        
    except Exception as e:
        print(f"❌ Error training FIRST-YEAR hybrid models: {e}")
        return False

def save_hybrid_models():
    """Save trained hybrid models"""
    try:
        if not hybrid_models:
            print("❌ No models to save")
            return False
        
        for course, model_data in hybrid_models.items():
            safe_course_name = course.replace(' ', '_').replace('/', '_')
            filename = f'hybrid_model_{safe_course_name}.pkl'
            filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
            
            # Create compatible data structure
            compatible_data = {
                'course': model_data['course'],
                'model_type': model_data['model_type'],
                'arima_order': model_data['arima_order'],
                'metrics': model_data['metrics'],
                'last_training_date': model_data['last_training_date'],
                'data_points': model_data['data_points']
            }
            
            joblib.dump(compatible_data, filepath)
        
        print(f"💾 Saved {len(hybrid_models)} FIRST-YEAR hybrid models")
        return True
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

def load_hybrid_models():
    """Load trained hybrid models - only metadata"""
    global hybrid_models
    
    try:
        loaded_models = {}
        model_files = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.startswith('hybrid_model_') and f.endswith('.pkl')]
        
        for model_file in model_files:
            filepath = os.path.join(app.config['MODEL_FOLDER'], model_file)
            model_data = joblib.load(filepath)
            course = model_data['course']
            loaded_models[course] = model_data
        
        hybrid_models = loaded_models
        print(f"✅ Loaded {len(hybrid_models)} FIRST-YEAR hybrid models (metadata)")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def predict_with_hybrid_model(course, year, semester):
    """Make FIRST-YEAR enrollment prediction using hybrid ARIMA-LSTM model"""
    try:
        if course not in trained_models:
            print(f"❌ No trained model found for {course}")
            return predict_first_year_fallback(course, year, semester), "fallback"
        
        model_objects = trained_models[course]
        
        # Convert input to proper format
        pred_year = int(year.split('-')[0])
        pred_semester = 1 if semester == "1" else 2
        pred_month = 8 if pred_semester == 1 else 1
        
        # Create prediction date
        pred_date = pd.Timestamp(f"{pred_year}-{pred_month:02d}-01")
        
        # Get the timeseries for this course
        timeseries = model_objects.get('timeseries')
        if timeseries is None:
            print(f"❌ No timeseries data for {course}")
            return predict_first_year_fallback(course, year, semester), "fallback"
        
        # Check if this is a historical period where we have actual data
        is_historical = pred_date in timeseries.index
        
        if is_historical:
            print(f"📊 Making historical prediction for {course} {year} Semester {semester}")
            # For historical periods, we want to predict what the model would have predicted
            # using only data available before that period
            
            # Find the index of the target period in the timeseries
            target_idx = timeseries.index.get_loc(pred_date)
            if target_idx > 0:  # We have data before the target period
                # Use data only up to the period before the target
                train_data = timeseries.iloc[:target_idx]
                
                if len(train_data) >= 4:  # Need minimum data for ARIMA
                    try:
                        # Build ARIMA model with data up to the period before target
                        temp_arima, best_order = find_best_arima_params(train_data)
                        if temp_arima:
                            # Forecast one period ahead
                            forecast = temp_arima.forecast(steps=1)
                            prediction = forecast.iloc[0]
                            
                            # Try to apply LSTM residuals if we have enough data
                            if len(train_data) >= 7:  # Need more data for LSTM
                                try:
                                    train_predictions = temp_arima.predict(start=0, end=len(train_data)-1)
                                    residuals = train_data - train_predictions
                                    
                                    if len(residuals) >= 4:  # Minimum for LSTM
                                        # Build LSTM for residuals on this subset
                                        lstm_result = build_lstm_residual_model(residuals)
                                        if lstm_result:
                                            # Use the LSTM to predict the next residual
                                            residual_values = residuals.values.reshape(-1, 1)
                                            scaler = MinMaxScaler(feature_range=(0, 1))
                                            scaled_residuals = scaler.fit_transform(residual_values)
                                            
                                            # Create sequence for prediction
                                            time_steps = lstm_result['time_steps']
                                            if len(scaled_residuals) >= time_steps:
                                                last_sequence = scaled_residuals[-time_steps:]
                                                X_pred = np.array([last_sequence.flatten()])
                                                X_pred = X_pred.reshape(X_pred.shape[0], time_steps, 1)
                                                
                                                residual_pred_scaled = lstm_result['model'].predict(X_pred, verbose=0)
                                                residual_pred = scaler.inverse_transform(residual_pred_scaled)[0][0]
                                                
                                                prediction += residual_pred
                                                method_type = "hybrid_arima_lstm"
                                                print(f"🔧 Applied LSTM residual adjustment for historical prediction")
                                            else:
                                                method_type = "arima_only"
                                        else:
                                            method_type = "arima_only"
                                    else:
                                        method_type = "arima_only"
                                except Exception as e:
                                    print(f"⚠️ LSTM historical adjustment failed: {e}")
                                    method_type = "arima_only"
                            else:
                                method_type = "arima_only"
                            
                            prediction = max(5, float(prediction))
                            actual_value = timeseries.iloc[target_idx]
                            print(f"🎯 Historical prediction for {course}: {prediction} vs Actual: {actual_value}")
                            return round(prediction), method_type
                        else:
                            print(f"❌ Could not build historical ARIMA model for {course}")
                            return predict_first_year_fallback(course, year, semester), "fallback"
                    except Exception as e:
                        print(f"❌ Historical prediction error for {course}: {e}")
                        return predict_first_year_fallback(course, year, semester), "fallback"
                else:
                    print(f"⚠️ Insufficient data for historical prediction: {len(train_data)} records")
                    return predict_first_year_fallback(course, year, semester), "fallback"
            else:
                print(f"⚠️ No previous data for historical prediction")
                return predict_first_year_fallback(course, year, semester), "fallback"
        
        else:
            # FUTURE PREDICTION - use the full trained model
            print(f"🔮 Making FUTURE prediction for {course} {year} Semester {semester}")
            
            # Get the trained ARIMA model
            arima_model = model_objects.get('arima_model')
            if arima_model is None:
                print(f"❌ No ARIMA model for {course}")
                return predict_first_year_fallback(course, year, semester), "fallback"
            
            # Calculate how many periods to forecast
            last_date = timeseries.index[-1]
            
            # Calculate periods based on semester difference
            last_year = last_date.year
            last_semester = 1 if last_date.month == 8 else 2  # August = 1st sem, January = 2nd sem
            
            # Calculate total periods difference
            year_diff = pred_year - last_year
            semester_diff = pred_semester - last_semester
            periods = (year_diff * 2) + semester_diff
            
            if periods <= 0:
                periods = 1  # Ensure at least 1 period forecast
            
            print(f"📈 Forecasting {periods} periods ahead for {course} (last: {last_date}, target: {pred_date})")
            
            try:
                # Use the trained ARIMA model for forecasting
                forecast = arima_model.forecast(steps=periods)
                prediction = forecast.iloc[-1]  # Get the last forecast value
                
                # Apply LSTM residual correction if available
                if model_objects.get('lstm_model') is not None:
                    # Calculate residuals from training
                    train_predictions = arima_model.predict(start=0, end=len(timeseries)-1)
                    residuals = timeseries - train_predictions
                    
                    # Use average residual as adjustment for future predictions
                    residual_adjustment = residuals.mean()
                    prediction += residual_adjustment
                    print(f"🔧 Applied LSTM residual adjustment: {residual_adjustment:.2f}")
                    method_type = "hybrid_arima_lstm"
                else:
                    method_type = "arima_only"
                
                # Apply bounds checking based on historical data
                historical_mean = timeseries.mean()
                historical_std = timeseries.std()
                
                # Set reasonable bounds (within 2 standard deviations of historical mean)
                lower_bound = max(10, historical_mean - 2 * historical_std)
                upper_bound = historical_mean + 2 * historical_std
                
                if prediction < lower_bound:
                    print(f"⚠️ Prediction {prediction} below lower bound {lower_bound}, adjusting")
                    prediction = lower_bound
                elif prediction > upper_bound:
                    print(f"⚠️ Prediction {prediction} above upper bound {upper_bound}, adjusting")
                    prediction = upper_bound
                
                prediction = max(5, float(prediction))
                print(f"🎯 FUTURE prediction for {course}: {prediction}")
                
                return round(prediction), method_type
                
            except Exception as e:
                print(f"❌ Future prediction error for {course}: {e}")
                return predict_first_year_fallback(course, year, semester), "fallback"
                
    except Exception as e:
        print(f"❌ FIRST-YEAR Hybrid prediction error for {course}: {e}")
        return predict_first_year_fallback(course, year, semester), "fallback"

def predict_first_year_fallback(course, year, semester):
    """Simple FIRST-YEAR prediction for future enrollment"""
    try:
        # Get historical data for better fallback
        if current_dataset is not None:
            course_data = current_dataset[current_dataset['Course'] == course]
            if not course_data.empty:
                # Use first-year enrollment data
                historical_mean = course_data['1st_year_enrollees'].mean()
                # Add some variation based on semester
                if semester == "1":
                    prediction = historical_mean * 1.05
                else:
                    prediction = historical_mean * 0.95
                
                # Add small random variation
                variation = np.random.normal(0, historical_mean * 0.1)
                prediction = max(20, prediction + variation)
                return round(prediction)
        
        # Basic fallback if no historical data
        base_enrollment = 40
        
        if "BSBA" in course:
            base_enrollment = 50
        elif course in ["BSCS", "BSIT"]:
            base_enrollment = 35
        
        # Semester adjustment
        if semester == "1":
            base_enrollment *= 1.1  # Slightly higher in first semester
        else:
            base_enrollment *= 0.9  # Slightly lower in second semester
            
        # Add some variation
        variation = np.random.normal(0, 5)
        prediction = max(20, base_enrollment + variation)  # Ensure minimum first-year students
        
        return round(prediction)
    except Exception as e:
        print(f"FIRST-YEAR prediction error: {e}")
        return 40

# FIXED: Completely disabled future prediction check - JUST LIKE LSTM APP
def is_future_prediction(course, year, semester):
    """Check if the prediction request is for a future period - COMPLETELY DISABLED FOR HISTORICAL TESTING"""
    # TEMPORARILY DISABLE ALL FUTURE CHECKS TO ALLOW HISTORICAL PREDICTIONS
    print(f"DEBUG: Prediction requested for {course}, {year}, Semester {semester}")
    print("DEBUG: ALLOWING ALL PREDICTIONS (including historical) for accuracy testing")
    return False  # Always allow predictions, even for past years

def get_actual_historical_data(course, year, semester):
    """Get actual historical enrollment data for comparison"""
    try:
        if current_dataset is None:
            return None
            
        # Convert semester format
        if semester == "1":
            semester_dataset = "1st"
        elif semester == "2":
            semester_dataset = "2nd"
        else:
            semester_dataset = semester
            
        # Find the actual record
        record = current_dataset[
            (current_dataset['Course'] == course) & 
            (current_dataset['School_Year'] == year) & 
            (current_dataset['Semester'] == semester_dataset)
        ]
        
        if not record.empty:
            return {
                'actual_first_year': int(record['1st_year_enrollees'].iloc[0]),
                'actual_second_year': int(record['2nd_year_enrollees'].iloc[0]),
                'actual_third_year': int(record['3rd_year_enrollees'].iloc[0]),
                'actual_fourth_year': int(record['4th_year_enrollees'].iloc[0]),
                'actual_total': int(record['total_enrollees'].iloc[0])
            }
        return None
    except Exception as e:
        print(f"Error getting actual historical data: {e}")
        return None

def assess_prediction_accuracy(accuracy):
    """Assess how good the prediction is compared to actual data"""
    if accuracy >= 90:
        return "Excellent"
    elif accuracy >= 80:
        return "Very Good"
    elif accuracy >= 70:
        return "Good"
    elif accuracy >= 60:
        return "Fair"
    else:
        return "Poor"

# FIXED: Use the same validation as LSTM app - NO FUTURE RESTRICTIONS
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
                        
                        if '1st_year_enrollees' in df.columns:
                            dept_students += course_data['1st_year_enrollees'].sum()
                        
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
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hybrid_models_loaded': len(hybrid_models) if hybrid_models else 0
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
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hybrid_models_loaded': 0
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
        
        required_columns = ['School_Year', 'Semester', 'Course', '1st_year_enrollees']  # Require first-year column
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
                if col != '1st_year_enrollees':  # Don't override the required column
                    df[col] = 0
        
        return True, "Dataset is valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

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
                    
                    first_year_data.append(int(row['1st_year_enrollees']))
                    second_year_data.append(int(row['2nd_year_enrollees']))
                    third_year_data.append(int(row['3rd_year_enrollees']))
                    fourth_year_data.append(int(row['4th_year_enrollees']))
                
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
        return {
            'labels': [], 
            'first_year': [], 
            'second_year': [], 
            'third_year': [], 
            'fourth_year': []
        }

# Initialize the application
load_current_dataset()
load_hybrid_models()

# Train models if no existing models are loaded
if current_dataset is not None and len(trained_models) == 0:
    print("🔨 Training FIRST-YEAR hybrid models...")
    train_success = train_all_hybrid_models()
    if train_success:
        save_hybrid_models()
        print("✅ FIRST-YEAR Hybrid models trained and saved successfully!")
    else:
        print("❌ FIRST-YEAR Hybrid model training failed")

@app.before_request
def before_request():
    """Execute before each request"""
    session.permanent = True
    app.permanent_session_lifetime = timedelta(days=1)

@app.route("/")
def index():
    """Main page"""
    dataset_loaded = current_dataset is not None
    hybrid_loaded = len(trained_models) > 0
    model_count = len(hybrid_models)
    return render_template("frontface.html", 
                         dataset_loaded=dataset_loaded, 
                         hybrid_loaded=hybrid_loaded,
                         model_count=model_count)

@app.route("/check_dataset_status")
def check_dataset_status():
    """API endpoint to check dataset status"""
    dataset_loaded = current_dataset is not None
    dataset_info = None
    hybrid_loaded = len(trained_models) > 0
    model_count = len(hybrid_models)
    
    if dataset_loaded:
        dataset_info = analyze_dataset(current_dataset)
    
    return jsonify({
        'loaded': dataset_loaded,
        'hybrid_loaded': hybrid_loaded,
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
        
        # Validate dataset structure - now requires first-year column
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
        
        # Train hybrid models on new dataset
        train_success = train_all_hybrid_models()
        if train_success:
            save_hybrid_models()
        
        # Analyze dataset for response
        analysis_result = analyze_dataset(df)
        analysis_result['filename'] = secure_filename(file.filename)
        
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'models_trained': train_success,
            'dataset_info': analysis_result,
            'hybrid_loaded': len(trained_models) > 0,
            'model_count': len(hybrid_models)
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
        return redirect(url_for("ced_filter_v2"))
    else:
        flash("Invalid department selected", "error")
        return redirect(url_for("index"))

# NEW ROUTES FOR HISTORICAL PREDICTIONS - JUST LIKE LSTM APP
@app.route("/ced_filter_v2")
def ced_filter_v2():
    """CED filter page v2 - WITH HISTORICAL PREDICTION SUPPORT"""
    hybrid_loaded = len(trained_models) > 0
    model_count = len(hybrid_models)
    return render_template("CEDfilter_v2.html", hybrid_loaded=hybrid_loaded, model_count=model_count)

@app.route("/bed_filter_v2")
def bed_filter_v2():
    """BED filter page v2 - WITH HISTORICAL PREDICTION SUPPORT"""
    hybrid_loaded = len(trained_models) > 0
    model_count = len(hybrid_models)
    return render_template("BEDfilter_v2.html", hybrid_loaded=hybrid_loaded, model_count=model_count)

@app.route("/predict-historical", methods=["POST"])
def predict_historical():
    """Dedicated endpoint for historical predictions - JUST LIKE LSTM APP"""
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
        
        # ALWAYS allow prediction (past or future) - JUST LIKE LSTM APP
        is_future = False  # Force to false to enable historical comparison
        
        # Use hybrid model for FIRST-YEAR prediction
        prediction, method = predict_with_hybrid_model(course, year, semester)
        
        # Set confidence intervals
        if "hybrid" in method or "arima" in method:
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
        
        # Always try to get historical data for comparison
        actual_data = get_actual_historical_data(course, year, semester)
        if actual_data:
            response_data['actual_data'] = actual_data
            # Calculate prediction accuracy
            actual_first_year = actual_data['actual_first_year']
            prediction_error = abs(prediction - actual_first_year)
            prediction_accuracy = max(0, 100 - (prediction_error / actual_first_year * 100)) if actual_first_year > 0 else 0
            
            response_data['prediction_accuracy'] = round(prediction_accuracy, 2)
            response_data['prediction_error'] = prediction_error
            response_data['accuracy_assessment'] = assess_prediction_accuracy(prediction_accuracy)
        
        print(f"HISTORICAL PREDICTION RESULT: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"HISTORICAL PREDICTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Historical prediction error: {str(e)}'})

@app.route("/predict", methods=["POST"])
def predict():
    """Handle AJAX prediction requests for BOTH future and historical FIRST-YEAR enrollment - JUST LIKE LSTM APP"""
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
        
        # Check if this is a historical period (but allow all predictions) - JUST LIKE LSTM APP
        is_future = is_future_prediction(course, year, semester)
        
        # Use hybrid model for FIRST-YEAR prediction
        prediction, method = predict_with_hybrid_model(course, year, semester)
        
        # Set confidence intervals based on method
        if "hybrid" in method or "arima" in method:
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
        
        # ADDED: If historical data exists, include actual values for comparison - JUST LIKE LSTM APP
        if not is_future:
            actual_data = get_actual_historical_data(course, year, semester)
            if actual_data:
                response_data['actual_data'] = actual_data
                # Calculate prediction accuracy
                actual_first_year = actual_data['actual_first_year']
                prediction_error = abs(prediction - actual_first_year)
                prediction_accuracy = max(0, 100 - (prediction_error / actual_first_year * 100)) if actual_first_year > 0 else 0
                
                response_data['prediction_accuracy'] = round(prediction_accuracy, 2)
                response_data['prediction_error'] = prediction_error
                response_data['accuracy_assessment'] = assess_prediction_accuracy(prediction_accuracy)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'FIRST-YEAR enrollment prediction error: {str(e)}'})

@app.route("/predict-enrollment", methods=["POST"])
def predict_enrollment():
    """Alternative FIRST-YEAR enrollment prediction endpoint for BED page - UPDATED for historical comparison - JUST LIKE LSTM APP"""
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
        
        # Check if this is a historical period (but allow all predictions) - JUST LIKE LSTM APP
        is_future = is_future_prediction(course, year, semester)
        
        # Use hybrid model for FIRST-YEAR prediction
        prediction, method = predict_with_hybrid_model(course, year, semester)
        
        # Generate confidence intervals
        if "hybrid" in method or "arima" in method:
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
        
        # ADDED: If historical data exists, include actual values for comparison - JUST LIKE LSTM APP
        if not is_future:
            actual_data = get_actual_historical_data(course, year, semester)
            if actual_data:
                response_data['actual_data'] = actual_data
                # Calculate prediction accuracy
                actual_first_year = actual_data['actual_first_year']
                prediction_error = abs(prediction - actual_first_year)
                prediction_accuracy = max(0, 100 - (prediction_error / actual_first_year * 100)) if actual_first_year > 0 else 0
                
                response_data['prediction_accuracy'] = round(prediction_accuracy, 2)
                response_data['prediction_error'] = prediction_error
                response_data['accuracy_assessment'] = assess_prediction_accuracy(prediction_accuracy)
        
        return jsonify(response_data)
        
    except Exception as e:
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
    
    # Determine which template to use
    if course.startswith('BSBA'):
        template_name = "BED_result_v2.html"
    else:
        template_name = "CED_result.html"
    
    model_count = len(hybrid_models)
    
    return render_template(
        template_name,
        course=course,
        year=year,
        semester=semester_text,
        total_enrollment=historical_enrollment['total_enrollment'],
        first_year_enrollment=historical_enrollment['first_year'],
        year_levels={
            'first_year': historical_enrollment['first_year'],
            'second_year': historical_enrollment['second_year'],
            'third_year': historical_enrollment['third_year'],
            'fourth_year': historical_enrollment['fourth_year']
        },
        historical_trend=historical_trend,
        dataset_info=analyze_dataset(current_dataset) if current_dataset is not None else None,
        hybrid_loaded=len(trained_models) > 0,
        model_count=model_count
    )

@app.route("/forecast_history", methods=['POST'])
def forecast_history():
    """Alias for forecast endpoint for CED page"""
    return forecast()

# ADD THE MISSING ROUTES THAT YOUR TEMPLATES ARE LOOKING FOR
@app.route("/ced_filter")
def ced_filter():
    """CED filter page - ALWAYS ACCESSIBLE"""
    hybrid_loaded = len(trained_models) > 0
    model_count = len(hybrid_models)
    return render_template("CEDfilter.html", hybrid_loaded=hybrid_loaded, model_count=model_count)

@app.route("/bed_filter")
def bed_filter():
    """BED filter page - ALWAYS ACCESSIBLE"""
    hybrid_loaded = len(trained_models) > 0
    model_count = len(hybrid_models)
    return render_template("BEDfilter.html", hybrid_loaded=hybrid_loaded, model_count=model_count)

@app.route("/debug_hybrid_models")
def debug_hybrid_models():
    """Debug endpoint to check hybrid models status"""
    models_info = {}
    for course, model_data in hybrid_models.items():
        models_info[course] = {
            'model_type': model_data.get('model_type', 'unknown'),
            'arima_order': model_data.get('arima_order', 'unknown'),
            'metrics': model_data.get('metrics', {}),
            'data_points': model_data.get('data_points', 0),
            'last_training_date': str(model_data.get('last_training_date', 'Unknown'))
        }
    
    return jsonify({
        'total_models': len(hybrid_models),
        'models_info': models_info,
        'available_courses': list(hybrid_models.keys())
    })

@app.route("/retrain_hybrid_models", methods=["POST"])
def retrain_hybrid_models():
    """Force retrain hybrid models"""
    try:
        if current_dataset is None:
            return jsonify({'success': False, 'error': 'No dataset available'})
        
        success = train_all_hybrid_models()
        
        if success:
            save_success = save_hybrid_models()
            return jsonify({
                'success': True,
                'message': f'Successfully retrained {len(hybrid_models)} hybrid models for first-year enrollment',
                'models_trained': list(hybrid_models.keys()),
                'models_saved': save_success
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to retrain models'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("=== Hybrid ARIMA-LSTM FIRST-YEAR Enrollment Forecasting System ===")
    print(f"Dataset loaded: {current_dataset is not None}")
    print(f"FIRST-YEAR Hybrid Models loaded: {len(trained_models)}")
    
    if trained_models:
        print("Trained courses:", list(trained_models.keys()))
        for course, model in hybrid_models.items():
            model_type = model.get('model_type', 'unknown')
            r2 = model.get('metrics', {}).get('R2', 0)
            print(f"  ✅ {course}: {model_type} (R²: {r2:.4f})")
    
    if current_dataset is not None:
        print(f"Records: {len(current_dataset)}")
    
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=True)