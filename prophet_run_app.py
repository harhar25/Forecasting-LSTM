from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
import joblib
import json
import re
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import pickle

# Import Prophet
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thesis'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
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
prophet_models = {}

def load_prophet_models():
    """Load trained Prophet models for each course"""
    global prophet_models
    
    try:
        courses = ['BSBA-FINANCIAL_MANAGEMENT', 'BSBA-MARKETING_MANAGEMENT', 'BSIT', 'BSCS']
        
        for course in courses:
            model_filename = f"prophet_model_{course.replace(' ', '_').replace('-', '_')}.pkl"
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
            
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        prophet_models[course] = pickle.load(f)
                    print(f"✅ Loaded Prophet model for {course}")
                except Exception as e:
                    print(f"❌ Error loading Prophet model for {course}: {e}")
            else:
                print(f"⚠️ Prophet model not found for {course}: {model_path}")
        
        print(f"✅ Successfully loaded {len(prophet_models)} Prophet models")
        return len(prophet_models) > 0
        
    except Exception as e:
        print(f"❌ Error loading Prophet models: {e}")
        return False

def prepare_prophet_data(course_data):
    """Prepare data in Prophet format - using first-year enrollment data"""
    # Create proper date column
    course_data = course_data.copy()
    course_data['year'] = course_data['School_Year'].str.split('-').str[0].astype(int)
    course_data['month'] = course_data['Semester'].map({'1st': 8, '2nd': 1})  # Aug for 1st sem, Jan for 2nd sem
    course_data['date'] = pd.to_datetime(course_data['year'].astype(str) + '-' + course_data['month'].astype(str) + '-01')
    
    # Sort by date
    course_data = course_data.sort_values('date')
    
    # Create Prophet format using FIRST-YEAR enrollment data
    prophet_df = pd.DataFrame({
        'ds': course_data['date'],
        'y': course_data['1st_year_enrollees']  # Using first-year enrollment for prediction
    })
    
    return prophet_df

def train_prophet_models():
    """Train Prophet models for all courses using current dataset - focused on first-year enrollment"""
    global prophet_models
    
    try:
        if current_dataset is None:
            print("❌ No dataset available for training Prophet models")
            return False
        
        courses = current_dataset['Course'].unique()
        trained_models = 0
        
        for course in courses:
            try:
                # Get course data
                course_data = current_dataset[current_dataset['Course'] == course].copy()
                
                if len(course_data) < 6:  # Need minimum data
                    print(f"⚠️ Insufficient data for {course}, skipping")
                    continue
                
                # Prepare Prophet data using first-year enrollment
                prophet_df = prepare_prophet_data(course_data)
                
                # Check for data quality
                if prophet_df['y'].std() == 0:
                    print(f"⚠️ No variance in first-year data for {course}, skipping")
                    continue
                
                # Create and train Prophet model with better parameters
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0
                )
                
                model.fit(prophet_df)
                
                # Store model with metadata
                prophet_models[course] = {
                    'model': model,
                    'last_training_date': prophet_df['ds'].max(),
                    'training_records': len(prophet_df),
                    'historical_mean': prophet_df['y'].mean(),
                    'historical_std': prophet_df['y'].std(),
                    'historical_min': prophet_df['y'].min(),
                    'historical_max': prophet_df['y'].max()
                }
                
                # Save model
                model_filename = f"prophet_model_{course.replace(' ', '_').replace('-', '_')}.pkl"
                model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(prophet_models[course], f)
                
                print(f"✅ Trained Prophet model for {course} first-year enrollment with {len(prophet_df)} records")
                trained_models += 1
                
            except Exception as e:
                print(f"❌ Error training Prophet model for {course}: {e}")
        
        print(f"✅ Successfully trained {trained_models} Prophet models for first-year enrollment prediction")
        return trained_models > 0
        
    except Exception as e:
        print(f"❌ Error training Prophet models: {e}")
        return False

def predict_with_prophet(course, year, semester):
    """Make prediction using Prophet model - predicting first-year enrollment"""
    try:
        if course not in prophet_models:
            print(f"❌ No Prophet model found for {course}")
            fallback_pred = predict_first_year_fallback(course, year, semester)
            return fallback_pred, "fallback"
        
        model_data = prophet_models[course]
        model = model_data['model']
        
        # Convert input to date
        start_year = int(year.split('-')[0])
        month = 8 if semester == "1" else 1  # August for 1st sem, January for 2nd sem
        target_date = pd.to_datetime(f"{start_year}-{month:02d}-01")
        
        print(f"DEBUG: Predicting first-year enrollment for date: {target_date}")
        
        # Create future dataframe for the specific date
        future_dates = pd.DataFrame({'ds': [target_date]})
        
        # Get forecast
        forecast = model.predict(future_dates)
        prediction = forecast['yhat'].iloc[0]
        
        print(f"DEBUG: Raw first-year prediction: {prediction}")
        
        # Get historical bounds for validation
        if current_dataset is not None:
            course_data = current_dataset[current_dataset['Course'] == course]
            if not course_data.empty:
                # Use first-year enrollment data for validation
                historical_mean = course_data['1st_year_enrollees'].mean()
                historical_min = course_data['1st_year_enrollees'].min()
                historical_max = course_data['1st_year_enrollees'].max()
                
                print(f"DEBUG: First-year historical range: [{historical_min}, {historical_max}], Mean: {historical_mean:.1f}")
                
                # Set reasonable bounds
                lower_bound = max(10, historical_min * 0.7)
                upper_bound = historical_max * 1.3
                
                # Validate prediction
                if prediction < lower_bound or prediction > upper_bound:
                    print(f"DEBUG: First-year prediction {prediction} outside reasonable range, using historical mean")
                    prediction = historical_mean
        
        # Ensure minimum prediction
        prediction = max(10, float(prediction))
        
        print(f"✅ Prophet first-year prediction for {course}: {prediction:.0f}")
        
        return round(prediction), "prophet"
        
    except Exception as e:
        print(f"❌ Prophet prediction error for {course}: {e}")
        import traceback
        traceback.print_exc()
        fallback_pred = predict_first_year_fallback(course, year, semester)
        return fallback_pred, "fallback"

def predict_first_year_fallback(course, year, semester):
    """Simple prediction for first-year enrollment (fallback)"""
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
        base_enrollment = 50  # Lower base for first-year students
        if "BSBA" in course:
            base_enrollment = 60
        elif course in ["BSCS", "BSIT"]:
            base_enrollment = 40
        
        if semester == "1":
            base_enrollment *= 1.1
        else:
            base_enrollment *= 0.9
            
        variation = np.random.normal(0, 8)
        prediction = max(25, base_enrollment + variation)
        
        return round(prediction)
    except Exception as e:
        print(f"First-year fallback prediction error: {e}")
        return 50

# FIXED: Completely disabled future prediction check - JUST LIKE LSTM APP
def is_future_prediction(course, year, semester):
    """Check if the prediction request is for a future period - COMPLETELY DISABLED FOR HISTORICAL TESTING"""
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

# Load or train Prophet models
print("Loading Prophet models for first-year enrollment prediction...")
if not load_prophet_models():
    print("Training Prophet models for first-year enrollment...")
    train_prophet_models()

@app.before_request
def before_request():
    """Execute before each request"""
    session.permanent = True
    app.permanent_session_lifetime = timedelta(days=1)

@app.route("/")
def index():
    """Main page"""
    dataset_loaded = current_dataset is not None
    prophet_loaded = len(prophet_models) > 0
    return render_template("frontface.html", 
                         dataset_loaded=dataset_loaded, 
                         prophet_loaded=prophet_loaded)

@app.route("/check_dataset_status")
def check_dataset_status():
    """API endpoint to check dataset status"""
    dataset_loaded = current_dataset is not None
    dataset_info = None
    prophet_loaded = len(prophet_models) > 0
    
    if dataset_loaded:
        dataset_info = analyze_dataset(current_dataset)
    
    return jsonify({
        'loaded': dataset_loaded,
        'prophet_loaded': prophet_loaded,
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
        
        # Train new Prophet models with the new dataset
        train_prophet_models()
        
        # Analyze dataset for response
        analysis_result = analyze_dataset(df)
        analysis_result['filename'] = secure_filename(file.filename)
        
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'dataset_info': analysis_result,
            'prophet_loaded': len(prophet_models) > 0
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

# ADD THE MISSING ROUTES THAT YOUR TEMPLATES ARE LOOKING FOR
@app.route("/ced_filter")
def ced_filter():
    """CED filter page - ALWAYS ACCESSIBLE"""
    prophet_loaded = len(prophet_models) > 0
    return render_template("CEDfilter_prophet.html", prophet_loaded=prophet_loaded)

@app.route("/bed_filter")
def bed_filter():
    """BED filter page - ALWAYS ACCESSIBLE"""
    prophet_loaded = len(prophet_models) > 0
    return render_template("BEDfilter_prophet.html", prophet_loaded=prophet_loaded)

@app.route("/ced_filter_v2")
def ced_filter_v2():
    """CED filter page v2 - ALWAYS ACCESSIBLE"""
    prophet_loaded = len(prophet_models) > 0
    return render_template("CEDfilter_prophet.html", prophet_loaded=prophet_loaded)

@app.route("/bed_filter_v2")
def bed_filter_v2():
    """BED filter page v2 - ALWAYS ACCESSIBLE"""
    prophet_loaded = len(prophet_models) > 0
    return render_template("BEDfilter_prophet.html", prophet_loaded=prophet_loaded)

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
        
        # Use Prophet model for FIRST-YEAR prediction
        prediction, method = predict_with_prophet(course, year, semester)
        
        # Set confidence intervals
        if method == "prophet":
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
        
        # Use Prophet model for FIRST-YEAR prediction
        prediction, method = predict_with_prophet(course, year, semester)
        
        # Set confidence intervals based on method
        if method == "prophet":
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
        return jsonify({'error': f'FIRST-YEAR prediction error: {str(e)}'})

@app.route("/predict-enrollment", methods=["POST"])
def predict_enrollment():
    """Alternative FIRST-YEAR prediction endpoint for BED page - UPDATED for historical comparison - JUST LIKE LSTM APP"""
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
        
        # Use Prophet model for FIRST-YEAR prediction
        prediction, method = predict_with_prophet(course, year, semester)
        
        # Generate confidence intervals
        if method == "prophet":
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
    
    # Get historical data from dataset - KEEP ORIGINAL FORMAT
    historical_enrollment = get_historical_enrollment(course, year, semester)
    
    if historical_enrollment is None:
        flash(f"No historical data found for {course} in {year} ({semester_text} Semester)", "error")
        if course.startswith('BSBA'):
            return redirect(url_for("bed_filter"))
        else:
            return redirect(url_for("ced_filter"))
    
    # Get historical trend for charts - KEEP ORIGINAL FORMAT
    historical_trend = get_historical_trend(course)
    
    # Determine which template to use
    if course.startswith('BSBA'):
        template_name = "BED_result.html"
    else:
        template_name = "CED_result.html"
    
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
        prophet_loaded=len(prophet_models) > 0
    )

@app.route("/forecast_history", methods=['POST'])
def forecast_history():
    """Alias for forecast endpoint for CED page"""
    return forecast()

@app.route("/debug_prophet_models")
def debug_prophet_models():
    """Debug endpoint to check Prophet models status"""
    models_info = {}
    for course, model_data in prophet_models.items():
        models_info[course] = {
            'has_model': 'model' in model_data,
            'last_training_date': str(model_data.get('last_training_date', 'Unknown')),
            'training_records': model_data.get('training_records', 0),
            'historical_mean': model_data.get('historical_mean', 0),
            'historical_min': model_data.get('historical_min', 0),
            'historical_max': model_data.get('historical_max', 0)
        }
    
    return jsonify({
        'total_models': len(prophet_models),
        'models_info': models_info,
        'available_courses': list(prophet_models.keys())
    })

@app.route("/retrain_prophet_models", methods=["POST"])
def retrain_prophet_models():
    """Force retrain Prophet models"""
    try:
        if current_dataset is None:
            return jsonify({'success': False, 'error': 'No dataset available'})
        
        success = train_prophet_models()
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully retrained {len(prophet_models)} Prophet models for first-year enrollment',
                'models_trained': list(prophet_models.keys())
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to retrain models'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("=== First-Year Enrollment Forecasting System (Prophet Version) ===")
    print(f"Dataset loaded: {current_dataset is not None}")
    print(f"Prophet Models loaded: {len(prophet_models)}")
    
    if prophet_models:
        print("Loaded Prophet models for first-year enrollment prediction for courses:", list(prophet_models.keys()))
    
    if current_dataset is not None:
        print(f"Records: {len(current_dataset)}")
    
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)