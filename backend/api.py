import datetime
import io
import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from fraud_model_training import JobFraudDetector  # Make sure this import works
import pandas as pd
import logging
from sklearn.metrics import precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model instance - Initialize as None first
fraud_detector = None
MODEL_PATH = 'job_fraud_model.pkl'

def initialize_model():
    """Initialize the fraud detection model"""
    global fraud_detector
    
    try:
        # Create the fraud detector instance
        fraud_detector = JobFraudDetector()
        logger.info("JobFraudDetector instance created")

        # Check if model file exists and is valid
        if os.path.exists(MODEL_PATH):
            try:
                logger.info("Loading existing model...")
                fraud_detector.load_model(MODEL_PATH)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load existing model: {str(e)}")
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                logger.info("Deleted invalid model file")
                raise

        # Train new model with correct CSV path
        if not os.path.exists(MODEL_PATH):
            logger.info("Training new model...")
            # Use absolute path to Training_Data.csv
            csv_path = os.path.join(os.path.dirname(__file__), 'Training_Data.csv')
            logger.info(f"Reading training data from: {csv_path}")
            
            if not os.path.exists(csv_path):
                logger.warning("Training_Data.csv not found, creating sample data...")
                train_data = create_sample_training_data()
            else:
                train_data = pd.read_csv(csv_path)
                
            logger.info(f"Loaded {len(train_data)} training records")
            
            metrics = fraud_detector.train(train_data)
            logger.info(f"Training metrics: {metrics}")
            
            fraud_detector.save_model(MODEL_PATH)
            
        logger.info("Model initialization complete")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        # Create a basic fraud detector even if training fails
        fraud_detector = JobFraudDetector()
        raise

def create_sample_training_data():
    """Create sample training data for the model"""
    np.random.seed(42)
    n_samples = 2000
    
    data = []
    for i in range(n_samples):
        is_fraud = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% fraud rate
        
        if is_fraud:
            # Fraudulent job characteristics
            titles = [
                'Work from Home - Make $5000/month!',
                'URGENT: Easy Money Opportunity',
                'No Experience Required - High Pay',
                'Quick Cash Job - Start Today',
                'Make Money Fast - Guaranteed Income',
                'Home-based Business Opportunity',
                'Earn $3000 Weekly - No Skills Needed'
            ]
            
            descriptions = [
                'Make money fast! No experience needed. Work from home and earn guaranteed income! Contact us now!',
                'Urgent hiring! Easy work, high pay. Send processing fee for starter kit. Guaranteed results!',
                'Amazing opportunity! Work from anywhere. Guaranteed $3000/week! No experience required!',
                'Join our team and make easy money! Work from home opportunity. Pay small fee to get started.',
                'Exclusive business opportunity! Make thousands weekly! No skills needed! Act fast!'
            ]
            
            title = np.random.choice(titles)
            description = np.random.choice(descriptions)
            salary_range = np.random.choice(['$100000-$200000', '$5000-$15000/week', '$200000+', ''])
            required_experience = np.random.choice(['No experience', 'Entry level', '', 'Any'])
            required_education = np.random.choice(['High school', '', 'Any', 'None'])
            telecommuting = np.random.choice([True, True, False], p=[0.8, 0.1, 0.1])
            has_company_logo = np.random.choice([False, True], p=[0.8, 0.2])
            location = np.random.choice(['Remote', 'Work from home', 'Anywhere', 'USA'])
            company_profile = np.random.choice(['', 'New company', 'Growing business'])
            
        else:
            # Legitimate job characteristics
            titles = [
                'Software Engineer - Python Developer',
                'Marketing Coordinator',
                'Data Analyst Position',
                'Customer Service Representative',
                'Project Manager',
                'Sales Associate',
                'Graphic Designer',
                'Financial Analyst',
                'HR Specialist',
                'Content Writer'
            ]
            
            descriptions = [
                'We are looking for a skilled software engineer to join our team. Requirements include 3+ years experience with Python and web frameworks.',
                'Marketing coordinator needed for growing company. Bachelor degree preferred. Experience with digital marketing required.',
                'Data analyst position available. Experience with SQL, Python, and data visualization tools required.',
                'Customer service representative needed. Excellent communication skills required. Full-time position with benefits.',
                'Project manager position available. PMP certification preferred. 5+ years experience required.',
                'Sales associate position. Experience in retail preferred. Competitive salary and commission.',
                'Graphic designer needed. Proficiency in Adobe Creative Suite required. Portfolio review required.',
                'Financial analyst position. CPA preferred. Experience with financial modeling and analysis.',
                'HR specialist needed. Experience with recruitment and employee relations. Bachelor degree required.',
                'Content writer position. Experience with SEO and content management systems preferred.'
            ]
            
            title = np.random.choice(titles)
            description = np.random.choice(descriptions)
            salary_range = np.random.choice(['$45000-$65000', '$55000-$75000', '$40000-$55000', '$60000-$85000'])
            required_experience = np.random.choice(['1-3 years', '2-4 years', '3-5 years', '1-2 years'])
            required_education = np.random.choice(['Bachelor degree', 'Associate degree', 'Bachelor degree preferred', 'High school'])
            telecommuting = np.random.choice([True, False], p=[0.3, 0.7])
            has_company_logo = np.random.choice([True, False], p=[0.9, 0.1])
            
            locations = ['New York, NY', 'San Francisco, CA', 'Chicago, IL', 'Los Angeles, CA', 
                        'Boston, MA', 'Seattle, WA', 'Austin, TX', 'Denver, CO', 'Remote']
            location = np.random.choice(locations)
            
            company_profiles = ['Tech startup', 'Fortune 500 company', 'Growing business', 
                              'Established corporation', 'Non-profit organization', 'Consulting firm']
            company_profile = np.random.choice(company_profiles)
        
        data.append({
            'title': title,
            'description': description,
            'salary_range': salary_range,
            'required_experience': required_experience,
            'required_education': required_education,
            'telecommuting': telecommuting,
            'has_company_logo': has_company_logo,
            'location': location,
            'company_profile': company_profile,
            'fraudulent': is_fraud
        })
    
    return pd.DataFrame(data)

def safe_probability_fix(probability_value):
    """Safely fix any probability value to be between 0-100%"""
    try:
        # Handle None or empty values
        if probability_value is None or probability_value == '':
            return 0.0
            
        # Convert to float if it's a string
        if isinstance(probability_value, str):
            probability_value = float(probability_value.replace('%', ''))
        
        # Handle NaN
        if np.isnan(probability_value):
            return 0.0
            
        # If it's already a percentage (>1), keep as is but cap at 100
        if probability_value > 1:
            return min(probability_value, 100.0)
        else:
            # If it's a decimal (0-1), convert to percentage
            return min(probability_value * 100, 100.0)
            
    except (ValueError, TypeError, AttributeError):
        logger.warning(f"Could not process probability value: {probability_value}")
        return 0.0

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'model_loaded': fraud_detector is not None and hasattr(fraud_detector, 'model') and fraud_detector.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
            
        file = request.files['file']
        df = pd.read_csv(file.stream)
        df = df.fillna('')
        
        # Make predictions
        predictions = fraud_detector.predict(df)
        
        results = []
        for i, pred in enumerate(predictions):
            row = df.iloc[i]
            
            # Extract salary information - FIXED VERSION
            salary_range = None
            
            # First, check if there's a direct salary_range column
            if 'salary_range' in row and row['salary_range'] and str(row['salary_range']).strip():
                salary_range = str(row['salary_range']).strip()
            
            # If no direct salary_range, try to extract from description
            elif 'description' in row and row['description']:
                desc = str(row['description'])
                
                # Look for various salary patterns
                salary_patterns = [
                    r'\$[\d,]+\s*-\s*\$[\d,]+(?:\s*(?:per|\/)\s*(?:hour|hr|year|yr|month|mo|week|wk))?',
                    r'\$[\d,]+(?:\s*(?:per|\/)\s*(?:hour|hr|year|yr|month|mo|week|wk))',
                    r'[\d,]+\s*-\s*[\d,]+\s*(?:per|\/)\s*(?:hour|hr|year|yr|month|mo|week|wk)',
                    r'\$[\d,]+\+',
                    r'up to \$[\d,]+',
                    r'starting at \$[\d,]+'
                ]
                
                for pattern in salary_patterns:
                    salary_matches = re.findall(pattern, desc, re.IGNORECASE)
                    if salary_matches:
                        salary_range = salary_matches[0]
                        break
                
                # If still no match, look for any dollar amounts
                if not salary_range and '$' in desc:
                    dollar_matches = re.findall(r'\$[\d,]+', desc)
                    if dollar_matches:
                        if len(dollar_matches) >= 2:
                            salary_range = f"{dollar_matches[0]} - {dollar_matches[1]}"
                        else:
                            salary_range = dollar_matches[0]
            
            # Check other possible salary columns
            if not salary_range:
                salary_columns = ['salary', 'compensation', 'pay', 'wage', 'salary_min', 'salary_max']
                for col in salary_columns:
                    if col in row and row[col] and str(row[col]).strip():
                        salary_range = str(row[col]).strip()
                        break
            
            # If we have salary_min and salary_max columns, combine them
            if not salary_range and 'salary_min' in row and 'salary_max' in row:
                if row['salary_min'] and row['salary_max']:
                    salary_range = f"${row['salary_min']} - ${row['salary_max']}"
            
            # Default fallback only if nothing found
            if not salary_range:
                salary_range = "Not specified"
            
            # Get company information
            company_info = str(row.get('company_profile', '')).strip()
            company_name = None
            
            # Try different company name columns
            if 'company' in row and row['company']:
                company_name = str(row['company']).strip()
            elif company_info:
                # Try to extract company name from first sentence
                sentences = company_info.split('.')
                if sentences:
                    company_name = sentences[0].strip()
            
            job_details = {
                'title': str(row.get('title', '')) or 'Title not specified',
                'company_profile': company_info or 'No company profile available',
                'location': str(row.get('location', '')) or 'Location not specified',
                'salary_range': salary_range, 
                'employment_type': str(row.get('employment_type', '')) or 'Not specified',
                'description': str(row.get('description', '')) or 'No description available',
                'fraud_probability': float(pred['fraud_probability'] * 100),
                'risk_level': str(pred['risk_level']),
                'is_fraud': bool(pred['is_fraud'])
            }
            results.append(job_details)
            
        suspicious_jobs = sorted(results, key=lambda x: x['fraud_probability'], reverse=True)[:10]
        
        return jsonify({
            'success': True,
            'predictions': results,
            'suspicious_jobs': suspicious_jobs,
            'summary': {
                'totalJobs': int(len(df)),
                'fraudulentJobs': int(sum(1 for p in predictions if p['is_fraud'])),
                'genuineJobs': int(sum(1 for p in predictions if not p['is_fraud']))
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if fraud_detector is None:
        return jsonify({'error': 'Model not initialized'}), 500
        
    if not hasattr(fraud_detector, 'model') or fraud_detector.model is None:
        return jsonify({'error': 'Model not trained'}), 500
    
    info = {
        'model_type': 'Random Forest Classifier',
        'features_count': len(fraud_detector.feature_names) if hasattr(fraud_detector, 'feature_names') and fraud_detector.feature_names else 0,
        'model_loaded': True,
        'last_updated': datetime.datetime.now().isoformat()
    }
    
    return jsonify(info)

@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample data for testing"""
    sample_jobs = [
        {
            'title': 'Work from Home - Make $5000/month!',
            'description': 'Make money fast! No experience needed. Work from home and earn guaranteed income!',
            'salary_range': '$100000-$200000',
            'required_experience': 'No experience',
            'required_education': 'High school',
            'telecommuting': True,
            'has_company_logo': False,
            'location': 'Remote',
            'company_profile': 'New company'
        },
        {
            'title': 'Software Engineer - Python Developer',
            'description': 'We are looking for a skilled software engineer to join our team. Requirements include 3+ years experience with Python and web frameworks.',
            'salary_range': '$75000-$95000',
            'required_experience': '3-5 years',
            'required_education': 'Bachelor degree',
            'telecommuting': False,
            'has_company_logo': True,
            'location': 'San Francisco, CA',
            'company_profile': 'Tech startup'
        },
        {
            'title': 'URGENT: Easy Money Opportunity',
            'description': 'Urgent hiring! Easy work, high pay. Send processing fee for starter kit. Guaranteed results!',
            'salary_range': '$5000-$15000/week',
            'required_experience': 'Any',
            'required_education': 'Any',
            'telecommuting': True,
            'has_company_logo': False,
            'location': 'Work from home',
            'company_profile': ''
        }
    ]
    
    return jsonify({
        'sample_jobs': sample_jobs,
        'description': 'Sample job postings for testing the fraud detection API'
    })

@app.route('/model/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with new data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No training file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Read the training data
        try:
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Validate required columns
        required_columns = ['title', 'fraudulent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing required columns: {missing_columns}'}), 400
        
        # Initialize new model instance
        global fraud_detector
        fraud_detector = JobFraudDetector()
        
        # Train the model
        fraud_detector.train(df)
        
        # Save the model
        fraud_detector.save_model(MODEL_PATH)
        
        logger.info("Model retrained successfully")
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'training_samples': len(df),
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in retraining: {str(e)}")
        return jsonify({'error': f'Model retraining failed: {str(e)}'}), 500
    
@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get comprehensive model performance statistics"""
    try:
        # Check if model is initialized
        if fraud_detector is None:
            logger.error("Model not initialized")
            return jsonify({
                'success': False, 
                'message': 'Model not initialized. Please train the model first.'
            }), 500
            
        # Check if model is trained
        if not hasattr(fraud_detector, 'model') or fraud_detector.model is None:
            logger.error("Model not trained")
            return jsonify({
                'success': False, 
                'message': 'Model not trained. Please train the model first.'
            }), 500
            
        logger.info("Fetching model statistics...")
        
        # Try to read training data
        train_data_path = os.path.join(os.path.dirname(__file__), 'Training_Data.csv')
        
        if os.path.exists(train_data_path):
            try:
                train_data = pd.read_csv(train_data_path)
                logger.info(f"Loaded training data from file: {len(train_data)} samples")
            except Exception as e:
                logger.warning(f"Failed to read training data file: {str(e)}")
                train_data = create_sample_training_data()
                logger.info(f"Created sample training data: {len(train_data)} samples")
        else:
            logger.info("Training data file not found, creating sample data")
            train_data = create_sample_training_data()
            
        # Ensure we have the required columns
        if 'fraudulent' not in train_data.columns:
            logger.error("Training data missing 'fraudulent' column")
            return jsonify({
                'success': False,
                'message': "Training data missing required 'fraudulent' column"
            }), 500
            
        # Fill missing values
        train_data = train_data.fillna('')
        
        try:
            # Create features using the model's method
            logger.info("Creating features for evaluation...")
            features = fraud_detector.create_features(train_data)
            
            # Get true labels
            y_true = train_data['fraudulent'].astype(int)
            
            # Make predictions
            logger.info("Making predictions for evaluation...")
            y_pred = fraud_detector.model.predict(features)
            y_pred_proba = fraud_detector.model.predict_proba(features)[:, 1]  # Probability of fraud class
            
            # Calculate comprehensive metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate additional metrics
            fraud_rate = float(y_true.sum() / len(y_true) * 100)
            total_samples = len(y_true)
            fraud_samples = int(y_true.sum())
            legitimate_samples = int(total_samples - fraud_samples)
            
            # Create comprehensive metrics response
            metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'fraud_rate': float(fraud_rate),
                'total_samples': int(total_samples),
                'fraud_samples': int(fraud_samples),
                'legitimate_samples': int(legitimate_samples),
                # Additional insights
                'model_accuracy': float(np.mean(y_true == y_pred)),
                'true_positives': int(np.sum((y_true == 1) & (y_pred == 1))),
                'false_positives': int(np.sum((y_true == 0) & (y_pred == 1))),
                'true_negatives': int(np.sum((y_true == 0) & (y_pred == 0))),
                'false_negatives': int(np.sum((y_true == 1) & (y_pred == 0))),
                'average_fraud_probability': float(np.mean(y_pred_proba[y_true == 1])) if fraud_samples > 0 else 0.0,
                'average_legitimate_probability': float(np.mean(y_pred_proba[y_true == 0])) if legitimate_samples > 0 else 0.0
            }
            
            logger.info(f"Calculated metrics successfully: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
            
            # Create response with proper CORS headers
            response = jsonify({
                'success': True,
                'metrics': metrics,
                'message': 'Model statistics retrieved successfully',
                'timestamp': datetime.datetime.now().isoformat(),
                'evaluation_note': 'F1-score is the primary metric due to dataset imbalance'
            })
            
            # Add CORS headers explicitly
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
            
            return response
            
        except Exception as feature_error:
            logger.error(f"Error in feature creation or prediction: {str(feature_error)}")
            return jsonify({
                'success': False,
                'message': f'Error evaluating model: {str(feature_error)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in /statistics endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Statistics calculation failed: {str(e)}',
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

# Add CORS preflight handler for statistics endpoint
@app.route('/statistics', methods=['OPTIONS'])
def statistics_options():
    """Handle preflight OPTIONS request for statistics endpoint"""
    response = jsonify({'status': 'OK'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
    return response

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Initialize the model when starting the app
        initialize_model()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.info("Starting app without model - some endpoints may not work")
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)