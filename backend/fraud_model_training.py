import pandas as pd
import numpy as np
import re
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from textblob import TextBlob
import joblib
import logging

logger = logging.getLogger(__name__)

class JobFraudDetector:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        logger.info("Initializing JobFraudDetector")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features_df = pd.DataFrame()
        
        # Text features
        df['title'] = df['title'].fillna('')
        df['description'] = df['description'].fillna('')
        df['company_profile'] = df['company_profile'].fillna('')
        df['requirements'] = df['requirements'].fillna('')
        df['benefits'] = df['benefits'].fillna('')
        
        # Combine text fields
        df['combined_text'] = (df['title'] + ' ' + df['description'] + ' ' + 
                             df['company_profile'] + ' ' + df['requirements'] + ' ' + 
                             df['benefits'])

        # Basic text features
        features_df['text_length'] = df['combined_text'].apply(len)
        features_df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))
        features_df['avg_word_length'] = df['combined_text'].apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x else 0
        )
        
        # Sentiment features
        features_df['sentiment'] = df['combined_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        
        # Suspicious patterns
        features_df['has_email'] = df['combined_text'].apply(
            lambda x: 1 if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', x) else 0
        )
        features_df['has_url'] = df['combined_text'].apply(
            lambda x: 1 if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x) else 0
        )
        features_df['has_phone'] = df['combined_text'].apply(
            lambda x: 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', x) else 0
        )
        
        # Binary features
        features_df['telecommuting'] = df['telecommuting'].fillna(0).astype(int)
        features_df['has_company_logo'] = df['has_company_logo'].fillna(0).astype(int)
        features_df['has_questions'] = df['has_questions'].fillna(0).astype(int)

        # Fill any missing values
        features_df = features_df.fillna(0)
        
        return features_df

    def train(self, df: pd.DataFrame) -> Dict:
        """Train the fraud detection model"""
        logger.info("Creating features for training...")
        features_df = self.create_features(df)
        
        # Store feature columns for prediction
        self.feature_columns = features_df.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, 
            df['fraudulent'].fillna(0),
            test_size=0.2,
            random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        return metrics

    def predict(self, df: pd.DataFrame) -> List[dict]:
        """Make predictions on new data"""
        try:
            # Clean input data
            df = df.fillna('')
            
            # Create features
            features = self.create_features(df)
            
            # Ensure features match training columns
            missing_cols = set(self.feature_columns) - set(features.columns)
            for col in missing_cols:
                features[col] = 0
                
            features = features[self.feature_columns]
            
            # Make predictions
            fraud_probs = self.model.predict_proba(features)
            predictions = self.model.predict(features)
            
            results = []
            for prob, is_fraud in zip(fraud_probs, predictions):
                fraud_probability = float(prob[1])
                results.append({
                    'is_fraud': bool(is_fraud),
                    'fraud_probability': fraud_probability,
                    'risk_level': 'High' if fraud_probability > 0.7 
                                 else 'Medium' if fraud_probability > 0.3 
                                 else 'Low'
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Failed to make predictions: {str(e)}")

    def save_model(self, filepath: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, filepath)

    def load_model(self, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']