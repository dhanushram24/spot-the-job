import pandas as pd
from fraud_model_training import JobFraudDetector
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model_with_data():
    try:
        # Read the training data
        logger.info("Loading training data...")
        df = pd.read_csv('Training_Data.csv')
        logger.info(f"Loaded {len(df)} job postings")

        # Initialize the fraud detector
        logger.info("Initializing fraud detector...")
        detector = JobFraudDetector()

        # Train the model
        logger.info("Training model...")
        metrics = detector.train(df)

        # Save the trained model
        logger.info("Saving model...")
        detector.save_model('job_fraud_model.pkl')

        # Print metrics
        logger.info("\nTraining Results:")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")

        return True

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    train_model_with_data()