import os
import numpy as np
import tensorflow as tf
from translator.sign_classifier import SignClassifier
from translator.data_collector import DataCollector

# Suppress TensorFlow warnings about AVX2 FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    """
    Train the sign language recognition model using collected data
    """
    print("Sign Language Recognition Model Training")
    print("----------------------------------------")
    
    # Initialize data collector
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    collector = DataCollector(data_dir=data_dir)
    
    # Check if data exists
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("No training data found. Please collect data first.")
        return
    
    # Prepare dataset
    print("Preparing dataset...")
    X, y = collector.prepare_dataset()
    
    if len(X) == 0:
        print("No samples found in the dataset.")
        return
    
    print(f"Dataset prepared with {len(X)} samples.")
    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize classifier
    classifier = SignClassifier()
    
    # Train model
    print("Training model...")
    history = classifier.train(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
    
    # Evaluate model
    print("Evaluating model...")
    loss, accuracy = classifier.model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {accuracy:.4f}")
    
    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), 'translator', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'sign_language_model.h5')
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

