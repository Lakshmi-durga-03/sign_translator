import tensorflow as tf
import numpy as np
import os

class SignClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the sign language classifier
        
        Args:
            model_path: Path to the pre-trained model (if available)
        """
        self.model = None
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      'space', 'delete', 'nothing']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """
        Build a TensorFlow model for sign language classification
        """
        # Input shape: 21 landmarks with x, y, z coordinates (21 * 3 = 63)
        input_shape = (63,)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, validation_data=None):
        """
        Train the model with the provided data
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation (used if validation_data is None)
            validation_data: Tuple of (X_val, y_val) for validation
        
        Returns:
            history: Training history
        """
        if self.model is None:
            self.build_model()
        
        if validation_data is not None:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split
            )
        
        return history
    
    def predict(self, features):
        """
        Predict the sign from hand landmarks
        
        Args:
            features: Extracted features from hand landmarks
            
        Returns:
            predicted_label: Predicted sign label
            confidence: Prediction confidence
        """
        if self.model is None:
            return "No model loaded", 0.0
        
        if features.size == 0 or np.all(features == 0):
            return "No hand detected", 0.0
        
        # Make prediction
        prediction = self.model.predict(features)
        predicted_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_index]
        
        return self.labels[predicted_index], float(confidence)
    
    def save_model(self, model_path):
        """
        Save the trained model
        
        Args:
            model_path: Path to save the model
        """
        if self.model:
            self.model.save(model_path)
    
    def load_model(self, model_path):
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the pre-trained model
        """
        self.model = tf.keras.models.load_model(model_path)

