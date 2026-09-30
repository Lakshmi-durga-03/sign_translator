import cv2
import mediapipe as mp
import numpy as np
import os
import time
from .hand_detector import HandDetector

class DataCollector:
    def __init__(self, data_dir='data'):
        """
        Initialize the data collector for sign language dataset creation
        
        Args:
            data_dir: Directory to save collected data
        """
        self.data_dir = data_dir
        self.hand_detector = HandDetector(detection_confidence=0.7)
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Create directories for each sign
        self.signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      'space', 'delete', 'nothing']
        
        for sign in self.signs:
            sign_dir = os.path.join(data_dir, sign)
            if not os.path.exists(sign_dir):
                os.makedirs(sign_dir)
    
    def collect_data(self, sign, num_samples=100):
        """
        Collect data for a specific sign
        
        Args:
            sign: The sign to collect data for
            num_samples: Number of samples to collect
            
        Returns:
            success: Whether data collection was successful
        """
        if sign not in self.signs:
            print(f"Sign '{sign}' is not in the list of supported signs.")
            return False
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        # Counter for collected samples
        counter = 0
        
        # Directory to save data
        sign_dir = os.path.join(self.data_dir, sign)
        
        print(f"Collecting data for sign '{sign}'. Press 'q' to quit.")
        print("Get ready...")
        time.sleep(2)
        
        while counter < num_samples:
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame from webcam.")
                break
            
            # Find hands and landmarks
            frame, results = self.hand_detector.find_hands(frame)
            
            # Get landmark positions
            landmark_list = self.hand_detector.find_positions(frame)
            
            # Display counter on frame
            cv2.putText(frame, f"Collecting: {sign} ({counter}/{num_samples})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Data Collection", frame)
            
            # Collect data if hand is detected
            if landmark_list:
                # Extract features
                features = self.hand_detector.extract_features(landmark_list)
                
                # Save features to file
                np.save(os.path.join(sign_dir, f"{sign}_{counter}.npy"), features)
                
                # Increment counter
                counter += 1
                
                # Small delay to avoid duplicate frames
                time.sleep(0.1)
            
            # Check for quit key
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Collected {counter} samples for sign '{sign}'.")
        return counter == num_samples
    
    def prepare_dataset(self):
        """
        Prepare the collected data for training
        
        Returns:
            X: Features
            y: Labels (one-hot encoded)
        """
        X = []
        y = []
        
        for i, sign in enumerate(self.signs):
            sign_dir = os.path.join(self.data_dir, sign)
            
            if not os.path.exists(sign_dir):
                continue
            
            # Get all .npy files in the sign directory
            files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
            
            for file in files:
                # Load features
                features = np.load(os.path.join(sign_dir, file))
                
                # Flatten features if needed
                if features.ndim > 1:
                    features = features.flatten()
                
                # Add to dataset
                X.append(features)
                
                # Create one-hot encoded label
                label = np.zeros(len(self.signs))
                label[i] = 1
                y.append(label)
        
        return np.array(X), np.array(y)

