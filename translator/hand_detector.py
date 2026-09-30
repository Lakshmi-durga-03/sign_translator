import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the hand detector with MediaPipe
        
        Args:
            static_mode: Whether to treat the input images as a batch or as a video stream
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
        """
        Detect hands in an image and optionally draw landmarks
        
        Args:
            img: Input image (BGR format)
            draw: Whether to draw hand landmarks on the image
            
        Returns:
            img: Image with or without drawings
            results: MediaPipe hand detection results
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if hands are detected and draw is True
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img, self.results
    
    def find_positions(self, img, hand_no=0):
        """
        Find the positions of hand landmarks
        
        Args:
            img: Input image
            hand_no: Index of the hand to find positions for (if multiple hands are detected)
            
        Returns:
            landmark_list: List of landmark positions [id, x, y, z]
        """
        img_height, img_width, _ = img.shape
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * img_width), int(lm.y * img_height)
                    # Add z coordinate (depth)
                    landmark_list.append([id, cx, cy, lm.z])
        
        return landmark_list
    
    def extract_features(self, landmark_list):
        """
        Extract features from landmark positions for sign recognition
        
        Args:
            landmark_list: List of landmark positions
            
        Returns:
            features: Numpy array of features for model input
        """
        if not landmark_list:
            return np.zeros((1, 63))  # Return zeros if no landmarks detected
        
        # Extract x, y, z coordinates for each landmark
        features = []
        for lm in landmark_list:
            features.extend([lm[1], lm[2], lm[3]])
        
        # Normalize features
        if features:
            features = np.array(features)
            # Normalize to range [0, 1]
            min_val = np.min(features)
            max_val = np.max(features)
            if max_val > min_val:
                features = (features - min_val) / (max_val - min_val)
        
        return np.array([features])

