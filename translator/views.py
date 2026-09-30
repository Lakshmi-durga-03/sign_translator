from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import json
import base64
import os
import time
import tempfile

# Suppress TensorFlow warnings about AVX2 FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from .hand_detector import HandDetector
from .sign_classifier import SignClassifier
from .speech_utils import TextToSpeech, SpeechToText

# Initialize the hand detector and sign classifier
hand_detector = HandDetector(detection_confidence=0.7)

# Path to the model - you'll need to train and save a model first
model_path = os.path.join(os.path.dirname(__file__), 'models', 'sign_language_model.h5')
model_dir = os.path.join(os.path.dirname(__file__), 'models')

# Create models directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Initialize classifier (it will build a new model if no model exists)
sign_classifier = SignClassifier(model_path if os.path.exists(model_path) else None)

# Initialize text-to-speech and speech-to-text engines
tts = TextToSpeech(use_gtts=True)  # Use Google TTS by default
stt = SpeechToText()

# Global variables for text accumulation
current_text = ""
last_prediction = ""
prediction_count = 0
min_consistent_predictions = 5  # Number of consistent predictions required to add a letter

def index(request):
    """Render the main page"""
    return render(request, 'translator/index.html')

def generate_frames():
    """Generator function for streaming video frames"""
    global current_text, last_prediction, prediction_count
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Find hands and landmarks
        frame, results = hand_detector.find_hands(frame)
        
        # Get landmark positions
        landmark_list = hand_detector.find_positions(frame)
        
        # Extract features and make prediction if landmarks are detected
        prediction_text = "No hand detected"
        confidence = 0.0
        
        if landmark_list:
            features = hand_detector.extract_features(landmark_list)
            prediction_text, confidence = sign_classifier.predict(features)
            
            # Accumulate text based on consistent predictions
            if prediction_text == last_prediction and prediction_text not in ["No hand detected", "nothing"]:
                prediction_count += 1
                if prediction_count >= min_consistent_predictions:
                    if prediction_text == "space":
                        current_text += " "
                    elif prediction_text == "delete" and current_text:
                        current_text = current_text[:-1]
                    else:
                        current_text += prediction_text
                    prediction_count = 0
            else:
                last_prediction = prediction_text
                prediction_count = 0
        
        # Display prediction and current text on frame
        cv2.putText(frame, f"Prediction: {prediction_text} ({confidence:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {current_text}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def video_feed(request):
    """Video streaming route"""
    return StreamingHttpResponse(generate_frames(),
                                content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def process_frame(request):
    """Process a single frame sent from the client"""
    global current_text, last_prediction, prediction_count
    
    # Handle direct GET requests by redirecting to the main page
    if request.method == 'GET':
        return redirect('index')
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image', '')
            
            # Decode base64 image
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Find hands and landmarks
            frame, results = hand_detector.find_hands(frame, draw=False)
            
            # Get landmark positions
            landmark_list = hand_detector.find_positions(frame)
            
            # Extract features and make prediction if landmarks are detected
            prediction_text = "No hand detected"
            confidence = 0.0
            
            if landmark_list:
                features = hand_detector.extract_features(landmark_list)
                prediction_text, confidence = sign_classifier.predict(features)
                
                # Accumulate text based on consistent predictions
                if prediction_text == last_prediction and prediction_text not in ["No hand detected", "nothing"]:
                    prediction_count += 1
                    if prediction_count >= min_consistent_predictions:
                        if prediction_text == "space":
                            current_text += " "
                        elif prediction_text == "delete" and current_text:
                            current_text = current_text[:-1]
                        else:
                            current_text += prediction_text
                        prediction_count = 0
                else:
                    last_prediction = prediction_text
                    prediction_count = 0
            
            return JsonResponse({
                'prediction': prediction_text,
                'confidence': float(confidence),
                'current_text': current_text
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def clear_text(request):
    """Clear the accumulated text"""
    global current_text
    
    # Accept both GET and POST requests for clearing text
    current_text = ""
    return JsonResponse({'status': 'success', 'current_text': current_text})

@csrf_exempt
def update_text(request):
    """Update the text manually"""
    global current_text
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            mode = data.get('mode', 'replace')  # 'replace' or 'append'
            
            if mode == 'replace':
                current_text = text
            elif mode == 'append':
                current_text += text
            
            return JsonResponse({
                'status': 'success',
                'current_text': current_text
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def text_to_speech(request):
    """Convert text to speech"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            
            if not text:
                return JsonResponse({'error': 'No text provided'}, status=400)
            
            # Speak the text asynchronously
            tts.speak_async(text)
            
            return JsonResponse({'status': 'success', 'message': 'Speaking text'})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def speech_to_text(request):
    """Convert speech to text"""
    global current_text
    
    if request.method == 'POST':
        try:
            # Listen for speech
            text, error = stt.listen()
            
            if error:
                return JsonResponse({'error': error}, status=400)
            
            # Update the current text
            if text:
                current_text = text
            
            return JsonResponse({
                'status': 'success',
                'text': text,
                'current_text': current_text
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def train_model(request):
    """Train the sign language model with collected data"""
    if request.method == 'POST':
        try:
            # In a real application, you would load your training data here
            # For this example, we'll just return a message
            return JsonResponse({
                'status': 'success',
                'message': 'Model training would start here. In a real application, you would need to collect and prepare training data.'
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

