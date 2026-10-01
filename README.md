# 👋 AI Sign Language Translator

An AI-powered web application that recognizes **sign language gestures in real time** using a webcam and converts them into text. The system uses **computer vision, hand landmark detection, and deep learning** to recognize hand signs and also provides speech-based communication features.

## 📌 Overview

The **AI Sign Language Translator** is designed to make communication more accessible between people who use sign language and people who communicate through speech or text.

The application captures hand gestures through a webcam, detects hand landmarks using **MediaPipe**, extracts normalized features, and uses a **TensorFlow neural network** to classify the detected sign.

The recognized signs are converted into text in real time. The application also supports **Text-to-Speech** and **Speech-to-Text**, enabling communication in both directions.

## ✨ Features

- 🤟 Real-time sign language recognition through webcam
- 🖐️ Hand detection and landmark extraction using MediaPipe
- 🧠 Deep learning-based sign classification using TensorFlow
- 🔤 Recognition of alphabet signs from **A–Z**
- ␣ Support for **Space**, **Delete**, and **Nothing** gestures
- 📝 Real-time text generation from recognized signs
- 🔊 Text-to-Speech conversion
- 🎤 Speech-to-Text conversion
- 📷 Live webcam processing using OpenCV
- 📊 Custom hand-landmark dataset collection
- 🌐 Django-based web application

## 🛠️ Tech Stack

**Frontend**
- HTML
- CSS
- JavaScript

**Backend**
- Python
- Django

**AI / Machine Learning**
- TensorFlow
- NumPy
- Scikit-learn

**Computer Vision**
- OpenCV
- MediaPipe

**Speech Processing**
- Google Text-to-Speech (gTTS)
- SpeechRecognition
- pyttsx3
- Pygame

**Database**
- SQLite

## 🏗️ System Workflow

```text
             Webcam Input
                  │
                  ▼
             OpenCV Capture
                  │
                  ▼
        MediaPipe Hand Detection
                  │
                  ▼
        Hand Landmark Extraction
                  │
                  ▼
        Feature Normalization
                  │
                  ▼
        TensorFlow Classifier
                  │
                  ▼
          Predicted Sign
                  │
                  ▼
          Text Generation
                  │
             ┌────┴────┐
             ▼         ▼
        Display Text   Text-to-Speech
             
Speech Input
     │
     ▼
Speech Recognition
     │
     ▼
     Text Output
```

## 🧠 Sign Recognition Pipeline

The sign recognition process consists of the following steps:

### 1. Capture

The webcam captures live video frames using **OpenCV**.

### 2. Hand Detection

**MediaPipe** detects the user's hand and extracts its landmark coordinates.

### 3. Feature Extraction

The hand landmarks are converted into numerical features and normalized before being passed to the classifier.

### 4. Sign Classification

A **TensorFlow neural network model** predicts the corresponding sign from the extracted features.

The trained model supports:

```text
A - Z
space
delete
nothing
```

### 5. Text Formation

Predicted signs are accumulated to form meaningful text.

For example:

```text
H → E → L → L → O
```

produces:

```text
HELLO
```

### 6. Voice Output

The generated text can be converted into speech using the integrated Text-to-Speech functionality.

## 📂 Project Structure

```text
sign_translator/
│
├── data/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── ...
│   ├── Z/
│   ├── space/
│   ├── delete/
│   └── nothing/
│
├── sign_translator/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── translator/
│   ├── data_collector.py
│   ├── hand_detector.py
│   ├── sign_classifier.py
│   ├── speech_utils.py
│   ├── views.py
│   ├── urls.py
│   └── models/
│       └── sign_language_model.h5
│
├── templates/
│   └── translator/
│       └── index.html
│
├── collect_data.py
├── train_model.py
├── manage.py
├── requirements.txt
└── README.md
```

## 📊 Dataset

The project uses a custom dataset consisting of **hand landmark features** collected through the webcam.

Each gesture is stored as a NumPy `.npy` file under its corresponding class directory.

```text
data/
├── A/
├── B/
├── C/
├── ...
├── Z/
├── space/
├── delete/
└── nothing/
```

Instead of directly training on raw images, the system uses **hand landmark coordinates extracted by MediaPipe**, providing a compact representation of the hand gesture.

## 🏋️ Model Training

The project includes a dedicated training pipeline through `train_model.py`.

The training process:

1. Loads the collected landmark data.
2. Prepares the feature and label arrays.
3. Splits the dataset into training and validation sets.
4. Trains the TensorFlow neural network.
5. Evaluates the model on validation data.
6. Saves the trained model as:

```text
translator/models/sign_language_model.h5
```

## 🎤 Speech-to-Text

The application also provides speech recognition functionality.

The microphone captures the user's speech and **SpeechRecognition** processes the audio to generate text.

This allows users to communicate with the system without manually typing.

## 🔊 Text-to-Speech

Recognized sign language text can be converted into speech.

The project supports:

- **gTTS** for online text-to-speech
- **pyttsx3** for offline text-to-speech fallback
- **Pygame** for audio playback

This provides an additional communication channel for users.

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Webcam
- Microphone
- Git

### Clone the Repository

```bash
git clone https://github.com/Lakshmi-durga-03/sign_translator.git
cd sign_translator
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Database Migrations

```bash
python manage.py migrate
```

### Start the Application

```bash
python manage.py runserver
```

Open the application in your browser:

```text
http://127.0.0.1:8000/
```

## 📸 Application

The application provides a web interface where users can interact with the sign recognition system and access the available speech features.

> Add screenshots of your application here to make the repository more informative.

```text
screenshots/
├── home.png
├── sign-recognition.png
└── speech-output.png
```

## 🎯 Use Cases

- Accessibility-focused communication
- Sign language learning and practice
- Assistive technology
- Human-computer interaction
- Computer vision applications
- AI/ML academic projects

## 🔮 Future Enhancements

- Support for complete words and sentences
- Dynamic gesture recognition
- Recognition of multiple sign languages
- Improved accuracy using larger and more diverse datasets
- Mobile application support
- Multilingual speech output
- Cloud-based model deployment
- Continuous sentence prediction

