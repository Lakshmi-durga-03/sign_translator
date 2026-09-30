import os
import tempfile
import threading
import time
import pygame
import speech_recognition as sr
from gtts import gTTS
import pyttsx3

class TextToSpeech:
    """Utility class for text-to-speech conversion"""
    
    def __init__(self, use_gtts=True):
        """
        Initialize the text-to-speech engine
        
        Args:
            use_gtts: Whether to use Google TTS (online) or pyttsx3 (offline)
        """
        self.use_gtts = use_gtts
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Initialize pyttsx3 for offline TTS
        if not use_gtts:
            self.engine = pyttsx3.init()
            # Set properties (optional)
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    
    def speak(self, text, lang='en'):
        """
        Convert text to speech and play it
        
        Args:
            text: Text to convert to speech
            lang: Language code (for gTTS)
        """
        if not text:
            return
        
        if self.use_gtts:
            # Use Google TTS (requires internet connection)
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_filename = fp.name
                
                # Generate speech
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(temp_filename)
                
                # Play the audio
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                
                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Clean up the temporary file
                os.unlink(temp_filename)
                
            except Exception as e:
                print(f"Error in Google TTS: {e}")
                # Fall back to offline TTS
                self.use_gtts = False
                self.speak(text, lang)
        else:
            # Use pyttsx3 (offline TTS)
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error in offline TTS: {e}")
    
    def speak_async(self, text, lang='en'):
        """
        Convert text to speech asynchronously
        
        Args:
            text: Text to convert to speech
            lang: Language code (for gTTS)
        """
        thread = threading.Thread(target=self.speak, args=(text, lang))
        thread.daemon = True
        thread.start()


class SpeechToText:
    """Utility class for speech-to-text conversion"""
    
    def __init__(self):
        """Initialize the speech recognition engine"""
        self.recognizer = sr.Recognizer()
        
        # Adjust for ambient noise level
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000
        
    def listen(self, timeout=5, phrase_time_limit=5):
        """
        Listen for speech and convert to text
        
        Args:
            timeout: Maximum time to wait for speech
            phrase_time_limit: Maximum time to listen for a phrase
            
        Returns:
            text: Recognized text
            error: Error message (if any)
        """
        text = ""
        error = None
        
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                
                print("Recognizing...")
                text = self.recognizer.recognize_google(audio)
                print(f"Recognized: {text}")
                
        except sr.WaitTimeoutError:
            error = "Listening timed out. Please try again."
        except sr.UnknownValueError:
            error = "Could not understand audio. Please try again."
        except sr.RequestError as e:
            error = f"Could not request results; {e}"
        except Exception as e:
            error = f"Error: {e}"
        
        return text, error
    
    def listen_async(self, callback, timeout=5, phrase_time_limit=5):
        """
        Listen for speech asynchronously and call the callback with the result
        
        Args:
            callback: Function to call with the result (text, error)
            timeout: Maximum time to wait for speech
            phrase_time_limit: Maximum time to listen for a phrase
        """
        def listen_thread():
            text, error = self.listen(timeout, phrase_time_limit)
            callback(text, error)
        
        thread = threading.Thread(target=listen_thread)
        thread.daemon = True
        thread.start()

