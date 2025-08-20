#!/usr/bin/env python3
"""
Real-Time Multimodal Emotion Recognition Interface

DONE Features:
  - Live audio processing from microphone
  - Live video processing from webcam
  - Real-time emotion prediction
  - Multimodal fusion of audio + visual features
  - Live visualization with confidence scores

Usage:
  python scripts/realtime_interface.py --model_ckpt results/ravdess_multimodal_cross_attention_20250805_031003/best_model.pth
"""

import argparse
import cv2
import numpy as np
import torch
import torchaudio
import threading
import queue
import time
import sys
import os
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
import pyaudio
import wave

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_zser_model import MultimodalZSERModel
from scripts.extract_features import extract_wav2vec2, extract_face_frame, extract_vit_embedding

# ================================
# CONSTANTS & CONFIGURATION
# ================================
EMOTION_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_COLORS = ['gray', 'blue', 'yellow', 'blue', 'red', 'purple', 'green', 'orange']

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Video settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FPS = 30

# Processing settings
PROCESSING_INTERVAL = 0.5  # seconds
BUFFER_SIZE = 10  # frames to average predictions

# ================================
# AUDIO PROCESSING
# ================================
class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(sample_rate * 2))  # 2 seconds buffer
        self.is_recording = False
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.stream = self.p.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
        
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.extend(audio_data)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def get_audio_features(self, wav2vec2_processor, wav2vec2_model):
        """Extract audio features from current buffer"""
        if len(self.audio_buffer) < self.sample_rate:  # Need at least 1 second
            return None
            
        # Convert buffer to numpy array
        audio_data = np.array(list(self.audio_buffer))
        
        # Extract features using existing function
        features = extract_wav2vec2(audio_data, wav2vec2_processor, wav2vec2_model)
        return features

# ================================
# VIDEO PROCESSING
# ================================
class VideoProcessor:
    def __init__(self, video_width=640, video_height=480):
        self.video_width = video_width
        self.video_height = video_height
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.vit_processor = None
        self.vit_model = None
        self.current_frame = None
        self.face_detected = False
        
    def start_camera(self):
        """Start webcam capture"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        
    def stop_camera(self):
        """Stop webcam capture"""
        if self.cap:
            self.cap.release()
            
    def get_frame(self):
        """Get current frame from webcam"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                return frame
        return None
        
    def detect_face(self, frame):
        """Detect face in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            self.face_detected = True
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face_img = frame[y:y+h, x:x+w]
            return face_img, (x, y, w, h)
        else:
            self.face_detected = False
            return None, None
            
    def get_vision_features(self, face_img):
        """Extract vision features from face image"""
        if face_img is None or self.vit_processor is None or self.vit_model is None:
            return None
            
        # Extract features using existing function
        features = extract_vit_embedding(face_img, self.vit_processor, self.vit_model)
        return features

# ================================
# EMOTION PREDICTOR
# ================================
class EmotionPredictor:
    def __init__(self, model_ckpt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = MultimodalZSERModel(
            input_dim_audio=768,
            input_dim_vision=768,
            fusion_type='cross_attention',
            use_temporal=False  # Use static mode for real-time
        ).to(self.device)
        
        # Load checkpoint
        ckpt = torch.load(model_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        # Load feature extractors
        self.wav2vec2_processor = None
        self.wav2vec2_model = None
        self.vit_processor = None
        self.vit_model = None
        
        self._load_feature_extractors()
        
    def _load_feature_extractors(self):
        """Load pre-trained feature extractors"""
        from transformers import Wav2Vec2Processor, Wav2Vec2Model, ViTImageProcessor, ViTModel
        
        # Load Wav2Vec2 for audio
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        
        # Load ViT for vision
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
        
    def predict_emotion(self, audio_features, vision_features):
        """Predict emotion from audio and vision features"""
        if audio_features is None and vision_features is None:
            return None, None
            
        with torch.no_grad():
            # Prepare inputs
            if audio_features is not None:
                audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                audio_tensor = torch.zeros(1, 768, dtype=torch.float32).to(self.device)
                
            if vision_features is not None:
                vision_tensor = torch.tensor(vision_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                vision_tensor = torch.zeros(1, 768, dtype=torch.float32).to(self.device)
            
            # Get predictions
            logits, _ = self.model(audio_tensor, vision_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predicted emotion
            predicted_emotion = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_emotion].item()
            
            return predicted_emotion, confidence

# ================================
# REAL-TIME INTERFACE
# ================================
class RealTimeInterface:
    def __init__(self, model_ckpt_path):
        self.predictor = EmotionPredictor(model_ckpt_path)
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        
        # Prediction history
        self.prediction_history = deque(maxlen=BUFFER_SIZE)
        self.current_emotion = None
        self.current_confidence = 0.0
        
        # GUI elements
        self.root = None
        self.canvas = None
        self.emotion_label = None
        self.confidence_bar = None
        
    def start(self):
        """Start the real-time interface"""
        # Start audio and video processing
        self.audio_processor.start_recording()
        self.video_processor.start_camera()
        
        # Create GUI
        self._create_gui()
        
        # Start processing loop
        self._processing_loop()
        
    def stop(self):
        """Stop the real-time interface"""
        self.audio_processor.stop_recording()
        self.video_processor.stop_camera()
        if self.root:
            self.root.quit()
            
    def _create_gui(self):
        """Create the GUI window"""
        self.root = tk.Tk()
        self.root.title("Real-Time Emotion Recognition")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video display
        video_frame = ttk.LabelFrame(main_frame, text="Live Video")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas = tk.Canvas(video_frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.canvas.pack()
        
        # Emotion display
        emotion_frame = ttk.LabelFrame(main_frame, text="Emotion Prediction")
        emotion_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.emotion_label = ttk.Label(emotion_frame, text="No emotion detected", font=("Arial", 16))
        self.emotion_label.pack(pady=10)
        
        # Confidence bar
        confidence_frame = ttk.Frame(emotion_frame)
        confidence_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(confidence_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence_bar = ttk.Progressbar(confidence_frame, length=200, mode='determinate')
        self.confidence_bar.pack(side=tk.LEFT, padx=(10, 0))
        
        # Status
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="Processing...")
        self.status_label.pack(side=tk.LEFT)
        
        # Quit button
        quit_button = ttk.Button(main_frame, text="Quit", command=self.stop)
        quit_button.pack(pady=10)
        
    def _processing_loop(self):
        """Main processing loop"""
        last_process_time = time.time()
        
        while True:
            current_time = time.time()
            
            # Process at regular intervals
            if current_time - last_process_time >= PROCESSING_INTERVAL:
                self._process_frame()
                last_process_time = current_time
                
            # Update GUI
            self._update_gui()
            
            # Check if window is closed
            try:
                self.root.update()
            except tk.TclError:
                break
                
    def _process_frame(self):
        """Process current audio and video frame"""
        # Get video frame
        frame = self.video_processor.get_frame()
        if frame is None:
            return
            
        # Detect face
        face_img, face_coords = self.video_processor.detect_face(frame)
        
        # Extract features
        audio_features = self.audio_processor.get_audio_features(
            self.predictor.wav2vec2_processor, 
            self.predictor.wav2vec2_model
        )
        
        vision_features = None
        if face_img is not None:
            vision_features = self.video_processor.get_vision_features(face_img)
            
        # Predict emotion
        emotion, confidence = self.predictor.predict_emotion(audio_features, vision_features)
        
        if emotion is not None:
            self.prediction_history.append((emotion, confidence))
            
            # Average predictions over buffer
            if len(self.prediction_history) >= BUFFER_SIZE // 2:
                emotions = [p[0] for p in self.prediction_history]
                confidences = [p[1] for p in self.prediction_history]
                
                # Get most common emotion
                from collections import Counter
                emotion_counts = Counter(emotions)
                most_common_emotion = emotion_counts.most_common(1)[0][0]
                
                # Average confidence for most common emotion
                avg_confidence = np.mean([c for e, c in self.prediction_history if e == most_common_emotion])
                
                self.current_emotion = most_common_emotion
                self.current_confidence = avg_confidence
                
    def _update_gui(self):
        """Update GUI elements"""
        # Update emotion label
        if self.current_emotion is not None:
            emotion_name = EMOTION_NAMES[self.current_emotion]
            self.emotion_label.config(text=f"Emotion: {emotion_name.upper()}")
            
            # Update confidence bar
            confidence_percent = int(self.current_confidence * 100)
            self.confidence_bar['value'] = confidence_percent
            
            # Update status
            status_text = f"Audio: {'✓' if self.audio_processor.audio_buffer else '✗'}, "
            status_text += f"Face: {'✓' if self.video_processor.face_detected else '✗'}"
            self.status_label.config(text=status_text)
        else:
            self.emotion_label.config(text="No emotion detected")
            self.confidence_bar['value'] = 0
            self.status_label.config(text="Waiting for input...")

# ================================
# MAIN FUNCTION
# ================================
def main():
    parser = argparse.ArgumentParser(description="Real-time emotion recognition interface")
    parser.add_argument("--model_ckpt", type=str, required=True, 
                       help="Path to trained model checkpoint")
    
    args = parser.parse_args()
    
    # Create and start interface
    interface = RealTimeInterface(args.model_ckpt)
    
    try:
        interface.start()
    except KeyboardInterrupt:
        print("\nStopping real-time interface...")
    finally:
        interface.stop()

if __name__ == "__main__":
    main() 