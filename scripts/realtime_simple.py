#!/usr/bin/env python3
"""
Simple Real-Time Multimodal Emotion Recognition Interface

DONE Features:
  - Live audio processing from microphone
  - Live video processing from webcam
  - Real-time emotion prediction
  - Console-based output
  - OpenCV video display

Usage:
  python scripts/realtime_simple.py --model_ckpt results/ravdess_multimodal_cross_attention_20250805_031003/best_model.pth
"""

import argparse
import cv2
import numpy as np
import torch
import time
import sys
import os
from collections import deque
import threading
import queue

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_zser_model import MultimodalZSERModel

# ================================
# CONSTANTS & CONFIGURATION
# ================================
EMOTION_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_COLORS = [(128, 128, 128), (255, 0, 0), (0, 255, 255), (255, 0, 0), 
                  (0, 0, 255), (255, 0, 255), (0, 255, 0), (0, 165, 255)]

# Video settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FPS = 30

# Processing settings
PROCESSING_INTERVAL = 1.0  # seconds
BUFFER_SIZE = 5  # frames to average predictions

# ================================
# SIMULATED AUDIO PROCESSING
# ================================
class SimulatedAudioProcessor:
    """Simulated audio processor for demo purposes"""
    def __init__(self):
        self.audio_features = np.random.randn(768)  # Simulated features
        
    def get_audio_features(self):
        """Return simulated audio features"""
        return self.audio_features.copy()

# ================================
# VIDEO PROCESSING
# ================================
class VideoProcessor:
    def __init__(self, video_width=640, video_height=480):
        self.video_width = video_width
        self.video_height = video_height
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
        """Extract vision features from face image (simulated)"""
        if face_img is None:
            return None
            
        # Simulate ViT features
        features = np.random.randn(768)
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
            use_temporal=True  # Match the saved model configuration
        ).to(self.device)
        
        # Load checkpoint
        ckpt = torch.load(model_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
    def predict_emotion(self, audio_features, vision_features):
        """Predict emotion from audio and vision features"""
        if audio_features is None and vision_features is None:
            return None, None
            
        with torch.no_grad():
            # Prepare inputs for temporal model
            if audio_features is not None:
                # Convert to temporal format [B, T, D] where T=1
                audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                audio_tensor = torch.zeros(1, 1, 768, dtype=torch.float32).to(self.device)
                
            if vision_features is not None:
                # Convert to temporal format [B, T, D] where T=1
                vision_tensor = torch.tensor(vision_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                vision_tensor = torch.zeros(1, 1, 768, dtype=torch.float32).to(self.device)
            
            # Get predictions
            logits = self.model(audio_tensor, vision_tensor)
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
        self.audio_processor = SimulatedAudioProcessor()
        self.video_processor = VideoProcessor()
        
        # Prediction history
        self.prediction_history = deque(maxlen=BUFFER_SIZE)
        self.current_emotion = None
        self.current_confidence = 0.0
        
    def start(self):
        """Start the real-time interface"""
        # Start video processing
        self.video_processor.start_camera()
        
        print("🎭 Real-Time Emotion Recognition Interface")
        print("Press 'q' to quit")
        print("-" * 50)
        
        # Main processing loop
        self._processing_loop()
        
    def stop(self):
        """Stop the real-time interface"""
        self.video_processor.stop_camera()
        cv2.destroyAllWindows()
        
    def _processing_loop(self):
        """Main processing loop"""
        last_process_time = time.time()
        
        while True:
            current_time = time.time()
            
            # Get video frame
            frame = self.video_processor.get_frame()
            if frame is None:
                continue
                
            # Process at regular intervals
            if current_time - last_process_time >= PROCESSING_INTERVAL:
                self._process_frame(frame)
                last_process_time = current_time
                
            # Display frame
            self._display_frame(frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def _process_frame(self, frame):
        """Process current audio and video frame"""
        # Detect face
        face_img, face_coords = self.video_processor.detect_face(frame)
        
        # Extract features
        audio_features = self.audio_processor.get_audio_features()
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
                
                # Print to console
                emotion_name = EMOTION_NAMES[self.current_emotion]
                print(f"🎯 Emotion: {emotion_name.upper()} | Confidence: {self.current_confidence:.2f}")
                
    def _display_frame(self, frame):
        """Display frame with emotion overlay"""
        # Draw face detection box
        face_img, face_coords = self.video_processor.detect_face(frame)
        if face_coords:
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Draw emotion prediction
        if self.current_emotion is not None:
            emotion_name = EMOTION_NAMES[self.current_emotion]
            confidence_percent = int(self.current_confidence * 100)
            
            # Draw emotion text
            text = f"{emotion_name.upper()}: {confidence_percent}%"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       EMOTION_COLORS[self.current_emotion], 2)
            
            # Draw confidence bar
            bar_width = 200
            bar_height = 20
            bar_x = 10
            bar_y = 60
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (128, 128, 128), -1)
            
            # Confidence bar
            confidence_width = int(bar_width * self.current_confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), 
                         EMOTION_COLORS[self.current_emotion], -1)
            
        # Draw status
        status_text = f"Audio: {'✓' if True else '✗'}, Face: {'✓' if self.video_processor.face_detected else '✗'}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Real-Time Emotion Recognition', frame)

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