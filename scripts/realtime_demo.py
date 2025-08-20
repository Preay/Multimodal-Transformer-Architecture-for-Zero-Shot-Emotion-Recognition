#!/usr/bin/env python3
"""
Real-time Emotion Recognition Demo

This script provides a live demonstration of the multimodal emotion recognition system
using webcam and microphone input. It can run in audio-only, video-only, or multimodal modes.

Usage:
    python scripts/realtime_demo.py --mode audio_only
    python scripts/realtime_demo.py --mode multimodal
    python scripts/realtime_demo.py --mode video_only
"""

import cv2
import pyaudio
import numpy as np
import torch
import torch.nn.functional as F
import threading
import queue
import time
import argparse
import os
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from models.multimodal_zser_model import create_model
from scripts.extract_audio import extract_wav2vec2_features
from scripts.extract_vision import extract_face_frame, extract_vit_embedding
from transformers import Wav2Vec2Processor, Wav2Vec2Model, ViTModel, ViTImageProcessor

class RealTimeEmotionRecognition:
    """Real-time emotion recognition system using webcam and microphone."""
    
    def __init__(self, model_path, mode='multimodal', device='cpu'):
        self.mode = mode
        self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = create_model(mode)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels
        self.emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Load feature extractors
        if mode in ['audio_only', 'multimodal']:
            self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_model.to(self.device)
            self.audio_model.eval()
        
        if mode in ['video_only', 'multimodal']:
            self.vision_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.vision_model.to(self.device)
            self.vision_model.eval()
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.audio_buffer = deque(maxlen=int(self.RATE * 2))  # 2 seconds buffer
        
        # Video settings
        self.video_capture = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Queues for thread communication
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Prediction history
        self.prediction_history = deque(maxlen=30)  # Last 30 predictions
        
        # Threading
        self.running = False
        self.audio_thread = None
        self.video_thread = None
        self.processing_thread = None
    
    def start_audio_stream(self):
        """Start audio capture thread."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("Audio stream started...")
        while self.running:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                self.audio_buffer.extend(audio_data)
                
                if len(self.audio_buffer) >= self.RATE:  # 1 second of audio
                    audio_chunk = np.array(list(self.audio_buffer)[-self.RATE:])
                    self.audio_queue.put(audio_chunk)
            except Exception as e:
                print(f"Audio error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def start_video_stream(self):
        """Start video capture thread."""
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            print("Error: Could not open video capture")
            return
        
        print("Video stream started...")
        while self.running:
            ret, frame = self.video_capture.read()
            if ret:
                self.video_queue.put(frame)
            else:
                break
        
        self.video_capture.release()
    
    def process_audio(self, audio_data):
        """Process audio data and extract features."""
        try:
            # Convert to tensor
            waveform = torch.tensor(audio_data, dtype=torch.float32)
            
            # Process with Wav2Vec2
            inputs = self.audio_processor(waveform, sampling_rate=self.RATE, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            with torch.no_grad():
                outputs = self.audio_model(input_values, attention_mask=attention_mask)
                features = outputs.last_hidden_state.mean(dim=1)
            
            return features.cpu()
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None
    
    def process_video(self, frame):
        """Process video frame and extract features."""
        try:
            # Extract face
            face = extract_face_frame(frame, self.face_cascade)
            if face is None:
                return None
            
            # Extract ViT features
            features = extract_vit_embedding(face, self.vision_processor, self.vision_model, self.device)
            return features.cpu()
        except Exception as e:
            print(f"Video processing error: {e}")
            return None
    
    def predict_emotion(self, audio_features=None, video_features=None):
        """Predict emotion from features."""
        try:
            with torch.no_grad():
                if self.mode == 'audio_only':
                    if audio_features is None:
                        return None
                    logits, _ = self.model(audio_features.to(self.device), return_projection=True)
                
                elif self.mode == 'video_only':
                    if video_features is None:
                        return None
                    logits, _ = self.model(video_features.to(self.device), return_projection=True)
                
                elif self.mode == 'multimodal':
                    if audio_features is None and video_features is None:
                        return None
                    
                    # Handle missing modalities
                    if audio_features is None:
                        audio_features = torch.zeros(1, 768)
                    if video_features is None:
                        video_features = torch.zeros(1, 768)
                    
                    logits, _ = self.model(audio_features.to(self.device), video_features.to(self.device), return_projection=True)
                
                # Get predictions
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                return {
                    'emotion': self.emotion_labels[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities[0].cpu().numpy()
                }
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def processing_loop(self):
        """Main processing loop."""
        while self.running:
            audio_features = None
            video_features = None
            
            # Get audio features
            if self.mode in ['audio_only', 'multimodal']:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    audio_features = self.process_audio(audio_data)
                except queue.Empty:
                    pass
            
            # Get video features
            if self.mode in ['video_only', 'multimodal']:
                try:
                    frame = self.video_queue.get(timeout=0.1)
                    video_features = self.process_video(frame)
                except queue.Empty:
                    pass
            
            # Make prediction
            if audio_features is not None or video_features is not None:
                prediction = self.predict_emotion(audio_features, video_features)
                if prediction:
                    self.prediction_history.append(prediction)
                    self.result_queue.put(prediction)
            
            time.sleep(0.1)  # 10 FPS
    
    def start(self):
        """Start the real-time recognition system."""
        self.running = True
        
        # Start threads
        if self.mode in ['audio_only', 'multimodal']:
            self.audio_thread = threading.Thread(target=self.start_audio_stream)
            self.audio_thread.start()
        
        if self.mode in ['video_only', 'multimodal']:
            self.video_thread = threading.Thread(target=self.start_video_stream)
            self.video_thread.start()
        
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()
    
    def stop(self):
        """Stop the real-time recognition system."""
        self.running = False
        
        if self.audio_thread:
            self.audio_thread.join()
        if self.video_thread:
            self.video_thread.join()
        if self.processing_thread:
            self.processing_thread.join()
        
        if self.video_capture:
            self.video_capture.release()

class EmotionRecognitionGUI:
    """GUI for real-time emotion recognition."""
    
    def __init__(self, model_path, mode='multimodal'):
        self.recognition_system = RealTimeEmotionRecognition(model_path, mode)
        self.mode = mode
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title(f"Real-time Emotion Recognition - {mode.upper()}")
        self.root.geometry("800x600")
        
        # Create figure for plotting
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Control buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.start_button = tk.Button(self.control_frame, text="Start", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = tk.Button(self.control_frame, text="Stop", command=self.stop_recognition)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.quit_button = tk.Button(self.control_frame, text="Quit", command=self.quit_app)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Status label
        self.status_label = tk.Label(self.control_frame, text="Ready")
        self.status_label.pack(side=tk.TOP, pady=5)
        
        # Current prediction label
        self.prediction_label = tk.Label(self.control_frame, text="No prediction", font=("Arial", 16))
        self.prediction_label.pack(side=tk.TOP, pady=5)
        
        # Initialize plots
        self.init_plots()
        
        # Animation
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=100, blit=False)
        
        self.running = False
    
    def init_plots(self):
        """Initialize the plots."""
        # Emotion history plot
        self.ax1.clear()
        self.ax1.set_title("Emotion Prediction History")
        self.ax1.set_ylabel("Confidence")
        self.ax1.set_ylim(0, 1)
        
        # Probability distribution plot
        self.ax2.clear()
        self.ax2.set_title("Current Emotion Probabilities")
        self.ax2.set_ylabel("Probability")
        self.ax2.set_ylim(0, 1)
        
        self.fig.tight_layout()
    
    def update_plots(self, frame):
        """Update the plots with new data."""
        if not self.running:
            return
        
        # Update emotion history
        if self.recognition_system.prediction_history:
            emotions = [p['emotion'] for p in self.recognition_system.prediction_history]
            confidences = [p['confidence'] for p in self.recognition_system.prediction_history]
            
            self.ax1.clear()
            self.ax1.plot(confidences, 'b-', linewidth=2)
            self.ax1.set_title("Emotion Prediction History")
            self.ax1.set_ylabel("Confidence")
            self.ax1.set_ylim(0, 1)
            self.ax1.grid(True, alpha=0.3)
        
        # Update probability distribution
        try:
            latest_prediction = self.recognition_system.result_queue.get_nowait()
            probabilities = latest_prediction['probabilities']
            emotions = self.recognition_system.emotion_labels
            
            self.ax2.clear()
            bars = self.ax2.bar(emotions, probabilities, color='skyblue', alpha=0.7)
            self.ax2.set_title("Current Emotion Probabilities")
            self.ax2.set_ylabel("Probability")
            self.ax2.set_ylim(0, 1)
            self.ax2.tick_params(axis='x', rotation=45)
            
            # Highlight the predicted emotion
            predicted_idx = self.recognition_system.emotion_labels.index(latest_prediction['emotion'])
            bars[predicted_idx].set_color('red')
            
            # Update prediction label
            self.prediction_label.config(
                text=f"{latest_prediction['emotion'].upper()} ({latest_prediction['confidence']:.2f})"
            )
            
        except queue.Empty:
            pass
        
        self.fig.tight_layout()
    
    def start_recognition(self):
        """Start the recognition system."""
        self.recognition_system.start()
        self.running = True
        self.status_label.config(text="Running...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
    
    def stop_recognition(self):
        """Stop the recognition system."""
        self.recognition_system.stop()
        self.running = False
        self.status_label.config(text="Stopped")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def quit_app(self):
        """Quit the application."""
        if self.running:
            self.stop_recognition()
        self.root.quit()
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition Demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--mode', type=str, default='multimodal', 
                       choices=['audio_only', 'video_only', 'multimodal'],
                       help='Recognition mode')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI (console only)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    if args.no_gui:
        # Console-only mode
        recognition_system = RealTimeEmotionRecognition(args.model_path, args.mode, args.device)
        
        print(f"Starting real-time emotion recognition in {args.mode} mode...")
        print("Press Ctrl+C to stop")
        
        try:
            recognition_system.start()
            
            while True:
                try:
                    prediction = recognition_system.result_queue.get(timeout=1.0)
                    print(f"Predicted emotion: {prediction['emotion']} (confidence: {prediction['confidence']:.2f})")
                except queue.Empty:
                    pass
                
        except KeyboardInterrupt:
            print("\nStopping...")
            recognition_system.stop()
    
    else:
        # GUI mode
        app = EmotionRecognitionGUI(args.model_path, args.mode)
        app.run()

if __name__ == "__main__":
    main() 