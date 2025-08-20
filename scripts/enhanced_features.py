#!/usr/bin/env python3
"""
Enhanced Feature Extraction for Better Zero-Shot Performance
"""

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, ViTModel, ViTImageProcessor
import cv2
import numpy as np

class EnhancedFeatureExtractor:
    def __init__(self):
        # Use larger, better pre-trained models
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
        self.vision_processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
        self.vision_model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        
    def extract_enhanced_audio_features(self, audio_path):
        """Enhanced audio feature extraction with multiple pooling strategies"""
        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        # Process with Wav2Vec2-Large
        inputs = self.audio_processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            
            # Multiple pooling strategies
            mean_pooled = outputs.last_hidden_state.mean(dim=1)  # [1, 1024]
            max_pooled = outputs.last_hidden_state.max(dim=1)[0]  # [1, 1024]
            attention_pooled = self.attention_pooling(outputs.last_hidden_state)  # [1, 1024]
            
            # Concatenate different pooling strategies
            enhanced_features = torch.cat([mean_pooled, max_pooled, attention_pooled], dim=-1)  # [1, 3072]
            
        return enhanced_features
    
    def extract_enhanced_vision_features(self, face_img):
        """Enhanced vision feature extraction with multi-layer features"""
        # Preprocess image
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Process with ViT-Large
        inputs = self.vision_processor(images=face_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.vision_model(**inputs, output_hidden_states=True)
            
            # Extract features from multiple layers
            layer_features = []
            for i in [6, 12, 18, 24]:  # Different layers
                layer_feat = outputs.hidden_states[i][:, 0, :]  # CLS token
                layer_features.append(layer_feat)
            
            # Concatenate multi-layer features
            enhanced_features = torch.cat(layer_features, dim=-1)  # [1, 4096]
            
        return enhanced_features
    
    def attention_pooling(self, hidden_states):
        """Attention-based pooling for better feature aggregation"""
        # Simple attention mechanism
        attention_weights = torch.softmax(torch.mean(hidden_states, dim=-1), dim=-1)
        attended_features = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return attended_features

# Usage
extractor = EnhancedFeatureExtractor()
enhanced_audio = extractor.extract_enhanced_audio_features("sample.wav")
enhanced_vision = extractor.extract_enhanced_vision_features(face_img) 