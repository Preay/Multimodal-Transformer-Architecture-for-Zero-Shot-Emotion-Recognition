#!/usr/bin/env python3
"""
Advanced Data Augmentation for Emotion Recognition
"""

import torch
import torchaudio
import cv2
import numpy as np
from torchvision import transforms
import random

class EmotionDataAugmentation:
    def __init__(self):
        self.audio_transforms = self._get_audio_transforms()
        self.vision_transforms = self._get_vision_transforms()
    
    def _get_audio_transforms(self):
        """Audio augmentation transforms"""
        return {
            'pitch_shift': lambda x: torchaudio.functional.pitch_shift(x, 16000, random.randint(-2, 2)),
            'time_stretch': lambda x: torchaudio.functional.speed(x, random.uniform(0.8, 1.2)),
            'add_noise': lambda x: x + torch.randn_like(x) * 0.01,
            'frequency_mask': lambda x: self._frequency_mask(x),
            'time_mask': lambda x: self._time_mask(x)
        }
    
    def _get_vision_transforms(self):
        """Vision augmentation transforms"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])
    
    def augment_audio(self, audio_features, augmentation_prob=0.5):
        """Apply audio augmentation to features"""
        if random.random() < augmentation_prob:
            # Apply random audio augmentation
            aug_type = random.choice(list(self.audio_transforms.keys()))
            audio_features = self.audio_transforms[aug_type](audio_features)
        
        return audio_features
    
    def augment_vision(self, face_img, augmentation_prob=0.5):
        """Apply vision augmentation to face image"""
        if random.random() < augmentation_prob:
            # Convert to PIL Image for transforms
            from PIL import Image
            pil_img = Image.fromarray(face_img)
            face_img = self.vision_transforms(pil_img)
            face_img = np.array(face_img)
        
        return face_img
    
    def _frequency_mask(self, audio, freq_mask_param=10):
        """Frequency masking for audio"""
        freq_mask = torch.ones(audio.shape[0], audio.shape[1])
        freq_mask[:, :freq_mask_param] = 0
        return audio * freq_mask
    
    def _time_mask(self, audio, time_mask_param=10):
        """Time masking for audio"""
        time_mask = torch.ones(audio.shape[0], audio.shape[1])
        time_mask[:time_mask_param, :] = 0
        return audio * time_mask

class BalancedDataAugmentation:
    """Augmentation focused on improving weak emotion classes"""
    
    def __init__(self):
        self.augmenter = EmotionDataAugmentation()
    
    def augment_weak_emotions(self, dataset, target_counts):
        """Augment weak emotion classes to balance dataset"""
        augmented_data = []
        
        for emotion_class, target_count in target_counts.items():
            current_count = len([x for x in dataset if x[1] == emotion_class])
            
            if current_count < target_count:
                # Find samples of this emotion
                emotion_samples = [x for x in dataset if x[1] == emotion_class]
                
                # Augment to reach target count
                needed_samples = target_count - current_count
                for _ in range(needed_samples):
                    sample = random.choice(emotion_samples)
                    augmented_sample = self._augment_sample(sample)
                    augmented_data.append(augmented_sample)
        
        return dataset + augmented_data
    
    def _augment_sample(self, sample):
        """Apply strong augmentation to a sample"""
        audio_feat, vision_feat, label = sample
        
        # Apply multiple augmentations
        audio_feat = self.augmenter.augment_audio(audio_feat, augmentation_prob=0.8)
        vision_feat = self.augmenter.augment_vision(vision_feat, augmentation_prob=0.8)
        
        return (audio_feat, vision_feat, label)

# Usage example
augmenter = EmotionDataAugmentation()
balanced_augmenter = BalancedDataAugmentation()

# Target counts for weak emotions
target_counts = {
    0: 100,  # neutral
    2: 100,  # happy  
    3: 100,  # sad
    5: 100,  # fearful
    7: 100   # surprised
}

# Augment weak emotion classes
augmented_dataset = balanced_augmenter.augment_weak_emotions(dataset, target_counts) 