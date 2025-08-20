#!/usr/bin/env python3
"""
Baseline Models for Zero-Shot Emotion Recognition Comparison

This script implements various baseline models to compare against the main
multimodal approach. Includes:
- Audio-only baselines
- Vision-only baselines  
- Simple fusion baselines
- Random baseline
- Majority class baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import build_dataset_df, EmotionAudioDataset, EmotionMultimodalDataset
from models.multimodal_zser_model import MultimodalZSERModel


# ================================
# BASELINE MODEL ARCHITECTURES
# ================================

class AudioOnlyBaseline(nn.Module):
    """Simple audio-only baseline using Wav2Vec2 features"""
    def __init__(self, input_dim=768, num_classes=8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, audio_features):
        return self.classifier(audio_features)


class VisionOnlyBaseline(nn.Module):
    """Simple vision-only baseline using ViT features"""
    def __init__(self, input_dim=768, num_classes=8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, vision_features):
        return self.classifier(vision_features)


class EarlyFusionBaseline(nn.Module):
    """Early fusion baseline: concatenate raw features"""
    def __init__(self, audio_dim=768, vision_dim=768, num_classes=8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(audio_dim + vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, audio_features, vision_features):
        combined = torch.cat([audio_features, vision_features], dim=-1)
        return self.classifier(combined)


class LateFusionBaseline(nn.Module):
    """Late fusion baseline: separate classifiers + average"""
    def __init__(self, audio_dim=768, vision_dim=768, num_classes=8):
        super().__init__()
        self.audio_classifier = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.vision_classifier = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, audio_features, vision_features):
        audio_logits = self.audio_classifier(audio_features)
        vision_logits = self.vision_classifier(vision_features)
        return (audio_logits + vision_logits) / 2.0


# ================================
# RULE-BASED BASELINES
# ================================

class RandomBaseline:
    """Random prediction baseline"""
    def __init__(self, num_classes=8):
        self.num_classes = num_classes
    
    def predict(self, X):
        return np.random.randint(0, self.num_classes, size=len(X))


class MajorityClassBaseline:
    """Majority class baseline"""
    def __init__(self):
        self.majority_class = None
    
    def fit(self, y):
        # Find most common class
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]
        return self
    
    def predict(self, X):
        return np.full(len(X), self.majority_class)


# ================================
# BASELINE EVALUATION
# ================================

def evaluate_baseline_model(model, test_loader, device, model_type="neural"):
    """Evaluate a baseline model"""
    if model_type == "neural":
        model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_type == "neural":
                if isinstance(model, (AudioOnlyBaseline, VisionOnlyBaseline)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    logits = model(features)
                elif isinstance(model, (EarlyFusionBaseline, LateFusionBaseline)):
                    # Handle multimodal dataset format: ((audio, vision), label)
                    features, labels = batch
                    audio_feat, vision_feat = features
                    audio_feat = audio_feat.to(device)
                    vision_feat = vision_feat.to(device)
                    labels = labels.to(device)
                    logits = model(audio_feat, vision_feat)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                # For rule-based baselines
                features, labels = batch
                preds = model.predict(features.cpu().numpy())
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'accuracy': accuracy,
        'macro_f1': f1,
        'predictions': all_preds,
        'true_labels': all_labels
    }


def run_all_baselines(dataset_name, dataset_path, features_dir, results_dir):
    """Run all baseline models and save results"""
    import json
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    df = build_dataset_df(dataset_name, dataset_path)
    
    # Split data (70% train, 30% test for zero-shot)
    test_split = int(0.3 * len(df))
    train_df = df.iloc[:-test_split]
    test_df = df.iloc[-test_split:]
    
    # Create datasets
    train_audio = EmotionAudioDataset(train_df, feature_dir=os.path.join(features_dir, "audio"))
    test_audio = EmotionAudioDataset(test_df, feature_dir=os.path.join(features_dir, "audio"))
    
    train_multimodal = EmotionMultimodalDataset(
        train_df,
        feature_dir_audio=os.path.join(features_dir, "audio"),
        feature_dir_vision=os.path.join(features_dir, "vision")
    )
    test_multimodal = EmotionMultimodalDataset(
        test_df,
        feature_dir_audio=os.path.join(features_dir, "audio"),
        feature_dir_vision=os.path.join(features_dir, "vision")
    )
    
    # Create data loaders
    train_loader = DataLoader(train_audio, batch_size=32, shuffle=True)
    train_loader_multimodal = DataLoader(train_multimodal, batch_size=32, shuffle=True)
    test_loader_audio = DataLoader(test_audio, batch_size=32, shuffle=False)
    test_loader_multimodal = DataLoader(test_multimodal, batch_size=32, shuffle=False)
    
    results = {}
    
    # 1. Audio-only baseline
    print("Training Audio-only baseline...")
    audio_model = AudioOnlyBaseline().to(device)
    # Train the model (simplified training)
    optimizer = torch.optim.Adam(audio_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):  # Quick training
        audio_model.train()
        for batch in train_loader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = audio_model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    audio_results = evaluate_baseline_model(audio_model, test_loader_audio, device)
    results['audio_only'] = audio_results
    print(f"Audio-only: Acc={audio_results['accuracy']:.4f}, F1={audio_results['macro_f1']:.4f}")
    
    # 2. Vision-only baseline (if vision features available)
    if os.path.exists(os.path.join(features_dir, "vision")):
        print("Training Vision-only baseline...")
        vision_model = VisionOnlyBaseline().to(device)
        # Train vision model (simplified)
        optimizer = torch.optim.Adam(vision_model.parameters(), lr=1e-4)
        
        for epoch in range(10):
            vision_model.train()
            for batch in train_loader:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                logits = vision_model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        
        vision_results = evaluate_baseline_model(vision_model, test_loader_audio, device)
        results['vision_only'] = vision_results
        print(f"Vision-only: Acc={vision_results['accuracy']:.4f}, F1={vision_results['macro_f1']:.4f}")
    
    # 3. Early fusion baseline
    print("Training Early fusion baseline...")
    early_model = EarlyFusionBaseline().to(device)
    optimizer = torch.optim.Adam(early_model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        early_model.train()
        for batch in train_loader_multimodal:
            # Handle multimodal dataset format: ((audio, vision), label)
            features, labels = batch
            audio_feat, vision_feat = features
            audio_feat = audio_feat.to(device)
            vision_feat = vision_feat.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = early_model(audio_feat, vision_feat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    early_results = evaluate_baseline_model(early_model, test_loader_multimodal, device)
    results['early_fusion'] = early_results
    print(f"Early fusion: Acc={early_results['accuracy']:.4f}, F1={early_results['macro_f1']:.4f}")
    
    # 4. Late fusion baseline
    print("Training Late fusion baseline...")
    late_model = LateFusionBaseline().to(device)
    optimizer = torch.optim.Adam(late_model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        late_model.train()
        for batch in train_loader_multimodal:
            # Handle multimodal dataset format: ((audio, vision), label)
            features, labels = batch
            audio_feat, vision_feat = features
            audio_feat = audio_feat.to(device)
            vision_feat = vision_feat.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = late_model(audio_feat, vision_feat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    late_results = evaluate_baseline_model(late_model, test_loader_multimodal, device)
    results['late_fusion'] = late_results
    print(f"Late fusion: Acc={late_results['accuracy']:.4f}, F1={late_results['macro_f1']:.4f}")
    
    # 5. Rule-based baselines
    print("Evaluating rule-based baselines...")
    
    # Random baseline
    random_baseline = RandomBaseline()
    random_results = evaluate_baseline_model(random_baseline, test_loader_audio, device, "rule")
    results['random'] = random_results
    print(f"Random: Acc={random_results['accuracy']:.4f}, F1={random_results['macro_f1']:.4f}")
    
    # Majority class baseline
    majority_baseline = MajorityClassBaseline()
    majority_baseline.fit(train_df['label'].values)
    majority_results = evaluate_baseline_model(majority_baseline, test_loader_audio, device, "rule")
    results['majority_class'] = majority_results
    print(f"Majority class: Acc={majority_results['accuracy']:.4f}, F1={majority_results['macro_f1']:.4f}")
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {
            'accuracy': float(value['accuracy']),
            'macro_f1': float(value['macro_f1']),
            'predictions': [int(x) for x in value['predictions']],
            'true_labels': [int(x) for x in value['true_labels']]
        }
    
    with open(os.path.join(results_dir, f"baseline_results_{dataset_name}.json"), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE RESULTS SUMMARY")
    print("="*50)
    for model_name, result in results.items():
        print(f"{model_name:15s}: Acc={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline models for comparison")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results/baselines")
    
    args = parser.parse_args()
    run_all_baselines(args.dataset_name, args.dataset_path, args.features_dir, args.results_dir) 