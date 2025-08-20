#!/usr/bin/env python3
"""
Simple but Effective Model Improvement Script

This script implements key improvements to boost zero-shot emotion recognition:

1. Focal Loss for class imbalance
2. Label Smoothing for generalization
3. Learning Rate Scheduling
4. Data Augmentation
5. Better Regularization

Usage:
    python scripts/simple_improvement.py --dataset_name ravdess --epochs 50
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import build_dataset_df, EmotionMultimodalDataset
from models.multimodal_zser_model import MultimodalZSERModel


# ===============================
# IMPROVED LOSS FUNCTIONS
# ===============================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing for better generalization"""
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class CombinedLoss(nn.Module):
    """Combined loss with focal and label smoothing"""
    def __init__(self, num_classes=8, focal_weight=0.7, smoothing=0.1):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.label_smoothing = LabelSmoothingLoss(num_classes, smoothing)
        self.focal_weight = focal_weight

    def forward(self, logits, labels):
        focal_loss = self.focal_loss(logits, labels)
        smooth_loss = self.label_smoothing(logits, labels)
        
        total_loss = self.focal_weight * focal_loss + (1 - self.focal_weight) * smooth_loss
        return total_loss


# ===============================
# SIMPLE DATA AUGMENTATION
# ===============================
def augment_features(features, noise_factor=0.05, dropout_rate=0.1):
    """Simple feature augmentation"""
    # Add noise
    noise = torch.randn_like(features) * noise_factor
    features = features + noise
    
    # Feature dropout
    mask = torch.rand_like(features) > dropout_rate
    features = features * mask
    
    return features


# ===============================
# TRAINING FUNCTIONS
# ===============================
def train_epoch_simple(model, train_loader, optimizer, criterion, device, augmentation=True):
    """Simple training with improvements"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, labels) in enumerate(pbar):
        # Handle data structure from EmotionMultimodalDataset
        if isinstance(data, tuple):
            audio_feats, vision_feats = data
            audio_feats = audio_feats.to(device)
            vision_feats = vision_feats.to(device)
        elif isinstance(data, list):
            # Handle list format
            audio_feats = data[0].to(device)
            vision_feats = data[1].to(device) if len(data) > 1 else None
        else:
            audio_feats = data.to(device)
            vision_feats = None
            
        labels = labels.to(device)
        
        # Simple augmentation
        if augmentation and np.random.random() < 0.3:
            audio_feats = augment_features(audio_feats)
            if vision_feats is not None:
                vision_feats = augment_features(vision_feats)
        
        optimizer.zero_grad()
        
        if vision_feats is not None:
            logits = model(audio_feats, vision_feats)
        else:
            logits = model(audio_feats)
        
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate_epoch_simple(model, val_loader, criterion, device):
    """Simple validation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Validation"):
            # Handle data structure from EmotionMultimodalDataset
            if isinstance(data, tuple):
                audio_feats, vision_feats = data
                audio_feats = audio_feats.to(device)
                vision_feats = vision_feats.to(device)
            elif isinstance(data, list):
                # Handle list format
                audio_feats = data[0].to(device)
                vision_feats = data[1].to(device) if len(data) > 1 else None
            else:
                audio_feats = data.to(device)
                vision_feats = None
                
            labels = labels.to(device)
            
            if vision_feats is not None:
                logits = model(audio_feats, vision_feats)
            else:
                logits = model(audio_feats)
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    f1 = f1_score(all_labels, all_predictions, average='macro') * 100
    
    return total_loss / len(val_loader), accuracy, f1


# ===============================
# MAIN TRAINING FUNCTION
# ===============================
def train_improved_model(args):
    """Main training function with improvements"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    df = build_dataset_df(args.dataset_name, args.dataset_path)
    print(f"Loaded {len(df)} samples")
    
    # Split data with stratification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(df, df['label']))
    
    # Use first fold for now
    train_idx, val_idx = splits[0]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    # Create datasets
    train_dataset = EmotionMultimodalDataset(
        train_df, 
        feature_dir_audio=f"{args.features_dir}/audio",
        feature_dir_vision=f"{args.features_dir}/vision",
        temporal_mode=args.use_temporal
    )
    
    val_dataset = EmotionMultimodalDataset(
        val_df,
        feature_dir_audio=f"{args.features_dir}/audio", 
        feature_dir_vision=f"{args.features_dir}/vision",
        temporal_mode=args.use_temporal
    )
    
    # Create weighted sampler for class balance
    class_counts = train_df['label'].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values
    sample_weights = [class_weights[label] for label in train_df['label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = MultimodalZSERModel(
        input_dim_audio=768,
        input_dim_vision=768,
        proj_dim=256,
        num_classes=8,
        fusion_type=args.fusion_type,
        use_temporal=args.use_temporal
    ).to(device)
    
    # Improved loss function
    criterion = CombinedLoss(num_classes=8, focal_weight=0.7)
    
    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.audio_encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': model.vision_encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': model.fusion.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Training loop
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch_simple(
            model, train_loader, optimizer, criterion, device, augmentation=True
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate_epoch_simple(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'val_f1': val_f1
            }, f"{args.save_dir}/best_improved_model.pth")
            print(f"DONE New best model saved! Acc: {best_acc:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= 15:
            print("Early stopping triggered!")
            break
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Simple Model Improvement")
    parser.add_argument("--dataset_name", type=str, default="ravdess", required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/improved")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fusion_type", type=str, default="cross_attention")
    parser.add_argument("--use_temporal", action="store_true")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train improved model
    best_acc = train_improved_model(args)
    
    print(f"\n🎯 Final Best Accuracy: {best_acc:.2f}%")
    
    # Save results
    results = {
        "best_accuracy": best_acc,
        "improvements": [
            "Focal Loss for class imbalance",
            "Label Smoothing for generalization", 
            "Simple data augmentation",
            "Weighted sampling for class balance",
            "Learning rate scheduling",
            "Gradient clipping"
        ]
    }
    
    with open(f"{args.save_dir}/improvement_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("DONE Training completed with improvements!")


if __name__ == "__main__":
    main() 