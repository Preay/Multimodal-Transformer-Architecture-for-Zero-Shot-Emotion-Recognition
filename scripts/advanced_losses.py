#!/usr/bin/env python3
"""
Advanced Loss Functions for Zero-Shot Emotion Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CenterLoss(nn.Module):
    """Center loss for compact class representations"""
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    
    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        loss = F.mse_loss(features, centers_batch)
        return loss

class ArcFaceLoss(nn.Module):
    """ArcFace loss for better feature discrimination"""
    def __init__(self, num_classes, embedding_dim, margin=0.5, scale=64):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
    
    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos_theta = F.linear(embeddings, weight)
        
        # Apply margin
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * torch.cos(self.margin) - sin_theta * torch.sin(self.margin)
        
        # Apply margin only to positive class
        cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        
        # Scale
        output = cos_theta_m * self.scale
        
        return F.cross_entropy(output, labels)

class PrototypicalLoss(nn.Module):
    """Prototypical loss for zero-shot learning"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels, prototypes):
        # Normalize embeddings and prototypes
        embeddings = F.normalize(embeddings, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Compute distances to prototypes
        distances = torch.cdist(embeddings, prototypes)
        
        # Convert to similarities
        similarities = -distances / self.temperature
        
        return F.cross_entropy(similarities, labels)

class ContrastiveEmotionLoss(nn.Module):
    """Contrastive loss specifically designed for emotions"""
    def __init__(self, temperature=0.1, margin=1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive and negative masks
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Positive pairs (same emotion)
        positive_mask = labels_matrix.float()
        positive_mask.fill_diagonal_(0)  # Remove self-similarity
        
        # Negative pairs (different emotions)
        negative_mask = (~labels_matrix).float()
        
        # Compute positive and negative similarities
        positive_sim = (similarity_matrix * positive_mask).sum(dim=1)
        negative_sim = similarity_matrix * negative_mask
        
        # Contrastive loss
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        targets = torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)
        
        return F.cross_entropy(logits, targets)

class EmotionSpecificLoss(nn.Module):
    """Loss function that gives different weights to different emotions"""
    def __init__(self, emotion_weights=None):
        super().__init__()
        if emotion_weights is None:
            # Weight weak emotions more heavily
            self.emotion_weights = torch.tensor([
                2.0,  # neutral (weak)
                1.0,  # calm (strong)
                2.5,  # happy (weak)
                3.0,  # sad (very weak)
                1.0,  # angry (strong)
                1.5,  # fearful (moderate)
                1.0,  # disgust (strong)
                1.5   # surprised (moderate)
            ])
        else:
            self.emotion_weights = emotion_weights
    
    def forward(self, logits, labels):
        # Apply emotion-specific weights
        weights = self.emotion_weights[labels]
        loss = F.cross_entropy(logits, labels, reduction='none')
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class CombinedAdvancedLoss(nn.Module):
    """Combination of multiple advanced loss functions"""
    def __init__(self, num_classes, embedding_dim, device):
        super().__init__()
        self.center_loss = CenterLoss(num_classes, embedding_dim, device)
        self.arcface_loss = ArcFaceLoss(num_classes, embedding_dim)
        self.contrastive_loss = ContrastiveEmotionLoss()
        self.emotion_specific_loss = EmotionSpecificLoss()
        
        # Loss weights
        self.center_weight = 0.1
        self.arcface_weight = 0.3
        self.contrastive_weight = 0.2
        self.emotion_weight = 0.4
    
    def forward(self, logits, embeddings, labels, prototypes=None):
        # Compute different losses
        center_loss = self.center_loss(embeddings, labels)
        arcface_loss = self.arcface_loss(embeddings, labels)
        contrastive_loss = self.contrastive_loss(embeddings, labels)
        emotion_loss = self.emotion_specific_loss(logits, labels)
        
        # Combine losses
        total_loss = (
            self.center_weight * center_loss +
            self.arcface_weight * arcface_loss +
            self.contrastive_weight * contrastive_loss +
            self.emotion_weight * emotion_loss
        )
        
        return total_loss, {
            'center_loss': center_loss.item(),
            'arcface_loss': arcface_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'emotion_loss': emotion_loss.item()
        }

# Usage example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
advanced_loss = CombinedAdvancedLoss(num_classes=8, embedding_dim=256, device=device)

# In training loop
loss, loss_components = advanced_loss(logits, embeddings, labels)
loss.backward() 