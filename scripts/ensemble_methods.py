#!/usr/bin/env python3
"""
Ensemble Methods for Zero-Shot Emotion Recognition
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

class ModelEnsemble:
    """Ensemble of multiple emotion recognition models"""
    
    def __init__(self, models, fusion_strategy='weighted_average'):
        self.models = models
        self.fusion_strategy = fusion_strategy
        self.weights = None
    
    def fit_weights(self, val_loader, device):
        """Learn optimal weights for ensemble"""
        all_predictions = []
        all_labels = []
        
        # Get predictions from all models
        for model in self.models:
            model.eval()
            predictions = []
            labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    audio_feat, vision_feat, batch_labels = batch
                    audio_feat = audio_feat.to(device)
                    vision_feat = vision_feat.to(device)
                    
                    logits, _ = model(audio_feat, vision_feat, return_projection=True)
                    pred = torch.softmax(logits, dim=1)
                    predictions.append(pred.cpu().numpy())
                    labels.append(batch_labels.numpy())
            
            all_predictions.append(np.concatenate(predictions))
            all_labels.append(np.concatenate(labels))
        
        # Learn optimal weights
        self.weights = self._learn_optimal_weights(all_predictions, all_labels[0])
    
    def _learn_optimal_weights(self, predictions_list, true_labels):
        """Learn optimal weights for ensemble"""
        from scipy.optimize import minimize
        
        def objective(weights):
            # Normalize weights
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            
            # Weighted ensemble prediction
            ensemble_pred = np.zeros_like(predictions_list[0])
            for i, pred in enumerate(predictions_list):
                ensemble_pred += weights[i] * pred
            
            # Compute accuracy
            ensemble_labels = np.argmax(ensemble_pred, axis=1)
            accuracy = accuracy_score(true_labels, ensemble_labels)
            return -accuracy  # Minimize negative accuracy
        
        # Initial weights (equal)
        initial_weights = np.ones(len(predictions_list)) / len(predictions_list)
        
        # Optimize weights
        result = minimize(objective, initial_weights, method='L-BFGS-B')
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        return optimal_weights
    
    def predict(self, audio_feat, vision_feat, device):
        """Ensemble prediction"""
        ensemble_logits = torch.zeros(audio_feat.size(0), 8).to(device)
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                logits, _ = model(audio_feat, vision_feat, return_projection=True)
                if self.weights is not None:
                    ensemble_logits += self.weights[i] * logits
                else:
                    ensemble_logits += logits / len(self.models)
        
        return ensemble_logits

class DiverseEnsemble:
    """Ensemble with diverse models for better generalization"""
    
    def __init__(self):
        self.models = []
        self.diversity_weights = []
    
    def add_model(self, model, diversity_score=1.0):
        """Add model with diversity score"""
        self.models.append(model)
        self.diversity_weights.append(diversity_score)
    
    def create_diverse_models(self, base_model_class, num_models=5):
        """Create diverse models with different architectures"""
        for i in range(num_models):
            # Different fusion strategies
            fusion_types = ['attention', 'mlp', 'cross_attention', 'gated', 'hierarchical']
            fusion_type = fusion_types[i % len(fusion_types)]
            
            # Different dropout rates
            dropout_rate = 0.2 + (i * 0.1)
            
            # Different learning rates
            lr = 1e-4 * (0.5 + i * 0.2)
            
            # Create model with different parameters
            model = base_model_class(
                fusion_type=fusion_type,
                dropout_rate=dropout_rate,
                learning_rate=lr
            )
            
            self.add_model(model, diversity_score=1.0 + i * 0.2)
    
    def predict_with_diversity(self, audio_feat, vision_feat, device):
        """Weighted prediction considering model diversity"""
        ensemble_logits = torch.zeros(audio_feat.size(0), 8).to(device)
        total_weight = sum(self.diversity_weights)
        
        for model, weight in zip(self.models, self.diversity_weights):
            model.eval()
            with torch.no_grad():
                logits, _ = model(audio_feat, vision_feat, return_projection=True)
                ensemble_logits += (weight / total_weight) * logits
        
        return ensemble_logits

class TemporalEnsemble:
    """Ensemble considering temporal information"""
    
    def __init__(self, models, temporal_window=5):
        self.models = models
        self.temporal_window = temporal_window
        self.predictions_history = []
    
    def predict_with_temporal_smoothing(self, audio_feat, vision_feat, device):
        """Prediction with temporal smoothing"""
        current_prediction = self._get_ensemble_prediction(audio_feat, vision_feat, device)
        
        # Add to history
        self.predictions_history.append(current_prediction)
        
        # Keep only recent predictions
        if len(self.predictions_history) > self.temporal_window:
            self.predictions_history.pop(0)
        
        # Temporal smoothing
        if len(self.predictions_history) > 1:
            smoothed_prediction = torch.stack(self.predictions_history).mean(dim=0)
        else:
            smoothed_prediction = current_prediction
        
        return smoothed_prediction
    
    def _get_ensemble_prediction(self, audio_feat, vision_feat, device):
        """Get ensemble prediction"""
        ensemble_logits = torch.zeros(audio_feat.size(0), 8).to(device)
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits, _ = model(audio_feat, vision_feat, return_projection=True)
                ensemble_logits += logits / len(self.models)
        
        return torch.softmax(ensemble_logits, dim=1)

class ConfidenceWeightedEnsemble:
    """Ensemble weighted by prediction confidence"""
    
    def __init__(self, models):
        self.models = models
    
    def predict_with_confidence_weighting(self, audio_feat, vision_feat, device):
        """Weight predictions by confidence"""
        all_predictions = []
        all_confidences = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits, _ = model(audio_feat, vision_feat, return_projection=True)
                probabilities = torch.softmax(logits, dim=1)
                
                # Compute confidence (max probability)
                confidence = torch.max(probabilities, dim=1)[0]
                
                all_predictions.append(probabilities)
                all_confidences.append(confidence)
        
        # Weight by confidence
        weighted_prediction = torch.zeros_like(all_predictions[0])
        total_confidence = torch.zeros(audio_feat.size(0)).to(device)
        
        for pred, conf in zip(all_predictions, all_confidences):
            weighted_prediction += pred * conf.unsqueeze(1)
            total_confidence += conf
        
        # Normalize
        weighted_prediction = weighted_prediction / total_confidence.unsqueeze(1)
        
        return weighted_prediction

# Usage example
def create_ensemble():
    # Create diverse models
    ensemble = DiverseEnsemble()
    ensemble.create_diverse_models(ImprovedMultimodalZSERModel, num_models=5)
    
    # Train each model
    for model in ensemble.models:
        train_model(model, train_loader, val_loader)
    
    return ensemble

# Temporal ensemble for real-time applications
temporal_ensemble = TemporalEnsemble(models, temporal_window=3)

# Confidence-weighted ensemble
confidence_ensemble = ConfidenceWeightedEnsemble(models) 