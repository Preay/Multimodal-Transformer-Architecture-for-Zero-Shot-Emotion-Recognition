"""
Audio-only Zero-Shot Emotion Recognition Model

DONE Features:
  - Works for static [B, D] audio embeddings
  - Optionally supports temporal [B, T, D] with BiLSTM + Attention pooling
  - Lightweight projection → 128-d contrastive space
  - Classification head for seen-class supervised training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================
# Optional Temporal Encoding
# ===========================
class TemporalEncoder(nn.Module):
    """BiLSTM for temporal contextualization of audio embeddings."""
    def __init__(self, input_dim=256, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, input_dim)  # back to original dim

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.lstm(x)
        out = self.proj(out)  # [B, T, D]
        return out


class TemporalAttention(nn.Module):
    """Temporal attention pooling → selects key time frames."""
    def __init__(self, dim=256):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=-1)  # [B, T]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)       # [B, D]
        return pooled, weights


# ===========================
# Audio Z-SER Model
# ===========================
class AudioZSERModel(nn.Module):
    """
    Audio-only Zero-Shot Emotion Recognition model.
    
    Args:
        input_dim:   Dimension of extracted audio features (default: 768 from Wav2Vec2)
        proj_dim:    Projection dimension for contrastive head
        num_classes: Number of supervised emotion classes
        use_temporal: If True, applies BiLSTM + attention before classification
    
    Input:
        x: [B, D]  (static)
           OR [B, T, D] (temporal sequence)
    """
    def __init__(self,
                 input_dim=768,
                 proj_dim=128,
                 num_classes=8,
                 use_temporal=False):
        super().__init__()
        self.use_temporal = use_temporal

        # Encode raw audio embeddings → 256-d
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        # Optional temporal modeling
        if use_temporal:
            self.temporal = TemporalEncoder(input_dim=256)
            self.attention = TemporalAttention(dim=256)

        # Projection head → contrastive space
        self.proj_head = nn.Sequential(
            nn.Linear(256, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )

        # Classification head for supervised emotions
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, x, return_projection=False):
        """
        Forward pass:
            x: [B, D] (static) OR [B, T, D] (temporal)
            
        Returns:
            logits: [B, num_classes]
            z:      [B, proj_dim] (if return_projection=True)
        """

        # === Temporal case ===
        if self.use_temporal:
            assert x.dim() == 3, "When use_temporal=True, expected [B, T, D] input"

            # Frame-level encode → [B, T, 256]
            seq = self.encoder(x)

            # Contextualize temporal dependencies → [B, T, 256]
            seq_ctx = self.temporal(seq)

            # Temporal attention pooling → [B, 256]
            pooled, _ = self.attention(seq_ctx)

            # Projection + classification
            z = self.proj_head(pooled)
            logits = self.classifier(z)

        # === Static case ===
        else:
            assert x.dim() == 2, "When use_temporal=False, expected [B, D] static input"

            # Encode → project → classify
            enc = self.encoder(x)         # [B, 256]
            z = self.proj_head(enc)       # [B, proj_dim]
            logits = self.classifier(z)   # [B, num_classes]

        if return_projection:
            return logits, z
        return logits
