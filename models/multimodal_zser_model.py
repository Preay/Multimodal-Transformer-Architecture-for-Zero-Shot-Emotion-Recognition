"""
Spatiotemporal Multimodal Transformer-based Zero-Shot Emotion Recognition Model.

DONE Features:
  - Audio + Vision fusion
  - Multiple fusion strategies:
        • 'mlp' → simple concatenation + MLP projection
        • 'cross_attention' → bi-directional attention between audio & vision
        • 'early' → fuse raw before encoding
        • 'late' → average logits from separate classifiers
  - Temporal encoding with BiLSTM + Attention pooling
  - Works for both temporal [B, T, D] and static [B, D] embeddings

Used in both training + zero-shot evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================
# Temporal Encoding Modules
# ===========================
class TemporalEncoder(nn.Module):
    """
    BiLSTM-based temporal encoder → contextualize sequential features.
    
    Input:  [B, T, D]
    Output: [B, T, D] (contextualized)
    """
    def __init__(self, input_dim=256, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, input_dim)  # project back to original dim

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.lstm(x)  # [B, T, 2*H]
        out = self.proj(out)   # back to D
        return out


class TemporalAttention(nn.Module):
    """
    Temporal attention pooling → learn which time steps matter.
    
    Input:  [B, T, D]
    Output: 
        pooled: [B, D]
        weights: [B, T]
    """
    def __init__(self, dim=256):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=-1)  # [B, T]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)       # [B, D]
        return pooled, weights


# ===========================
# Cross-Modal Fusion
# ===========================
class CrossModalGating(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(2 * dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_a: torch.Tensor, z_v: torch.Tensor):
        # z_a, z_v: [B, D]
        g = self.sigmoid(self.fc(torch.cat([z_a, z_v], dim=-1)))  # [B, 2]
        g_a, g_v = g[:, :1], g[:, 1:]                              # [B,1], [B,1]
        z_fused = g_a * z_a + g_v * z_v
        return z_fused, g_a, g_v

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention:
      - Audio attends to Vision
      - Vision attends to Audio
    """
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, audio, vision):
        """
        audio, vision: [B, D] → we treat as single "token"
        """
        audio = audio.unsqueeze(1)   # [B, 1, D]
        vision = vision.unsqueeze(1)

        # Audio attends to vision
        audio_attn, _ = self.attn(audio, vision, vision)

        # Vision attends to audio
        vision_attn, _ = self.attn(vision, audio, audio)

        # Concatenate attended features
        fused = torch.cat([audio_attn.squeeze(1), vision_attn.squeeze(1)], dim=-1)  # [B, 2D]
        return fused


# ===========================
# Main Multimodal Model
# ===========================
class MultimodalZSERModel(nn.Module):
    """
    Multimodal Zero-Shot Emotion Recognition Model
    
    Args:
        input_dim_audio  → dimension of audio embeddings
        input_dim_vision → dimension of visual embeddings
        proj_dim         → projection dimension for contrastive head
        num_classes      → 8 emotions
        fusion_type      → 'mlp', 'cross_attention', 'early', 'late'
        use_temporal     → enable temporal BiLSTM + attention pooling

    Input:
        audio:  [B, D] (static) OR [B, T_a, D] (temporal)
        vision: [B, D] OR [B, T_v, D]
    """
    def __init__(self,
                 input_dim_audio=768,
                 input_dim_vision=768,
                 proj_dim=128,
                 num_classes=8,
                 fusion_type='mlp',
                 use_temporal=True,
                 use_gating=True):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_temporal = use_temporal
        self.use_gating = use_gating

        # Base encoders → project raw embeddings → 256-d
        self.audio_encoder = nn.Sequential(
            nn.Linear(input_dim_audio, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        self.vision_encoder = nn.Sequential(
            nn.Linear(input_dim_vision, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        # ===== Temporal modules =====
        if use_temporal:
            self.audio_temporal = TemporalEncoder(input_dim=256)
            self.vision_temporal = TemporalEncoder(input_dim=256)
            self.audio_attention = TemporalAttention(dim=256)
            self.vision_attention = TemporalAttention(dim=256)

        # ===== Fusion Layers =====
        if fusion_type in ['mlp', 'cross_attention']:
            self.fusion = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, proj_dim),
                nn.ReLU(),
                nn.LayerNorm(proj_dim)
            )
            self.classifier = nn.Linear(proj_dim, num_classes)

            # Projections for per-modality gating head
            self.audio_proj = nn.Sequential(
                nn.Linear(256, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim)
            )
            self.vision_proj = nn.Sequential(
                nn.Linear(256, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim)
            )
            if self.use_gating:
                self.gate = CrossModalGating(dim=proj_dim)

        elif fusion_type == 'early':
            self.fusion = nn.Sequential(
                nn.Linear(input_dim_audio + input_dim_vision, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, proj_dim),
                nn.ReLU(),
                nn.LayerNorm(proj_dim)
            )
            self.classifier = nn.Linear(proj_dim, num_classes)

        elif fusion_type == 'late':
            self.audio_classifier = nn.Linear(256, num_classes)
            self.vision_classifier = nn.Linear(256, num_classes)

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # ===== Cross-modal attention (only if needed) =====
        if fusion_type == 'cross_attention':
            self.cross_attn = CrossModalAttentionFusion(dim=256, heads=4)

    # ===========================
    # Forward Pass
    # ===========================
    def forward(self, audio, vision=None, return_projection=False):
        """
        Forward pass for multimodal fusion.
        
        Args:
            audio: [B, D] or [B, T, D] audio features
            vision: [B, D] or [B, T, D] vision features (optional for audio-only)
            return_projection: whether to return projection embeddings
            
        Returns:
          logits: [B, num_classes]
          z:      [B, proj_dim] (if return_projection=True)
        """
        
        # Handle audio-only mode
        if vision is None:
            # For audio-only, use audio as both inputs
            vision = audio

        # ===== Temporal Encoding Mode =====
        if self.use_temporal:
            # Expect sequences [B, T, D_raw]
            assert audio.dim() == 3 and vision.dim() == 3, \
                "When use_temporal=True, expected temporal inputs [B, T, D]"

            # Encode frame-level embeddings
            audio_seq = self.audio_encoder(audio)    # [B, T_a, 256]
            vision_seq = self.vision_encoder(vision) # [B, T_v, 256]

            # Contextualize temporal dynamics with BiLSTM
            audio_ctx = self.audio_temporal(audio_seq)     # [B, T_a, 256]
            vision_ctx = self.vision_temporal(vision_seq) # [B, T_v, 256]

            # Attention pooling → single vector [B, 256]
            a, _ = self.audio_attention(audio_ctx)
            v, _ = self.vision_attention(vision_ctx)

        # ===== Static Mode =====
        else:
            # Expect single embeddings [B, D_raw]
            assert audio.dim() == 2 and vision.dim() == 2, \
                "When use_temporal=False, expected static inputs [B, D]"
            a = self.audio_encoder(audio)
            v = self.vision_encoder(vision)

        # ===== Fusion Strategies =====
        if self.fusion_type == 'mlp':
            fused = torch.cat([a, v], dim=-1)
            z_base = self.fusion(fused)
            logits = self.classifier(z_base)

            # Optional gating head for alternative projection used by contrastive/analysis
            if self.use_gating:
                z_a = self.audio_proj(a)
                z_v = self.vision_proj(v)
                z_gated, g_a, g_v = self.gate(z_a, z_v)
                z = z_gated
            else:
                z = z_base

        elif self.fusion_type == 'cross_attention':
            fused = self.cross_attn(a, v)    # audio↔vision attended
            z_base = self.fusion(fused)
            logits = self.classifier(z_base)

            if self.use_gating:
                # build per-modality projections from pre-attn summaries
                z_a = self.audio_proj(a)
                z_v = self.vision_proj(v)
                z_gated, g_a, g_v = self.gate(z_a, z_v)
                z = z_gated
            else:
                z = z_base

        elif self.fusion_type == 'early':
            # Fuse before encoding
            early_fused = torch.cat([audio, vision], dim=-1)
            z = self.fusion(early_fused)
            logits = self.classifier(z)

        elif self.fusion_type == 'late':
            logits_audio = self.audio_classifier(a)
            logits_vision = self.vision_classifier(v)
            logits = (logits_audio + logits_vision) / 2.0
            z = (a + v) / 2.0  # For contrastive head, avg embeddings

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        if return_projection:
            # attach aux gate stats when available
            if getattr(self, 'use_gating', False) and self.fusion_type in ['mlp','cross_attention'] and 'z' in locals():
                if 'g_a' in locals() and 'g_v' in locals():
                    aux = {
                        "g_a_mean": g_a.mean().detach(),
                        "g_v_mean": g_v.mean().detach(),
                        "gates": torch.cat([g_a, g_v], dim=1).detach()
                    }
                    return logits, z, aux
            return logits, z
        return logits
