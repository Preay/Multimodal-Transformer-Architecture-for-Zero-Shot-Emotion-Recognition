
Model Card
==========

Task: Multimodal Zero-Shot Emotion Recognition on RAVDESS.

Key design:
- Fusion: cross_attention
- Temporal: False
- Gating: True
- Loss weights: CE + contrastive(None) + focal(0.0) + proto_align(0.0)

ZSL Protocol:
- Held-out classes (if any): 
- Prototype-based cosine similarity on normalized embeddings.

Implementation notes:
- Actor-aware splits; AMP; early stopping on val macro-F1; ReduceLROnPlateau.
