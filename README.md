# Multimodal Zero-Shot Emotion Recognition (ZSER)

This repository implements a multimodal zero-shot emotion recognition system that classifies human emotions using audio and visual data from the [RAVDESS dataset](https://zenodo.org/record/1188976). The system leverages pretrained Wav2Vec2 and Vision Transformer (ViT) models, enhanced with cross-modal attention, temporal modeling, and prototype-based zero-shot classification.

Note: Dataset files are not included due to size and licensing constraints. Please download the dataset manually (instructions below).

## Key Results (Zero-Shot on RAVDESS)

- Accuracy: 39.1%
- Macro F1-score: 37.2%
- Parameters: ~2.5M
- GPU Memory: ~4GB during training

## Project Structure

multimodal_zser/
├── models/
│ ├── audio_zser_model.py # Audio-only architecture
│ └── multimodal_zser_model.py # Cross-modal fusion model
├── scripts/
│ ├── extract_features.py # Feature extraction
│ ├── train_model.py # Training loop
│ └── evaluate.py # Evaluation & visualizations
├── utils/
│ └── interpretability.py # t-SNE & attention maps
├── results/
│ └── improved/ # Output visualizations/logs
├── main.py # Unified pipeline
├── requirements.txt # Dependencies
└── README.md



## Requirements

Install dependencies with:


pip install -r requirements.txt

Key Libraries
Python 3.9+

PyTorch ≥ 2.0

torchvision

torchaudio

transformers

scikit-learn

OpenCV

matplotlib



# Dataset Setup (RAVDESS)
Download the RAVDESS dataset and organize it as follows:


data/
└── ravdess/
    ├── audio/
    │   ├── audio_speech_actors_01-24/
    │   └── audio_song_actors_01-24/
    └── vision/
        ├── video_speech_actors_01-24/
        └── video_song_actors_01-24/
# How to Run
Step 1: Extract Features

python scripts/extract_features.py --modality both

Step 2: Train Model

python scripts/train_model.py --fusion cross_attn --epochs 30

Step 3: Evaluate Model (Zero-Shot)


python scripts/evaluate.py --use_zero_shot

# Performance Summary

Metric	Value
Accuracy	39.1%
Macro F1	37.2%

# Visual outputs are saved in results/improved/:

Confusion Matrix: confusion_matrix_ravdess.png

Embedding Space (t-SNE): tsne_ravdess.png

Attention Maps: attn_audio_ravdess.png, attn_vision_ravdess.png

# Key Features
Zero-shot classification using prototype learning

Wav2Vec2 audio embeddings

ViT-based facial feature extraction

BiLSTM + attention pooling

Multiple fusion strategies (early, late, MLP, cross-attn)

Contrastive learning for robust embedding space

Visual analytics for model interpretability

# Citation & Acknowledgments

This repository was developed as part of a master's dissertation at Queen Mary University of London.

If you use this work, please cite the RAVDESS dataset and relevant papers referenced in this project.

# To Do

 Add text modality (trimodal ZSER)

 Ensemble fusion strategies

 Support multi-label emotion classification

 Enable real-time inference (e.g., webcam demo)

