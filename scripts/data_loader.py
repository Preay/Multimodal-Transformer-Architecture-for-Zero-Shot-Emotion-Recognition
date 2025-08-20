"""
Multi-dataset loader for Multimodal Emotion Recognition.
Supports:
- RAVDESS
- CREMA-D
- SAVEE
Returns audio + vision features for each sample
"""

import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd

# ==== Unified Emotion Mapping ====
UNIFIED_EMOTIONS = {
    'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
    'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
}

# Reverse mapping utilities
EMOTION_ID_TO_NAME = {v: k for k, v in UNIFIED_EMOTIONS.items()}

def labels_to_names(label_ids):
    """
    Convert iterable of label ids to emotion names using UNIFIED_EMOTIONS.
    """
    return [EMOTION_ID_TO_NAME.get(int(l), str(l)) for l in label_ids]

def names_to_labels(names):
    """
    Convert iterable of emotion names to label ids using UNIFIED_EMOTIONS.
    Ignores names not present in mapping.
    """
    return [UNIFIED_EMOTIONS[n] for n in names if n in UNIFIED_EMOTIONS]

# ==== RAVDESS Parsing ====
RAVDESS_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def parse_ravdess_filename(filename):
    # Example: 03-01-05-02-02-02-12.wav
    parts = filename.split('-')
    if len(parts) != 7:
        return None
    emotion_code = parts[2]
    return RAVDESS_MAP.get(emotion_code, None)


# ==== CREMA-D Parsing ====
CREMA_MAP = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgust"
}

def parse_crema_filename(filename):
    # Example: 1016_IEO_HAP_HI.wav
    parts = filename.split('_')
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return CREMA_MAP.get(emotion_code, None)


# ==== SAVEE Parsing ====
SAVEE_MAP = {
    'a': 'angry',
    'd': 'disgust',
    'f': 'fearful',
    'h': 'happy',
    'n': 'neutral',
    'sa': 'sad',
    'su': 'surprised'
}

def parse_savee_filename(filename):
    # Example: DC_a01.wav
    base = filename.split('.')[0]
    parts = base.split('_')
    if len(parts) < 2:
        return None
    emo_code = parts[1]
    # handle multi-char codes like 'sa', 'su'
    emo_key = 'sa' if emo_code.startswith('sa') else 'su' if emo_code.startswith('su') else emo_code[0]
    return SAVEE_MAP.get(emo_key, None)


# ==== Generic Loader ====
def build_dataset_df(dataset_name, data_dir):
    """
    Return a dataframe with [filepath, emotion_label]
    Supports .wav audio files
    """
    all_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    records = []
    for f in all_files:
        fname = os.path.basename(f)
        if dataset_name == "ravdess":
            emotion = parse_ravdess_filename(fname)
        elif dataset_name == "crema":
            emotion = parse_crema_filename(fname)
        elif dataset_name == "savee":
            emotion = parse_savee_filename(fname)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if emotion and emotion in UNIFIED_EMOTIONS:
            records.append([f, UNIFIED_EMOTIONS[emotion]])

    df = pd.DataFrame(records, columns=["path", "label"])
    return df


def train_test_split_by_classes(df, holdout_labels):
    """
    Class-held-out split for true zero-shot evaluation.

    Args:
        df: pandas.DataFrame with columns ["path", "label"]
        holdout_labels: list of emotion names to hold out (e.g., ["fearful", "disgust"]).

    Returns:
        train_df: dataframe containing only classes NOT in holdout_labels
        test_df: dataframe containing only classes IN holdout_labels
    """
    import pandas as pd  # local import to avoid top-level dependency issues

    holdout_ids = set(names_to_labels(holdout_labels))
    if len(holdout_ids) == 0:
        # No holdout, return original as train and empty test
        return df.copy().reset_index(drop=True), pd.DataFrame(columns=df.columns)

    mask_holdout = df["label"].isin(holdout_ids)
    test_df = df[mask_holdout].copy().reset_index(drop=True)
    train_df = df[~mask_holdout].copy().reset_index(drop=True)

    return train_df, test_df


# ==== Unified Dataset Class ====
class EmotionMultimodalDataset(Dataset):
    """
    Dataset that loads BOTH audio & vision features for multimodal training.
    Returns:
      (audio_features, vision_features), label
    If feature_dir_vision is None → returns only audio features
    Supports temporal sequences if features are sequences.
    """

    def __init__(self,
                 df,
                 feature_dir_audio=None,
                 feature_dir_vision=None,
                 temporal_mode=False):
        self.df = df
        self.feature_dir_audio = feature_dir_audio
        self.feature_dir_vision = feature_dir_vision
        self.temporal_mode = temporal_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get the base filename from the path
        wav_path = row["path"]
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        label = row["label"]
        
        # Load audio features
        if self.feature_dir_audio:
            audio_feat_path = os.path.join(self.feature_dir_audio, f"{base_name}.pt")
            try:
                audio_features = torch.load(audio_feat_path, weights_only=True)
                # Ensure consistent shape: [768] for non-temporal, [T, 768] for temporal
                if audio_features.dim() == 1:
                    if self.temporal_mode:
                        # For temporal mode, repeat the feature to create a sequence
                        audio_features = audio_features.unsqueeze(0).repeat(10, 1)  # [10, 768]
                    else:
                        audio_features = audio_features  # [768]
                elif audio_features.dim() == 2 and not self.temporal_mode:
                    audio_features = audio_features.squeeze(0)  # [768]
                elif audio_features.dim() == 2 and self.temporal_mode:
                    audio_features = audio_features  # [T, 768]
            except Exception as e:
                print(f"Error loading audio features from {audio_feat_path}: {e}")
                # Return a zero tensor as fallback with correct shape
                if self.temporal_mode:
                    audio_features = torch.zeros(10, 768)  # [10, 768] for temporal
                else:
                    audio_features = torch.zeros(768)  # [768] for non-temporal
        else:
            audio_features = None
            
        # Load vision features  
        if self.feature_dir_vision:
            vision_feat_path = os.path.join(self.feature_dir_vision, f"{base_name}.pt")
            try:
                vision_features = torch.load(vision_feat_path, weights_only=True)
                # Ensure consistent shape: [768] for non-temporal, [T, 768] for temporal
                if vision_features.dim() == 1:
                    if self.temporal_mode:
                        # For temporal mode, repeat the feature to create a sequence
                        vision_features = vision_features.unsqueeze(0).repeat(10, 1)  # [10, 768]
                    else:
                        vision_features = vision_features  # [768]
                elif vision_features.dim() == 2 and not self.temporal_mode:
                    vision_features = vision_features.squeeze(0)  # [768]
                elif vision_features.dim() == 2 and self.temporal_mode:
                    vision_features = vision_features  # [T, 768]
            except Exception as e:
                print(f"Error loading vision features from {vision_feat_path}: {e}")
                # Return a zero tensor as fallback with correct shape
                if self.temporal_mode:
                    vision_features = torch.zeros(10, 768)  # [10, 768] for temporal
                else:
                    vision_features = torch.zeros(768)  # [768] for non-temporal
        else:
            vision_features = None

        return (audio_features, vision_features), torch.tensor(label, dtype=torch.long)


# ==== Audio-Only Dataset Class ====
class EmotionAudioDataset(Dataset):
    """
    Dataset that loads ONLY audio features for audio-only training.
    Returns:
      audio_features, label
    """
    
    def __init__(self, df, feature_dir=None, temporal_mode=False):
        self.df = df
        self.feature_dir = feature_dir
        self.temporal_mode = temporal_mode
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["path"]
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        label = row["label"]
        
        # Load audio features
        if self.feature_dir:
            audio_feat_path = os.path.join(self.feature_dir, f"{base_name}.pt")
            try:
                audio_features = torch.load(audio_feat_path, weights_only=True)
                # Ensure consistent shape: [768] for non-temporal, [T, 768] for temporal
                if audio_features.dim() == 1:
                    if self.temporal_mode:
                        # For temporal mode, repeat the feature to create a sequence
                        audio_features = audio_features.unsqueeze(0).repeat(10, 1)  # [10, 768]
                    else:
                        audio_features = audio_features  # [768]
                elif audio_features.dim() == 2 and not self.temporal_mode:
                    audio_features = audio_features.squeeze(0)  # [768]
                elif audio_features.dim() == 2 and self.temporal_mode:
                    audio_features = audio_features  # [T, 768]
            except Exception as e:
                print(f"Error loading audio features from {audio_feat_path}: {e}")
                # Return a zero tensor as fallback with correct shape
                if self.temporal_mode:
                    audio_features = torch.zeros(10, 768)  # [10, 768] for temporal
                else:
                    audio_features = torch.zeros(768)  # [768] for non-temporal
        else:
            # No precomputed features - use random placeholder with correct shape
            if self.temporal_mode:
                audio_features = torch.randn(10, 768)  # [10, 768] for temporal
            else:
                audio_features = torch.randn(768)  # [768] for non-temporal
            
        return audio_features, torch.tensor(label, dtype=torch.long)
