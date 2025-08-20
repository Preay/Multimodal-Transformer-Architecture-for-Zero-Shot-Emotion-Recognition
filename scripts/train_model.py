"""
Unified Trainer for Zero-Shot Emotion Recognition

- Supports audio-only, vision-only, or multimodal fusion
- Uses pre-extracted Wav2Vec2 (audio) & ViT (vision) embeddings
- Works with RAVDESS, CREMA-D, SAVEE datasets
- Logs metrics to TensorBoard & saves best models

Usage:
  python scripts/train_model.py \
    --dataset_name ravdess \
    --mode multimodal \
    --features_dir features/ravdess \
    --epochs 30 \
    --batch_size 32 \
    --fusion_type cross_attention
"""

import os
import json
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import sys
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import (
    build_dataset_df,
    EmotionAudioDataset,
    EmotionMultimodalDataset,
    train_test_split_by_classes,
    labels_to_names,
)
from models.multimodal_zser_model import MultimodalZSERModel

try:
    import yaml
except Exception:
    yaml = None


# ===============================
# LOSSES
# ===============================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, proj, labels):
        sim = torch.matmul(proj, proj.T) / self.temp  # [B,B]
        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        # positive = sum of same-class sims
        pos_sim = (sim * pos_mask).sum(dim=1)
        neg_sim = sim * (1 - pos_mask)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        targets = torch.zeros(proj.size(0), dtype=torch.long, device=proj.device)
        return self.ce(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logpt = torch.nn.functional.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        logpt = (logpt * torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()).sum(dim=1)
        pt = (pt * torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()).sum(dim=1)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, cls_weight: float = 1.0, contrastive_weight: float = 0.2, focal_weight: float = 0.0, proto_align_weight: float = 0.0, class_weights_tensor: torch.Tensor | None = None):
        super().__init__()
        self.cls_weight = cls_weight
        self.con_weight = contrastive_weight
        self.focal_weight = focal_weight
        self.proto_weight = proto_align_weight
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
        self.focal_loss = FocalLoss()
        self.con_loss = ContrastiveLoss()

    def forward(self, logits: torch.Tensor, proj: torch.Tensor, labels: torch.Tensor):
        # Cross-entropy
        ce = self.cls_loss(logits, labels)
        # Optional focal
        fl = self.focal_loss(logits, labels) if self.focal_weight > 0.0 else torch.tensor(0.0, device=logits.device)
        # Contrastive
        con = self.con_loss(proj, labels) if self.con_weight > 0.0 else torch.tensor(0.0, device=logits.device)
        # Prototype alignment: pull sample toward its class centroid
        if self.proto_weight > 0.0:
            with torch.no_grad():
                # ensure normalized embeddings for cosine similarity
                proj_n = torch.nn.functional.normalize(proj, dim=-1)
                unique_labels = labels.unique()
                centroids = {}
                for ul in unique_labels:
                    mask = (labels == ul)
                    if mask.any():
                        centroids[int(ul.item())] = proj_n[mask].mean(dim=0)
                # stack centroids for gather
            # Build centroid vector for each sample
            centroid_vecs = torch.stack([centroids[int(y.item())] for y in labels], dim=0)
            proto_align = (1.0 - torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(proj, dim=-1), torch.nn.functional.normalize(centroid_vecs, dim=-1), dim=-1)).mean()
        else:
            proto_align = torch.tensor(0.0, device=logits.device)

        total = self.cls_weight * ce + self.focal_weight * fl + self.con_weight * con + self.proto_weight * proto_align
        return total, ce, con, fl, proto_align


# ===============================
# UTILS
# ===============================
def temporal_collate_fn(batch):
    """
    Custom collate function to handle temporal and non-temporal tensors.
    Ensures consistent shapes: [B, T, D] for temporal, [B, D] for non-temporal.
    """
    audio_feats = []
    vision_feats = []
    labels = []
    
    for i, item in enumerate(batch):
        
        # The dataset returns ((audio_feat, vision_feat), label) for multimodal
        if isinstance(item[0], tuple):
            audio_feat, vision_feat = item[0]
            label = item[1]
            audio_feats.append(audio_feat)
            vision_feats.append(vision_feat)
        else:
            # For audio-only: (audio_feat, label)
            audio_feat, label = item
            audio_feats.append(audio_feat)
            vision_feats.append(audio_feat)  # Use audio as vision for audio-only
        labels.append(label)
    
    # Stack tensors
    audio_feats = torch.stack(audio_feats)
    vision_feats = torch.stack(vision_feats)
    labels = torch.stack(labels)
    
    return audio_feats, vision_feats, labels


# ===============================
# TRAIN/VALIDATION LOOPS
# ===============================
def run_epoch(model, loader, optimizer, criterion, device, mode="train", input_mode="audio_only", normalize_z=True, gate_entropy_weight: float = 0.0, writer: "SummaryWriter" = None, epoch: int = 0, p_drop: float = 0.0):
    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss, total_cls, total_con = 0.0, 0.0, 0.0
    correct, total = 0, 0

    loop = tqdm(loader, desc=f"{mode.capitalize()}")

    with torch.set_grad_enabled(mode == "train"):
        for batch in loop:
            # Our collate function returns (audio_feats, vision_feats, labels)
            audio_feat, vision_feat, labels = batch
            labels = labels.to(device)
            audio_feat = audio_feat.to(device)
            vision_feat = vision_feat.to(device)
            
            # Remove noisy debug prints for production

            optimizer.zero_grad() if mode == "train" else None

            # Modality dropout (training only)
            if mode == "train" and p_drop > 0.0:
                if random.random() < p_drop:
                    if random.random() < 0.5:
                        audio_feat = torch.zeros_like(audio_feat)
                    else:
                        vision_feat = torch.zeros_like(vision_feat)

            # Forward pass
            if input_mode == "audio_only":
                out = model(audio_feat, return_projection=True)
            elif input_mode == "vision_only":
                out = model(vision_feat, return_projection=True)
            else:  # multimodal
                out = model(audio_feat, vision_feat, return_projection=True)

            # Unpack optional aux
            g_a_mean = None
            g_v_mean = None
            if isinstance(out, tuple) and len(out) == 3:
                logits, proj, aux = out
                if isinstance(aux, dict):
                    if "g_a_mean" in aux:
                        g_a_mean = aux["g_a_mean"].item() if hasattr(aux["g_a_mean"], 'item') else float(aux["g_a_mean"])
                    if "g_v_mean" in aux:
                        g_v_mean = aux["g_v_mean"].item() if hasattr(aux["g_v_mean"], 'item') else float(aux["g_v_mean"])
                    gates_tensor = aux.get("gates", None)
                else:
                    logits, proj = out  # fallback
                    gates_tensor = None
            else:
                logits, proj = out
                gates_tensor = None

            if normalize_z and proj is not None:
                proj = nn.functional.normalize(proj, dim=-1)
                # Recompute logits on normalized embedding when applicable
                if hasattr(model, "classifier") and input_mode != "late":
                    logits = model.classifier(proj)

            # Loss
            loss, cls_loss, con_loss, focal_loss, proto_align = criterion(logits, proj, labels)

            # Optional gate entropy regularizer
            if gate_entropy_weight > 0.0 and gates_tensor is not None:
                # gates_tensor: [B, 2]
                g = gates_tensor.clamp(1e-6, 1 - 1e-6)
                entropy = -(g * torch.log(g) + (1 - g) * torch.log(1 - g)).mean()
                loss = loss + gate_entropy_weight * entropy

            if mode == "train":
                loss.backward()
                optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_con += con_loss.item()
            _, preds = torch.max(logits, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # Log gate means per batch (last value will be indicative per epoch)
            if writer is not None and g_a_mean is not None and g_v_mean is not None:
                split = "Train" if mode == "train" else "Val"
                writer.add_scalar(f"Gates/{split}_g_a_mean", g_a_mean, epoch)
                writer.add_scalar(f"Gates/{split}_g_v_mean", g_v_mean, epoch)
                # Extra loss components (last batch gives indicative trend)
                writer.add_scalar(f"Loss/{split}_Focal", float(focal_loss.item()) if isinstance(focal_loss, torch.Tensor) else float(focal_loss), epoch)
                writer.add_scalar(f"Loss/{split}_ProtoAlign", float(proto_align.item()) if isinstance(proto_align, torch.Tensor) else float(proto_align), epoch)

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct/total:.2f}%")

    avg_loss = total_loss / len(loader)
    avg_cls = total_cls / len(loader)
    avg_con = total_con / len(loader)
    acc = 100 * correct / total
    return {"loss": avg_loss, "cls_loss": avg_cls, "con_loss": avg_con, "accuracy": acc}


# ===============================
# MAIN TRAIN FUNCTION
# ===============================
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_feature_dim_from_dir(feature_dir: str, temporal_mode: bool) -> int:
    """Infer feature dimension by loading one .pt file from directory."""
    try:
        for fname in os.listdir(feature_dir):
            if fname.endswith(".pt"):
                tensor = torch.load(os.path.join(feature_dir, fname), weights_only=True)
                if tensor.dim() == 1:
                    return int(tensor.shape[-1])
                elif tensor.dim() == 2:
                    return int(tensor.shape[-1])
        return 768
    except Exception:
        return 768


def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DONE Using {device}")

    # Seeds
    set_seeds(args.seed)

    # Dataset
    df = build_dataset_df(args.dataset_name, args.dataset_path)
    n_total = len(df)
    print(f"DONE Loaded {n_total} samples for {args.dataset_name.upper()}")

    # Class-held-out split if provided
    if args.held_out:
        train_df, test_df = train_test_split_by_classes(df, args.held_out)
    else:
        # Fallback random split
        test_split = int(0.2 * n_total)
        train_df = df.iloc[:-test_split]
        test_df = df.iloc[-test_split:]

    # Print split label diagnostics
    train_labels_unique = sorted(set(train_df["label"].tolist()))
    test_labels_unique = sorted(set(test_df["label"].tolist()))
    print(f"Train classes (ids): {train_labels_unique}")
    print(f"Train classes (names): {labels_to_names(train_labels_unique)}")
    print(f"Test classes (ids): {test_labels_unique}")
    print(f"Test classes (names): {labels_to_names(test_labels_unique)}")

    # Dataset class -> loads .pt features
    if args.mode == "multimodal":
        # For multimodal, use the multimodal dataset
        train_ds = EmotionMultimodalDataset(
            train_df, 
            feature_dir_audio=os.path.join(args.features_dir, "audio"),
            feature_dir_vision=os.path.join(args.features_dir, "vision"),
            temporal_mode=args.use_temporal
        )
        test_ds = EmotionMultimodalDataset(
            test_df, 
            feature_dir_audio=os.path.join(args.features_dir, "audio"),
            feature_dir_vision=os.path.join(args.features_dir, "vision"),
            temporal_mode=args.use_temporal
        )
    else:
        # For audio-only or vision-only
        if args.mode == "audio_only":
            feature_dir = os.path.join(args.features_dir, "audio")
        else:  # vision_only
            feature_dir = os.path.join(args.features_dir, "vision")
            
        train_ds = EmotionAudioDataset(train_df, feature_dir=feature_dir, temporal_mode=args.use_temporal)
        test_ds = EmotionAudioDataset(test_df, feature_dir=feature_dir, temporal_mode=args.use_temporal)

    # Use fewer workers on Windows to avoid issues
    num_workers = 0 if os.name == 'nt' else 4  # 0 for Windows, 4 for Unix
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, collate_fn=temporal_collate_fn)
    val_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, collate_fn=temporal_collate_fn)

    # Model
    # Infer input dims if not provided
    if args.input_dim_audio is None:
        audio_dir = os.path.join(args.features_dir, "audio")
        args.input_dim_audio = infer_feature_dim_from_dir(audio_dir, args.use_temporal)
    if args.input_dim_vision is None:
        vision_dir = os.path.join(args.features_dir, "vision")
        if os.path.isdir(vision_dir):
            args.input_dim_vision = infer_feature_dim_from_dir(vision_dir, args.use_temporal)
        else:
            args.input_dim_vision = args.input_dim_audio

    model = MultimodalZSERModel(
        input_dim_audio=args.input_dim_audio,
        input_dim_vision=args.input_dim_vision,
        fusion_type=args.fusion_type,
        num_classes=args.num_classes,
        use_temporal=args.use_temporal,
        use_gating=args.use_gating
    ).to(device)

    # Loss & Optimizer
    # Class weights for balanced CE
    class_weights_tensor = None
    if args.balanced_ce:
        counts = train_df["label"].value_counts().sort_index()
        weights_series = 1.0 / counts
        # Normalize nonzero weights to mean=1.0
        weights_series = weights_series / weights_series.mean()
        import numpy as _np
        full_weights = _np.zeros(args.num_classes, dtype=_np.float32)
        for cls_id, w in weights_series.items():
            if int(cls_id) < args.num_classes:
                full_weights[int(cls_id)] = float(w)
        class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32, device=device)

    criterion = CombinedLoss(
        contrastive_weight=getattr(args, "lambda_contrastive", args.contrastive_weight),
        focal_weight=getattr(args, "lambda_focal", 0.0),
        proto_align_weight=getattr(args, "lambda_proto_align", 0.0),
        class_weights_tensor=class_weights_tensor
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # Save dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.dataset_name}_{args.mode}_{args.fusion_type}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, "logs"))

    best_acc = 0.0

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, "train", args.mode, args.normalize_z, args.gate_entropy_weight, writer, epoch, getattr(args, "p_drop", 0.0))
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, "validate", args.mode, args.normalize_z, 0.0, writer, epoch, 0.0)

        # LR schedule
        scheduler.step(val_metrics["loss"])

        # Log to TB
        for split, metrics in [("Train", train_metrics), ("Val", val_metrics)]:
            writer.add_scalar(f"{split}/Loss", metrics["loss"], epoch)
            writer.add_scalar(f"{split}/Acc", metrics["accuracy"], epoch)
            writer.add_scalar(f"{split}/ClsLoss", metrics["cls_loss"], epoch)
            writer.add_scalar(f"{split}/ConLoss", metrics["con_loss"], epoch)

        # Save best
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_acc,
                "args": vars(args)
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"🎯 New best model saved! Val Acc={best_acc:.2f}%")

    writer.close()
    # Ensure a checkpoint exists even if best never improved
    best_ckpt_path = os.path.join(save_dir, "best_model.pth")
    if not os.path.exists(best_ckpt_path):
        torch.save({
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": best_acc,
            "args": vars(args)
        }, best_ckpt_path)
    print(f"DONE Training done! Best Val Acc: {best_acc:.2f}% | Model saved in {save_dir}")


# ===============================
# CLI
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Train multimodal/audio-only ZSER model")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["ravdess", "crema", "savee"])
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory with extracted .pt features")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["audio_only", "vision_only", "multimodal"])
    parser.add_argument("--fusion_type", type=str, default="mlp", choices=["mlp", "cross_attention", "early", "late"])
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--use_temporal", action="store_true", help="Enable temporal modeling")
    parser.add_argument("--held_out", nargs="*", default=None, help="Emotion names to hold out for ZSL (e.g., fearful disgust)")
    parser.add_argument("--balanced_ce", action="store_true", help="Use class-weighted cross entropy")
    parser.add_argument("--normalize_z", action="store_true", default=True, help="L2-normalize projection z before loss and classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_dim_audio", type=int, default=None)
    parser.add_argument("--input_dim_vision", type=int, default=None)
    # Cross-modal gating toggle (default: True). Use --no_use_gating to disable
    parser.add_argument("--use_gating", dest="use_gating", action="store_true", help="Enable cross-modal gating for mlp/cross_attention")
    parser.add_argument("--no_use_gating", dest="use_gating", action="store_false", help="Disable cross-modal gating")
    parser.set_defaults(use_gating=True)
    parser.add_argument("--gate_entropy_weight", type=float, default=0.0, help="Optional entropy regularizer for gates")
    # YAML config for advanced settings
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config with training hyperparams")
    # Exposed advanced knobs (can also be provided via YAML)
    parser.add_argument("--lambda_contrastive", type=float, default=None)
    parser.add_argument("--lambda_focal", type=float, default=0.0)
    parser.add_argument("--lambda_proto_align", type=float, default=0.0)
    parser.add_argument("--p_drop", type=float, default=0.0, help="Modality dropout probability during training")
    parser.add_argument("--alpha_semantic", type=float, default=1.0, help="Blend factor for semantic-aware prototypes (used in eval)")

    args = parser.parse_args()

    # Merge YAML config if provided
    if args.config is not None and yaml is not None and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        # Map known fields
        for key in [
            "lambda_contrastive", "lambda_focal", "lambda_proto_align", "p_drop",
            "seed", "lr", "weight_decay", "epochs", "batch_size", "normalize_z"
        ]:
            if key in cfg:
                setattr(args, key, cfg[key])
    train(args)


if __name__ == "__main__":
    main()
