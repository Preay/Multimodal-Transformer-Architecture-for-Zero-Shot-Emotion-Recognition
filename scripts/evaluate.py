#!/usr/bin/env python3
"""
Evaluate multimodal zero-shot emotion recognition model.

DONE Computes:
  - Accuracy, Macro-F1
  - Confusion Matrix
  - t-SNE visualization for embeddings
  - Optional temporal attention visualization (for interpretability)

Usage Example:
--------------
python scripts/evaluate.py \
  --dataset_name ravdess \
  --dataset_path data/ravdess \
  --mode multimodal \
  --fusion_type cross_attention \
  --features_dir features/ravdess \
  --model_ckpt results/multimodal_2025xxxx/best_model.pth \
  --results_dir results/eval_ravdess \
  --use_temporal
"""

import argparse
import os
import torch
import numpy as np
import sys
import json
import random
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

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
from utils.interpretability import (
    plot_confusion_matrix,
    plot_tsne_embeddings,
    plot_temporal_attention
)

# ================================
# UTILS
# ================================
def temporal_collate_fn(batch):
    """
    Custom collate function to handle temporal and non-temporal tensors.
    Ensures consistent shapes: [B, T, D] for temporal, [B, D] for non-temporal.
    """
    audio_feats = []
    vision_feats = []
    labels = []
    
    for item in batch:
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


CLASS_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


# ==========================
# Extract Embeddings
# ==========================
def set_seeds(seed: int = 42):
    import numpy as _np
    import random as _random
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_embeddings(model, loader, device, use_temporal=False, normalize_z=True):
    model.eval()
    all_emb = []
    all_labels = []
    attn_audio = []
    attn_vision = []
    all_gates = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            # Our collate function returns (audio_feats, vision_feats, labels)
            audio_feat, vision_feat, labels = batch
            labels = labels.to(device)
            audio_feat = audio_feat.to(device)
            vision_feat = vision_feat.to(device)
            
            out = model(audio_feat, vision_feat, return_projection=True)
            if isinstance(out, tuple) and len(out) == 3 and isinstance(out[2], dict) and ("gates" in out[2]):
                logits, z, aux = out
                if aux.get("gates") is not None:
                    all_gates.append(aux["gates"].detach().cpu().numpy())
            else:
                logits, z = out
            if normalize_z and z is not None:
                z = torch.nn.functional.normalize(z, dim=-1)

            all_emb.append(z.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Capture temporal attention if model supports it
            if use_temporal and hasattr(model, "audio_attention"):
                # Process through encoders first, then get attention weights
                audio_encoded = model.audio_encoder(audio_feat)  # [B, T, 256]
                vision_encoded = model.vision_encoder(vision_feat)  # [B, T, 256]
                
                # Get attention weights from TemporalAttention modules
                _, a_w = model.audio_attention(audio_encoded)
                _, v_w = model.vision_attention(vision_encoded)
                attn_audio.extend(a_w.cpu().numpy())
                attn_vision.extend(v_w.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Save gates if any
    if len(all_gates) > 0:
        try:
            import numpy as _np
            gates_concat = _np.concatenate(all_gates, axis=0)
        except Exception:
            gates_concat = None
    else:
        gates_concat = None

    return embeddings, labels, attn_audio, attn_vision, gates_concat


# ==========================
# Compute Class Centroids
# ==========================
def compute_class_centroids(embeddings, labels):
    centroids = {}
    for lbl in np.unique(labels):
        cls_emb = embeddings[labels == lbl]
        centroid = cls_emb.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids[lbl] = centroid
    return centroids


# ==========================
# Zero-Shot Prediction
# ==========================
def zero_shot_predict(test_embeddings, centroids):
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    class_ids = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in class_ids], axis=0)
    sims = np.dot(test_embeddings, centroid_matrix.T)
    preds = np.argmax(sims, axis=1)
    pred_labels = np.array([class_ids[i] for i in preds])
    return pred_labels


# ==========================
# Main Evaluation Routine
# ==========================
def evaluate_model(config):
    set_seeds(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["results_dir"], exist_ok=True)

    # ==== 1. Build dataset split ====
    df = build_dataset_df(config["dataset_name"], config["dataset_path"])

    # Class-held-out split
    holdout = config.get("held_out", []) or []
    seen_df, unseen_df = train_test_split_by_classes(df, holdout)

    # Diagnostics
    train_labels_unique = sorted(set(seen_df["label"].tolist()))
    test_labels_unique = sorted(set(unseen_df["label"].tolist()))
    print(f"Train(seen) classes (ids): {train_labels_unique}")
    print(f"Train(seen) classes (names): {labels_to_names(train_labels_unique)}")
    print(f"Test(unseen) classes (ids): {test_labels_unique}")
    print(f"Test(unseen) classes (names): {labels_to_names(test_labels_unique)}")

    # Use appropriate dataset class based on mode
    if config.get("mode", "multimodal") == "multimodal":
        train_set = EmotionMultimodalDataset(
            seen_df, 
            feature_dir_audio=os.path.join(config["features_dir"], "audio"),
            feature_dir_vision=os.path.join(config["features_dir"], "vision"),
            temporal_mode=config.get("use_temporal", False)
        )
        test_set = EmotionMultimodalDataset(
            unseen_df, 
            feature_dir_audio=os.path.join(config["features_dir"], "audio"),
            feature_dir_vision=os.path.join(config["features_dir"], "vision"),
            temporal_mode=config.get("use_temporal", False)
        )
    else:
        train_set = EmotionAudioDataset(
            seen_df, 
            feature_dir=config["features_dir"],
            temporal_mode=config.get("use_temporal", False)
        )
        test_set = EmotionAudioDataset(
            unseen_df, 
            feature_dir=config["features_dir"],
            temporal_mode=config.get("use_temporal", False)
        )

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, collate_fn=temporal_collate_fn)

    # ==== 2. Load model ====
    baseline = config.get("baseline", None)
    normalize_z = config.get("normalize_z", True)

    if baseline == "nearest_centroid":
        # Use raw features from dataset as embeddings
        def get_representation(batch, mode):
            a, v, _ = batch
            if mode == "audio_only":
                rep = a
            elif mode == "vision_only":
                rep = v
            else:
                # Mean-pool temporal then concatenate
                rep_a = a.mean(dim=1) if a.dim() == 3 else a
                rep_v = v.mean(dim=1) if v.dim() == 3 else v
                rep = torch.cat([rep_a, rep_v], dim=-1)
            return rep

        # Collect train representations
        all_train_rep = []
        all_train_labels = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Collect train reps"):
                rep = get_representation(batch, config.get("mode", "multimodal"))
                rep = rep.to(device)
                if normalize_z:
                    rep = torch.nn.functional.normalize(rep, dim=-1)
                all_train_rep.append(rep.cpu().numpy())
                all_train_labels.append(batch[2].numpy())
        train_emb = np.concatenate(all_train_rep, axis=0)
        train_labels = np.concatenate(all_train_labels, axis=0)

        # Collect test representations
        all_test_rep = []
        all_test_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Collect test reps"):
                rep = get_representation(batch, config.get("mode", "multimodal"))
                rep = rep.to(device)
                if normalize_z:
                    rep = torch.nn.functional.normalize(rep, dim=-1)
                all_test_rep.append(rep.cpu().numpy())
                all_test_labels.append(batch[2].numpy())
        test_emb = np.concatenate(all_test_rep, axis=0)
        test_labels = np.concatenate(all_test_labels, axis=0)

        attn_audio, attn_vision = [], []
    else:
        # ==== Build and load model ====
        # Try to infer input dims from first batch
        first_batch = next(iter(train_loader))
        a0, v0, _ = first_batch
        dim_a = a0.shape[-1] if a0.dim() >= 2 else a0.shape[0]
        dim_v = v0.shape[-1] if v0.dim() >= 2 else v0.shape[0]

        model = MultimodalZSERModel(
            input_dim_audio=int(dim_a),
            input_dim_vision=int(dim_v),
            fusion_type=config["fusion_type"],
            use_temporal=config.get("use_temporal", False)
        ).to(device)

        ckpt = torch.load(config["model_ckpt"], map_location=device)
        # Allow loading older checkpoints that may not have gating params
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # ==== 3. Extract embeddings ====
        train_emb, train_labels, _, _, _ = extract_embeddings(model, train_loader, device, config["use_temporal"], normalize_z)
        test_emb, test_labels, attn_audio, attn_vision, gates_concat = extract_embeddings(model, test_loader, device, config["use_temporal"], normalize_z)

    # ==== 4. Save evaluation context for later t-SNE plotting ====
    try:
        seen_class_ids = np.unique(train_labels)
        seen_mask = np.isin(test_labels, seen_class_ids).astype(np.int32)
        np.savez(
            os.path.join(config["results_dir"], f"eval_context_{config['dataset_name']}.npz"),
            test_emb=test_emb,
            test_labels=test_labels,
            seen_mask=seen_mask,
        )
    except Exception:
        pass

    # ==== 5. Zero-shot classification ====
    centroids = compute_class_centroids(train_emb, train_labels)
    pred_labels = zero_shot_predict(test_emb, centroids)

    # ==== 6. Metrics (ZSL) ====
    acc = accuracy_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels, average="macro")
    print(f"\nDONE Zero-Shot Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # Optional: Generalized ZSL (compute on seen+unseen)
    # Create a small seen-eval split
    from sklearn.model_selection import train_test_split
    seen_train_idx, seen_eval_idx = train_test_split(np.arange(len(train_emb)), test_size=0.2, random_state=42, stratify=train_labels)
    seen_eval_emb = train_emb[seen_eval_idx]
    seen_eval_labels = train_labels[seen_eval_idx]
    # Predict seen-eval with centroids limited to seen classes
    seen_class_ids = sorted(set(train_labels.tolist()))
    seen_centroids = {c: centroids[c] for c in seen_class_ids if c in centroids}
    seen_preds = zero_shot_predict(seen_eval_emb, seen_centroids)
    acc_seen = accuracy_score(seen_eval_labels, seen_preds)
    acc_unseen = acc
    harmonic = (2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)) if (acc_seen + acc_unseen) > 0 else 0.0

    # ==== 7. Save raw results & metrics ====
    np.savez(
        os.path.join(config["results_dir"], f"eval_results_{config['dataset_name']}.npz"),
        test_labels=test_labels, pred_labels=pred_labels
    )

    metrics = {
        "acc": float(acc),
        "f1_macro": float(f1),
        "acc_seen": float(acc_seen),
        "acc_unseen": float(acc_unseen),
        "harmonic": float(harmonic),
        "held_out": holdout,
        "mode": config.get("mode", "multimodal"),
        "fusion_type": config.get("fusion_type"),
        "baseline": baseline or "model",
    }
    with open(os.path.join(config["results_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Append to leaderboard
    import csv
    leaderboard_path = os.path.join(config["results_dir"], "leaderboard.csv")
    header = ["dataset", "mode", "fusion", "baseline", "held_out", "acc", "f1_macro", "acc_seen", "acc_unseen", "harmonic"]
    row = [
        config["dataset_name"],
        config.get("mode", "multimodal"),
        config.get("fusion_type"),
        baseline or "model",
        ";".join(holdout),
        acc,
        f1,
        acc_seen,
        acc_unseen,
        harmonic,
    ]
    write_header = not os.path.exists(leaderboard_path)
    with open(leaderboard_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    # ==== 8. Confusion Matrix ====
    plot_confusion_matrix(
        test_labels,
        pred_labels,
        CLASS_NAMES,
        os.path.join(config["results_dir"], f"confusion_matrix_{config['dataset_name']}.png")
    )

    # ==== 9. t-SNE visualization ====
    plot_tsne_embeddings(
        train_emb,
        train_labels,
        CLASS_NAMES,
        os.path.join(config["results_dir"], f"tsne_seen.png"),
        title=f"t-SNE: Seen (Train) Embeddings ({config['dataset_name']})"
    )
    plot_tsne_embeddings(
        test_emb,
        test_labels,
        CLASS_NAMES,
        os.path.join(config["results_dir"], f"tsne_unseen.png"),
        title=f"t-SNE: Unseen (Test) Embeddings ({config['dataset_name']})"
    )

    # ==== 10. Temporal Attention visualization ====
    if config["use_temporal"] and len(attn_audio) > 0:
        plot_temporal_attention(
            attn_audio[0],
            title=f"Audio Temporal Attention ({config['dataset_name']})",
            save_path=os.path.join(config["results_dir"], f"attn_audio_{config['dataset_name']}.png")
        )
        if len(attn_vision) > 0:
            plot_temporal_attention(
                attn_vision[0],
                title=f"Vision Temporal Attention ({config['dataset_name']})",
                save_path=os.path.join(config["results_dir"], f"attn_vision_{config['dataset_name']}.png")
            )

    # ==== 11. Prototype similarity bars for first few samples ====
    from utils.visualization import plot_similarity_bars
    os.makedirs(os.path.join(config["results_dir"], "explanations"), exist_ok=True)
    class_ids = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in class_ids], axis=0)
    sims_all = np.dot(test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True), centroid_matrix.T)
    num_viz = min(20, sims_all.shape[0])
    for i in range(num_viz):
        plot_similarity_bars(
            sims_all[i],
            CLASS_NAMES,
            os.path.join(config["results_dir"], "explanations", f"sample_{i}_similarities.png"),
        )

    # ==== 12. Optional: visualize gates distribution ====
    try:
        if 'gates_concat' in locals() and gates_concat is not None:
            import matplotlib.pyplot as plt
            import numpy as _np
            gates_dir = os.path.join(config["results_dir"], "explanations")
            os.makedirs(gates_dir, exist_ok=True)
            plt.figure()
            plt.hist(gates_concat[:, 0], bins=20, alpha=0.6, label='g_a')
            plt.hist(gates_concat[:, 1], bins=20, alpha=0.6, label='g_v')
            plt.legend()
            plt.title('Gate distributions')
            plt.savefig(os.path.join(gates_dir, 'gates_hist.png'))
            plt.close()

            # random samples
            idxs = _np.random.choice(gates_concat.shape[0], size=min(4, gates_concat.shape[0]), replace=False)
            for i, idx in enumerate(idxs):
                plt.figure()
                plt.bar(['g_a', 'g_v'], gates_concat[idx])
                plt.ylim(0, 1)
                plt.title(f'Gates sample {i}')
                plt.savefig(os.path.join(gates_dir, f'gates_sample_{i}.png'))
                plt.close()
    except Exception:
        pass

    return acc, f1


# ==========================
# CLI Entry Point
# ==========================
def main():
    parser = argparse.ArgumentParser(description="Evaluate ZSER model on unseen emotions")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, required=True, choices=["ravdess", "crema", "savee"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--features_dir", type=str, required=True, help="Path to extracted features")

    # Model args
    parser.add_argument("--model_ckpt", type=str, help="Path to best_model.pth checkpoint")
    parser.add_argument("--fusion_type", type=str, default="mlp", choices=["mlp", "cross_attention", "early", "late"])
    parser.add_argument("--mode", type=str, default="multimodal", choices=["audio_only", "vision_only", "multimodal"])
    parser.add_argument("--use_temporal", action="store_true", help="Visualize temporal attention if available")
    parser.add_argument("--held_out", nargs="*", default=None, help="Emotion names to hold out for ZSL (e.g., fearful disgust)")
    parser.add_argument("--baseline", type=str, default=None, choices=[None, "nearest_centroid"], help="Use baseline instead of model")
    parser.add_argument("--normalize_z", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation params
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--results_dir", type=str, default="results/eval")

    args = parser.parse_args()

    # Build config dict
    config = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "features_dir": args.features_dir,
        "model_ckpt": args.model_ckpt,
        "fusion_type": args.fusion_type,
        "mode": args.mode,
        "use_temporal": args.use_temporal,
        "batch_size": args.batch_size,
        "results_dir": args.results_dir
    }

    # Run evaluation
    if args.baseline != "nearest_centroid" and not args.model_ckpt:
        print("Model checkpoint required unless using --baseline nearest_centroid")
        sys.exit(1)

    config.update({
        "held_out": args.held_out or [],
        "baseline": args.baseline,
        "normalize_z": args.normalize_z,
        "seed": args.seed,
    })

    evaluate_model(config)


if __name__ == "__main__":
    main()
