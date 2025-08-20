#!/usr/bin/env python3
"""
Export audio-only, vision-only, and fused embeddings for seen train/val and unseen test.

Saves CSVs to artifacts/embeddings/{split}_{modality}.csv with columns:
  id, label, split, modality, e1..eD

Usage:
  python scripts/export_embeddings.py --checkpoint path/to/best_model.pth \
    --dataset_name ravdess --dataset_path data/ravdess --features_dir features/ravdess \
    --split unseen --modality fused
"""

import argparse
import os
import csv
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import build_dataset_df, EmotionAudioDataset, EmotionMultimodalDataset, train_test_split_by_classes
from models.multimodal_zser_model import MultimodalZSERModel


def set_seeds(seed: int = 42):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_dims(features_dir: str, temporal: bool) -> int:
    try:
        for root, _dirs, files in os.walk(features_dir):
            for f in files:
                if f.endswith('.pt'):
                    t = torch.load(os.path.join(root, f), weights_only=True)
                    return int(t.shape[-1]) if t.dim() >= 1 else 768
    except Exception:
        return 768
    return 768


def build_loaders(args):
    df = build_dataset_df(args.dataset_name, args.dataset_path)
    seen_df, unseen_df = train_test_split_by_classes(df, args.held_out or [])
    if args.split in ["train", "seen_train", "seen_val"]:
        # We don't have explicit val; split seen into train/val for export consistency
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(seen_df))
        tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=seen_df["label"])
        if args.split in ["train", "seen_train"]:
            target_df = seen_df.iloc[tr_idx]
        else:
            target_df = seen_df.iloc[va_idx]
    elif args.split in ["seen", "seen_all"]:
        target_df = seen_df
    elif args.split in ["unseen", "test"]:
        target_df = unseen_df
    else:
        raise ValueError(f"Unknown split: {args.split}")

    temporal = args.use_temporal
    if args.modality == "audio":
        ds = EmotionAudioDataset(target_df, feature_dir=os.path.join(args.features_dir, "audio"), temporal_mode=temporal)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    elif args.modality == "vision":
        # Reuse audio dataset class but point to vision dir
        ds = EmotionAudioDataset(target_df, feature_dir=os.path.join(args.features_dir, "vision"), temporal_mode=temporal)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    else:
        ds = EmotionMultimodalDataset(
            target_df,
            feature_dir_audio=os.path.join(args.features_dir, "audio"),
            feature_dir_vision=os.path.join(args.features_dir, "vision"),
            temporal_mode=temporal,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    return loader, target_df


def export(args):
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, target_df = build_loaders(args)

    # Model if fused requested
    if args.modality == "fused":
        # infer dims
        dim_a = infer_dims(os.path.join(args.features_dir, "audio"), args.use_temporal)
        dim_v = infer_dims(os.path.join(args.features_dir, "vision"), args.use_temporal)
        model = MultimodalZSERModel(
            input_dim_audio=dim_a,
            input_dim_vision=dim_v,
            fusion_type=args.fusion_type,
            use_temporal=args.use_temporal,
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.split}_{args.modality}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_written = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if args.modality == "fused":
                    # Handle multiple collate shapes:
                    # - ((a, v), labels)
                    # - (a, v, labels)
                    # - dict-like not supported here
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 2 and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
                            (a, v), labels = batch
                        elif len(batch) == 3:
                            a, v, labels = batch
                        else:
                            raise ValueError(f"Unexpected batch structure for fused modality: types={[type(x) for x in batch]} lens={[len(x) if hasattr(x,'__len__') else None for x in batch]}")
                    else:
                        raise ValueError(f"Unsupported batch type: {type(batch)}")
                    a = a.to(device)
                    v = v.to(device)
                    labels = labels.to(device)
                    out = model(a, v, return_projection=True)
                    if isinstance(out, (list, tuple)) and len(out) == 3:
                        logits, z, _ = out
                    else:
                        logits, z = out
                    z = torch.nn.functional.normalize(z, dim=-1)
                    emb = z.cpu().numpy()
                else:
                    # audio OR vision single input
                    # Expect (x, labels)
                    x, labels = batch
                    x = x.to(device)
                    labels = labels.to(device)
                    # If temporal, mean-pool time dim
                    if x.dim() == 3:
                        x = x.mean(dim=1)
                    emb = torch.nn.functional.normalize(x, dim=-1).cpu().numpy()

                # Build rows
                start = batch_idx * args.batch_size
                end = start + emb.shape[0]
                subset_df = target_df.iloc[start:end]
                for i in range(emb.shape[0]):
                    base_id = os.path.splitext(os.path.basename(subset_df.iloc[i]["path"]))[0]
                    label = int(subset_df.iloc[i]["label"])
                    row = [base_id, label, args.split, args.modality]
                    row.extend(list(map(float, emb[i].tolist())))
                    if not header_written:
                        header = ["id", "label", "split", "modality"] + [f"e{j+1}" for j in range(emb.shape[1])]
                        writer.writerow(header)
                        header_written = True
                    writer.writerow(row)

    # Save metadata
    meta = {
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset_name,
        "split": args.split,
        "modality": args.modality,
        "fusion_type": args.fusion_type,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as mf:
        json.dump(meta, mf, indent=2)

    print(f"DONE Saved {out_path}")


def cli():
    p = argparse.ArgumentParser(description="Export embeddings to CSV")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True, choices=["ravdess", "crema", "savee"])
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="unseen", choices=["train", "seen_train", "seen_val", "seen", "seen_all", "unseen", "test"])
    p.add_argument("--modality", type=str, default="fused", choices=["audio", "vision", "fused"])
    p.add_argument("--fusion_type", type=str, default="mlp", choices=["mlp", "cross_attention", "early", "late"])
    p.add_argument("--use_temporal", action="store_true")
    p.add_argument("--held_out", nargs="*", default=["fearful", "surprised"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="artifacts/embeddings")
    args = p.parse_args()

    export(args)


if __name__ == "__main__":
    cli()


