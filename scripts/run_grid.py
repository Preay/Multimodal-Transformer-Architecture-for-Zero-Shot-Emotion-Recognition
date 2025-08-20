#!/usr/bin/env python3
"""
Lightweight grid runner for fusion_type, contrastive_weight, lr.

Aggregates results to results/summary/summary.csv and plots f1 vs contrastive.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def run_cmd(cmd):
    print("CMD:", cmd)
    res = subprocess.run(cmd, shell=True)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--mode", default="multimodal", choices=["audio_only", "vision_only", "multimodal"])
    parser.add_argument("--held_out", nargs="*", default=[])
    parser.add_argument("--results_dir", default="results/summary")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_temporal", action="store_true")
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    fusion_types = ["mlp", "cross_attention", "late"]
    contrastive_weights = [0.1, 0.2, 0.3]
    lrs = [1e-4, 5e-5]

    rows = []
    for fusion, cw, lr in itertools.product(fusion_types, contrastive_weights, lrs):
        # Train
        train_cmd = (
            f"python scripts/train_model.py --dataset_name {args.dataset_name} --dataset_path {args.dataset_path} "
            f"--features_dir {args.features_dir} --mode {args.mode} --fusion_type {fusion} "
            f"--held_out {' '.join(args.held_out)} --epochs {args.epochs} --batch_size {args.batch_size} "
            f"--lr {lr} --contrastive_weight {cw}"
        )
        if args.use_temporal:
            train_cmd += " --use_temporal"
        if not run_cmd(train_cmd):
            print("Train failed, skipping config")
            continue

        # Find latest model dir
        model_root = Path("results")
        subdirs = [d for d in model_root.iterdir() if d.is_dir() and args.dataset_name in d.name and args.mode in d.name and fusion in d.name]
        if not subdirs:
            print("No model dir found, skipping")
            continue
        latest = max(subdirs, key=lambda p: p.stat().st_ctime)
        ckpt = latest / "best_model.pth"
        if not ckpt.exists():
            print("No checkpoint found, skipping")
            continue

        # Evaluate
        eval_dir = latest
        eval_cmd = (
            f"python scripts/evaluate.py --dataset_name {args.dataset_name} --dataset_path {args.dataset_path} "
            f"--features_dir {args.features_dir} --mode {args.mode} --fusion_type {fusion} "
            f"--model_ckpt {ckpt.as_posix()} --held_out {' '.join(args.held_out)}"
        )
        if args.use_temporal:
            eval_cmd += " --use_temporal"
        if not run_cmd(eval_cmd):
            print("Eval failed, skipping config")
            continue

        # Read metrics
        metrics_path = eval_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
                rows.append({
                    "fusion": fusion,
                    "contrastive_weight": cw,
                    "lr": lr,
                    **m,
                })

    if not rows:
        print("No results gathered.")
        return

    df = pd.DataFrame(rows)
    out_csv = Path(args.results_dir) / "summary.csv"
    df.to_csv(out_csv, index=False)

    # Plot F1 vs contrastive
    plt.figure(figsize=(6,4))
    for fusion in df["fusion_type"].unique():
        sub = df[df["fusion_type"] == fusion]
        plt.plot(sub["contrastive_weight"], sub["f1_macro"], marker="o", label=fusion)
    plt.xlabel("contrastive_weight")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.results_dir) / "f1_vs_contrastive.png", dpi=200)


if __name__ == "__main__":
    main()

