#!/usr/bin/env python3
"""
Plot t-SNE of test embeddings with seen/unseen markers.

Usage example:
python scripts/plot_tsne.py \
  --npz results/eval_ravdess/eval_context_ravdess.npz \
  --out figures/tsne_plot.png \
  --perplexity 30 \
  --max_points 4000
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Make project modules importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # If available, use the same class names
    from scripts.evaluate import CLASS_NAMES as EVAL_CLASS_NAMES
except Exception:
    EVAL_CLASS_NAMES = None


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization for ZSER test embeddings")
    parser.add_argument("--npz", type=str, required=True, help="Path to eval_context_<dataset>.npz")
    parser.add_argument("--out", type=str, default="figures/tsne_plot.png", help="Output PNG path")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--max_points", type=int, default=5000, help="Max points to plot (downsample)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data = np.load(args.npz)
    test_emb = data["test_emb"]
    test_labels = data["test_labels"]
    seen_mask = data["seen_mask"].astype(np.int32)

    # Downsample if needed to keep plot readable and fast
    num_points = test_emb.shape[0]
    if num_points > args.max_points:
        idx = np.random.RandomState(42).choice(num_points, size=args.max_points, replace=False)
        test_emb = test_emb[idx]
        test_labels = test_labels[idx]
        seen_mask = seen_mask[idx]

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=42)
    emb_2d = tsne.fit_transform(test_emb)

    # Prepare class name mapping
    unique_labels = sorted(np.unique(test_labels).tolist())
    if EVAL_CLASS_NAMES is not None and max(unique_labels) < len(EVAL_CLASS_NAMES):
        id_to_name = {i: EVAL_CLASS_NAMES[i] for i in unique_labels}
    else:
        id_to_name = {i: f"cls{i}" for i in unique_labels}

    # Colors per class
    cmap = plt.get_cmap("tab10")
    label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_labels)}

    plt.figure(figsize=(6, 4))
    for lbl in unique_labels:
        mask_lbl = (test_labels == lbl)
        # seen vs unseen
        for seen_flag, marker, edgecolor in [
            (1, "o", None),
            (0, "X", "black"),
        ]:
            mask = mask_lbl & (seen_mask == seen_flag)
            if not np.any(mask):
                continue
            plt.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=18,
                c=[label_to_color[lbl]],
                label=f"{id_to_name[lbl]}{' (unseen)' if seen_flag == 0 else ''}",
                marker=marker,
                edgecolors=edgecolor,
                linewidths=0.5 if edgecolor else 0.0,
                alpha=0.9,
            )

    # Compact legend suitable for two-column papers
    handles, labels = plt.gca().get_legend_handles_labels()
    # Deduplicate while preserving order
    seen = set()
    filtered = []
    for h, l in zip(handles, labels):
        if l not in seen:
            filtered.append((h, l))
            seen.add(l)
    if filtered:
        handles, labels = zip(*filtered)
        plt.legend(handles, labels, loc="best", fontsize=8, ncol=2, frameon=True)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()



