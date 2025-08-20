"""
Interpretability & Visualization Utilities for Multimodal Z-SER
--------------------------------------------------------------
Includes:
DONE Temporal attention visualization
DONE Cross-modal attention heatmaps
DONE t-SNE visualization (global, per-class, and evolution across epochs)
DONE Confusion matrix
DONE Similarity matrix heatmap
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

sns.set(style="whitegrid")


# ===============================
# TEMPORAL & CROSS-MODAL ATTENTION
# ===============================

def plot_temporal_attention(weights, title, save_path=None):
    """
    Plot attention weights over time steps.
    weights: [T] numpy array of attention weights.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(weights, marker='o', color='royalblue', linewidth=2)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Attention Weight")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cross_modal_attention(audio_weights, vision_weights, save_path=None):
    """
    Visualize audio↔vision cross-modal attention weights as a heatmap.
    audio_weights: [A, D]
    vision_weights: [V, D]
    """
    plt.figure(figsize=(6, 6))
    heatmap = audio_weights @ vision_weights.T
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar()
    plt.title("Cross-Modal Attention Heatmap")
    plt.xlabel("Vision Features")
    plt.ylabel("Audio Features")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ===============================
# T-SNE VISUALIZATION
# ===============================

def plot_tsne_embeddings(embeddings, labels, class_names, save_path, title="t-SNE Visualization"):
    """
    Global t-SNE plot of all embeddings.

    embeddings: [N, D] numpy array
    labels: [N] int labels
    class_names: list of emotion class names
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        mask = (labels == i)
        if np.any(mask):
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=name, alpha=0.7, s=40)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne_per_class(embeddings, labels, class_names, save_dir):
    """
    Generate separate t-SNE plots for each class.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    os.makedirs(save_dir, exist_ok=True)
    for i, name in enumerate(class_names):
        mask = (labels == i)
        if not np.any(mask):
            continue
        plt.figure(figsize=(8, 6))
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=name, alpha=0.8, s=50)
        plt.title(f"t-SNE for {name}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/tsne_{name}.png", dpi=300)
        plt.close()


def plot_tsne_evolution(embeddings_list, labels, class_names, save_dir):
    """
    Plot t-SNE at different training epochs to visualize embedding evolution.
    embeddings_list: list of (epoch, embeddings)
    """
    os.makedirs(save_dir, exist_ok=True)

    for epoch, emb in embeddings_list:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(emb)

        plt.figure(figsize=(10, 8))
        for i, name in enumerate(class_names):
            mask = (labels == i)
            if np.any(mask):
                plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=name, alpha=0.7, s=40)
        plt.title(f"t-SNE at Epoch {epoch}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/tsne_epoch_{epoch}.png", dpi=300)
        plt.close()


# ===============================
# CONFUSION & SIMILARITY MATRIX
# ===============================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrix as a heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_similarity_matrix(similarities, class_names, save_path, title="Similarity Matrix"):
    """
    similarities: [N, C] similarity scores (N test samples × C classes)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarities, cmap="viridis", xticklabels=class_names, yticklabels=False)
    plt.title(title)
    plt.xlabel("Class Prototypes")
    plt.ylabel("Test Samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
