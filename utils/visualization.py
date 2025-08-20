"""
Lightweight visualization utilities.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_similarity_bars(sim_vector, class_names, save_path):
    """
    Plot cosine similarity scores as a horizontal bar chart.

    sim_vector: [C] numpy array of cosine similarities per class
    class_names: list of length C with class display names
    """
    scores = np.array(sim_vector).astype(float)
    idx = np.arange(len(scores))
    labels = class_names[: len(scores)]

    plt.figure(figsize=(8, 4))
    order = np.argsort(scores)
    plt.barh(range(len(scores)), scores[order], color="steelblue")
    plt.yticks(range(len(scores)), [labels[i] for i in order])
    plt.xlabel("Cosine similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

