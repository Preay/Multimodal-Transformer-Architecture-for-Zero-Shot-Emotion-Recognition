#!/usr/bin/env python3
"""
Unified Experiment Runner for Multimodal Zero-Shot Emotion Recognition

Supports:
- Multi-dataset experiments (RAVDESS, CREMA-D, SAVEE)
- Ablation on fusion_type (mlp, cross_attention, early, late)
- Temporal vs static models
- Modality comparisons (audio-only, video-only, multimodal)
- Structured logging (JSON + CSV) + dissertation-ready visualizations
"""

import os
import json
import argparse
import pandas as pd
from itertools import product
import torch

from scripts.data_loader import build_dataset_df, EmotionMultimodalDataset
from models.multimodal_zser_model import MultimodalZSERModel
from scripts.train_model import train_one_epoch, evaluate_model as simple_eval
from scripts.evaluate import evaluate_model as detailed_eval  # for zero-shot + visualizations

# ===============================
# CONFIGURABLE PARAMETERS
# ===============================
DEFAULT_RESULTS_DIR = "results/experiments"
FUSION_TYPES = ["mlp", "cross_attention", "early", "late"]
TEMPORAL_MODES = [True, False]
DATASETS = ["ravdess", "crema", "savee"]
MODALITIES = ["audio_only", "video_only", "multimodal"]
LOSS_TYPES = ["cross_entropy", "combined"]

# ===============================
# TRAINING WRAPPER
# ===============================
def run_single_experiment(
    dataset_name,
    dataset_path,
    feature_dir_audio,
    feature_dir_vision,
    fusion_type="mlp",
    use_temporal=True,
    modality="multimodal",
    loss_type="combined",
    epochs=5,
    batch_size=16,
    lr=1e-4,
    results_dir="results/experiments",
    device=None,
):
    """
    Runs one experiment config: trains, evaluates, and returns metrics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== 1. Load dataset ====
    df = build_dataset_df(dataset_name, dataset_path)
    dataset = EmotionMultimodalDataset(
        df,
        feature_dir_audio=feature_dir_audio,
        feature_dir_vision=feature_dir_vision,
        temporal_mode=use_temporal,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ==== 2. Build model ====
    model = MultimodalZSERModel(
        input_dim_audio=768,
        input_dim_vision=768,
        fusion_type=fusion_type,
        use_temporal=use_temporal,
    ).to(device)

    # ==== 3. Optimizer & Loss ====
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_type == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        from scripts.train_model import CombinedLoss
        criterion = CombinedLoss()

    # ==== 4. Train ====
    for epoch in range(epochs):
        print(
            f"[{dataset_name}] {fusion_type} | Temporal={use_temporal} | "
            f"{modality} | {loss_type} | Epoch {epoch+1}/{epochs}"
        )
        train_one_epoch(model, loader, optimizer, criterion, device)

    # ==== 5. Evaluate ====
    acc, macro_f1 = simple_eval(model, loader, device)

    # Save a temporary checkpoint to evaluate with t-SNE + attention maps
    ckpt_path = os.path.join(results_dir, f"temp_model_{dataset_name}.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    # Detailed evaluation (zero-shot + visuals)
    detailed_cfg = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "model_ckpt": ckpt_path,
        "fusion_type": fusion_type,
        "use_temporal": use_temporal,
        "feature_dir_audio": feature_dir_audio,
        "feature_dir_vision": feature_dir_vision,
        "batch_size": batch_size,
        "results_dir": results_dir,
    }
    # Runs zero-shot evaluation + confusion matrix + t-SNE + attention plots
    detailed_acc, detailed_f1 = detailed_eval(detailed_cfg)

    return {
        "dataset": dataset_name,
        "fusion_type": fusion_type,
        "use_temporal": use_temporal,
        "modality": modality,
        "loss_type": loss_type,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "zero_shot_acc": detailed_acc,
        "zero_shot_f1": detailed_f1,
    }


# ===============================
# EXPERIMENT GRID GENERATOR
# ===============================
def generate_experiment_grid(
    datasets=DATASETS,
    fusion_types=FUSION_TYPES,
    temporal_modes=TEMPORAL_MODES,
    modalities=MODALITIES,
    loss_types=LOSS_TYPES,
):
    configs = []
    for d, f, t, m, l in product(
        datasets, fusion_types, temporal_modes, modalities, loss_types
    ):
        configs.append(
            {
                "dataset": d,
                "fusion_type": f,
                "use_temporal": t,
                "modality": m,
                "loss_type": l,
            }
        )
    return configs


# ===============================
# MAIN EXPERIMENT RUNNER
# ===============================
def run_all_experiments(
    dataset_root_map,
    feature_dir_audio_map,
    feature_dir_vision_map,
    results_dir=DEFAULT_RESULTS_DIR,
    max_epochs=3,
    batch_size=16,
    lr=1e-4,
):
    os.makedirs(results_dir, exist_ok=True)
    results = []

    configs = generate_experiment_grid()

    for cfg in configs:
        dataset_name = cfg["dataset"]
        dataset_path = dataset_root_map[dataset_name]
        feat_audio = feature_dir_audio_map.get(dataset_name, None)
        feat_vision = feature_dir_vision_map.get(dataset_name, None)

        print(
            f"\n=== Running Experiment ===\n"
            f"Dataset: {dataset_name}\nFusion: {cfg['fusion_type']}\n"
            f"Temporal: {cfg['use_temporal']}\nModality: {cfg['modality']}\nLoss: {cfg['loss_type']}"
        )

        metrics = run_single_experiment(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            feature_dir_audio=feat_audio,
            feature_dir_vision=feat_vision,
            fusion_type=cfg["fusion_type"],
            use_temporal=cfg["use_temporal"],
            modality=cfg["modality"],
            loss_type=cfg["loss_type"],
            epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            results_dir=results_dir,
        )

        results.append(metrics)

        # Save interim JSON
        with open(os.path.join(results_dir, "partial_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # ==== Save Final Results ====
    results_file_json = os.path.join(results_dir, "all_experiments.json")
    results_file_csv = os.path.join(results_dir, "all_experiments.csv")

    with open(results_file_json, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(results_file_csv, index=False)

    print(
        f"\nDONE All experiments completed! Saved to:\n{results_file_json}\n{results_file_csv}"
    )
    return results


# ===============================
# CLI ENTRYPOINT
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Run full multimodal emotion experiments with temporal + multi-dataset support"
    )
    parser.add_argument(
        "--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Where to save results"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Max epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Dataset paths
    parser.add_argument("--ravdess_path", type=str, required=True)
    parser.add_argument("--crema_path", type=str, required=True)
    parser.add_argument("--savee_path", type=str, required=True)

    # Feature directories (audio + vision)
    parser.add_argument("--ravdess_audio", type=str, required=True)
    parser.add_argument("--crema_audio", type=str, required=True)
    parser.add_argument("--savee_audio", type=str, required=True)

    parser.add_argument("--ravdess_vision", type=str, default=None)
    parser.add_argument("--crema_vision", type=str, default=None)
    parser.add_argument("--savee_vision", type=str, default=None)

    args = parser.parse_args()

    # Dataset root mapping
    dataset_root_map = {
        "ravdess": args.ravdess_path,
        "crema": args.crema_path,
        "savee": args.savee_path,
    }

    feature_dir_audio_map = {
        "ravdess": args.ravdess_audio,
        "crema": args.crema_audio,
        "savee": args.savee_audio,
    }

    feature_dir_vision_map = {
        "ravdess": args.ravdess_vision,
        "crema": args.crema_vision,
        "savee": args.savee_vision,
    }

    results = run_all_experiments(
        dataset_root_map=dataset_root_map,
        feature_dir_audio_map=feature_dir_audio_map,
        feature_dir_vision_map=feature_dir_vision_map,
        results_dir=args.results_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print("\n=== Experiment Summary ===")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
