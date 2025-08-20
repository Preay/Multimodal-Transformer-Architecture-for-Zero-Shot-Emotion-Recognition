#!/usr/bin/env python3
"""
Post-training dynamic quantization for linear/LSTM layers.
Saves int8 quantized state dict and basic efficiency metrics.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from pathlib import Path

from models.multimodal_zser_model import MultimodalZSERModel


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--fusion_type", default="mlp")
    parser.add_argument("--input_dim_audio", type=int, default=768)
    parser.add_argument("--input_dim_vision", type=int, default=768)
    parser.add_argument("--use_temporal", action="store_true")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = MultimodalZSERModel(
        input_dim_audio=args.input_dim_audio,
        input_dim_vision=args.input_dim_vision,
        fusion_type=args.fusion_type,
        use_temporal=args.use_temporal,
    )
    ckpt = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    params_fp = count_parameters(model)

    qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qpath = out_dir / "best_model_int8.pth"
    torch.save(qmodel.state_dict(), qpath.as_posix())

    # Disk sizes
    fp_size_mb = os.path.getsize(args.model_ckpt) / (1024 * 1024)
    int8_size_mb = os.path.getsize(qpath) / (1024 * 1024)

    metrics = {
        "params": int(params_fp),
        "disk_mb_fp": round(fp_size_mb, 3),
        "disk_mb_int8": round(int8_size_mb, 3),
    }
    with open(out_dir / "efficiency.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", qpath)
    print("Efficiency:", metrics)


if __name__ == "__main__":
    main()

