#!/usr/bin/env python3
"""
Benchmark CPU inference latency and estimate MACs if available.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.multimodal_zser_model import MultimodalZSERModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--fusion_type", default="mlp")
    parser.add_argument("--input_dim_audio", type=int, default=768)
    parser.add_argument("--input_dim_vision", type=int, default=768)
    parser.add_argument("--use_temporal", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--reps", type=int, default=50)
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
    model.to(device)

    # Dummy inputs
    if args.use_temporal:
        a = torch.randn(args.batch_size, 10, args.input_dim_audio, device=device)
        v = torch.randn(args.batch_size, 10, args.input_dim_vision, device=device)
    else:
        a = torch.randn(args.batch_size, args.input_dim_audio, device=device)
        v = torch.randn(args.batch_size, args.input_dim_vision, device=device)

    # Warmup
    for _ in range(5):
        model(a, v)

    # Timing
    times = []
    for _ in range(args.reps):
        t0 = time.time()
        with torch.no_grad():
            model(a, v)
        times.append((time.time() - t0) * 1000.0)

    cpu_ms = float(np.mean(times))

    # Estimate MACs if thop available
    macs = None
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(a, v))
    except Exception:
        macs = None

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / "efficiency.json"
    data = {}
    if out_path.exists():
        try:
            data = json.load(open(out_path))
        except Exception:
            data = {}

    data.update({"cpu_ms_per_sample": cpu_ms, "macs": macs})
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Latency (ms/sample):", cpu_ms)
    if macs is not None:
        print("MACs:", macs)


if __name__ == "__main__":
    main()

