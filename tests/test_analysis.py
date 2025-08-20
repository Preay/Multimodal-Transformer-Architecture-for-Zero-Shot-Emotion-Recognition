import os
import numpy as np
import pandas as pd


def test_tsne_pipeline_smoke(tmp_path):
    # Create tiny synthetic embeddings CSV
    rng = np.random.RandomState(42)
    n = 60
    d = 32
    X0 = rng.randn(n // 3, d) + 0.0
    X1 = rng.randn(n // 3, d) + 3.0
    X2 = rng.randn(n - 2 * (n // 3), d) - 3.0
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)))
    rows = []
    for i in range(n):
        row = [f"id{i}", int(y[i]), "unseen", "fused"] + list(map(float, X[i].tolist()))
        rows.append(row)
    cols = ["id", "label", "split", "modality"] + [f"e{j+1}" for j in range(d)]
    df = pd.DataFrame(rows, columns=cols)
    emb_dir = tmp_path / "artifacts" / "embeddings"
    emb_dir.mkdir(parents=True)
    emb_csv = str(emb_dir / "unseen_fused.csv")
    df.to_csv(emb_csv, index=False)

    # Run analysis
    import subprocess, sys
    out_dir = str(tmp_path / "artifacts")
    cfg_path = os.path.join(os.getcwd(), "configs", "analysis.yaml")
    cmd = [sys.executable, os.path.join("analysis", "tsne_and_metrics.py"), "--emb", emb_csv, "--out_dir", out_dir, "--config", cfg_path]
    subprocess.check_call(cmd)

    # Check outputs exist
    assert os.path.exists(os.path.join(out_dir, "metrics", "tsne_grid_metrics.csv"))
    # Find a coords file
    coords_files = [f for f in os.listdir(os.path.join(out_dir, "embeddings")) if f.startswith("tsne_unseen_fused")]
    assert len(coords_files) >= 1


