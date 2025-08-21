## Multimodal Zero-Shot Emotion Recognition (Z-SER)

End-to-end Python/PyTorch pipeline for Multimodal Zero‑Shot Emotion Recognition on RAVDESS, with unified feature extraction, actor‑aware splits, reproducible training, and evaluation/visualization.

### 0) Environment

Windows PowerShell example:
```bash
python -m venv zser_env
zser_env\Scripts\activate
pip install -r requirements.txt
```
Linux/macOS:
```bash
python -m venv zser_env
source zser_env/bin/activate
pip install -r requirements.txt
```

### 1) Dataset and paths

Expected layout:
```
data/ravdess/
  audio/{audio_speech_actors_01-24, audio_song_actors_01-24}/Actor_XX/*.wav
  vision/{video_speech_actors_01-24, video_song_actors_01-24}/Actor_XX/*.mp4
```

Provide a config at `configs/data_paths.json` (edit as needed):
```json
{
  "data_root": "data/ravdess",
  "audio_roots": [
    "data/ravdess/audio/audio_speech_actors_01-24",
    "data/ravdess/audio/audio_song_actors_01-24"
  ],
  "video_roots": [
    "data/ravdess/vision/video_speech_actors_01-24",
    "data/ravdess/vision/video_song_actors_01-24"
  ]
}
```
If a key is missing, the extractor falls back to sensible defaults based on `data_root`.

### 2) Feature extraction (idempotent)

Extract audio (Wav2Vec2) and vision (ViT) features. Files are named by the WAV stem and saved as `.pt` tensors.
```bash
python scripts/extract_features.py \
  --dataset_name ravdess \
  --data_config configs/data_paths.json \
  --modality both \
  --save_dir results/improved/features \
  --device cuda
```
Useful flags:
- `--audio_backbone {wav2vec2_base, distilhubert}`
- `--vision_backbone {vit_base, vit_tiny, mobilevit}`
- `--force` re-extracts even if files already exist

Outputs:
- `results/improved/features/audio/<stem>.pt`
- `results/improved/features/vision/<stem>.pt`

### 3) Train (multimodal)

Train with actor‑aware 60/20/20 splits, early stopping on validation macro‑F1, AMP, and ReduceLROnPlateau. The alias `--fusion cross_attn` maps to `--fusion_type cross_attention`.
```bash
python scripts/train_model.py \
  --dataset_name ravdess \
  --data_config configs/data_paths.json \
  --features_dir results/improved/features \
  --mode multimodal \
  --fusion cross_attn \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --seed 42 \
  --save_dir results/improved/runs \
  --amp \
  --balanced_ce \
  --use_weighted_sampler
```

Training artifacts (under `results/improved/runs/<timestamp>/`):
- `best.ckpt` (weights)
- `run_meta.json` (args, seed, environment, CUDA, git commit)
- `train_log.csv` (epoch, loss, val metrics)
- `val_metrics.json` and `val_metrics.csv`
- `test_metrics.json` and `test_metrics.csv`
- `model_card.md`
- `SUMMARY.md` (also printed to console)

### 4) Evaluate best checkpoint

Standard evaluation + zero-shot prototype test; writes confusion matrix and t‑SNE figures.
```bash
python scripts/evaluate.py \
  --dataset_name ravdess \
  --data_config configs/data_paths.json \
  --features_dir results/improved/features \
  --mode multimodal \
  --fusion_type cross_attention \
  --checkpoint results/improved/runs/<timestamp>/best.ckpt \
  --batch_size 32 \
  --save_dir results/improved/eval_best \
  --normalize_z
```

Evaluation artifacts (under `results/improved/eval_best/`):
- `metrics.json` (ZSL accuracy/macro‑F1 and GZSL breakdown)
- `confusion_matrix_ravdess.png`
- `tsne_seen.png`, `tsne_unseen.png`
- `explanations/sample_*_similarities.png` (prototype similarity bars)

### 5) Lightweight hyperparameter sweep (optional)

Runs a small, reproducible grid and evaluates the best run automatically.
```bash
python scripts/small_sweep.py \
  --dataset_name ravdess \
  --data_config configs/data_paths.json \
  --features_dir results/improved/features \
  --save_dir results/improved/runs \
  --eval_dir results/improved/eval_best \
  --seed 42 \
  --epochs 30 \
  --amp
```

Grid (by default):
- fusion in {cross_attn, mlp, late, early}
- lr in {1e-3, 5e-4, 1e-4}
- batch_size in {16, 32}
- weight_decay in {0, 1e-4}
- contrastive on/off

### 6) Zero-shot protocol

Class‑held‑out zero‑shot can be enabled by specifying held‑out emotion names during training/evaluation:
```bash
# train with held-out classes (example)
python scripts/train_model.py ... --held_out fearful disgust

# evaluate using the same held-out list
python scripts/evaluate.py ... --held_out fearful disgust --use_zero_shot
```
Cosine‑similarity classification uses L2‑normalized embeddings and class centroids computed from the training (seen) split.

### 7) Where things are saved

- Features: `results/improved/features/{audio,vision}/*.pt`
- Training run (timestamped): `results/improved/runs/<timestamp>/`
  - `best.ckpt`, `run_meta.json`, `train_log.csv`, `val_metrics.*`, `test_metrics.*`, `model_card.md`, `SUMMARY.md`
- Evaluation: `results/improved/eval_best/`
  - `metrics.json`, `confusion_matrix_ravdess.png`, `tsne_seen.png`, `tsne_unseen.png`, `explanations/*`

### 8) Reproducibility

- Fixed seeds for `random`, `numpy`, and `torch`; CuDNN deterministic on when available.
- All scripts accept `--seed` and log the used seed in `run_meta.json`.
- Actor‑aware stratified splits (no actor leakage between train/val/test) with fixed seed.

### 9) Troubleshooting

- Windows console encoding: logs avoid non‑ASCII characters.
- Vision pairing: videos are resolved from audio stems across the configured `video_roots`. Missing matches are logged and saved as zero vectors.
- Face detection: if no face is detected in the middle frame, a zero vector is saved (logged as "No face").
- Performance: ensure features exist for both modalities; prefer `--amp` and `--balanced_ce`, and consider running `scripts/small_sweep.py` for a quick hyper‑param search.

### 10) CLI help

```bash
python scripts/extract_features.py --help
python scripts/train_model.py --help
python scripts/evaluate.py --help
```

### Citation
If you use this code, please cite the RAVDESS dataset and relevant backbone models (Wav2Vec2, ViT).