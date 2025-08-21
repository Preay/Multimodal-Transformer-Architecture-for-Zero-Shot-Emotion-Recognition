{
  "best_val_macro_f1": 0.053301438429173065,
  "final_test": {
    "accuracy": 7.532051282051282,
    "macro_f1": 0.044413519878361486,
    "precision": 0.15732015680172362,
    "recall": 0.07682291666666667
  },
  "zero_shot": {
    "accuracy": null,
    "macro_f1": null
  },
  "fusion": "cross_attention",
  "hyperparams": {
    "lr": 0.0001,
    "batch_size": 32,
    "weight_decay": 0.0001,
    "use_contrastive": false
  },
  "notes": "Auto-generated summary. Actor-aware splits; centralized label parsing; AMP; early stopping by val macro-F1; weighted sampler (train only)."
}