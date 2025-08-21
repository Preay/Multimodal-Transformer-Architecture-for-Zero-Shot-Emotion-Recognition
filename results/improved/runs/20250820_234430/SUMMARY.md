{
  "best_val_macro_f1": 0.052892114929804265,
  "final_test": {
    "accuracy": 13.301282051282051,
    "macro_f1": 0.05702389125107036,
    "precision": 0.038296365861032126,
    "recall": 0.12630208333333334
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