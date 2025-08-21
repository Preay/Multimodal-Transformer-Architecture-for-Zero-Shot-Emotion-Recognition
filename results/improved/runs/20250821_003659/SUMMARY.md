{
  "best_val_macro_f1": 0.07085282667015638,
  "final_test": {
    "accuracy": 11.538461538461538,
    "macro_f1": 0.07054779376831827,
    "precision": 0.06477536008031819,
    "recall": 0.10026041666666666
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