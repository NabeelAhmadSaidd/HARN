Legend:
  [seed]      = best_model.pt, last_model.pt, training_history.csv
  [ablation]  = best_model.pt, last_model.pt, training_history.csv
  [processed] = test.csv, test.pkl, train.csv, train.pkl, val.csv, val.pkl

HARN/
├── project-structure.md
├── checkpoints/
│   ├── Baselines/
│   │   ├── PatchTST/
│   │   │   ├── AAPL/
│   │   │   │   ├── 151/   # [seed]
│   │   │   │   ├── 359/   # [seed]
│   │   │   │   └── 42/    # [seed]
│   │   │   ├── EURUSD/
│   │   │   │   ├── 151/   # [seed]
│   │   │   │   ├── 359/   # [seed]
│   │   │   │   └── 42/    # [seed]
│   │   │   ├── USDCHF/
│   │   │   │   ├── 151/   # [seed]
│   │   │   │   ├── 359/   # [seed]
│   │   │   │   └── 42/    # [seed]
│   │   │   └── XAUUSD/
│   │   │       ├── 151/   # [seed]
│   │   │       ├── 359/   # [seed]
│   │   │       └── 42/    # [seed]
│   │   └── TimeXer/
│   │       ├── AAPL/
│   │       │   ├── 151/   # [seed]
│   │       │   ├── 359/   # [seed]
│   │       │   └── 42/    # [seed]
│   │       ├── EURUSD/
│   │       │   ├── 151/   # [seed]
│   │       │   ├── 359/   # [seed]
│   │       │   └── 42/    # [seed]
│   │       ├── USDCHF/
│   │       │   ├── 151/   # [seed]
│   │       │   ├── 359/   # [seed]
│   │       │   └── 42/    # [seed]
│   │       └── XAUUSD/
│   │           ├── 151/   # [seed]
│   │           ├── 359/   # [seed]
│   │           └── 42/    # [seed]
│   ├── HARN/
│   │   ├── AAPL/
│   │   │   ├── 151/       # [seed]
│   │   │   ├── 359/       # [seed]
│   │   │   └── 42/        # [seed]
│   │   ├── EURUSD/
│   │   │   ├── 151/       # [seed]
│   │   │   ├── 359/       # [seed]
│   │   │   └── 42/        # [seed]
│   │   ├── USDCHF/
│   │   │   ├── 151/       # [seed]
│   │   │   ├── 359/       # [seed]
│   │   │   └── 42/        # [seed]
│   │   └── XAUUSD/
│   │       ├── 151/       # [seed]
│   │       ├── 359/       # [seed]
│   │       └── 42/        # [seed]
│   └── HARN_Ablations/
│       ├── AAPL/
│       │   ├── A0/        # [ablation]
│       │   ├── A1/        # [ablation]
│       │   ├── A2/        # [ablation]
│       │   └── A3/        # [ablation]
│       └── EURUSD/
│           ├── A0/        # [ablation]
│           ├── A1/        # [ablation]
│           ├── A2/        # [ablation]
│           └── A3/        # [ablation]
├── data/
│   ├── processed/
│   │   ├── AAPL/          # [processed]
│   │   ├── EURUSD/        # [processed]
│   │   ├── USDCHF/        # [processed]
│   │   └── XAUUSD/        # [processed]
│   └── raw/
│       ├── AAPL/
│       │   ├── AAPL_H1.csv
│       │   └── AAPL_M15.csv
│       ├── EURUSD/
│       │   ├── EURUSD_H1.csv
│       │   ├── EURUSD_H4.csv
│       │   └── EURUSD_M15.csv
│       ├── USDCHF/
│       │   ├── USDCHF_H1.csv
│       │   ├── USDCHF_M15.csv
│       │   └── USDCHF_M30.csv
│       └── XAUUSD/
│           ├── XAUUSD_M15.csv
│           └── XAUUSD_M5.csv
├── evaluation/
│   ├── baselines/
│   │   ├── baseline_metrics_aggregated.csv
│   │   └── baseline_metrics_per_seed.csv
│   ├── harn/
│   │   ├── harn_metrics_aggregated.csv
│   │   └── harn_metrics_per_seed.csv
│   └── harn_ablation/
│       └── harn_ablation_results.csv
├── notebooks/
│   ├── harn_evaluation.ipynb
│   ├── preprocessing_validation.ipynb
│   └── sample_visualization_and_alignment_debugging.ipynb
├── paper/
│   ├── forecast_error_distributions.png
│   ├── harn_architecture_overview.png
│   ├── harn_associative_memory.png
│   ├── harn_causal_multiscale_encoder.png
│   ├── harn_event_timeline.png
│   ├── harn_resonance_and_evidence.png
│   ├── man.aux
│   ├── man.fdb_latexmk
│   ├── man.fls
│   ├── man.log
│   ├── man.out
│   ├── man.pdf
│   ├── man.synctex.gz
│   ├── man.tex
│   ├── relative_component_degradation.png
│   └── representative_forecasts.png
└── src/
    ├── __init__.py
    ├── ablation/
    │   ├── ablations.md
    │   ├── ablations.png
    │   ├── evaluate_harn_ablations.py
    │   ├── harn_ablations.py
    │   └── train_harn_ablations.py
    ├── datasets/
    │   ├── preprocess.md
    │   ├── preprocessing.py
    │   └── timeframes.py
    └── models/
        ├── evaluate_harn.py
        ├── harn.md
        ├── HARN.png
        ├── harn.py
        ├── train_harn.py
        └── baselines/
            ├── base.py
            ├── evaluate_baselines.py
            ├── PatchTST.py
            ├── TimeXer.py
            ├── train_baselines.py
            ├── __init__.py
            └── layers/
                ├── Embed.py
                ├── PatchTST_backbone.py
                ├── PatchTST_layers.py
                ├── RevIN.py
                ├── SelfAttention_Family.py
                └── utils.py