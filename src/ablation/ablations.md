# HARN Ablation Study

This directory contains the ablation implementation of the **Hierarchical Associative Resonance Network (HARN)** used to evaluate the contribution of its major architectural components.

## Variants

| Variant | Description |
|---------|-------------|
| **A0** | Full HARN (baseline) |
| **A1** | Removes the **Surprise-Gated Associative Memory** and replaces it with a simple recurrent state update. |
| **A2** | Removes **Cross-Scale Resonance**, disabling communication between hierarchy levels. |
| **A3** | Removes **Bottom-Up Evidence Aggregation**, preventing higher levels from receiving summarized lower-level information. |

## Files

- `harn_ablations.py` — HARN implementation with configurable ablation variants.
- `train_harn_ablations.py` — Training script for all ablation experiments.

## Training

Train the desired variant using:

```bash
python -m src.ablation.train_harn_ablations --variant A0
python -m src.ablation.train_harn_ablations --variant A1
python -m src.ablation.train_harn_ablations --variant A2
python -m src.ablation.train_harn_ablations --variant A3
```

## Checkpoints

Model checkpoints are saved automatically under:

```
checkpoints/HARN_Ablations/{ASSET} [EURUSD, AAPL]
├── A0/
├── A1/
├── A2/
└── A3/
```

Each directory contains the best and latest model checkpoints along with the corresponding training logs.

## Purpose

The ablation study isolates the contribution of each proposed component while keeping the dataset, preprocessing, optimizer, hyperparameters, and training protocol identical across all variants. This enables a fair evaluation of the importance of each architectural component.