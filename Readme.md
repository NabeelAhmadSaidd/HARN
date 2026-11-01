# HARN

**Hierarchical Associative Resonance for Event-Driven Multi-Timeframe Forecasting**

HARN is a research implementation of an event-driven, multi-timeframe forecasting architecture. The project includes the HARN model, PatchTST and TimeXer baselines, multi-timeframe preprocessing, controlled ablations, evaluation scripts, notebooks, and the manuscript artifacts used to document the experiments.

The repository is organized to keep **raw data, processed datasets, model checkpoints, evaluation results, notebooks, source code, and paper artifacts** separate.

---

## Repository Structure

```text
HARN/
├── project-structure.md
│
├── checkpoints/
│   ├── Baselines/
│   │   ├── PatchTST/
│   │   │   ├── AAPL/
│   │   │   │   ├── 151/
│   │   │   │   ├── 359/
│   │   │   │   └── 42/
│   │   │   ├── EURUSD/
│   │   │   │   ├── 151/
│   │   │   │   ├── 359/
│   │   │   │   └── 42/
│   │   │   ├── USDCHF/
│   │   │   │   ├── 151/
│   │   │   │   ├── 359/
│   │   │   │   └── 42/
│   │   │   └── XAUUSD/
│   │   │       ├── 151/
│   │   │       ├── 359/
│   │   │       └── 42/
│   │   │
│   │   └── TimeXer/
│   │       ├── AAPL/
│   │       ├── EURUSD/
│   │       ├── USDCHF/
│   │       └── XAUUSD/
│   │
│   ├── HARN/
│   │   ├── AAPL/
│   │   │   ├── 151/
│   │   │   ├── 359/
│   │   │   └── 42/
│   │   ├── EURUSD/
│   │   │   ├── 151/
│   │   │   ├── 359/
│   │   │   └── 42/
│   │   ├── USDCHF/
│   │   │   ├── 151/
│   │   │   ├── 359/
│   │   │   └── 42/
│   │   └── XAUUSD/
│   │       ├── 151/
│   │       ├── 359/
│   │       └── 42/
│   │
│   └── HARN_Ablations/
│       ├── AAPL/
│       │   ├── A0/
│       │   ├── A1/
│       │   ├── A2/
│       │   └── A3/
│       └── EURUSD/
│           ├── A0/
│           ├── A1/
│           ├── A2/
│           └── A3/
│
├── data/
│   ├── processed/
│   │   ├── AAPL/
│   │   ├── EURUSD/
│   │   ├── USDCHF/
│   │   └── XAUUSD/
│   │
│   └── raw/
│       ├── AAPL/
│       ├── EURUSD/
│       ├── USDCHF/
│       └── XAUUSD/
│
├── evaluation/
│   ├── baselines/
│   ├── harn/
│   └── harn_ablation/
│
├── notebooks/
│   ├── harn_evaluation.ipynb
│   ├── preprocessing_validation.ipynb
│   └── sample_visualization_and_alignment_debugging.ipynb
│
├── paper/
│   ├── man.tex
│   ├── man.pdf
│   └── figures and LaTeX build artifacts
│
└── src/
    ├── ablation/
    ├── datasets/
    └── models/
```

---

## Checkpoint and Dataset Legend

The repository uses the following conventions.

| Label         | Contents                                                               |
| ------------- | ---------------------------------------------------------------------- |
| `[seed]`      | `best_model.pt`, `last_model.pt`, `training_history.csv`               |
| `[ablation]`  | `best_model.pt`, `last_model.pt`, `training_history.csv`               |
| `[processed]` | `test.csv`, `test.pkl`, `train.csv`, `train.pkl`, `val.csv`, `val.pkl` |

### Seeds

The primary experiments use three random seeds:

```text
42
151
359
```

Seed-specific checkpoints are stored independently so that individual training runs can be inspected or reproduced.

For example:

```text
checkpoints/HARN/EURUSD/42/
├── best_model.pt
├── last_model.pt
└── training_history.csv
```

The same structure is used for PatchTST and TimeXer.

---

# 1. Project Components

## HARN

The main model is implemented in:

```text
src/models/harn.py
```

The implementation contains the main HARN architecture, including its multi-timeframe processing and associative/resonance components.

Supporting documentation and visualization are located at:

```text
src/models/harn.md
src/models/HARN.png
```

## Baselines

The repository contains two baseline architectures:

* PatchTST
* TimeXer

Their implementations are located under:

```text
src/models/baselines/
```

Relevant files include:

```text
PatchTST.py
TimeXer.py
train_baselines.py
evaluate_baselines.py
base.py
```

The baseline directory also contains the model layers required by the implementations.

---

# 2. Data

## Raw Data

Raw OHLCV data is stored under:

```text
data/raw/
```

The current repository contains:

```text
data/raw/
├── AAPL/
│   ├── AAPL_H1.csv
│   └── AAPL_M15.csv
│
├── EURUSD/
│   ├── EURUSD_H1.csv
│   ├── EURUSD_H4.csv
│   └── EURUSD_M15.csv
│
├── USDCHF/
│   ├── USDCHF_H1.csv
│   ├── USDCHF_M15.csv
│   └── USDCHF_M30.csv
│
└── XAUUSD/
    ├── XAUUSD_M15.csv
    └── XAUUSD_M5.csv
```

Raw files use the general naming convention:

```text
{SYMBOL}_{TIMEFRAME}.csv
```

The expected OHLCV fields are:

```text
datetime
open
high
low
close
volume
```

---

## Processed Data

Processed datasets are stored under:

```text
data/processed/
```

Each asset contains:

```text
train.csv
train.pkl

val.csv
val.pkl

test.csv
test.pkl
```

The preprocessing implementation is located at:

```text
src/datasets/preprocessing.py
```

Timeframe definitions are maintained in:

```text
src/datasets/timeframes.py
```

Additional preprocessing documentation is available in:

```text
src/datasets/preprocess.md
```

---

# 3. Multi-Timeframe Processing

HARN uses an event-driven multi-timeframe representation.

The preprocessing pipeline aligns observations according to completed bars rather than simply assuming that all timeframes update simultaneously.

For a given anchor event:

1. The anchor timeframe identifies the current event.
2. Each configured timeframe is aligned to its latest completed bar available at that event.
3. A timeframe is marked as updated only when its corresponding bar has completed.
4. The model receives both timeframe windows and update information.
5. The forecasting target is constructed from the subsequent completed observation according to the configured preprocessing pipeline.

This structure is intended to preserve the temporal ordering of information across different resolutions.

---

# 4. Preprocessing

The preprocessing entry point is:

```text
src/datasets/preprocessing.py
```

The module can be invoked as a Python module.

For a single asset:

```bash
python -m src.datasets.preprocessing --symbol USDCHF
```

For all configured assets:

```bash
python -m src.datasets.preprocessing --all
```

The resulting datasets are written to:

```text
data/processed/{SYMBOL}/
```

For example:

```text
data/processed/EURUSD/
├── train.csv
├── train.pkl
├── val.csv
├── val.pkl
├── test.csv
└── test.pkl
```

---

# 5. HARN Training

The HARN training entry point is:

```text
src/models/train_harn.py
```

Checkpoints are organized by asset and random seed:

```text
checkpoints/HARN/{ASSET}/{SEED}/
```

Each completed run produces:

```text
best_model.pt
last_model.pt
training_history.csv
```

For example:

```text
checkpoints/HARN/AAPL/42/
├── best_model.pt
├── last_model.pt
└── training_history.csv
```

The training configuration is defined in the training script and associated model implementation.

---

# 6. Baseline Training

PatchTST and TimeXer are trained through:

```text
src/models/baselines/train_baselines.py
```

The resulting checkpoints follow:

```text
checkpoints/Baselines/
├── PatchTST/
│   └── {ASSET}/{SEED}/
│
└── TimeXer/
    └── {ASSET}/{SEED}/
```

Each seed directory contains:

```text
best_model.pt
last_model.pt
training_history.csv
```

---

# 7. Evaluation

Evaluation scripts are separated from training.

## HARN

HARN evaluation is implemented in:

```text
src/models/evaluate_harn.py
```

Results are stored under:

```text
evaluation/harn/
```

with:

```text
harn_metrics_aggregated.csv
harn_metrics_per_seed.csv
```

## Baselines

Baseline evaluation is implemented in:

```text
src/models/baselines/evaluate_baselines.py
```

Results are stored under:

```text
evaluation/baselines/
```

with:

```text
baseline_metrics_aggregated.csv
baseline_metrics_per_seed.csv
```

Keeping per-seed and aggregated results separate allows the underlying runs to be inspected independently from summary statistics.

---

# 8. HARN Ablations

The ablation implementation is located in:

```text
src/ablation/
```

Main files:

```text
harn_ablations.py
train_harn_ablations.py
evaluate_harn_ablations.py
ablations.md
ablations.png
```

The repository contains four ablation variants:

```text
A0
A1
A2
A3
```

for:

```text
AAPL
EURUSD
```

The checkpoint structure is:

```text
checkpoints/HARN_Ablations/
├── AAPL/
│   ├── A0/
│   ├── A1/
│   ├── A2/
│   └── A3/
│
└── EURUSD/
    ├── A0/
    ├── A1/
    ├── A2/
    └── A3/
```

Each ablation directory contains:

```text
best_model.pt
last_model.pt
training_history.csv
```

Ablation evaluation produces:

```text
evaluation/harn_ablation/harn_ablation_results.csv
```

---

# 9. Notebooks

The repository includes three notebooks for inspection and analysis.

### HARN Evaluation

```text
notebooks/harn_evaluation.ipynb
```

Used for inspecting and analyzing HARN evaluation results.

### Preprocessing Validation

```text
notebooks/preprocessing_validation.ipynb
```

Used to inspect processed datasets and validate preprocessing behavior.

### Visualization and Alignment Debugging

```text
notebooks/sample_visualization_and_alignment_debugging.ipynb
```

Used for visual inspection of samples and debugging temporal alignment.

---

# 10. Evaluation Outputs

The `evaluation/` directory contains generated experiment results rather than model implementations.

```text
evaluation/
├── baselines/
│   ├── baseline_metrics_aggregated.csv
│   └── baseline_metrics_per_seed.csv
│
├── harn/
│   ├── harn_metrics_aggregated.csv
│   └── harn_metrics_per_seed.csv
│
└── harn_ablation/
    └── harn_ablation_results.csv
```

The distinction between **per-seed** and **aggregated** results is intentional:

* `*_per_seed.csv` preserves individual experimental runs.
* `*_aggregated.csv` contains the corresponding aggregated results.

---

# 11. Paper

The manuscript and associated figures are stored under:

```text
paper/
```

Main manuscript files:

```text
man.tex
man.pdf
```

The directory also contains figures used by the manuscript, including:

```text
forecast_error_distributions.png
harn_architecture_overview.png
harn_associative_memory.png
harn_causal_multiscale_encoder.png
harn_event_timeline.png
harn_resonance_and_evidence.png
relative_component_degradation.png
representative_forecasts.png
```

LaTeX build artifacts such as:

```text
man.aux
man.fdb_latexmk
man.fls
man.log
man.out
man.synctex.gz
```

may also be present.

---

# 12. Source Code Organization

The main source tree is:

```text
src/
├── __init__.py
│
├── ablation/
│   ├── ablations.md
│   ├── ablations.png
│   ├── evaluate_harn_ablations.py
│   ├── harn_ablations.py
│   └── train_harn_ablations.py
│
├── datasets/
│   ├── preprocess.md
│   ├── preprocessing.py
│   └── timeframes.py
│
└── models/
    ├── evaluate_harn.py
    ├── harn.md
    ├── HARN.png
    ├── harn.py
    ├── train_harn.py
    │
    └── baselines/
        ├── base.py
        ├── evaluate_baselines.py
        ├── PatchTST.py
        ├── TimeXer.py
        ├── train_baselines.py
        ├── __init__.py
        │
        └── layers/
            ├── Embed.py
            ├── PatchTST_backbone.py
            ├── PatchTST_layers.py
            ├── RevIN.py
            ├── SelfAttention_Family.py
            └── utils.py
```

---

# 13. Reproducibility Layout

The repository separates the experimental pipeline into four stages:

```text
Raw Data
   │
   ▼
Preprocessing
   │
   ▼
Processed Train / Validation / Test Data
   │
   ├───────────────┐
   ▼               ▼
HARN           Baselines
   │               │
   ▼               ▼
Checkpoints     Checkpoints
   │               │
   └───────┬───────┘
           ▼
       Evaluation
           │
           ▼
      CSV Results
           │
           ▼
      Paper / Analysis
```

This separation makes it possible to distinguish:

* data preparation,
* model training,
* checkpoint selection,
* evaluation,
* aggregation,
* and manuscript visualization.

---

# 14. Experimental Assets

The repository currently contains experimental results for:

```text
AAPL
EURUSD
USDCHF
XAUUSD
```

HARN and the baselines use the available processed datasets for each asset. Ablation experiments are organized separately under:

```text
checkpoints/HARN_Ablations/
```

and currently cover:

```text
AAPL
EURUSD
```

---

# 15. Random Seeds

The primary seed set is:

```text
42
151
359
```

Each seed is stored independently rather than overwriting another run.

For example:

```text
checkpoints/HARN/XAUUSD/
├── 42/
├── 151/
└── 359/
```

This allows seed-level results to be retained and subsequently aggregated.

---

# 16. Recommended Workflow

A typical experiment proceeds as follows:

### Step 1 — Prepare raw data

Place the required OHLCV files under:

```text
data/raw/{SYMBOL}/
```

using the expected filename convention:

```text
{SYMBOL}_{TIMEFRAME}.csv
```

### Step 2 — Generate processed datasets

Run the preprocessing pipeline:

```bash
python -m src.datasets.preprocessing --symbol SYMBOL
```

or process all configured assets:

```bash
python -m src.datasets.preprocessing --all
```

### Step 3 — Train HARN

Run:

```text
src/models/train_harn.py
```

and retain the resulting seed-specific checkpoints.

### Step 4 — Train baselines

Run:

```text
src/models/baselines/train_baselines.py
```

for the configured baseline models.

### Step 5 — Train ablations

Use:

```text
src/ablation/train_harn_ablations.py
```

for the configured ablation variants.

### Step 6 — Evaluate models

Run the corresponding HARN and baseline evaluation scripts.

### Step 7 — Evaluate ablations

Run:

```text
src/ablation/evaluate_harn_ablations.py
```

### Step 8 — Inspect results

Use the generated CSV files and notebooks:

```text
evaluation/
notebooks/
```

### Step 9 — Build or inspect the manuscript

The manuscript source is:

```text
paper/man.tex
```

and the compiled manuscript is:

```text
paper/man.pdf
```

---

# 17. Important Directory Conventions

The repository follows these conventions:

```text
data/raw/          Original input data
data/processed/    Preprocessed datasets
checkpoints/       Trained model checkpoints
evaluation/        Evaluation outputs
notebooks/         Interactive analysis and debugging
src/               Source code
paper/             Manuscript and figures
```

Generated outputs should remain separated from source code wherever possible.

---


# 18. Status

This repository contains the implementation, datasets in processed form, trained checkpoints, evaluation outputs, ablation artifacts, notebooks, and manuscript materials associated with the HARN experiments.

The repository is intended primarily for **research, experimentation, reproducibility, and academic evaluation**.
