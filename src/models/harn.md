# Hierarchical Associative Resonance Network (HARN)

## Overview

The **Hierarchical Associative Resonance Network (HARN)** is a causal deep learning architecture designed for **online multi-timeframe financial forecasting**. The model processes streaming market data across multiple temporal resolutions (e.g., M15, H1, H4) using hierarchical memory, cross-scale communication, and event-driven state updates.

Unlike conventional sequence models that process each timeframe independently, HARN maintains persistent hidden states and associative memories that evolve continuously as new market data arrives, enabling efficient real-time forecasting.

---

## Architecture

HARN consists of five sequential components:

### 1. Causal Multi-Scale Encoder
Each timeframe is encoded independently using:

- Raw OHLCV projection
- Derived candlestick features
- Multi-branch gated dilated convolutions
- Learnable branch fusion
- Exponential decay temporal pooling
- Residual feature aggregation

This produces a compact latent representation for each hierarchy level.

---

### 2. Surprise-Gated Associative Memory

Each hierarchy maintains a persistent matrix-valued memory that stores long-term market information.

Memory updates are driven by prediction surprise:

- associative key-value retrieval
- bounded memory writes
- gated residual state updates
- persistent streaming memory

This enables continual adaptation without storing complete historical sequences.

---

### 3. Cross-Scale Resonance

Latent representations from all hierarchy levels interact through a shared low-rank attention mechanism.

This module allows information exchange between fast and slow market dynamics while preserving each timeframe's independent representation.

---

### 4. Bottom-Up Evidence Aggregation

Higher timeframes receive summarized evidence from recent lower-timeframe hidden states.

A lightweight attention mechanism aggregates historical information, allowing coarse resolutions to incorporate fine-grained market behavior.

---

### 5. Forecasting Head

Each hierarchy level predicts the future price movement as a stationary basis-point change relative to the latest observed close.

Absolute price forecasts are reconstructed from the predicted relative change and the most recent market price.

---

## Key Features

- Fully causal architecture
- Online streaming inference
- Event-driven hierarchical updates
- Persistent associative memory
- Cross-timeframe information sharing
- Multi-scale temporal feature extraction
- End-to-end differentiable training
- Supports arbitrary hierarchy depths

---

## Training

The accompanying `train_harn.py` script provides a complete end-to-end training pipeline including:

- Dynamic multi-timeframe dataset loading
- Stateful streaming training
- Truncated Backpropagation Through Time (TBPTT)
- Mixed precision training
- AdamW optimizer
- Cosine annealing learning rate scheduling
- Early stopping
- Automatic checkpointing
- Validation and test evaluation

The training pipeline automatically adapts to any hierarchy specified through the configured timeframe list.

---

## Default Configuration

| Parameter | Value |
|-----------|------:|
| Timeframes | M15, H1, H4 |
| Input Features | OHLCV |
| Hidden Dimension | 72 |
| Memory Key Dimension | 18 |
| Coupling Rank | 36 |
| Dilations | 1, 2, 4, 8 |
| History Length | 4 |
| Dropout | 0.1 |

---

## Output

For each hierarchy level, HARN predicts:

- Stationary price change (basis points)
- Reconstructed absolute future price
- Baseline reference price

These predictions are generated simultaneously while maintaining persistent hierarchical states across streaming updates.

---

## Applications

HARN is suitable for:

- Online financial forecasting
- Multi-timeframe price prediction
- Algorithmic trading research
- Streaming time-series modeling
- Hierarchical sequence learning
- Adaptive market representation learning

---

## Model Characteristics

- **Architecture:** Hierarchical Memory Network
- **Learning Paradigm:** Online Sequential Learning
- **Inference:** Streaming
- **Memory:** Persistent Associative Memory
- **Temporal Modeling:** Multi-Scale Causal Convolutions
- **Cross-Level Interaction:** Low-Rank Attention Resonance
- **Prediction Target:** Stationary Price Change