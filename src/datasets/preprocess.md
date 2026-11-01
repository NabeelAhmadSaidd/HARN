The processed dataset is a chronologically split, normalized, event-driven multi-timeframe dataset prepared for HTSN. Each sample is anchored at a completed **M15** candle and contains:

* **Input sequences**

  * M15: **32** historical OHLCV candles
  * H1: **16** historical OHLCV candles
  * H4: **8** historical OHLCV candles
* **Prediction targets**

  * Next M15 close
  * Next H1 close
  * Next H4 close
* **Metadata**

  * Anchor timestamp
  * Target timestamps for each timeframe
  * `is_h1_update` and `is_h4_update` event flags
* **Normalization**

  * Z-score normalization using statistics computed **only from the training split** to prevent data leakage.
* **Data splits**

  * Chronological Train (70%), Validation (15%), and Test (15%) splits, saved in both **Pickle (.pkl)** and **CSV (.csv)** formats.
