# Temporal Fusion Transformer (TFT) – An Intuitive Guide

## Overview

**Temporal Fusion Transformer (TFT)** is a deep learning model designed for **interpretable multi-horizon time series forecasting**. It outperforms traditional models and gives insights into **what**, **when**, and **why** a prediction was made.

This guide walks through:
- A clear breakdown of the TFT architecture
- A dry-run of how data flows through the model
- Example datasets and input formatting
- Role of exogenous variables
- Simplified explanation of key formulas from the research paper

---

## Architecture Explained Simply

TFT combines powerful sequence modeling (like LSTM and attention) with interpretability tools. It's structured like a forecasting control center:

### Key Components

| Component                    | What it Does                                                                 |
|-----------------------------|------------------------------------------------------------------------------|
| **Gating (GRN)**            | Turns off unnecessary parts of the network (like circuit breakers).         |
| **Variable Selection**      | Learns which variables are useful at each step.                             |
| **Static Covariate Encoder**| Adds information about each entity (store ID, patient, etc.) everywhere.   |
| **Sequence Encoder (LSTM)** | Captures recent patterns (e.g., last 4 days of demand).                     |
| **Multi-head Attention**    | Focuses on important time steps.                                            |
| **Quantile Regression**     | Predicts a range of possible future outcomes (p10, p50, p90).               |

---

## Data Flow (Dry Run Walkthrough)

### Let's Forecast Electricity Demand

You have:
- `y`: past electricity usage (target)
- `Exogenous1`, `Exogenous2`: extra features like temp or external load
- `hour`, `day`, `month`: known future features

```python
# Convert to Darts TimeSeries
series = TimeSeries.from_dataframe(df, value_cols='y')
future_covariates = TimeSeries.from_dataframe(df[['Exogenous1', 'Exogenous2', 'hour', 'day']])
```

1. **TFT Looks Back** 4 days (input_chunk_length=96): sees past `y`, `Exogenous1/2`, time features
2. **TFT Looks Forward** 1 day (output_chunk_length=24): uses known future exogenous values
3. **Learns What Matters**: dynamically selects which features to use
4. **Predicts Quantiles**: p10 (lower), p50 (median), p90 (upper)

---

## Dataset Format

| ds                  | y     | Exogenous1 | Exogenous2 | hour | day |
|---------------------|--------|------------|------------|------|-----|
| 2023-01-01 00:00:00 | 70     | 49593      | 57253      | 0    | 1   |
| 2023-01-01 01:00:00 | 65     | 48000      | 55000      | 1    | 1   |

### Exogenous Variables:
- Provide external context
- Help improve accuracy
- Must be known into the future (e.g., weather forecasts, holidays)

---

## Learning Objective (Formula Simplified)

From the paper:

```
ŷ(q, t, τ) = f_q(τ, y_{t-k:t}, z_{t-k:t}, x_{t-k:t+τ}, s)
```

Where:
- `y`: past target values
- `z`: past observed features (Exogenous1, etc.)
- `x`: known future inputs (e.g., hour, day)
- `s`: static info (store ID, etc.)
- `q`: quantile (like 10%, 50%, 90%)
- `τ`: forecast horizon (steps ahead)

> The model learns to estimate a range of likely future values based on all this information.

---

## Final Dry Run Example

```python
# Train best model after tuning
model = TFTModel(
    input_chunk_length=96,
    output_chunk_length=24,
    hidden_size=64,
    lstm_layers=2,
    dropout=0.1,
    num_attention_heads=4,
    batch_size=32,
    n_epochs=50,
)

model.fit(series, future_covariates=future_covariates)
forecast = model.predict(n=24, future_covariates=future_covariates)
```

---

## Quantile Loss Explained

```math
QL(y, ŷ, q) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)
```

- Penalizes over- and under-prediction differently
- Predicts ranges, not just point forecasts

---

## When to Use TFT

| Scenario                       | Is TFT a Good Fit? |
|-------------------------------|---------------------|
| Multi-step forecasting        | Yes                 |
| External/known future inputs  | Yes                 |
| Need for interpretability     | Yes                 |
| Irregular or missing data     | Not ideal           |

---

## References

> Bryan Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* - https://arxiv.org/abs/1912.09363
