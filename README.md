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

Excellent — here’s an expanded and refined section you can add to your `README.md` that not only previews the data but also explains the **importance** of each processing step and the **intermediate objects** created (`series`, `covariates`, `future_covariates`). This helps users **understand what’s happening, why it’s necessary**, and **how it helps the model**.

---

## 🔧 Data Processing – Why It Matters

TFT requires **structured, complete, and aligned** time series inputs. Simply throwing raw CSV data at it won’t work. Proper preprocessing ensures:
- ✅ No duplicate timestamps (which would break temporal order)
- ✅ All timestamps are aligned (missing hours are filled)
- ✅ Past and future features are formatted correctly
- ✅ Exogenous features (covariates) are available where needed

This section highlights how each transformation feeds into the model and **why it's essential**.

---

## 📥 Processing Flow and Intermediate Objects

### Step-by-step Breakdown with Output

#### 📌 Step 1: Load and clean the main data

```python
df = pd.read_csv("/kaggle/input/dummydata/electricity.csv", index_col='ds', parse_dates=True)
df = df[~df.index.duplicated(keep='first')]
```

- **Purpose**: Ensures time index is clean, with unique, ordered timestamps.
- Without this, Darts' resampling (`fill_missing_dates=True`) would fail.

---

#### 📌 Step 2: Create the main target `series`

```python
series = TimeSeries.from_dataframe(df, value_cols='y', fill_missing_dates=True, freq='h')
```

- **Object:** `series`
- **Contains:** The time series of electricity usage (`y`) as the target variable.
- **Why important?** This is what the model learns to predict.
- **Internally:** Looks like:

```
TimeSeries
start: 2016-10-22 00:00:00
end:   2016-12-30 23:00:00
data:
                            y
time                         
2016-10-22 00:00:00     70.00
2016-10-22 01:00:00     37.10
...                        ...
```

---

#### 📌 Step 3: Extract past exogenous features (covariates)

```python
X_past = df[['Exogenous1', 'Exogenous2']]
covariates = TimeSeries.from_dataframe(X_past, fill_missing_dates=True, freq='h')
```

- **Object:** `covariates`
- **Contains:** Features known in the past that may influence `y` (e.g., load in other regions).
- **Why important?** Helps the model learn correlations like:
  > “When `Exogenous1` goes up, `y` tends to increase.”

- **Internally:**

```
TimeSeries
columns: ['Exogenous1', 'Exogenous2']
data:
                            Exogenous1  Exogenous2
time                                            
2016-10-22 00:00:00        49593       57253
2016-10-22 01:00:00        46073       51887
...
```

---

#### 📌 Step 4: Prepare known future inputs

```python
future_df = pd.read_csv('/kaggle/input/dummydata/electricity-future.csv', index_col='ds', parse_dates=True)
future_df = future_df[~future_df.index.duplicated(keep='first')]
X_future = future_df[['Exogenous1', 'Exogenous2']]
```

- **Why important?**
  - You don’t know `y` for the future (that’s what you're predicting).
  - But you *do* know things like calendar info or scheduled events.
- These are essential for multi-step forecasting.

---

#### 📌 Step 5: Combine past and future into `future_covariates`

```python
X = pd.concat([X_past, X_future])
future_covariates = TimeSeries.from_dataframe(X, fill_missing_dates=True, freq='H')
```

- **Object:** `future_covariates`
- **Contains:** Covariates that are available from past *and* known in future.
- **Why important?**
  - TFT uses these for the decoder (to condition predictions).

- **Internally:**

```
TimeSeries
columns: ['Exogenous1', 'Exogenous2']
data:
                            Exogenous1  Exogenous2
time                                            
...                            ...         ...
2016-12-31 00:00:00            64108       70318
2016-12-31 01:00:00            62492       67898
...
```

---

## 🎯 Summary: Why These Objects Matter

| Object               | Purpose                                   | Used for        |
|----------------------|-------------------------------------------|------------------|
| `series`             | Main target values to predict (`y`)       | Training & eval  |
| `covariates`         | Past context features (Exogenous1/2)      | Encoder          |
| `future_covariates`  | Known future values (Exogenous1/2)        | Decoder          |

- Without these structures, TFT cannot:
  - Learn relationships between variables
  - Predict into the future
  - Handle multiple horizons properly

---

Would you like me to embed this directly into your README file for you?

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
