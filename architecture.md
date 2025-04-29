# 🧠 Temporal Fusion Transformer (TFT) Architecture – Deep Dive

This section walks you through the full **TFT architecture** (based on the image above) using:

- 📊 Visual explanation (linked to components in the diagram)
- 📚 Plain language with **real-world examples**
- 🧮 Simplified **math formulas and intuitions**
- 🔍 Dummy data to **see what's happening**

---

## 📌 Overview of the Flow

![Image](https://github.com/user-attachments/assets/b757eda9-37da-410e-93cd-98861f9ede59)

The model is split into **three parts**:
1. **Input Encoding**
2. **Temporal Fusion Decoder**
3. **Forecasting Head (Quantile Predictions)**

It learns:
- What features matter (Feature Selection)
- When it should pay attention (Temporal Attention)
- How to interpret uncertainty (Quantile Regression)

---

## 1️⃣ Variable Selection Network

![Image](https://github.com/user-attachments/assets/b830ff1f-e073-4def-a168-88ced5461133)

### 🔍 What It Does
Learns which input variables to focus on **at each time step** — automatically.

### 🧮 Intuition (Simplified)

Let’s say you input:

```python
x_t = [temperature, humidity, holiday, load_zone1, load_zone2]
```

But for the hour `t = 12:00 PM`, only `temperature` and `load_zone1` really matter.

The Variable Selection Network learns **weights** like:

```
[0.8, 0.05, 0.05, 0.9, 0.05] → Softmax → attention-like scores
```

It **filters** irrelevant inputs dynamically.

### 🧪 Dummy Example

Imagine:

```
At 8 AM → electricity load depends more on: [day, holiday, load_zone1]
At 2 PM → it's more about: [temperature, humidity]
```

TFT learns this mapping automatically using:

```math
αₜ = Softmax(W₁ GRN₁(xₜ), ..., Wₙ GRNₙ(xₜ))
```

Where:
- Each variable is passed through a **Gated Residual Network**
- Their outputs are weighted and summed
- The softmax layer highlights the **most important variables**

---

## 2️⃣ Gated Residual Network (GRN)

![Image](https://github.com/user-attachments/assets/e5877b0a-fbef-4e39-b18b-d7209e318cf1)

### 🔍 What It Does
A smart block that decides how much of the transformed signal to **pass through or suppress**.

### 🧠 Analogy

Imagine a **valve in a pipe**: water (information) flows in. The GRN learns **how much to open or close the valve** depending on whether that information is useful.

### 🧮 Formula (Simplified)

```math
GRN(x) = Gate(x) ⊙ (LayerNorm(Dense₂(ELU(Dense₁(x)))))
```

- `Dense₁`, `Dense₂` = Linear transformations
- `ELU` = Activation function (non-linear twist)
- `Gate(x)` = Learnable sigmoid-based switch (outputs 0–1)

### 🧪 Dummy Example

Input = `temperature = 35°C`

If temperature isn’t relevant now, the GRN might learn:

```
Gate(temperature) = 0.1 → suppress
```

If temperature is very predictive:

```
Gate(temperature) = 0.9 → amplify
```

---

## 3️⃣ Static Covariate Encoders

![Image](https://github.com/user-attachments/assets/0e1f2468-9c63-4d62-8c5d-50991654e0a4)

### 🔍 What It Does
Takes features that don’t change over time — like store ID, region, or type — and injects them into the entire model.

- Helps personalize the model across entities.
- For example: *“Region A always peaks at 6 PM, Region B at 8 PM”*

---

## 4️⃣ LSTM Encoder-Decoder

![Image](https://github.com/user-attachments/assets/3916662c-61e0-44bc-bd63-14223c6f6dcd)

### 🔁 Purpose
These are **sequence models** that:
- **Encode past time steps**
- **Decode future known inputs** to predict the target

### 🧠 Think of it like:
> LSTM Encoder: “Here's what happened in the last 4 days.”  
> LSTM Decoder: “Given that and what's planned (e.g., calendar), here’s what might happen tomorrow.”

---

## 5️⃣ Static Enrichment

![Image](https://github.com/user-attachments/assets/b470c018-3610-49c2-9946-8c84da7d4e95)

- Combines static features with each temporal step.
- Ensures that **personalization** affects both past and future modeling.

---

## 6️⃣ Temporal Self-Attention (Masked Multi-head)

![Image](https://github.com/user-attachments/assets/4d15acf7-115b-4991-a7fe-0149b6361807)

### 🔍 What It Does
Let’s the model **attend to important past steps** across the time sequence.

### 🧠 Example

Imagine predicting electricity usage for `t+1`.

- Attention might find:
  - `t-1` → useful (recent trend)
  - `t-24` → useful (same hour yesterday)
  - `t-168` → very useful (same hour last week)

### 🧮 Attention Score

For each time step `t`:
```math
AttentionScore(i) = Qₜ · Kᵢᵀ / sqrt(d_k)
```

- `Q`, `K` = query/key vectors from temporal inputs
- Masking ensures it doesn’t peek into the future

---

## 7️⃣ Position-wise Feed-Forward Layers

![image](https://github.com/user-attachments/assets/6f152b61-dedc-49a8-9279-c07f2f541c9a)

- Applies dense transformations to each time step
- Helps model **non-linear interactions** over time

---

## 8️⃣ Output Layer: Quantile Forecasts

![Image](https://github.com/user-attachments/assets/9afd876a-e313-4867-9b57-e9354395c0f7)

The model predicts a **range** instead of just a point.

### 🔍 Example

```
For t+1 → 10th percentile: 52.1
           50th percentile (median): 58.4
           90th percentile: 64.3
```

This gives you a **confidence band** rather than a single guess.

### 🧮 Quantile Loss Function

For quantile `q`:

```math
QL(y, ŷ, q) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)
```

- Penalizes underestimates more if `q` is high (e.g., p90)
- Helps model learn **risk-aware** forecasts

---

## 🧾 Full Dummy Forecast Example

Input:
- Past `y`: [50, 52, 55, 60, 58]
- Future Exogenous1 (temperature): [35, 36, 37]
- Future Exogenous2 (hour): [14, 15, 16]

Output:

| Time   | p10  | p50  | p90  |
|--------|------|------|------|
| t+1    | 55.2 | 58.4 | 62.7 |
| t+2    | 56.8 | 60.1 | 64.9 |
| t+3    | 57.5 | 61.2 | 65.5 |

---

## ✅ Summary: Why TFT Excels

| Feature                     | Benefit                                       |
|----------------------------|-----------------------------------------------|
| Variable Selection         | Learns *which inputs* matter when             |
| Gated Residual Networks    | Learns *how much* of each signal to use       |
| Temporal Attention         | Focuses on *important time steps*             |
| LSTM Encoder-Decoder       | Understands *short-term patterns*             |
| Static Enrichment          | Adapts to *individual series/entities*        |
| Quantile Forecasts         | Captures *uncertainty* in predictions         |

---
