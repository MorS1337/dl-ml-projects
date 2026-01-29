# Deep Learning Portfolio


This repository contains my projects implementing core Neural Network architectures **from scratch** to understand the mathematics behind SOTA models (Transformers, RNNs), as well as solutions for competitive Data Science tasks.

## 🛠 Tech Stack & Tools
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-Math-013243?style=for-the-badge&logo=numpy)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FL402C?style=for-the-badge&logo=xgboost)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas)

---

## 📂 Featured Projects

### 1. 🪢 Transformer Components & RoPE Implementation (From Scratch)
**File:** `attention.ipynb`

A deep dive into the architecture of Modern LLMs (LLaMA-like). Instead of using high-level `nn.Transformer` layers, I implemented the core mechanisms manually using PyTorch and linear algebra.

* **Key Implementations:**
    * **Multi-Head Attention (MHA):** Manual implementation of Query, Key, Value projections and scaled dot-product attention.
    * **Rotary Positional Embeddings (RoPE):** Implemented complex rotation logic (as seen in PaLM/LLaMA) and compared it with classic Sinusoidal embeddings.
    * **Custom Training Loop:** Built a trainer for a character-level Language Model trained on the "Tiny Shakespeare" dataset.

**📉 Results:**
* Demonstrated that RoPE provides faster convergence compared to absolute sinusoidal embeddings.
* **Metric:** Optimized Cross-Entropy Loss and Perplexity on validation set.

---

### 2. 🗣️ Neural Machine Translation with Bahdanau Attention
**File:** `seq2seq_rnn.ipynb`

Implementation of a Sequence-to-Sequence (Seq2Seq) model for text normalization/translation tasks.

* **Architecture:** Encoder-Decoder based on **Bi-Directional GRU**.
* **Attention Mechanism:** Implemented **Bahdanau (Additive) Attention** to handle long-term dependencies and solve the bottleneck problem.
* **Inference:** Implemented **Beam Search** algorithm (width=5) to improve generation quality over Greedy Decoding.

**TL;DR Metrics:**
* **Exact Match (Greedy):** ~99.08%
* **Exact Match (Beam Search):** ~99.10%

---

### 3. 🏃‍♂️ Activity Recognition from Wearable Sensors (Kaggle)
**File:** `aikc-idas-kaggle.ipynb`

A solution for classifying human physical activities (Walking, Jogging, Sitting, etc.) based on accelerometer and gyroscope data.

* **Approach:** Hybrid pipeline combining Feature Engineering and Gradient Boosting.
* **Feature Engineering:**
    * **Spectral Analysis:** Fast Fourier Transform (FFT) features (Dominant frequency, Energy).
    * **Statistical Features:** Rolling mean, std, quantiles, and signal jerk (derivative of acceleration).
* **Model:** **XGBoost** Classifier with class weights to handle imbalance.
* **Validation:** Strict **GroupKFold** strategy (grouped by `subject_id`) to prevent data leakage between users.

**📊 Result:**
* Achieved high **F1-Macro** score on the leaderboard.

---

### 4. 🧮 Custom TF-IDF Vectorizer & Logistic Regression
**File:** `tf_idf.ipynb`

Recreating classic NLP tools to understand the math behind `sklearn`.

* Implemented **TF-IDF** vectorizer using only `NumPy` (including n-gram generation, IDF smoothing, and L2 normalization).
* Compared performance with `sklearn.feature_extraction.text.TfidfVectorizer` on the AG News dataset.
* **Result:** My custom implementation achieved identical Accuracy (**~90-91%**) to the library version, proving correctness of the mathematical logic.

---
