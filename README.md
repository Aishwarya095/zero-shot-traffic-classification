# Zero-Shot Traffic Classification for Unseen Applications Using Convolutional Neural Network and Word2Vec

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Publication](https://img.shields.io/badge/Publication-IEEE-00629B.svg)](https://ieeexplore.ieee.org/)

> 📄 **IEEE Published Paper:** *Zero-Shot Traffic Classification for Unseen Applications Using Convolutional Neural Network and Word2Vec*  
> **Authors:** Aishwarya A, Prerna Awati, Nishchita N, Sonu Vaishnavi, Dr. Uday Kulkarni, Dr. Veena Hunasimarad  
> **Institution:** School of Computer Science and Engineering, KLE Technological University, Hubballi, India



## 📌 Executive Summary

With the rapid emergence of encrypted communication protocols—such as **TLS 1.3, QUIC, and VPN obfuscation**—traditional packet inspection methods (Deep Packet Inspection / Port-based matching) have become largely ineffective. Furthermore, standard supervised machine learning models operate under a **closed-world assumption**: they require labeled training data for every application class and fail when newly launched, unobserved network applications appear on the network.

This repository presents an end-to-end **Metric Learning and Zero-Shot Classification Framework** for encrypted network traffic classification. The framework integrates:
1. **Symbolic Feature Quantization:** Continuous flow statistics are discretized into symbolic categorical tokens to capture non-linear behavioral profiles.
2. **Word2Vec Semantic Embedding:** Flow token sequences are mapped into a dense continuous semantic vector space via Continuous Bag-of-Words (CBOW).
3. **1D-ResNet (Deep Residual CNN):** A lightweight 1-Dimensional ResNet projects semantic flow embeddings into a discriminative latent metric space.
4. **Prototypical Networks:** Class predictions are computed using geometric Euclidean distance to class prototype centroids, enabling few-shot and unseen application recognition.

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------------------------+
|                       ISCX VPN-nonVPN (Scenario-B)                            |
|             (23 Flow-level Statistical Features across 7 Classes)             |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
|                Step 1: Symbolic Feature Quantization                          |
|         e.g., duration -> [duration_low, duration_high], packets_very_high    |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
|                 Step 2: Word2Vec Continuous Embedding (CBOW)                  |
|          Token sequences -> Continuous dense 100-dimensional flow vectors     |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
|                  Step 3: 1D-ResNet Latent Space Projection                    |
|         Conv1D -> BatchNorm -> Residual Blocks (1D) -> 64-dim Latent Space    |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
|                     Step 4: Prototypical Metric Classifier                    |
|             c_k = (1 / |S_k|) ∑ f_θ(x_i)   [Class Centroids]                  |
|             Prediction: argmin_k || f_θ(x_query) - c_k ||_2                   |
+-------------------------------------------------------------------------------+
```

---

## 📊 Experimental Results

Evaluated on **10,782 cleaned network flow records** from the **ISCX VPN-nonVPN Scenario-B** dataset using an 80/20 train/test evaluation:

### Class-Wise Performance Breakdown

| Application Class | Category Role | Test Samples | Accuracy (%) | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **VOIP** | Unseen Target | 156 | **85.90%** | 0.89 | 0.86 | 0.88 |
| **BROWSING** | Unseen Target | 1000 | **78.50%** | 0.79 | 0.79 | 0.79 |
| **MAIL** | Seen Baseline | 181 | **65.19%** | 0.67 | 0.65 | 0.66 |
| **P2P** | Seen Baseline | 363 | **62.81%** | 0.60 | 0.63 | 0.61 |
| **FT (File Transfer)** | Seen Baseline | 268 | **34.70%** | 0.50 | 0.35 | 0.41 |
| **CHAT** | Seen Baseline | 118 | **33.05%** | 0.18 | 0.33 | 0.23 |
| **STREAMING** | Unseen Target | 71 | **32.39%** | 0.40 | 0.32 | 0.36 |
| **Overall Model** | **All 7 Classes** | **2,157** | **65.83%** | **0.67** | **0.66** | **0.66** |

> *Note on Research Formulation:* The 65.83% benchmark represents the end-to-end metric learning capability of Prototypical ResNet-1D across all application categories. For true zero-shot transfer (training strictly on Seen `CHAT, MAIL, FT, P2P`), the model provides geometric proximity mapping to discover which seen application profiles closely align with unobserved traffic flows.

---

## 📁 Repository Structure

```text
ML-Project/
├── README.md                      # Project documentation and benchmark report
├── requirements.txt               # Pinned dependencies for reproduction
├── .gitignore                     # Git ignore rules for datasets & checkpoints
├── notebooks/
│   └── zero_shot_traffic_prototypical_resnet.ipynb  # Interactive walkthrough notebook
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # ARFF loading, symbolic quantization & Word2Vec
│   ├── models.py                  # 1D-ResNet & Prototypical Network implementation
│   ├── train.py                   # CLI training and evaluation runner
│   └── evaluate.py                # Metric evaluation & Confusion Matrix generation
└── data/                          # Dataset folder (ARFF / ZIP archives)
    └── .gitkeep
```

---

## 🚀 Quickstart & Reproduction

### 1. Clone the Repository & Install Dependencies

```bash
git clone https://github.com/<your-username>/zero-shot-traffic-classification.git
cd zero-shot-traffic-classification

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the **ISCX VPN-nonVPN Scenario-B** dataset (`Scenario B-ARFF.zip` or `TimeBasedFeatures-Dataset-120s-AllinOne.arff`) from the [Canadian Institute for Cybersecurity (UNB)](https://www.unb.ca/cic/datasets/vpn.html).

Place the `.arff` or `.zip` file inside the `data/` directory:
```text
data/
└── Scenario B-ARFF/
    └── TimeBasedFeatures-Dataset-120s-AllinOne.arff
```

### 3. Run Training & Evaluation

#### Mode A: 7-Class Prototypical Metric Learning Benchmark (Replicating Paper Results)
```bash
python -m src.train --mode supervised --epochs 15 --batch_size 32
```

#### Mode B: Seen-to-Unseen Zero-Shot Geometric Proximity Mapping
```bash
python -m src.train --mode zero_shot --epochs 15 --batch_size 32
```

### 4. Interactive Jupyter Notebook
To run the full end-to-end pipeline interactively:
```bash
jupyter lab notebooks/zero_shot_traffic_prototypical_resnet.ipynb
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

