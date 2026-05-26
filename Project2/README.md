# 🌸 Data Classification Using AI
### Project 2 — DecodeLabs Industrial Training Kit | Batch 2026

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=for-the-badge)

---

## 📌 Overview

This project is part of the **DecodeLabs AI Engineering Internship Track**. The goal is to build a complete supervised learning pipeline from scratch — loading raw data, preprocessing it, training a classification model, and validating results with professional metrics.

> *"We do not write the rules. We provide history, and the machine derives the logic."*

---

## 🎯 Project Goal

Build a **K-Nearest Neighbors (KNN) classifier** on the Iris dataset to predict flower species based on physical measurements — following the full IPO (Input → Process → Output) framework.

---

## 🧠 What You'll Learn

- The difference between **heuristic rules** vs **supervised learning**
- How to apply **feature scaling** (StandardScaler) and why it matters for KNN
- How to **split data** into training and testing sets properly
- How to **tune hyperparameter K** using the Elbow Method
- How to interpret a **Confusion Matrix**, **F1 Score**, and **Classification Report**
- Why **accuracy alone is misleading** on real-world data

---

## 🗂️ Project Structure

```
📦 iris-classification-knn/
├── 📓 iris_classification_knn.ipynb   # Main notebook (full pipeline)
├── 📄 README.md                        # This file
```

---

## 🔬 Dataset

**The Iris Benchmark** — one of the most iconic datasets in machine learning.

| Property    | Value                              |
|-------------|------------------------------------|
| Source      | [UCI ML / Kaggle](https://www.kaggle.com/datasets/uciml/iris) |
| Samples     | 150 (balanced)                     |
| Classes     | 3 (Setosa, Versicolor, Virginica)  |
| Features    | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| Missing Data| None                               |

---

## ⚙️ Pipeline — The IPO Framework

```
INPUT                   PROCESS                  OUTPUT
─────────────────       ─────────────────────    ──────────────────────
Iris Dataset        →   Train-Test Split (80/20)  →   Confusion Matrix
Feature Scaling         KNN Algorithm (K tuned)       F1 Score
                        Elbow Method for K            Classification Report
```

### Steps at a Glance

| # | Step | Description |
|---|------|-------------|
| 1 | Import Libraries | numpy, pandas, matplotlib, seaborn, sklearn |
| 2 | Load Dataset | Iris via sklearn (identical to Kaggle UCI) |
| 3 | EDA | Pairplots, heatmaps, boxplots |
| 4 | Feature Scaling | StandardScaler → mean=0, variance=1 |
| 5 | Train-Test Split | 80/20 stratified split, shuffled |
| 6 | Tune K | Elbow Method across K=1 to K=30 |
| 7 | Train Model | KNeighborsClassifier.fit() |
| 8 | Evaluate | Confusion Matrix + F1 Score |
| 9 | Decision Boundary | 2D visualization of KNN regions |
| 10 | Predict | Run on new unseen flower measurements |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/iris-classification-knn.git
cd iris-classification-knn
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

### 3. Launch the Notebook

```bash
jupyter notebook iris_classification_knn.ipynb
```

> **Alternative:** Upload directly to [Google Colab](https://colab.research.google.com) or [Kaggle Notebooks](https://www.kaggle.com/code) — no local setup needed.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~96–100% |
| F1 Score (weighted) | ~0.96–1.00 |
| Optimal K | Found via Elbow Method |

> *Note: Exact scores may vary slightly depending on the random state.*

### Sample Confusion Matrix Output

```
               setosa  versicolor  virginica
setosa            10           0          0
versicolor         0           9          1
virginica          0           0         10
```

---

## 🧩 Key Concepts Explained

### Why Feature Scaling?
KNN uses **Euclidean distance** to find neighbors. Without scaling, a feature measured in large units (e.g., 0–10 cm) will dominate a feature in small units (e.g., 0–1 cm), creating biased results. `StandardScaler` eliminates this by normalizing all features to the same scale.

### Why Not Just Use Accuracy?
In imbalanced datasets, a model can achieve 99% accuracy by predicting only the majority class — never catching the rare class at all. **F1 Score** balances Precision and Recall, giving a true picture of model performance.

### How KNN Works
Given a new data point:
1. Calculate the distance to every training point
2. Select the K nearest neighbors
3. Assign the class by **majority vote**

The algorithm stores no mathematical model — it memorizes the training data and reasons at prediction time.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting & visualization |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML pipeline (scaler, model, metrics) |

---

## 📈 Visualizations Included

- ✅ Pairplot — all feature relationships by species
- ✅ Correlation Heatmap
- ✅ Boxplots per feature
- ✅ Raw vs Scaled data scatter plot
- ✅ Elbow Method (K vs Error Rate)
- ✅ Confusion Matrix heatmap
- ✅ KNN Decision Boundary (2D)

---

## 🔮 Next Steps (Project 3 Preview)

> *"From Tabular Data... to Computer Vision."*

- Deep Learning & Convolutional Neural Networks (CNNs)
- Image classification with pixel data
- Moving beyond classical ML into neural architectures

---

## 👤 Author

**[Your Name]**
AI Engineering Intern — DecodeLabs Batch 2026

---

## 🏢 About DecodeLabs

DecodeLabs is an industrial training organization based in Greater Lucknow, India, focused on building real-world AI engineering skills through hands-on project-based learning.

📞 +91 89330 06408 | ✉️ decodelabs.tech@gmail.com | 🌐 [www.decodelabs.tech](http://www.decodelabs.tech)

---

*Project 2 of the DecodeLabs AI Industrial Training Kit — Batch 2026* 🛡️
