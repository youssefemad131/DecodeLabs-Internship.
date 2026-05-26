# 🤖 Project 3 — AI Recommendation Logic
### Tech Stack Recommender | DecodeLabs Industrial Training Kit | Batch 2026

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-green?style=flat-square&logo=scikit-learn)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

> A content-based job recommendation engine that maps your skills to real job roles using **TF-IDF vectorization** and **Cosine Similarity** — no neural networks, no black boxes, pure algorithmic logic.

---

## 📌 Project Overview

This project is the **Project 3 milestone** of the DecodeLabs AI Industrial Training Kit. The goal is to build a recommendation system from scratch that:

- Takes a user's tech skills as input
- Converts skills into weighted TF-IDF vectors
- Measures similarity between the user profile and real job postings
- Returns the **Top-N most relevant job roles** with match scores

The system uses **Content-Based Filtering** — meaning it recommends based on item attributes (job skills), not on what other users did.

---

## 🗂️ Repository Structure

```
project3-ai-recommendation/
│
├── Project3_AI_Recommendation.ipynb   # Main Jupyter notebook
├── glassdoor_jobs.csv                 # Kaggle dataset (956 real job postings)
└── README.md                          # This file
```

---

## 📦 Dataset

**Source:** Glassdoor Data Science Jobs — available on [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/data-science-job-posting-on-glassdoor)

| Property | Value |
|----------|-------|
| Total job postings | 956 |
| Unique job titles | 328 |
| Industries covered | 63 |
| Key columns | Job Title, Job Description, Salary Estimate, Company Name, Location |

Skills are **extracted automatically** from raw job descriptions using regex keyword matching against a dictionary of 50+ tech skills.

---

## 🧠 Algorithm

The recommendation pipeline follows the **Input → Process → Output (IPO)** model:

```
User Skills Input
      │
      ▼
[1] INGESTION     →  Collect 3+ skills from user
      │
      ▼
[2] TF-IDF        →  Convert skills to weighted numerical vectors
      │               TF  = skill frequency in one job posting
      │               IDF = log(total posts / posts with that skill)
      │               Rare skills score HIGHER than common skills
      ▼
[3] COSINE SIM    →  Measure angle between user vector & each job vector
      │               cos(θ) = (A · B) / (||A|| × ||B||)
      │               Score 1.0 = perfect match | Score 0.0 = no overlap
      ▼
[4] SORT          →  Rank all 848 postings by descending score
      │
      ▼
[5] FILTER        →  Return Top-N unique job titles
      │
      ▼
Ranked Recommendations Output
```

### Why Cosine Similarity (not Euclidean)?

A job posting with 15 skills listed vs one with 3 — Euclidean distance penalizes the shorter one unfairly. Cosine similarity measures **orientation**, not magnitude. A user with 3 skills and one with 10 skills pointing in the same direction get the same score.

### Why TF-IDF (not raw binary vectors)?

Binary vectors treat `python` (appears in 80% of postings) the same as `pytorch` (appears in 5%). TF-IDF **down-weights** generic skills and **up-weights** rare, specific ones — making matches more meaningful.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/project3-ai-recommendation.git
cd project3-ai-recommendation
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook Project3_AI_Recommendation.ipynb
```

### 4. Run all cells — then edit Cell 9 with your own skills

```python
MY_SKILLS = [
    "python",
    "machine learning",
    "sql",
    "deep learning",
    "pytorch",
]
```

---

## 📓 Notebook Walkthrough

| Cell | Section | Description |
|------|---------|-------------|
| 1 | Imports | Load all libraries |
| 2 | Load Data | Read `glassdoor_jobs.csv` from Kaggle |
| 3 | EDA | Job title distribution, sector breakdown charts |
| 4 | Feature Extraction | Regex-based skill extraction from descriptions |
| 5 | Skills Analysis | Top 20 skills frequency, skills-per-role boxplot |
| 6 | TF-IDF | Build 848 × 625 TF-IDF matrix |
| 7 | Recommend Function | Core similarity engine |
| 8 | Test Cases | 3 profiles tested (Data Science / DevOps / BI) |
| 9 | Interactive | Enter your own skills, get your matches |
| 10 | Deep Dive | Visualize similarity score distribution |

---

## 🎯 Sample Output

```
USER SKILLS: ['python', 'machine learning', 'sql', 'tensorflow', 'statistics']

🥇 #1 Data Scientist
   Match  : [████████████░░░░░░░░] 61.1%
   Company: Lorven Technologies Inc
   Salary : $53K–$91K (Glassdoor est.)
   ✔ Matched: python, machine learning, statistics

🥈 #2 Senior Data Scientist
   Match  : [██████████░░░░░░░░░░] 51.3%

🥉 #3 Machine Learning Engineer
   Match  : [████████░░░░░░░░░░░░] 43.7%
```

---

## 📊 Key Results

- **TF-IDF matrix size:** 848 job postings × 625 unique skill terms
- **Top skill in dataset:** `python` (appears in ~82% of postings)
- **Most specific high-value skill:** `pytorch`, `kubernetes`, `snowflake`
- **Average skills extracted per posting:** 5.9

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | TF-IDF vectorization, cosine similarity |
| `matplotlib` | Charts and visualizations |
| `re` | Regex-based skill extraction |
| `jupyter` | Interactive notebook environment |

---

## 💡 Key Concepts Covered

- **Content-Based Filtering** vs Collaborative Filtering
- **TF-IDF** feature extraction and weighting
- **Cosine Similarity** as a magnitude-invariant distance metric
- **Cold Start problem** and bypass strategies
- **Top-N filtering** to prevent choice overload
- **Vector space model** — shared vocabulary between user and items

---

## 🔮 Possible Extensions

- Add a Streamlit web UI for live skill input
- Expand the dataset to LinkedIn Jobs (1.3M postings on Kaggle)
- Implement collaborative filtering to complement content-based
- Add salary range filtering before recommendations
- Use word embeddings (Word2Vec / BERT) instead of TF-IDF for semantic matching

---

## 👤 Author

Built as part of the **DecodeLabs AI Industrial Training Kit — Batch 2026**

---

## 📄 License

This project is licensed under the MIT License.
