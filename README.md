# 🎫 Support Ticket Topic Classification

> Automatic classification of IT support tickets by topic using NLP and machine learning — built as part of the **Future Interns ML Program (Task 2, 2026)**.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-blueviolet?style=flat-square)]()
[![Task](https://img.shields.io/badge/Future_Interns-ML_Task_2-green?style=flat-square)]()

---

## 📌 Overview

This project builds an end-to-end **text classification pipeline** that automatically assigns a **topic category** to IT support tickets. It covers data cleaning, exploratory analysis, TF-IDF vectorization, multi-model comparison, and final evaluation with cross-validation.

**Dataset:** `all_tickets_processed_improved_v3.csv`  
**Columns used:** `Document` (ticket text) · `Topic_group` (target label)

---

## 📁 Project Structure

```
support-ticket-classification/
│
├── FUTURE_ML_02.ipynb                          # Main notebook
└── all_tickets_processed_improved_v3.csv.zip   # Dataset (not versioned)
```

---

## 🔍 Dataset

| Field | Description |
|---|---|
| `Document` | Raw text of the support ticket (lowercased, cleaned) |
| `Topic_group` | Target category (multi-class label) |

**Preprocessing applied:**
- Dropped rows with missing `Document` or `Topic_group`
- Lowercased and stripped whitespace
- Removed empty strings

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Class distribution bar chart (topic group frequency)
- Ticket length distribution (number of words)
- Baseline accuracy (majority class)

### 2. Text Vectorization — TF-IDF

```python
TfidfVectorizer(
    stop_words='english',
    max_features=12000,
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=3,
    sublinear_tf=True        # log normalization
)
```

### 3. Model Comparison

Three classifiers were compared on an 80/20 stratified split:

| Model | Description |
|---|---|
| **MultinomialNB** | Naive Bayes — fast probabilistic baseline |
| **SGDClassifier** | Linear SVM trained with stochastic gradient descent |
| **LinearSVC** ✅ | Support Vector Classifier — best performance |

### 4. Final Model: LinearSVC

```python
Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=12000,
                               ngram_range=(1,2), min_df=3, sublinear_tf=True)),
    ('clf',   LinearSVC(max_iter=3000, random_state=42))
])
```

---

## 📊 Evaluation

- ✅ **Accuracy** on held-out test set (20%)
- ✅ **Weighted F1-score** (handles class imbalance)
- ✅ **Full classification report** (precision, recall, F1 per class)
- ✅ **Confusion matrix** heatmap
- ✅ **5-Fold Cross-Validation** (mean accuracy ± std)

### Live Prediction Demo

The notebook includes real prediction examples, e.g.:

```python
"please reset my password because i can no longer access my external account"
→  Predicted: Access / Account Management

"i need a new laptop charger and a replacement docking station"
→  Predicted: Hardware Request

"the shared drive is full and we cannot upload new files"
→  Predicted: Storage / Infrastructure
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Run
```bash
jupyter notebook FUTURE_ML_02.ipynb
```

Update the `zip_path` variable in the first cell to point to your local dataset location.

---

## 💡 Key Takeaways

- **LinearSVC** consistently outperforms Naive Bayes and SGD for multi-class text classification on support tickets
- **TF-IDF with bigrams** (`ngram_range=(1,2)`) captures meaningful two-word expressions (e.g., "password reset", "hard drive")
- **Sublinear TF scaling** (`sublinear_tf=True`) reduces the influence of very frequent terms and improves performance
- **Stratified split** ensures each class is proportionally represented in train/test sets
