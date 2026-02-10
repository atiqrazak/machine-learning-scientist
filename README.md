# Bank Transaction Category Prediction (Machine Learning) Challenge

## Overview
This project builds a **machine learning pipeline to predict Uncategorized transaction categories** from raw bank transaction data.  
The goal is to automatically label transactions (e.g. *Loans*, *Restaurants*, *Payroll*) so that user spending and income
patterns can be analysed and used for personalised financial insights.

The solution combines:
- **Natural Language Processing (NLP)** on transaction descriptions
- **Numerical transaction features**
- **Seasonality features** (day, month, weekday)
- **User profile information**
- A **Logistic Regression classifier** implemented using a robust scikit-learn pipeline

---

## Project Objectives
- Predict the `Uncategorized` `category` of each transaction in `bank_transaction.csv`
- Handle **class imbalance** and **missing labels**
- Generate **confidence scores** for predicted categories
- Enable **manual review** of low-confidence predictions
- Provide a clean, reproducible ML workflow suitable for real-world use

---

## Datasets

### 1. `bank_transaction.csv`
Contains transaction-level data.

Key columns:
- `client_id`
- `txn_date`
- `description`
- `amount`
- `category` (target variable)

### 2. `user_profile.csv`
Contains user-level financial preferences such as:
- Interest in paying off debt
- Interest in managing spending
- Interest in growing savings

These features are merged into transaction data to improve prediction quality.

---

## Feature Engineering

### Feature sets for different models
- Simple numeric-only model
- NLP text features added model
- Seasonal features (day/month/weekday) added model
- Complex model: text + seasonal + numeric features combined

### Text Features (NLP)
- Transaction descriptions are cleaned and vectorized using **TF-IDF**
- Uses unigrams and bigrams (`ngram_range=(1,2)`)
- Rare words are filtered to reduce noise
- Stopwords are removed to improve signal quality

### Numeric Features
- `amount` (transaction value)
- User profile boolean indicators (converted to integers)

### Seasonal / Time Features
- `txn_day` — day of month
- `txn_month` — month
- `txn_weekday` — day of week (0 = Monday)

These help capture recurring patterns such as salaries, rent, or subscriptions.

---

## Modeling Approach

### Pipeline Design
A **scikit-learn Pipeline** is used to ensure:
- No data leakage
- Clean separation of preprocessing and modeling
- Reproducibility

Pipeline steps:
1. **ColumnTransformer**
   - TF-IDF for text (`desc_clean`)
   - StandardScaler for numeric features
2. **Logistic Regression**
   - Chosen as a strong, interpretable baseline
   - Uses `class_weight='balanced'` to handle imbalanced categories

---

## Training & Evaluation

- Labeled data is split into **training and validation sets** using stratified sampling
- Metrics used:
  - **Accuracy**
  - **Macro F1 Score** (important for imbalanced classes)
- A **confusion matrix** is generated to visualise misclassifications

---

## Handling Uncategorized Transactions

Transactions with missing labels (`Uncategorized`) are:
1. Predicted using the trained model
2. Assigned a **confidence score** based on maximum predicted probability

### Confidence Thresholding
- **High confidence (≥ 0.75)**  
  → Safe for auto-labeling and downstream analytics
- **Low confidence (< 0.75)**  
  → Flagged for manual review or future model improvement

---

## Outputs
The final dataset includes:
- `predicted_category`
- `prediction_confidence`

These outputs enable:
- Automated categorisation
- Quality control through confidence filtering
- Iterative improvement via human-in-the-loop review

---

## How to Run

### Requirements
- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk

