# RabTech Academy Internship — Task 4
## Supervised Classification / Regression Modeling & Hyperparameter Tuning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 1. Project Overview

This repository contains the implementation of **Task 4 of the RabTech Academy Internship**.

The objective is to build a supervised machine learning modeling pipeline for **customer churn prediction** using the Telco Customer Churn dataset.

The project goes beyond training a single machine learning model. It compares multiple classification algorithms, performs hyperparameter optimization using **GridSearchCV with Stratified K-Fold Cross-Validation**, evaluates the models using multiple classification metrics, creates ROC-AUC curves and a confusion matrix, selects a champion model, and serializes that model for future use.

---

# 2. Task Requirements

The RabTech Academy task requires the following:

- Train at least 4 distinct model architectures on the preprocessed training set.
- Perform hyperparameter optimization using GridSearchCV or Optuna.
- Use Stratified K-Fold cross-validation.
- Evaluate models using:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix
- Select a champion model based on validation metrics.
- Provide a Jupyter Notebook with model comparison and ROC-AUC curves.
- Provide a serialized model (`.pkl` or `.joblib`).

This repository implements all of these requirements.

---

# 3. Dataset

## Dataset Used

**IBM Telco Customer Churn Dataset**

The dataset contains customer-level information such as:

- Demographics
- Tenure
- Contract type
- Internet service
- Payment method
- Monthly charges
- Total charges
- Services subscribed
- Churn status

### Target Variable

```text
Churn
```

The target is converted to binary format:

```text
No  -> 0
Yes -> 1
```

The customer identifier (`customerID`) is removed because it is an identifier rather than a useful predictive feature.

`TotalCharges` is converted from text to numeric values, with invalid/blank values handled as missing values.

---

# 4. Machine Learning Problem

This is a **binary supervised classification problem**.

The objective is:

> Predict whether a customer is likely to churn.

### Classes

| Value | Meaning |
|---|---|
| 0 | No Churn |
| 1 | Churn |

---

# 5. Models Implemented

Four distinct classification architectures are trained:

## 5.1 Logistic Regression

A linear classification algorithm used as an interpretable baseline.

It provides a strong baseline for comparison against tree-based ensemble models.

---

## 5.2 Random Forest Classifier

An ensemble of decision trees trained using randomized subsets of observations and features.

The model can capture nonlinear relationships and feature interactions.

---

## 5.3 Gradient Boosting Classifier

An ensemble method that builds trees sequentially, where each new tree attempts to improve the previous model.

---

## 5.4 XGBoost Classifier

XGBoost is a gradient-boosted decision tree implementation designed for efficient and accurate supervised learning.

The project uses the scikit-learn-compatible `XGBClassifier` interface.

---

# 6. Project Workflow

The complete workflow is:

```text
Raw Dataset
     |
     v
Data Loading
     |
     v
Data Cleaning
     |
     v
Train/Test Split
     |
     v
Feature Identification
     |
     v
Preprocessing Pipeline
     |
     +-------------------------------+
     |                               |
     v                               v
Logistic Regression             Random Forest
     |                               |
     v                               v
Hyperparameter Tuning             Hyperparameter Tuning
     |                               |
     +---------------+---------------+
                     |
                     v
             Gradient Boosting
                     |
                     v
             Hyperparameter Tuning
                     |
                     v
                  XGBoost
                     |
                     v
             Hyperparameter Tuning
                     |
                     v
           Model Comparison
                     |
                     v
              ROC-AUC Curves
                     |
                     v
           Champion Model
                     |
          +----------+----------+
          |                     |
          v                     v
    Confusion Matrix      Serialized Model
                              (.pkl)
```

---

# 7. Data Preprocessing

The project uses an sklearn `Pipeline` and `ColumnTransformer`.

This is important because preprocessing is included inside the cross-validation workflow.

## Numerical Features

Numerical columns are processed using:

1. Median imputation
2. StandardScaler

Example:

```text
Numerical Feature
       |
       v
Missing Value Imputation
       |
       v
Standard Scaling
```

---

## Categorical Features

Categorical columns are processed using:

1. Most-frequent-value imputation
2. One-Hot Encoding

Example:

```text
Categorical Feature
       |
       v
Missing Value Imputation
       |
       v
One-Hot Encoding
```

Unknown categories are ignored during transformation.

---

# 8. Train/Test Split

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

The split uses:

```python
stratify=y
```

This helps preserve the target-class distribution between training and testing sets.

A fixed random state is used:

```python
RANDOM_STATE = 42
```

This improves reproducibility.

---

# 9. Cross-Validation

Hyperparameter optimization uses:

```text
5-Fold Stratified K-Fold Cross-Validation
```

Configuration:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Stratification is used so that each fold approximately maintains the class distribution of the target variable.

---

# 10. Hyperparameter Optimization

The project uses:

```text
GridSearchCV
```

with:

```text
Scoring Metric = ROC-AUC
CV = 5-fold StratifiedKFold
```

The search evaluates multiple parameter combinations for every model.

---

## Logistic Regression Parameters

The search includes:

```text
C
solver
class_weight
```

---

## Random Forest Parameters

The search includes:

```text
n_estimators
max_depth
min_samples_split
class_weight
```

---

## Gradient Boosting Parameters

The search includes:

```text
n_estimators
learning_rate
max_depth
```

---

## XGBoost Parameters

The search includes:

```text
n_estimators
max_depth
learning_rate
subsample
```

The exact best parameters are automatically saved after execution.

---

# 11. Evaluation Metrics

The models are evaluated using several metrics.

## Accuracy

Measures the proportion of correct predictions:

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## Precision

Measures how many predicted positive cases are actually positive.

```text
Precision =
True Positives / (True Positives + False Positives)
```

For this project, the positive class is:

```text
Churn = 1
```

---

## Recall

Measures how many actual churn customers are successfully detected.

```text
Recall =
True Positives / (True Positives + False Negatives)
```

---

## F1-Score

The harmonic mean of precision and recall:

```text
F1 =
2 × Precision × Recall
----------------------
Precision + Recall
```

---

## ROC-AUC

ROC-AUC summarizes classification performance across decision thresholds.

It is used as the **primary model-selection metric** in this project.

The same metric is used during GridSearchCV:

```python
scoring="roc_auc"
```

---

# 12. Confusion Matrix

The final champion model produces a confusion matrix containing:

```text
True Negative
False Positive
False Negative
True Positive
```

It is saved as:

```text
outputs/confusion_matrix.png
```

The visualization makes it easy to inspect the types of classification errors produced by the champion model.

---

# 13. ROC-AUC Curves

ROC curves are generated for all four tuned models.

The resulting visualization is saved as:

```text
outputs/roc_auc_curves.png
```

The plot includes the ROC-AUC value for every model so that the models can be compared visually and numerically.

---

# 14. Champion Model Selection

The project does **not** manually decide in advance that Random Forest or XGBoost should be the winner.

Instead, the code compares the tuned models and selects the model with the highest ROC-AUC:

```python
champion_name = comparison_df.iloc[0]["Model"]
```

This makes the selection data-driven.

The champion model is saved as:

```text
models/champion_model.pkl
```

---

# 15. Model Serialization

The selected champion model is serialized using Joblib:

```python
joblib.dump(
    champion_model,
    "models/champion_model.pkl"
)
```

The saved model contains the complete sklearn pipeline, including preprocessing and the trained classifier.

Therefore, a future prediction workflow can load the single `.pkl` file and pass raw feature data through the same preprocessing and model pipeline.

The script also reloads the serialized model after saving it to verify that it can be successfully restored.

---

# 16. Project Structure

```text
RabTech_Task4_Modeling_Tuning/
│
├── data/
│   └── telco_customer_churn.csv
│
├── notebooks/
│   └── Task4_Supervised_Modeling_Tuning.ipynb
│
├── src/
│   └── train_models.py
│
├── outputs/
│   ├── model_comparison.csv
│   ├── best_model_params.csv
│   ├── champion_model_summary.csv
│   ├── champion_classification_report.txt
│   ├── roc_auc_curves.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   ├── X_train_processed.csv
│   ├── X_test_processed.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── cv_results_*.csv
│
├── models/
│   └── champion_model.pkl
│
├── reports/
│   └── Task4_Modeling_Report.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 17. Installation

## Step 1 — Open the project

Open the project folder in VS Code:

```text
RabTech_Task4_Modeling_Tuning
```

---

## Step 2 — Create virtual environment

Open the VS Code terminal:

```powershell
python -m venv venv
```

---

## Step 3 — Activate virtual environment

Windows PowerShell:

```powershell
venv\Scripts\activate
```

You should see:

```text
(venv)
```

at the beginning of the terminal.

---

# 18. Install Dependencies

Run:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, use:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
notebook
joblib
xgboost
tabulate
```

Then run:

```powershell
pip install -r requirements.txt
```

`tabulate` is included because the generated Markdown report uses pandas' `to_markdown()` method.

---

# 19. Dataset Setup

Place your dataset here:

```text
data/telco_customer_churn.csv
```

The script expects exactly this filename.

If your CSV currently has another filename, rename it to:

```text
telco_customer_churn.csv
```

---

# 20. Run the Python Training Pipeline

From the **project root**, run:

```powershell
python src/train_models.py
```

Do not run:

```powershell
python train_models.py
```

unless you have moved the script out of the `src` directory.

---

# 21. Expected Terminal Workflow

The script will display sections similar to:

```text
================================================================================
RABTECH ACADEMY - TASK 4
SUPERVISED CLASSIFICATION MODELING & TUNING
================================================================================

================================================================================
DATA LOADING
================================================================================

================================================================================
PREPROCESSING
================================================================================

================================================================================
MODELS
================================================================================

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

================================================================================
HYPERPARAMETER OPTIMIZATION
================================================================================

Tuning: Logistic Regression
Tuning: Random Forest
Tuning: Gradient Boosting
Tuning: XGBoost

================================================================================
MODEL EVALUATION
================================================================================
```

At the end, the script prints the selected champion model and the locations of the generated files.

---

# 22. Jupyter Notebook

The evaluator-facing notebook should be:

```text
notebooks/Task4_Supervised_Modeling_Tuning.ipynb
```

The notebook should demonstrate:

1. Project objective
2. Dataset loading
3. Dataset inspection
4. Data cleaning
5. Target distribution
6. Train/test split
7. Preprocessing
8. Four model architectures
9. Stratified K-Fold setup
10. GridSearchCV
11. Best hyperparameters
12. Model comparison
13. Precision
14. Recall
15. F1-Score
16. ROC-AUC
17. ROC curves
18. Champion model
19. Confusion matrix
20. Model serialization
21. Final conclusion

The notebook provides the visual proof, while `src/train_models.py` provides a reproducible executable pipeline.

---

# 23. Required Deliverables

After successful execution, verify these files exist.

## Model Comparison

```text
outputs/model_comparison.csv
```

Contains:

```text
Model
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

---

## Best Hyperparameters

```text
outputs/best_model_params.csv
```

Contains the best cross-validation ROC-AUC and parameter combination for each model.

---

## ROC-AUC Curves

```text
outputs/roc_auc_curves.png
```

---

## Confusion Matrix

```text
outputs/confusion_matrix.png
```

---

## Champion Classification Report

```text
outputs/champion_classification_report.txt
```

---

## Champion Model

```text
models/champion_model.pkl
```

---

## Final Report

```text
reports/Task4_Modeling_Report.md
```

---

# 24. Requirement-to-Implementation Mapping

| RabTech Requirement | Implementation |
|---|---|
| At least 4 models | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| Preprocessed training set | ColumnTransformer + Pipeline |
| Hyperparameter optimization | GridSearchCV |
| Stratified K-Fold | StratifiedKFold, 5 folds |
| Precision | `precision_score()` |
| Recall | `recall_score()` |
| F1-Score | `f1_score()` |
| ROC-AUC | `roc_auc_score()` |
| ROC-AUC curves | `roc_auc_curves.png` |
| Confusion Matrix | `confusion_matrix.png` |
| Model comparison | `model_comparison.csv` |
| Champion selection | Highest ROC-AUC |
| Serialized model | `champion_model.pkl` |
| Jupyter proof | Task 4 notebook |
| Reproducibility | `random_state=42` |

---

# 25. Important Machine Learning Practice

The preprocessing operations are placed inside the sklearn pipeline:

```text
Pipeline
   |
   +-- Preprocessor
   |      |
   |      +-- Numeric Imputation
   |      +-- Scaling
   |      +-- Categorical Imputation
   |      +-- One-Hot Encoding
   |
   +-- Classifier
```

The pipeline is then passed to `GridSearchCV`.

This prevents the preprocessing stage from being fitted globally before cross-validation and helps reduce data leakage during model selection.

---

# 26. Why ROC-AUC Is Used for Tuning

The project uses:

```python
scoring="roc_auc"
```

for GridSearchCV.

ROC-AUC is useful for binary classification because it evaluates how well the model separates the two classes over different classification thresholds.

The final model comparison still reports:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

so the evaluator can inspect several aspects of model performance.

---

# 27. Reproducibility

The project uses:

```python
RANDOM_STATE = 42
```

for the train/test split, cross-validation, and applicable models.

This makes the experiment easier to reproduce.

---

# 28. Common Errors and Solutions

## Error: Dataset not found

If you see:

```text
Dataset not found
```

check:

```text
data/telco_customer_churn.csv
```

The filename must match exactly.

---

## Error: `No module named xgboost`

Run:

```powershell
pip install xgboost
```

---

## Error: `No module named tabulate`

Run:

```powershell
pip install tabulate
```

---

## Error: `No module named joblib`

Run:

```powershell
pip install joblib
```

---

## Error: `OneHotEncoder.__init__() got an unexpected keyword argument 'sparse_output'`

Your scikit-learn version is old.

Upgrade:

```powershell
pip install --upgrade scikit-learn
```

Then run the project again.

---

## Error: PowerShell does not allow virtual environment activation

Try:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
venv\Scripts\activate
```

---

# 29. GitHub Submission

After confirming that the project runs successfully:

```powershell
git init
git add .
git commit -m "Complete RabTech Task 4 supervised modeling and tuning"
```

Create a GitHub repository, then connect it:

```powershell
git remote add origin YOUR_GITHUB_REPOSITORY_URL
```

Then:

```powershell
git branch -M main
git push -u origin main
```

Before pushing, verify that the repository contains:

```text
README.md
requirements.txt
src/train_models.py
notebooks/Task4_Supervised_Modeling_Tuning.ipynb
outputs/
models/champion_model.pkl
reports/
```

---

# 30. Final Evaluator Checklist

Before submitting the public GitHub link, verify:

- [ ] Dataset is available or documented.
- [ ] Notebook runs from beginning to end.
- [ ] Four different classification algorithms are present.
- [ ] GridSearchCV is demonstrated.
- [ ] StratifiedKFold is demonstrated.
- [ ] Best hyperparameters are displayed.
- [ ] Precision is reported.
- [ ] Recall is reported.
- [ ] F1-Score is reported.
- [ ] ROC-AUC is reported.
- [ ] Model comparison table is visible.
- [ ] ROC-AUC curves are visible.
- [ ] Confusion matrix is visible.
- [ ] Champion model is identified using the metric-based selection rule.
- [ ] Champion model `.pkl` file exists.
- [ ] Final report exists.
- [ ] README explains how to reproduce the project.
- [ ] `requirements.txt` exists.
- [ ] GitHub repository is public if the evaluator requires a public link.

---

# 31. Final Submission Link

The RabTech evaluator requests one public link.

Submit the **public GitHub repository URL** containing the complete project.

Example:

```text
https://github.com/YOUR_USERNAME/RabTech_Task4_Modeling_Tuning
```

If the evaluator instead requests the deployed notebook or another specific public URL, provide that URL according to the submission portal instructions.

---

## Author

**Atharva Mahesh Joshi**

B.Tech — Artificial Intelligence & Data Science

RabTech Academy Internship — Task 4

---

## Summary

This project demonstrates a complete supervised machine learning workflow:

```text
Data
 ↓
Cleaning
 ↓
Train/Test Split
 ↓
Preprocessing Pipeline
 ↓
4 Classification Models
 ↓
Stratified K-Fold CV
 ↓
GridSearchCV
 ↓
Metric Evaluation
 ↓
ROC-AUC Curves
 ↓
Confusion Matrix
 ↓
Champion Model
 ↓
Model Serialization
 ↓
Final Report
```

The project is designed to satisfy the documented RabTech Task 4 deliverables while maintaining a reproducible and evaluator-friendly structure.
