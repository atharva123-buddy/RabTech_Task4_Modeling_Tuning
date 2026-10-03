# RabTech Academy Internship - Task 4

## Supervised Classification / Regression Modeling & Tuning

### Project
**Telco Customer Churn Prediction**

---

## Objective

Train and compare at least four supervised classification models,
perform hyperparameter optimization using GridSearchCV with
Stratified K-Fold cross-validation, evaluate the models using
Precision, Recall, F1-Score and ROC-AUC, create ROC-AUC curves
and a confusion matrix, and select a champion model based on
validation performance.

---

## Models

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost

---

## Validation Strategy

- Train/Test Split: 80/20
- Cross-Validation: 5-fold StratifiedKFold
- GridSearch scoring: ROC-AUC
- Random State: 42

---

## Model Comparison

| Model               |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| XGBoost             |     0.8055 |      0.6701 |   0.5267 |     0.5898 |    0.8474 |
| Gradient Boosting   |     0.7991 |      0.6608 |   0.5000 |     0.5693 |    0.8457 |
| Logistic Regression |     0.8048 |      0.6552 |   0.5588 |     0.6032 |    0.8411 |
| Random Forest       |     0.7594 |      0.5364 |   0.6898 |     0.6035 |    0.8316 |

---

## Champion Model

**Champion Model:** XGBoost

The champion model was selected based on the highest ROC-AUC
among the evaluated tuned models.

### Champion Metrics

- Accuracy: 0.8055
- Precision: 0.6701
- Recall: 0.5267
- F1-Score: 0.5898
- ROC-AUC: 0.8474

---

## Hyperparameter Tuning

| Model               |   Best CV ROC-AUC | Best Parameters                                                                                                        |   Combinations Tested |
|:--------------------|------------------:|:-----------------------------------------------------------------------------------------------------------------------|----------------------:|
| Logistic Regression |          0.846326 | {'model__C': 10.0, 'model__class_weight': None, 'model__solver': 'liblinear'}                                          |                     6 |
| Random Forest       |          0.833916 | {'model__class_weight': 'balanced', 'model__max_depth': 15, 'model__min_samples_split': 5, 'model__n_estimators': 200} |                    16 |
| Gradient Boosting   |          0.849905 | {'model__learning_rate': 0.05, 'model__max_depth': 2, 'model__n_estimators': 200}                                      |                     8 |
| XGBoost             |          0.849981 | {'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__n_estimators': 100, 'model__subsample': 0.8}             |                    16 |

---

## Required Evidence

- `outputs/model_comparison.csv`
- `outputs/best_model_params.csv`
- `outputs/roc_auc_curves.png`
- `outputs/confusion_matrix.png`
- `outputs/champion_classification_report.txt`
- `outputs/champion_model_summary.csv`
- `models/champion_model.pkl`

---

## Conclusion

Four supervised classification models were trained and compared.
GridSearchCV with StratifiedKFold was used for hyperparameter
optimization. The tuned models were evaluated using Accuracy,
Precision, Recall, F1-Score and ROC-AUC.

The model with the highest ROC-AUC was selected as the champion
model and serialized using Joblib for future use.
