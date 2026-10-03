from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "telco_customer_churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================

def load_data():
    """Load and clean the Telco Customer Churn dataset."""

    print("\n" + "=" * 80)
    print("RABTECH ACADEMY - TASK 4")
    print("SUPERVISED CLASSIFICATION MODELING & TUNING")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("1. DATA LOADING")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            "Please place telco_customer_churn.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset path : {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")

    # Convert TotalCharges safely to numeric.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # Check target.
    if "Churn" not in df.columns:
        raise ValueError("The dataset does not contain a 'Churn' column.")

    # Robust target conversion for pandas 3.x string dtype.
    churn_values = (
        df["Churn"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["Churn"] = churn_values.map({
        "no": 0,
        "yes": 1
    })

    # Remove customer ID because it is an identifier.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Remove rows where target conversion failed.
    df = df.dropna(subset=["Churn"]).copy()
    df["Churn"] = df["Churn"].astype(int)

    print(f"Rows after cleaning: {len(df)}")

    print("\nTarget distribution:")
    print(df["Churn"].value_counts())

    print("\nTarget percentage:")
    print(
        (df["Churn"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    return df


# ============================================================
# 2. PREPROCESS DATA
# ============================================================

def preprocess_data(df):
    """Create train/test sets and preprocessing pipeline."""

    print("\n" + "=" * 80)
    print("2. PREPROCESSING")
    print("=" * 80)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Stratified train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    print(f"Training samples   : {len(X_train)}")
    print(f"Testing samples    : {len(X_test)}")
    print(f"Numeric features   : {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Numerical preprocessing.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# 3. BUILD MODELS
# ============================================================

def build_models():
    """Create four different classification models."""

    print("\n" + "=" * 80)
    print("3. MODEL ARCHITECTURES")
    print("=" * 80)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1
        )
    }

    for model_name in models:
        print(f"- {model_name}")

    return models


# ============================================================
# 4. HYPERPARAMETER GRIDS
# ============================================================

def get_parameter_grids():
    """Return hyperparameter grids for GridSearchCV."""

    parameter_grids = {

        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear"],
            "model__class_weight": [None, "balanced"]
        },

        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 15],
            "model__min_samples_split": [2, 5],
            "model__class_weight": [None, "balanced"]
        },

        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.10],
            "model__max_depth": [2, 3]
        },

        "XGBoost": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.10],
            "model__subsample": [0.8, 1.0]
        }
    }

    return parameter_grids


# ============================================================
# 5. HYPERPARAMETER TUNING
# ============================================================

def tune_models(models, preprocessor, X_train, y_train):
    """Tune all models using GridSearchCV and StratifiedKFold."""

    print("\n" + "=" * 80)
    print("4. HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    print(
        "Method: GridSearchCV | "
        "Cross-validation: 5-fold StratifiedKFold | "
        "Scoring: ROC-AUC"
    )

    parameter_grids = get_parameter_grids()

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    best_models = {}
    tuning_results = {}

    for name, model in models.items():

        print("\n" + "-" * 80)
        print(f"Tuning model: {name}")
        print("-" * 80)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grids[name],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_

        tuning_results[name] = grid_search

        print(f"\nBest CV ROC-AUC: {grid_search.best_score_:.4f}")

        print("Best parameters:")
        print(grid_search.best_params_)

    # Save tuning summary.
    tuning_rows = []

    for name, grid in tuning_results.items():
        tuning_rows.append({
            "Model": name,
            "Best CV ROC-AUC": grid.best_score_,
            "Best Parameters": str(grid.best_params_),
            "Combinations Tested": len(grid.cv_results_["params"])
        })

        # Save detailed GridSearch results.
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        cv_results_df = pd.DataFrame(grid.cv_results_)

        cv_results_df.to_csv(
            OUTPUT_DIR / f"cv_results_{safe_name}.csv",
            index=False
        )

    tuning_df = pd.DataFrame(tuning_rows)

    tuning_df.to_csv(
        OUTPUT_DIR / "best_model_params.csv",
        index=False
    )

    return best_models, tuning_df


# ============================================================
# 6. EVALUATE MODELS
# ============================================================

def evaluate_models(best_models, X_test, y_test):
    """Calculate required classification metrics."""

    print("\n" + "=" * 80)
    print("5. MODEL EVALUATION")
    print("=" * 80)

    results = {}

    for name, model in best_models.items():

        y_pred = model.predict(X_test)
        y_probability = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Predictions": y_pred,
            "Probabilities": y_probability
        }

    comparison_rows = []

    for name, metrics in results.items():
        comparison_rows.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"]
        })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df = comparison_df.sort_values(
        by="ROC-AUC",
        ascending=False
    ).reset_index(drop=True)

    print("\nModel comparison:")
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    return comparison_df, results


# ============================================================
# 7. ROC-AUC CURVES
# ============================================================

def create_roc_curve(best_models, X_test, y_test):
    """Create ROC curves for all tuned models."""

    print("\n" + "=" * 80)
    print("6. ROC-AUC CURVES")
    print("=" * 80)

    plt.figure(figsize=(10, 7))

    for name, model in best_models.items():

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test,
            probabilities
        )

        auc_value = roc_auc_score(
            y_test,
            probabilities
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{name} (AUC = {auc_value:.3f})"
        )

    # Random classifier baseline.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curves - Tuned Classification Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "roc_auc_curves.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"ROC-AUC curve saved: {output_path}")


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(champion_name, champion_model, X_test, y_test):
    """Create and save confusion matrix for champion model."""

    print("\n" + "=" * 80)
    print("7. CONFUSION MATRIX")
    print("=" * 80)

    predictions = champion_model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("Confusion Matrix:")
    print(matrix)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(
        f"Confusion Matrix - {champion_name}"
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Confusion matrix saved: {output_path}")

    return predictions


# ============================================================
# 9. SAVE CHAMPION MODEL
# ============================================================

def save_champion_model(
    champion_name,
    champion_model,
    comparison_df,
    X_train,
    X_test,
    y_train,
    y_test
):
    """Save champion model and all required evidence."""

    print("\n" + "=" * 80)
    print("8. CHAMPION MODEL")
    print("=" * 80)

    model_path = MODEL_DIR / "champion_model.pkl"

    # Serialize model.
    joblib.dump(
        champion_model,
        model_path
    )

    print(f"Champion model: {champion_name}")
    print(f"Saved model   : {model_path}")

    # Test loading the saved model.
    loaded_model = joblib.load(model_path)

    loaded_predictions = loaded_model.predict(
        X_test
    )

    print(
        "Serialized model verification: SUCCESS"
    )

    # Classification report.
    report = classification_report(
        y_test,
        loaded_predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    report_path = (
        OUTPUT_DIR /
        "champion_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Champion Model: {champion_name}\n\n"
        )

        file.write(report)

    # Champion summary.
    champion_row = comparison_df[
        comparison_df["Model"] == champion_name
    ].iloc[0]

    summary = pd.DataFrame([{
        "Champion Model": champion_name,
        "Accuracy": champion_row["Accuracy"],
        "Precision": champion_row["Precision"],
        "Recall": champion_row["Recall"],
        "F1-Score": champion_row["F1-Score"],
        "ROC-AUC": champion_row["ROC-AUC"]
    }])

    summary.to_csv(
        OUTPUT_DIR / "champion_model_summary.csv",
        index=False
    )

    # Save transformed training/testing data.
    preprocessor = champion_model.named_steps[
        "preprocessor"
    ]

    X_train_processed = preprocessor.transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    X_train_processed_df.to_csv(
        OUTPUT_DIR / "X_train_processed.csv"
    )

    X_test_processed_df.to_csv(
        OUTPUT_DIR / "X_test_processed.csv"
    )

    y_train.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False
    )

    print(
        f"Classification report saved: {report_path}"
    )


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

def create_feature_importance(
    champion_name,
    champion_model,
    X_train
):
    """Create feature importance when supported by the champion model."""

    model = champion_model.named_steps["model"]
    preprocessor = champion_model.named_steps["preprocessor"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # Tree models expose feature_importances_.
    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        importance_df.to_csv(
            OUTPUT_DIR / "feature_importance.csv",
            index=False
        )

        top_features = importance_df.head(15)

        plt.figure(figsize=(10, 7))

        sns.barplot(
            data=top_features,
            x="Importance",
            y="Feature"
        )

        plt.title(
            f"Top 15 Features - {champion_name}"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "feature_importance.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Feature importance saved successfully."
        )

    else:
        print(
            "Champion model does not expose "
            "feature_importances_. Skipping plot."
        )


# ============================================================
# 11. FINAL MARKDOWN REPORT
# ============================================================

def create_final_report(
    champion_name,
    comparison_df,
    tuning_df
):
    """Create a final Markdown report."""

    champion_row = comparison_df[
        comparison_df["Model"] == champion_name
    ].iloc[0]

    report_path = (
        REPORT_DIR /
        "Task4_Modeling_Report.md"
    )

    comparison_markdown = comparison_df.to_markdown(
        index=False,
        floatfmt=".4f"
    )

    tuning_markdown = tuning_df.to_markdown(
        index=False
    )

    report = f"""# RabTech Academy Internship - Task 4

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

{comparison_markdown}

---

## Champion Model

**Champion Model:** {champion_name}

The champion model was selected based on the highest ROC-AUC
among the evaluated tuned models.

### Champion Metrics

- Accuracy: {champion_row["Accuracy"]:.4f}
- Precision: {champion_row["Precision"]:.4f}
- Recall: {champion_row["Recall"]:.4f}
- F1-Score: {champion_row["F1-Score"]:.4f}
- ROC-AUC: {champion_row["ROC-AUC"]:.4f}

---

## Hyperparameter Tuning

{tuning_markdown}

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
"""

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    print(
        f"Final report saved: {report_path}"
    )


# ============================================================
# 12. MAIN FUNCTION
# ============================================================

def main():
    """Execute the complete Task 4 workflow."""

    # Step 1: Load data.
    df = load_data()

    # Step 2: Preprocess data.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data(df)

    # Step 3: Build four models.
    models = build_models()

    # Step 4: Tune models.
    (
        best_models,
        tuning_df
    ) = tune_models(
        models,
        preprocessor,
        X_train,
        y_train
    )

    # Step 5: Evaluate all models.
    (
        comparison_df,
        evaluation_results
    ) = evaluate_models(
        best_models,
        X_test,
        y_test
    )

    # Step 6: Create ROC curves.
    create_roc_curve(
        best_models,
        X_test,
        y_test
    )

    # Step 7: Select champion using highest ROC-AUC.
    champion_name = comparison_df.iloc[0]["Model"]

    champion_model = best_models[
        champion_name
    ]

    # Step 8: Create confusion matrix.
    create_confusion_matrix(
        champion_name,
        champion_model,
        X_test,
        y_test
    )

    # Step 9: Save champion model.
    save_champion_model(
        champion_name,
        champion_model,
        comparison_df,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Step 10: Feature importance.
    create_feature_importance(
        champion_name,
        champion_model,
        X_train
    )

    # Step 11: Final report.
    create_final_report(
        champion_name,
        comparison_df,
        tuning_df
    )

    # Final output.
    print("\n" + "=" * 80)
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nChampion model:")
    print(champion_name)

    print("\nGenerated folders:")
    print(f"Outputs : {OUTPUT_DIR}")
    print(f"Models  : {MODEL_DIR}")
    print(f"Reports : {REPORT_DIR}")

    print("\nMain deliverables:")
    print("- model_comparison.csv")
    print("- best_model_params.csv")
    print("- roc_auc_curves.png")
    print("- confusion_matrix.png")
    print("- champion_model.pkl")
    print("- Task4_Modeling_Report.md")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()