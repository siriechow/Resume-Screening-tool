# train_model.py

"""
Training script for the Resume Screening ML model.

Expected files:
- data/training_jd.txt
- data/training_data.csv with columns: resume_text, label

Usage:
    python train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from config import MODEL_PATH
from nlp_utils import build_features


def load_training_data(
    jd_path: str = "data/training_jd.txt",
    csv_path: str = "data/training_data.csv",
):
    if not os.path.exists(jd_path):
        raise FileNotFoundError(f"Job description file not found: {jd_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    df = pd.read_csv(csv_path)

    if "resume_text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "CSV must contain 'resume_text' and 'label' columns."
        )

    return jd_text, df


def build_feature_matrix(jd_text: str, df: pd.DataFrame):
    X = []
    y = []

    for _, row in df.iterrows():
        resume_text = str(row["resume_text"])
        label = int(row["label"])

        features, _ = build_features(jd_text, resume_text)
        X.append(features)
        y.append(label)

    X = np.vstack(X)
    y = np.array(y, dtype=int)

    return X, y


def train_and_save_model():
    jd_text, df = load_training_data()
    X, y = build_feature_matrix(jd_text, df)

    n_samples = len(y)
    n_classes = len(np.unique(y))

    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 classes to train a classifier, but got {n_classes}. "
            "Check that your CSV has both label 0 and 1."
        )

    print(f"Loaded {n_samples} samples with {n_classes} classes.")

    # If dataset is very small, avoid fancy splits
    if n_samples < 6:
        print(
            "Dataset is very small. Skipping train/test split and training on all data."
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        print("Model trained on all available data (no test evaluation).")
    else:
        # Choose a test_size that guarantees at least 1 sample per class in test set
        # test_size must satisfy: n_samples * test_size >= n_classes
        min_test_size = n_classes / n_samples
        test_size = max(0.2, min_test_size)

        print(
            f"Using test_size={test_size:.2f} with stratification "
            f"to ensure each class appears in the test set."
        )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )
        except ValueError as e:
            print(
                "Stratified split failed due to small or imbalanced data. "
                "Falling back to non-stratified split."
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=None,
            )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_pred_prob)
        print(f"AUC: {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
