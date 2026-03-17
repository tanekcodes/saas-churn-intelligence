"""
Churn prediction model training, evaluation, and persistence.

We compare Logistic Regression (interpretable baseline) vs. XGBoost (best accuracy),
then use the XGBoost model for SHAP analysis.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


# ── train / evaluate ──────────────────────────────────────────────────────────

def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    C=0.5,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train) -> XGBClassifier:
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def evaluate(model, X_test, y_test, model_name: str = "") -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    report = classification_report(y_test, pred, output_dict=True)
    return {
        "name": model_name,
        "roc_auc": roc_auc_score(y_test, proba),
        "avg_precision": average_precision_score(y_test, proba),
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
        "proba": proba,
        "pred": pred,
    }


def cross_validate(model, X, y, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return {"mean": scores.mean(), "std": scores.std(), "scores": scores}


# ── persistence ───────────────────────────────────────────────────────────────

def save_model(model, name: str = "churn_model.pkl") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    joblib.dump(model, path)
    print(f"Model saved -> {path}")
    return path


def load_model(name: str = "churn_model.pkl"):
    return joblib.load(MODEL_DIR / name)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_roc_curves(eval_results: list[dict]) -> go.Figure:
    """Overlay ROC curves for multiple models."""
    fig = go.Figure()
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    # Diagonal baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random Baseline",
            showlegend=True,
        )
    )
    for i, res in enumerate(eval_results):
        fpr, tpr, _ = roc_curve(res["y_test"], res["proba"])
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{res['name']} (AUC = {res['roc_auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )
    fig.update_layout(
        title="ROC Curves — Logistic Regression vs. XGBoost",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_confusion_matrix(y_test, y_pred, model_name: str = "") -> go.Figure:
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Did Not Churn", "Churned"]
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {model_name}",
    )
    fig.update_layout(height=400)
    return fig


def plot_feature_importance(model: XGBClassifier, feature_names: list[str]) -> go.Figure:
    importance = model.feature_importances_
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=True)
        .tail(20)
    )
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="XGBoost Feature Importance (Top 20)",
        labels={"importance": "Importance Score", "feature": ""},
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=600, showlegend=False)
    return fig
