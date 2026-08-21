"""
Data drift detection and model retraining triggers.

A model trained once on a fixed dataset silently goes stale as the real
customer population's characteristics shift over time -- new customers skew
toward different contract types, pricing changes shift the MonthlyCharges
distribution, a product change alters which add-ons are common. None of that
throws an error. The model keeps making predictions; they just get quietly
less trustworthy. This module gives the pipeline a way to notice.

Two kinds of drift are checked, because they catch different failure modes:

1. **Feature drift** (a.k.a. covariate shift): are the INPUTS the model sees
   today shaped differently than the inputs it was trained on? Detected via
   Population Stability Index (PSI), the standard metric for this in credit
   risk and churn modeling specifically (borrowed from the same toolkit
   real credit-risk teams use to monitor scorecards -- worth noting given
   this project's audience).
2. **Performance drift** (a.k.a. concept drift): even if the inputs look the
   same, has the actual relationship between inputs and outcomes changed?
   This can only be checked once new data has enough elapsed time to have
   known outcomes (i.e. you need to wait and see who actually churned), so
   it lags feature drift monitoring by design -- flagged explicitly here
   rather than silently treated as equivalent to feature drift.

Retraining triggers on either check breaching a threshold, and logs a full
before/after comparison rather than silently swapping the model -- consistent
with this project's audit-trail habit (see src/database.py's ingestion_log).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone

# PSI interpretation thresholds, industry-standard bands (same bands used in
# credit scorecard monitoring):
#   < 0.10           -> no significant shift
#   0.10 - 0.25       -> moderate shift, worth watching
#   > 0.25            -> significant shift, investigate / retrain
PSI_WATCH_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# If model AUC on new labeled data drops more than this (absolute) below its
# training-time AUC, that's flagged as concept drift regardless of PSI.
AUC_DROP_ALERT_THRESHOLD = 0.03


def _psi_for_feature(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index for one feature between a baseline
    ('expected', i.e. training-time distribution) and new data ('actual').

    PSI = sum over bins of (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

    Works for numeric features directly (quantile-binned against the
    baseline's own bin edges, so 'actual' is compared against a fixed
    reference frame rather than re-binned on its own scale, which would
    hide the shift). Categorical features are treated as their own bins.
    """
    if pd.api.types.is_numeric_dtype(expected):
        # Bin edges from the baseline distribution, not from `actual` --
        # this is what makes the shift visible instead of dividing it away.
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(expected.dropna(), quantiles))
        if len(edges) < 2:
            return 0.0
        expected_binned = pd.cut(expected, bins=edges, include_lowest=True)
        actual_binned = pd.cut(actual, bins=edges, include_lowest=True)
    else:
        expected_binned = expected
        actual_binned = actual

    expected_pct = expected_binned.value_counts(normalize=True, sort=False)
    actual_pct = actual_binned.value_counts(normalize=True, sort=False)

    # align on the same set of bins, filling absent bins with a small
    # epsilon rather than 0 (avoids divide-by-zero / log(0))
    all_bins = expected_pct.index.union(actual_pct.index)
    eps = 1e-4
    expected_pct = expected_pct.reindex(all_bins, fill_value=eps).clip(lower=eps)
    actual_pct = actual_pct.reindex(all_bins, fill_value=eps).clip(lower=eps)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_feature_drift(
    baseline: pd.DataFrame, new_data: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """PSI for each feature, with a plain-language flag per feature."""
    rows = []
    for feat in features:
        if feat not in baseline.columns or feat not in new_data.columns:
            continue
        psi = _psi_for_feature(baseline[feat], new_data[feat])
        if psi < PSI_WATCH_THRESHOLD:
            flag = "stable"
        elif psi < PSI_ALERT_THRESHOLD:
            flag = "watch"
        else:
            flag = "alert"
        rows.append({"feature": feat, "psi": psi, "flag": flag})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def compute_performance_drift(
    baseline_auc: float, model, X_new: pd.DataFrame, y_new: pd.Series
) -> dict:
    """
    Concept drift check: score the existing model against newly-arrived,
    newly-LABELED data (i.e. customers whose churn outcome is now known),
    and compare against the AUC it achieved at training time.

    This is deliberately kept separate from feature drift (see module
    docstring) -- a model can look perfectly fine on PSI while quietly
    degrading in actual predictive power, if the *relationship* between
    features and churn has shifted even though the features' raw
    distributions haven't.
    """
    from sklearn.metrics import roc_auc_score

    proba = model.predict_proba(X_new)[:, 1]
    new_auc = roc_auc_score(y_new, proba)
    drop = baseline_auc - new_auc

    return {
        "baseline_auc": baseline_auc,
        "new_data_auc": new_auc,
        "auc_drop": drop,
        "flag": "alert" if drop > AUC_DROP_ALERT_THRESHOLD else "stable",
    }


def should_retrain(feature_drift: pd.DataFrame, performance_drift: dict | None) -> dict:
    """
    Combine both checks into a single retrain decision with a documented
    reason -- this is the audit-trail habit again: a retraining pipeline
    that can't say *why* it retrained is exactly the kind of black-box
    behavior this project has avoided everywhere else.
    """
    reasons = []

    n_alert_features = (feature_drift["flag"] == "alert").sum()
    n_watch_features = (feature_drift["flag"] == "watch").sum()
    if n_alert_features > 0:
        alert_feats = feature_drift[feature_drift["flag"] == "alert"]["feature"].tolist()
        reasons.append(f"{n_alert_features} feature(s) with PSI > {PSI_ALERT_THRESHOLD} (significant shift): {alert_feats}")

    if performance_drift is not None and performance_drift["flag"] == "alert":
        reasons.append(
            f"AUC dropped {performance_drift['auc_drop']:.3f} on new labeled data "
            f"({performance_drift['baseline_auc']:.3f} -> {performance_drift['new_data_auc']:.3f})"
        )

    return {
        "should_retrain": len(reasons) > 0,
        "reasons": reasons,
        "n_alert_features": int(n_alert_features),
        "n_watch_features": int(n_watch_features),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def retrain_and_compare(
    train_fn, X_train_new: pd.DataFrame, y_train_new: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, old_model
) -> dict:
    """
    Retrain on the updated dataset and report a clear before/after
    comparison rather than silently swapping models -- a retrain that makes
    things worse should be visible, not hidden by the act of retraining
    itself.
    """
    from sklearn.metrics import roc_auc_score

    new_model = train_fn(X_train_new, y_train_new)

    old_auc = roc_auc_score(y_test, old_model.predict_proba(X_test)[:, 1])
    new_auc = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])

    return {
        "old_model_auc": old_auc,
        "new_model_auc": new_auc,
        "improved": new_auc > old_auc,
        "delta": new_auc - old_auc,
        "new_model": new_model,
        "retrained_at": datetime.now(timezone.utc).isoformat(),
    }
