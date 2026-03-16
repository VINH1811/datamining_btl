"""
src/evaluation/metrics.py
──────────────────────────
Tính toán tất cả metrics cho pipeline chất lượng nước.

Classification metrics:
  - F1-macro, F1-class, Precision, Recall
  - ROC-AUC, PR-AUC (Average Precision)
  - Confusion Matrix (TP, FP, FN, TN)

Regression metrics:
  - MAE, RMSE, R², sMAPE

Clustering metrics:
  - Silhouette Score, Davies-Bouldin, Calinski-Harabasz

Usage:
    from src.evaluation.metrics import (
        compute_classification_metrics,
        compute_regression_metrics,
        compute_clustering_metrics,
    )
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from typing import Dict, Any, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    labels: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Tính đầy đủ metrics phân lớp.

    Parameters
    ----------
    y_true : array
        Nhãn thực tế
    y_pred : array
        Nhãn dự đoán
    y_proba : array, optional
        Xác suất lớp 1 (cần cho ROC-AUC, PR-AUC)
    threshold : float
        Ngưỡng phân lớp đã dùng
    labels : list, optional
        Tên các lớp (ví dụ: ['Unsafe', 'Safe'])

    Returns
    -------
    metrics : dict
        Dictionary đầy đủ các metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics = {
        "n_samples": len(y_true),
        "threshold": threshold,
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "f1_class0": round(float(f1_score(y_true, y_pred, pos_label=0, average="binary")), 4),
        "f1_class1": round(float(f1_score(y_true, y_pred, pos_label=1, average="binary")), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # Diễn giải
        "fp_rate": round(float(fp / max(fp + tn, 1)), 4),  # False Alarm Rate
        "fn_rate": round(float(fn / max(fn + tp, 1)), 4),  # Miss Rate
    }

    # Các metrics cần xác suất
    if y_proba is not None:
        y_proba = np.array(y_proba)
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    # Classification report đầy đủ
    label_names = labels if labels else [f"Class {i}" for i in range(len(np.unique(y_true)))]
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=label_names if len(label_names) == len(np.unique(y_true)) else None,
        zero_division=0,
    )

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Tính đầy đủ metrics hồi quy.

    Parameters
    ----------
    y_true : array
        Giá trị thực tế (WQI)
    y_pred : array
        Giá trị dự đoán

    Returns
    -------
    metrics : dict
        MAE, RMSE, R², sMAPE, và thống kê residual
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residuals = y_true - y_pred

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # sMAPE
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_val = float(np.mean(numerator / (denominator + 1e-8)) * 100)

    return {
        "n_samples": len(y_true),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "smape": round(smape_val, 2),
        "mean_residual": round(float(residuals.mean()), 4),
        "std_residual": round(float(residuals.std()), 4),
        "max_overestimate": round(float(residuals.min()), 4),
        "max_underestimate": round(float(residuals.max()), 4),
        "pct_within_5": round(float((np.abs(residuals) <= 5).mean() * 100), 2),
        "pct_within_10": round(float((np.abs(residuals) <= 10).mean() * 100), 2),
    }


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Tính đầy đủ metrics đánh giá clustering (không cần nhãn thực tế).

    Parameters
    ----------
    X : array
        Feature matrix đã chuẩn hoá
    labels : array
        Nhãn cụm (có thể chứa -1 cho noise)

    Returns
    -------
    metrics : dict
        Silhouette, Davies-Bouldin, Calinski-Harabasz, + thống kê cụm
    """
    # Loại bỏ noise points (-1)
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    n_clusters = len(set(labels_valid))
    n_noise = int((labels == -1).sum())

    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_samples": len(labels),
    }

    if n_clusters < 2:
        metrics.update({"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None})
        return metrics

    try:
        metrics["silhouette"] = round(
            float(silhouette_score(X_valid, labels_valid)), 4
        )
        metrics["davies_bouldin"] = round(
            float(davies_bouldin_score(X_valid, labels_valid)), 4
        )
        metrics["calinski_harabasz"] = round(
            float(calinski_harabasz_score(X_valid, labels_valid)), 2
        )
    except Exception as e:
        metrics["error"] = str(e)

    # Phân bố cụm
    cluster_sizes = {}
    for cid in sorted(set(labels_valid)):
        n = int((labels_valid == cid).sum())
        cluster_sizes[f"cluster_{cid}"] = {
            "n": n,
            "pct": round(n / len(labels_valid) * 100, 1),
        }
    metrics["cluster_distribution"] = cluster_sizes

    return metrics


def print_classification_summary(metrics: Dict[str, Any]) -> None:
    """In tóm tắt kết quả phân lớp."""
    print(f"\n{'='*50}")
    print("CLASSIFICATION EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Samples:    {metrics.get('n_samples', '?')}")
    print(f"  Accuracy:   {metrics.get('accuracy', 0):.4f}")
    print(f"  F1-macro:   {metrics.get('f1_macro', 0):.4f}")
    print(f"  F1-class1:  {metrics.get('f1_class1', 0):.4f}")
    print(f"  Precision:  {metrics.get('precision', 0):.4f}")
    print(f"  Recall:     {metrics.get('recall', 0):.4f}")

    if metrics.get("roc_auc"):
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    if metrics.get("pr_auc"):
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics.get('tp',0):>5}  FP={metrics.get('fp',0):>5}")
    print(f"    FN={metrics.get('fn',0):>5}  TN={metrics.get('tn',0):>5}")

    print(f"\n  ⚠ False Positive Rate (báo sai an toàn): {metrics.get('fp_rate',0):.3f}")
    print(f"  ⚠ False Negative Rate (bỏ sót nguy hiểm): {metrics.get('fn_rate',0):.3f}")


def print_regression_summary(metrics: Dict[str, Any]) -> None:
    """In tóm tắt kết quả hồi quy."""
    print(f"\n{'='*50}")
    print("REGRESSION EVALUATION SUMMARY (WQI)")
    print(f"{'='*50}")
    print(f"  Samples:         {metrics.get('n_samples', '?')}")
    print(f"  MAE:             {metrics.get('mae', 0):.4f}")
    print(f"  RMSE:            {metrics.get('rmse', 0):.4f}")
    print(f"  R²:              {metrics.get('r2', 0):.4f}")
    print(f"  sMAPE:           {metrics.get('smape', 0):.2f}%")
    print(f"  Mean Residual:   {metrics.get('mean_residual', 0):.4f}")
    print(f"  Std Residual:    {metrics.get('std_residual', 0):.4f}")
    print(f"  Within ±5 units: {metrics.get('pct_within_5', 0):.1f}%")
    print(f"  Within ±10 units:{metrics.get('pct_within_10', 0):.1f}%")
