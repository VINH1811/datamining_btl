"""
src/models/supervised.py
─────────────────────────
Phân lớp (Classification) và Hồi quy (Regression) cho chất lượng nước:
  - Phân lớp: an toàn / không an toàn (Potability 0/1)
  - Hồi quy: dự báo WQI liên tục
  - ≥ 2 baselines: ZeroR + Logistic Regression
  - k-Fold Cross-Validation (5-fold mặc định)
  - Metrics đúng: F1, PR-AUC, ROC-AUC (phân lớp); MAE, RMSE, sMAPE, R² (hồi quy)

Usage:
    from src.models.supervised import WaterClassifier, WaterRegressor

    # Phân lớp
    clf = WaterClassifier(algorithm="RandomForest", cv_folds=5)
    results = clf.fit(X_train, y_train)
    metrics = clf.evaluate(X_test, y_test)

    # Hồi quy
    reg = WaterRegressor(algorithm="XGBoost", cv_folds=5)
    reg.fit(X_train, wqi_train)
    reg_metrics = reg.evaluate(X_test, wqi_test)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from typing import Dict, Any, Optional, List, Tuple
import joblib
import warnings

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(numerator / (denominator + 1e-8)) * 100)


class WaterClassifier:
    """
    Phân lớp nguồn nước an toàn / không an toàn.

    Parameters
    ----------
    algorithm : str
        'RandomForest', 'XGBoost', 'LogisticRegression', 'SVM'
    cv_folds : int
        Số folds cross-validation (default 5)
    class_weight : str or None
        'balanced' để xử lý mất cân bằng lớp
    threshold : float
        Ngưỡng phân lớp (default 0.5 — có thể điều chỉnh để giảm FN)
    random_seed : int
    """

    BASELINES = ["ZeroR", "DummyStratified", "LogisticRegression"]

    def __init__(
        self,
        algorithm: str = "RandomForest",
        cv_folds: int = 5,
        class_weight: str = "balanced",
        threshold: float = 0.5,
        random_seed: int = 42,
    ):
        self.algorithm = algorithm
        self.cv_folds = cv_folds
        self.class_weight = class_weight
        self.threshold = threshold
        self.random_seed = random_seed

        self._model = None
        self._cv_results: Dict[str, Any] = {}
        self._test_metrics: Dict[str, Any] = {}
        self._baselines: Dict[str, Any] = {}

    def _build_model(self, name: Optional[str] = None):
        """Tạo model theo tên thuật toán."""
        algo = name or self.algorithm
        cw = self.class_weight

        model_map = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight=cw, random_state=self.random_seed, n_jobs=-1
            ),
            "XGBoost": (
                XGBClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    scale_pos_weight=2, random_state=self.random_seed,
                    eval_metric="logloss", verbosity=0
                ) if XGBOOST_AVAILABLE
                else GradientBoostingClassifier(n_estimators=100, random_state=self.random_seed)
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_seed
            ),
            "LogisticRegression": LogisticRegression(
                class_weight=cw, max_iter=1000, random_state=self.random_seed
            ),
            "SVM": SVC(
                kernel="rbf", class_weight=cw, probability=True, random_state=self.random_seed
            ),
            "ZeroR": DummyClassifier(strategy="most_frequent"),
            "DummyStratified": DummyClassifier(
                strategy="stratified", random_state=self.random_seed
            ),
        }

        if algo not in model_map:
            raise ValueError(f"algorithm phải là: {list(model_map.keys())}")
        return model_map[algo]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train model với k-Fold CV và tất cả baselines.

        Returns
        -------
        cv_results : dict
            Cross-validation scores + baseline comparison
        """
        X = X_train.fillna(X_train.median())
        y = y_train.astype(int)

        # ── Cross-validation ──
        self._model = self._build_model()
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_out = cross_validate(
                self._model, X, y, cv=cv,
                scoring=["f1_macro", "roc_auc", "average_precision"],
                return_train_score=True,
            )

        self._cv_results = {
            "algorithm": self.algorithm,
            "cv_folds": self.cv_folds,
            "f1_macro_mean": round(float(cv_out["test_f1_macro"].mean()), 4),
            "f1_macro_std": round(float(cv_out["test_f1_macro"].std()), 4),
            "roc_auc_mean": round(float(cv_out["test_roc_auc"].mean()), 4),
            "pr_auc_mean": round(float(cv_out["test_average_precision"].mean()), 4),
            "train_f1_mean": round(float(cv_out["train_f1_macro"].mean()), 4),
            "overfit_gap": round(float(
                cv_out["train_f1_macro"].mean() - cv_out["test_f1_macro"].mean()
            ), 4),
        }

        # Fit model toàn bộ train set
        self._model.fit(X, y)

        # ── Baselines ──
        self._fit_baselines(X, y)

        print(f"✅ {self.algorithm}: F1={self._cv_results['f1_macro_mean']:.4f} "
              f"(±{self._cv_results['f1_macro_std']:.4f}), "
              f"ROC-AUC={self._cv_results['roc_auc_mean']:.4f}, "
              f"PR-AUC={self._cv_results['pr_auc_mean']:.4f}")
        return self._cv_results

    def _fit_baselines(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train và đánh giá các baselines."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

        for bl_name in ["ZeroR", "DummyStratified", "LogisticRegression"]:
            bl_model = self._build_model(bl_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bl_cv = cross_validate(
                    bl_model, X, y, cv=cv,
                    scoring=["f1_macro", "roc_auc"], return_train_score=False
                )
            bl_model.fit(X, y)
            self._baselines[bl_name] = {
                "model": bl_model,
                "f1_macro": round(float(bl_cv["test_f1_macro"].mean()), 4),
                "roc_auc": round(float(bl_cv["test_roc_auc"].mean()), 4),
            }

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Đánh giá model trên test set.
        Bao gồm: F1, PR-AUC, ROC-AUC, Precision, Recall, Confusion Matrix.
        """
        if self._model is None:
            raise RuntimeError("Gọi fit() trước.")

        X = X_test.fillna(X_test.median())
        y = y_test.astype(int)

        y_proba = self._model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        cm = confusion_matrix(y, y_pred)

        self._test_metrics = {
            "n_test": len(y),
            "threshold": self.threshold,
            "accuracy": round(float((y == y_pred).mean()), 4),
            "f1_macro": round(float(f1_score(y, y_pred, average="macro")), 4),
            "f1_class1": round(float(f1_score(y, y_pred, pos_label=1)), 4),
            "precision": round(float(precision_score(y, y_pred)), 4),
            "recall": round(float(recall_score(y, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, y_proba)), 4),
            "pr_auc": round(float(average_precision_score(y, y_proba)), 4),
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        }

        return self._test_metrics

    def get_baseline_comparison(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """So sánh model chính với baselines."""
        y = y_test.astype(int)
        X = X_test.fillna(X_test.median())

        rows = []
        # Best model
        if self._test_metrics:
            rows.append({
                "Model": f"{self.algorithm} ⭐",
                "F1-macro": self._test_metrics["f1_macro"],
                "ROC-AUC": self._test_metrics["roc_auc"],
                "PR-AUC": self._test_metrics["pr_auc"],
                "Rank": "Best",
            })

        # Baselines
        for bl_name, bl_info in self._baselines.items():
            bl_pred = bl_info["model"].predict(X)
            bl_proba = (bl_info["model"].predict_proba(X)[:, 1]
                       if hasattr(bl_info["model"], "predict_proba") else None)

            row = {
                "Model": bl_name,
                "F1-macro": round(float(f1_score(y, bl_pred, average="macro")), 4),
                "ROC-AUC": round(float(roc_auc_score(y, bl_proba)) if bl_proba is not None else 0.5, 4),
                "PR-AUC": round(float(average_precision_score(y, bl_proba)) if bl_proba is not None else 0.0, 4),
                "Rank": "Baseline",
            }
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("F1-macro", ascending=False).reset_index(drop=True)
        df.index += 1
        return df

    def analyze_threshold_errors(
        self, X_test: pd.DataFrame, y_test: pd.Series,
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Phân tích lỗi gần ngưỡng quyết định (Rubric G).
        Cho thấy FP/FN thay đổi khi điều chỉnh threshold.
        """
        if self._model is None:
            raise RuntimeError("Gọi fit() trước.")

        if thresholds is None:
            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        X = X_test.fillna(X_test.median())
        y = y_test.astype(int)
        y_proba = self._model.predict_proba(X)[:, 1]

        rows = []
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()
            rows.append({
                "threshold": thr,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "f1": round(float(f1_score(y, y_pred, average="macro")), 4),
                "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
            })

        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Lưu model."""
        joblib.dump(self._model, path)
        print(f"✅ Model saved → {path}")


class WaterRegressor:
    """
    Hồi quy để dự báo WQI (Water Quality Index) liên tục.

    Parameters
    ----------
    algorithm : str
        'XGBoost', 'RandomForest', 'Ridge'
    cv_folds : int
    random_seed : int
    """

    def __init__(
        self,
        algorithm: str = "XGBoost",
        cv_folds: int = 5,
        random_seed: int = 42,
    ):
        self.algorithm = algorithm
        self.cv_folds = cv_folds
        self.random_seed = random_seed

        self._model = None
        self._cv_results: Dict[str, Any] = {}
        self._test_metrics: Dict[str, Any] = {}

    def _build_model(self, name: Optional[str] = None):
        algo = name or self.algorithm
        model_map = {
            "XGBoost": (
                XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    random_state=self.random_seed, verbosity=0
                ) if XGBOOST_AVAILABLE
                else RandomForestRegressor(n_estimators=100, random_state=self.random_seed)
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=200, random_state=self.random_seed, n_jobs=-1
            ),
            "Ridge": Ridge(alpha=1.0),
            "MeanPredictor": DummyRegressor(strategy="mean"),
            "LinearRegression": Ridge(alpha=0.001),
        }
        if algo not in model_map:
            raise ValueError(f"algorithm phải là: {list(model_map.keys())}")
        return model_map[algo]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train với k-Fold CV."""
        X = X_train.fillna(X_train.median())
        y = y_train.astype(float)

        self._model = self._build_model()
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_out = cross_validate(
                self._model, X, y, cv=cv,
                scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
                return_train_score=True,
            )

        self._cv_results = {
            "algorithm": self.algorithm,
            "mae_mean": round(float(-cv_out["test_neg_mean_absolute_error"].mean()), 4),
            "rmse_mean": round(float(-cv_out["test_neg_root_mean_squared_error"].mean()), 4),
            "r2_mean": round(float(cv_out["test_r2"].mean()), 4),
        }

        self._model.fit(X, y)

        print(f"✅ {self.algorithm} Regressor: "
              f"MAE={self._cv_results['mae_mean']:.4f}, "
              f"RMSE={self._cv_results['rmse_mean']:.4f}, "
              f"R²={self._cv_results['r2_mean']:.4f}")
        return self._cv_results

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Đánh giá trên test set: MAE, RMSE, R², sMAPE."""
        X = X_test.fillna(X_test.median())
        y = y_test.astype(float).values
        y_pred = self._model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        smape_val = smape(y, y_pred)

        self._test_metrics = {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "smape": round(float(smape_val), 2),
        }
        return self._test_metrics

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)
        print(f"✅ Model saved → {path}")
