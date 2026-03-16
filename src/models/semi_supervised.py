"""
src/models/semi_supervised.py
──────────────────────────────
Học bán giám sát cho phân lớp chất lượng nước:
  - Giả lập ít nhãn: giữ p% nhãn, ẩn phần còn lại
  - Label Spreading k-NN graph (sklearn)
  - Learning curve: F1 vs % nhãn
  - Phân tích pseudo-label sai ở vùng ít mẫu
  - So sánh supervised-only vs semi-supervised

Usage:
    from src.models.semi_supervised import WaterSemiSupervisedLearner

    ssl = WaterSemiSupervisedLearner(labeled_pct=0.20, n_neighbors=7)
    results = ssl.fit(X, y)
    learning_curve = ssl.compute_learning_curve(X, y)
    ssl.print_summary()
"""

import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score
)
from typing import List, Dict, Any, Optional, Tuple
import warnings


class WaterSemiSupervisedLearner:
    """
    Học bán giám sát cho dataset chất lượng nước.

    Mô phỏng kịch bản thực tế: chỉ có p% mẫu có nhãn Potability
    (việc xét nghiệm nước tốn kém → ít nhãn).

    Parameters
    ----------
    labeled_pct : float
        Tỷ lệ mẫu có nhãn (0.05 = 5%, 0.20 = 20%)
    algorithm : str
        'label_spreading' hoặc 'label_propagation'
    kernel : str
        'knn' hoặc 'rbf'
    n_neighbors : int
        Số láng giềng cho k-NN graph
    alpha : float
        Hệ số clamping (0=hoàn toàn giám sát, 1=hoàn toàn vô giám sát)
    confidence_threshold : float
        Ngưỡng chấp nhận pseudo-label
    random_seed : int
    """

    def __init__(
        self,
        labeled_pct: float = 0.20,
        algorithm: str = "label_spreading",
        kernel: str = "knn",
        n_neighbors: int = 7,
        alpha: float = 0.20,
        confidence_threshold: float = 0.80,
        random_seed: int = 42,
    ):
        self.labeled_pct = labeled_pct
        self.algorithm = algorithm
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.random_seed = random_seed

        self._model = None
        self._results: Dict[str, Any] = {}
        self._pseudo_label_analysis: Dict[str, Any] = {}

    def _create_partial_labels(
        self, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Giả lập ít nhãn: giữ labeled_pct% nhãn, đánh dấu còn lại là -1.

        Returns
        -------
        y_partial : np.ndarray
            Mảng nhãn với -1 cho mẫu không có nhãn
        labeled_idx : np.ndarray
            Chỉ số của các mẫu có nhãn
        """
        y_arr = y.values.astype(int)
        n_total = len(y_arr)
        n_labeled = max(10, int(n_total * self.labeled_pct))

        rng = np.random.RandomState(self.random_seed)

        # Phân tầng: đảm bảo cả 2 lớp đều có mặt
        labeled_idx = []
        for cls in [0, 1]:
            cls_idx = np.where(y_arr == cls)[0]
            n_cls = max(5, int(len(cls_idx) * self.labeled_pct))
            chosen = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
            labeled_idx.extend(chosen.tolist())

        labeled_idx = np.array(labeled_idx)

        y_partial = np.full(n_total, -1, dtype=int)
        y_partial[labeled_idx] = y_arr[labeled_idx]

        return y_partial, labeled_idx

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train semi-supervised model và so sánh với supervised-only.

        Returns
        -------
        results : dict
            Kết quả so sánh: semi_f1, supervised_f1, improvement
        """
        X_arr = X.fillna(X.median()).values
        y_arr = y.astype(int)

        # Tạo partial labels
        y_partial, labeled_idx = self._create_partial_labels(y_arr)
        n_labeled = (y_partial != -1).sum()
        n_unlabeled = (y_partial == -1).sum()

        # ── Supervised-only baseline (chỉ dùng nhãn đã biết) ──
        X_labeled = X_arr[y_partial != -1]
        y_labeled = y_partial[y_partial != -1]

        sup_model = RandomForestClassifier(
            n_estimators=100, random_state=self.random_seed, class_weight="balanced"
        )
        sup_model.fit(X_labeled, y_labeled)

        # ── Semi-supervised model ──
        if self.algorithm == "label_spreading":
            self._model = LabelSpreading(
                kernel=self.kernel,
                n_neighbors=self.n_neighbors,
                alpha=self.alpha,
                max_iter=1000,
            )
        else:
            self._model = LabelPropagation(
                kernel=self.kernel,
                n_neighbors=self.n_neighbors,
                max_iter=1000,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_arr, y_partial)

        # Đánh giá trên test set (nếu có) hoặc trên labeled set
        if X_test is not None and y_test is not None:
            X_eval = X_test.fillna(X_test.median()).values
            y_eval = y_test.astype(int).values
        else:
            # Dùng mẫu không nhãn làm pseudo test
            unlabeled_mask = y_partial == -1
            X_eval = X_arr[unlabeled_mask]
            y_eval = y_arr.values[unlabeled_mask]

        # Predictions
        semi_pred = self._model.predict(X_eval)
        sup_pred = sup_model.predict(X_eval)

        semi_f1 = float(f1_score(y_eval, semi_pred, average="macro", zero_division=0))
        sup_f1 = float(f1_score(y_eval, sup_pred, average="macro", zero_division=0))

        # Phân tích pseudo-labels
        self._pseudo_label_analysis = self._analyze_pseudo_labels(
            X_arr, y_arr.values, y_partial
        )

        self._results = {
            "labeled_pct": self.labeled_pct,
            "n_labeled": int(n_labeled),
            "n_unlabeled": int(n_unlabeled),
            "n_total": len(y_arr),
            "supervised_f1": round(sup_f1, 4),
            "semi_f1": round(semi_f1, 4),
            "improvement": round((semi_f1 - sup_f1) * 100, 2),
            "algorithm": self.algorithm,
        }

        print(f"✅ Semi-supervised ({self.labeled_pct*100:.0f}% labeled): "
              f"Supervised F1={sup_f1:.4f} → Semi F1={semi_f1:.4f} "
              f"(+{self._results['improvement']:.2f}%)")

        return self._results

    def _analyze_pseudo_labels(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_partial: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Phân tích chất lượng pseudo-labels ở vùng ít mẫu.
        """
        if self._model is None:
            return {}

        # Lấy các mẫu không có nhãn
        unlabeled_mask = y_partial == -1
        if not unlabeled_mask.any():
            return {}

        X_unlabeled = X[unlabeled_mask]
        y_true_unlabeled = y_true[unlabeled_mask]
        y_pseudo = self._model.predict(X_unlabeled)

        # Lấy xác suất nếu có (LabelSpreading lưu trong label_distributions_)
        try:
            proba = self._model.label_distributions_[unlabeled_mask]
            confidence = proba.max(axis=1)
        except Exception:
            confidence = np.ones(len(X_unlabeled)) * 0.8

        # Pseudo-label đúng và sai
        correct_mask = y_pseudo == y_true_unlabeled
        n_correct = int(correct_mask.sum())
        n_incorrect = int((~correct_mask).sum())
        n_total_pseudo = len(y_pseudo)
        error_rate = round(n_incorrect / max(n_total_pseudo, 1) * 100, 2)

        # Phân tích vùng ít mẫu: low-confidence pseudo-labels
        low_conf_mask = confidence < self.confidence_threshold
        n_low_conf = int(low_conf_mask.sum())
        low_conf_error_rate = round(
            (y_pseudo[low_conf_mask] != y_true_unlabeled[low_conf_mask]).mean() * 100, 2
        ) if n_low_conf > 0 else 0.0

        high_conf_error_rate = round(
            (y_pseudo[~low_conf_mask] != y_true_unlabeled[~low_conf_mask]).mean() * 100, 2
        ) if (~low_conf_mask).any() else 0.0

        return {
            "n_pseudo_labels": n_total_pseudo,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "error_rate_pct": error_rate,
            "n_low_confidence": n_low_conf,
            "low_conf_error_rate_pct": low_conf_error_rate,
            "high_conf_error_rate_pct": high_conf_error_rate,
            "avg_confidence": round(float(confidence.mean()), 3),
            "confidence_threshold": self.confidence_threshold,
        }

    def compute_learning_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        labeled_pct_list: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Tính learning curve: F1 theo % nhãn.

        Parameters
        ----------
        labeled_pct_list : list
            Danh sách tỷ lệ nhãn cần thử nghiệm

        Returns
        -------
        curve_df : pd.DataFrame
            Bảng: labeled_pct, supervised_f1, semi_f1, improvement
        """
        if labeled_pct_list is None:
            labeled_pct_list = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]

        results = []
        original_pct = self.labeled_pct

        X_arr = X.fillna(X.median()).values
        y_arr = y.astype(int)

        # Sử dụng phần test cố định (20%)
        rng = np.random.RandomState(self.random_seed)
        test_idx = rng.choice(len(y_arr), size=int(len(y_arr) * 0.2), replace=False)
        train_idx = np.setdiff1d(np.arange(len(y_arr)), test_idx)

        X_train_full = X_arr[train_idx]
        y_train_full = y_arr.values[train_idx]
        X_test = X_arr[test_idx]
        y_test = y_arr.values[test_idx]

        for pct in labeled_pct_list:
            self.labeled_pct = pct

            try:
                # Supervised-only
                n_labeled = max(5, int(len(y_train_full) * pct))
                labeled_idx = rng.choice(len(y_train_full), size=n_labeled, replace=False)
                sup_model = RandomForestClassifier(
                    n_estimators=50, random_state=self.random_seed, class_weight="balanced"
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sup_model.fit(X_train_full[labeled_idx], y_train_full[labeled_idx])
                sup_pred = sup_model.predict(X_test)
                sup_f1 = float(f1_score(y_test, sup_pred, average="macro", zero_division=0))

                if pct < 1.0:
                    # Semi-supervised
                    y_partial = np.full(len(y_train_full), -1, dtype=int)
                    y_partial[labeled_idx] = y_train_full[labeled_idx]

                    ssl_model = LabelSpreading(
                        kernel=self.kernel,
                        n_neighbors=self.n_neighbors,
                        alpha=self.alpha, max_iter=500
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ssl_model.fit(X_train_full, y_partial)
                    semi_pred = ssl_model.predict(X_test)
                    semi_f1 = float(f1_score(y_test, semi_pred, average="macro", zero_division=0))
                else:
                    semi_f1 = sup_f1

            except Exception as e:
                warnings.warn(f"Lỗi tại pct={pct}: {e}")
                sup_f1 = 0.0
                semi_f1 = 0.0

            results.append({
                "labeled_pct": pct,
                "labeled_pct_str": f"{int(pct*100)}%",
                "supervised_f1": round(sup_f1, 4),
                "semi_f1": round(semi_f1, 4),
                "improvement": round((semi_f1 - sup_f1) * 100, 2),
            })

        self.labeled_pct = original_pct

        df = pd.DataFrame(results)
        print(f"\n📈 Learning Curve ({len(labeled_pct_list)} điểm):")
        print(df[["labeled_pct_str", "supervised_f1", "semi_f1", "improvement"]].to_string(index=False))
        return df

    def print_summary(self) -> None:
        """In tóm tắt kết quả bán giám sát."""
        r = self._results
        pa = self._pseudo_label_analysis

        print(f"\n{'='*55}")
        print(f"SEMI-SUPERVISED LEARNING SUMMARY")
        print(f"Algorithm: {self.algorithm} | Kernel: {self.kernel}-NN (k={self.n_neighbors})")
        print(f"{'='*55}")
        print(f"  Labeled samples:    {r.get('n_labeled', '?'):>5} ({self.labeled_pct*100:.0f}%)")
        print(f"  Unlabeled samples:  {r.get('n_unlabeled', '?'):>5}")
        print(f"  Supervised F1:      {r.get('supervised_f1', 0):.4f}")
        print(f"  Semi-supervised F1: {r.get('semi_f1', 0):.4f}")
        print(f"  Improvement:        +{r.get('improvement', 0):.2f}%")

        if pa:
            print(f"\n  📊 Pseudo-label Analysis:")
            print(f"    Total pseudo-labels:    {pa.get('n_pseudo_labels', 0):>5}")
            print(f"    Correct:                {pa.get('n_correct', 0):>5} ({100 - pa.get('error_rate_pct', 0):.1f}%)")
            print(f"    Incorrect (noise):      {pa.get('n_incorrect', 0):>5} ({pa.get('error_rate_pct', 0):.1f}%)")
            print(f"    Low-confidence (<{self.confidence_threshold}):   {pa.get('n_low_confidence', 0):>5}")
            print(f"    Low-conf error rate:    {pa.get('low_conf_error_rate_pct', 0):.1f}%")
            print(f"    High-conf error rate:   {pa.get('high_conf_error_rate_pct', 0):.1f}%")
