"""
src/mining/clustering.py
────────────────────────
Phân cụm nguồn nước theo profile chỉ số:
  - K-Means, DBSCAN, Hierarchical (Ward)
  - Elbow method + Silhouette analysis để chọn k tối ưu
  - Profiling từng cụm (mean, std, percentile)
  - Cảnh báo cụm rủi ro cao (unsafe ratio > threshold)
  - Đánh giá: Silhouette, Davies-Bouldin, Calinski-Harabasz

Usage:
    from src.mining.clustering import WaterClusterer

    clusterer = WaterClusterer(algorithm="kmeans", k=3)
    labels = clusterer.fit(X)
    profile = clusterer.get_cluster_profiles(df)
    risk_map = clusterer.flag_risk_clusters(profile)
    clusterer.print_summary()
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, List, Tuple
import warnings

FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

WHO_THRESHOLDS = {
    "ph": (6.5, 8.5), "Hardness": (50, 300), "Solids": (0, 500),
    "Chloramines": (0, 4.0), "Sulfate": (0, 250), "Conductivity": (0, 400),
    "Organic_carbon": (0, 2.0), "Trihalomethanes": (0, 80), "Turbidity": (0, 4.0),
}


class WaterClusterer:
    """
    Phân cụm nguồn nước theo profile chỉ số chất lượng.

    Parameters
    ----------
    algorithm : str
        'kmeans', 'dbscan', 'hierarchical'
    k : int
        Số cụm (chỉ dùng cho kmeans và hierarchical)
    random_seed : int
    risk_threshold_pct : float
        Tỷ lệ mẫu unsafe trong cụm để gán nhãn "High Risk"
    """

    def __init__(
        self,
        algorithm: str = "kmeans",
        k: int = 3,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
        random_seed: int = 42,
        risk_threshold_pct: float = 0.6,
    ):
        self.algorithm = algorithm
        self.k = k
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_seed = random_seed
        self.risk_threshold_pct = risk_threshold_pct

        self._model = None
        self._labels = None
        self._metrics: Dict[str, Any] = {}
        self._feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame) -> np.ndarray:
        """
        Phân cụm dữ liệu.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (đã chuẩn hoá)

        Returns
        -------
        labels : np.ndarray
            Nhãn cụm cho từng mẫu
        """
        self._feature_cols = list(X.columns)
        X_arr = X.values

        if self.algorithm == "kmeans":
            self._model = KMeans(
                n_clusters=self.k,
                random_state=self.random_seed,
                n_init=10,
            )
        elif self.algorithm == "dbscan":
            self._model = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                n_jobs=-1,
            )
        elif self.algorithm == "hierarchical":
            self._model = AgglomerativeClustering(
                n_clusters=self.k,
                linkage="ward",
            )
        else:
            raise ValueError(f"algorithm phải là: kmeans, dbscan, hierarchical")

        self._labels = self._model.fit_predict(X_arr)
        self._compute_metrics(X_arr)

        n_clusters = len(set(self._labels)) - (1 if -1 in self._labels else 0)
        n_noise = int((self._labels == -1).sum())
        print(f"✅ Phân cụm xong: {n_clusters} cụm, {n_noise} noise points — "
              f"Silhouette={self._metrics.get('silhouette', 0):.3f}")

        return self._labels

    def elbow_analysis(
        self, X: pd.DataFrame, k_range: Tuple[int, int] = (2, 8)
    ) -> pd.DataFrame:
        """
        Phân tích Elbow + Silhouette để chọn k tối ưu.

        Returns
        -------
        results : pd.DataFrame
            Bảng: k, inertia, silhouette, davies_bouldin
        """
        k_min, k_max = k_range
        results = []
        X_arr = X.values

        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
            labels = km.fit_predict(X_arr)

            sil = silhouette_score(X_arr, labels) if k > 1 else 0.0
            dbi = davies_bouldin_score(X_arr, labels) if k > 1 else np.inf
            ch = calinski_harabasz_score(X_arr, labels) if k > 1 else 0.0

            results.append({
                "k": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette": round(float(sil), 4),
                "davies_bouldin": round(float(dbi), 4),
                "calinski_harabasz": round(float(ch), 2),
            })

        df_results = pd.DataFrame(results)

        # Đề xuất k tối ưu (silhouette cao nhất)
        best_k = df_results.loc[df_results["silhouette"].idxmax(), "k"]
        print(f"\n📊 Elbow Analysis kết quả:")
        print(df_results.to_string(index=False))
        print(f"\n✅ Đề xuất k tối ưu: k={best_k} (Silhouette cao nhất)")

        return df_results

    def get_cluster_profiles(
        self, df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Tạo bảng profiling cho từng cụm.

        Trả về mean của từng chỉ số cho mỗi cụm + thông tin thêm:
        - Kích thước cụm (n, %)
        - Số chỉ số vượt ngưỡng WHO trung bình
        - Tỷ lệ unsafe (nếu có nhãn Potability)
        """
        if self._labels is None:
            raise RuntimeError("Gọi fit() trước.")

        df = df.copy()
        df["cluster"] = self._labels

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

        profile_data = []
        for cluster_id in sorted(df["cluster"].unique()):
            if cluster_id == -1:
                continue  # Bỏ qua noise (DBSCAN)

            mask = df["cluster"] == cluster_id
            cluster_df = df[mask]

            row = {"cluster": int(cluster_id), "n": len(cluster_df)}
            row["pct"] = round(len(cluster_df) / len(df) * 100, 1)

            # Mean của từng chỉ số
            for col in feature_cols:
                row[f"{col}_mean"] = round(float(cluster_df[col].mean()), 3)

            # Số chỉ số vượt ngưỡng WHO
            n_violations = 0
            for col, (lo, hi) in WHO_THRESHOLDS.items():
                if col in cluster_df.columns:
                    vals = cluster_df[col]
                    n_violations += int(((vals < lo) | (vals > hi)).sum())
            row["total_who_violations"] = n_violations
            row["avg_violations_per_sample"] = round(n_violations / len(cluster_df), 2)

            # Tỷ lệ unsafe nếu có Potability
            if "Potability" in df.columns:
                unsafe_ratio = 1 - cluster_df["Potability"].mean()
                row["unsafe_ratio"] = round(float(unsafe_ratio), 3)
            else:
                row["unsafe_ratio"] = None

            profile_data.append(row)

        profile_df = pd.DataFrame(profile_data)

        # Thêm nhãn rủi ro
        if "unsafe_ratio" in profile_df.columns and profile_df["unsafe_ratio"].notna().any():
            profile_df["risk_level"] = profile_df["unsafe_ratio"].apply(
                lambda x: "🔴 High Risk" if x > self.risk_threshold_pct
                else ("🟡 Medium Risk" if x > 0.4 else "🟢 Low Risk")
            )

        return profile_df

    def flag_risk_clusters(self, profile: pd.DataFrame) -> Dict[int, str]:
        """
        Trả về dict mapping cluster_id → mức rủi ro.
        """
        risk_map = {}
        if "risk_level" in profile.columns:
            for _, row in profile.iterrows():
                risk_map[int(row["cluster"])] = row["risk_level"]
        return risk_map

    def _compute_metrics(self, X_arr: np.ndarray) -> None:
        """Tính các metric đánh giá clustering."""
        valid_labels = self._labels[self._labels != -1]
        X_valid = X_arr[self._labels != -1]

        if len(set(valid_labels)) < 2:
            return

        try:
            self._metrics["silhouette"] = round(
                float(silhouette_score(X_valid, valid_labels)), 4
            )
            self._metrics["davies_bouldin"] = round(
                float(davies_bouldin_score(X_valid, valid_labels)), 4
            )
            self._metrics["calinski_harabasz"] = round(
                float(calinski_harabasz_score(X_valid, valid_labels)), 2
            )
            if hasattr(self._model, "inertia_"):
                self._metrics["inertia"] = round(float(self._model.inertia_), 2)
        except Exception as e:
            warnings.warn(f"Lỗi tính metrics: {e}")

    def print_summary(self) -> None:
        """In tóm tắt kết quả phân cụm."""
        if self._labels is None:
            print("Chưa fit. Gọi fit() trước.")
            return

        n_clusters = len(set(self._labels)) - (1 if -1 in self._labels else 0)
        print(f"\n{'='*50}")
        print(f"CLUSTERING SUMMARY — {self.algorithm.upper()} (k={self.k})")
        print(f"{'='*50}")
        print(f"Số cụm: {n_clusters}")

        for k, v in self._metrics.items():
            label = {
                "silhouette": "Silhouette Score  (↑ tốt hơn)",
                "davies_bouldin": "Davies-Bouldin     (↓ tốt hơn)",
                "calinski_harabasz": "Calinski-Harabasz  (↑ tốt hơn)",
                "inertia": "Inertia (within-SS)",
            }.get(k, k)
            print(f"  {label}: {v}")

        # Phân bố cụm
        print(f"\nPhân bố cụm:")
        unique, counts = np.unique(self._labels, return_counts=True)
        for cid, cnt in zip(unique, counts):
            label = "noise" if cid == -1 else f"cụm {cid}"
            print(f"  {label}: {cnt:>5} mẫu ({cnt/len(self._labels)*100:.1f}%)")
