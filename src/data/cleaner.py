"""
src/data/cleaner.py  (v2 — nâng cấp)
────────────────────
Tiền xử lý dữ liệu chất lượng nước:
  1. Xử lý missing values (KNN imputation — tốt hơn median)
  2. Winsorization thay IQR removal (giữ 100% dữ liệu, chỉ clip)
  3. RobustScaler (tốt hơn StandardScaler với Solids skewed)
  4. Phân chia train/test có phân tầng

Usage:
    from src.data.cleaner import WaterDataCleaner

    cleaner = WaterDataCleaner(config)
    df_clean = cleaner.fit_transform(df_raw)
    X_train, X_test, y_train, y_test = cleaner.split(df_clean)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Tuple, Dict, Any, Optional
import joblib
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
TARGET_COL = "Potability"


class WaterDataCleaner:
    """
    Pipeline tiền xử lý nâng cấp cho dataset chất lượng nước.

    Cải tiến v2:
    - KNN imputation thay median (giữ được correlation giữa features)
    - Winsorization thay IQR removal (giữ 100% dữ liệu, chỉ clip outlier)
    - RobustScaler mặc định (tốt hơn với Solids phân phối lệch mạnh)
    - Báo cáo chi tiết before/after cho từng bước

    Parameters
    ----------
    missing_strategy : str
        'knn' (khuyến nghị) | 'median' | 'mean' | 'drop'
    knn_neighbors : int
        Số láng giềng KNN imputation (mặc định 5)
    outlier_method : str
        'winsor' (khuyến nghị) | 'iqr' | 'zscore' | 'none'
    winsor_lower : float
        Percentile dưới để clip (mặc định 0.01 = 1%)
    winsor_upper : float
        Percentile trên để clip (mặc định 0.99 = 99%)
    scaling : str
        'robust' (khuyến nghị) | 'standard' | 'minmax' | 'none'
    """

    def __init__(
        self,
        missing_strategy: str = "knn",
        knn_neighbors: int = 5,
        outlier_method: str = "winsor",
        outlier_threshold: float = 3.0,
        winsor_lower: float = 0.01,
        winsor_upper: float = 0.99,
        scaling: str = "robust",
        test_size: float = 0.20,
        random_seed: int = 42,
    ):
        self.missing_strategy = missing_strategy
        self.knn_neighbors = knn_neighbors
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.winsor_lower = winsor_lower
        self.winsor_upper = winsor_upper
        self.scaling = scaling
        self.test_size = test_size
        self.random_seed = random_seed

        self._imputer: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._winsor_bounds: Dict[str, Tuple[float, float]] = {}
        self._fitted = False
        self.cleaning_report: Dict[str, Any] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit và transform toàn bộ dataset."""
        df = df.copy()
        n_initial = len(df)
        n_missing_before = int(df[FEATURE_COLS].isna().sum().sum())

        # Bước 1: Missing values
        df, n_imputed = self._handle_missing(df, fit=True)

        # Bước 2: Outlier
        df, n_outlier_affected = self._handle_outliers(df, fit=True)

        # Bước 3: Scale
        df = self._scale_features(df, fit=True)

        self._fitted = True
        self.cleaning_report = {
            "n_initial": n_initial,
            "n_imputed_cells": n_imputed,
            "n_outlier_affected": n_outlier_affected,
            "n_final": len(df),
            "retention_pct": round(len(df) / n_initial * 100, 1),
            "missing_strategy": self.missing_strategy,
            "outlier_method": self.outlier_method,
            "scaling": self.scaling,
        }
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform tập mới (không fit lại)."""
        if not self._fitted:
            raise RuntimeError("Gọi fit_transform() trước transform().")
        df = df.copy()
        df, _ = self._handle_missing(df, fit=False)
        df = self._handle_outliers_transform(df)
        df = self._scale_features(df, fit=False)
        return df

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Phân chia train/test có phân tầng theo Potability."""
        if TARGET_COL not in df.columns:
            raise ValueError(f"Không tìm thấy cột target '{TARGET_COL}'")
        df_valid = df.dropna(subset=[TARGET_COL])
        X = df_valid[FEATURE_COLS]
        y = df_valid[TARGET_COL].astype(int)
        return train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_seed,
        )

    # ── Private methods ────────────────────────────────────────────

    def _handle_missing(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, int]:
        """Xử lý missing values — KNN hoặc median."""
        feature_data = df[FEATURE_COLS].copy()
        n_missing = int(feature_data.isna().sum().sum())

        if self.missing_strategy == "drop":
            mask = feature_data.notna().all(axis=1)
            df = df[mask].reset_index(drop=True)
            return df, n_missing

        if self.missing_strategy == "knn":
            if fit:
                self._imputer = KNNImputer(
                    n_neighbors=self.knn_neighbors,
                    weights="distance",
                )
                imputed = self._imputer.fit_transform(feature_data)
            else:
                imputed = self._imputer.transform(feature_data)
        else:
            strategy = self.missing_strategy
            if fit:
                self._imputer = SimpleImputer(strategy=strategy)
                imputed = self._imputer.fit_transform(feature_data)
            else:
                imputed = self._imputer.transform(feature_data)

        df = df.copy()
        df[FEATURE_COLS] = imputed
        return df, n_missing

    def _handle_outliers(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, int]:
        """Xử lý outlier — Winsorization (clip) hoặc IQR removal."""
        if self.outlier_method == "none":
            return df, 0

        n_before = len(df)
        feature_data = df[FEATURE_COLS]

        if self.outlier_method == "winsor":
            # Winsorization: clip tại percentile thay vì xóa dòng
            n_clipped = 0
            df = df.copy()
            for col in FEATURE_COLS:
                if fit:
                    lo = float(feature_data[col].quantile(self.winsor_lower))
                    hi = float(feature_data[col].quantile(self.winsor_upper))
                    self._winsor_bounds[col] = (lo, hi)
                else:
                    lo, hi = self._winsor_bounds.get(col, (
                        feature_data[col].min(), feature_data[col].max()
                    ))
                n_clipped += int(
                    ((feature_data[col] < lo) | (feature_data[col] > hi)).sum()
                )
                df[col] = df[col].clip(lower=lo, upper=hi)
            return df, n_clipped

        elif self.outlier_method == "iqr":
            mask = pd.Series([True] * len(df), index=df.index)
            for col in FEATURE_COLS:
                q1 = feature_data[col].quantile(0.25)
                q3 = feature_data[col].quantile(0.75)
                iqr = q3 - q1
                lo = q1 - 1.5 * iqr
                hi = q3 + 1.5 * iqr
                mask &= (feature_data[col] >= lo) & (feature_data[col] <= hi)
            df = df[mask].reset_index(drop=True)
            return df, n_before - len(df)

        elif self.outlier_method == "zscore":
            z_scores = np.abs(
                (feature_data - feature_data.mean()) / feature_data.std()
            )
            mask = (z_scores < self.outlier_threshold).all(axis=1)
            df = df[mask].reset_index(drop=True)
            return df, n_before - len(df)

        return df, 0

    def _handle_outliers_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization bounds đã fit lên tập mới."""
        if self.outlier_method != "winsor" or not self._winsor_bounds:
            return df
        df = df.copy()
        for col in FEATURE_COLS:
            if col in df.columns and col in self._winsor_bounds:
                lo, hi = self._winsor_bounds[col]
                df[col] = df[col].clip(lower=lo, upper=hi)
        return df

    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Chuẩn hoá đặc trưng — RobustScaler mặc định."""
        if self.scaling == "none":
            return df

        scaler_map = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        if self.scaling not in scaler_map:
            raise ValueError(f"scaling phải là: {list(scaler_map)}")

        feature_data = df[FEATURE_COLS].copy()
        if fit:
            self._scaler = scaler_map[self.scaling]()
            scaled = self._scaler.fit_transform(feature_data)
        else:
            scaled = self._scaler.transform(feature_data)

        df = df.copy()
        df[FEATURE_COLS] = scaled
        return df

    def save_artifacts(self, output_dir: str) -> None:
        """Lưu imputer và scaler để dùng lại sau."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        if self._imputer:
            joblib.dump(self._imputer, f"{output_dir}/imputer.pkl")
        if self._scaler:
            joblib.dump(self._scaler, f"{output_dir}/scaler.pkl")
        if self._winsor_bounds:
            joblib.dump(self._winsor_bounds, f"{output_dir}/winsor_bounds.pkl")
        print(f"✅ Artifacts saved to {output_dir}/")

    @classmethod
    def load_artifacts(cls, output_dir: str) -> "WaterDataCleaner":
        """Khôi phục cleaner từ artifacts đã lưu."""
        obj = cls()
        try:
            obj._imputer = joblib.load(f"{output_dir}/imputer.pkl")
            obj._scaler  = joblib.load(f"{output_dir}/scaler.pkl")
            obj._fitted  = True
            try:
                obj._winsor_bounds = joblib.load(f"{output_dir}/winsor_bounds.pkl")
            except FileNotFoundError:
                pass
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Không tìm thấy artifact: {e}")
        return obj

    def print_report(self) -> None:
        """In báo cáo tiền xử lý."""
        r = self.cleaning_report
        method_label = {
            "winsor": f"winsorization [p{int(self.winsor_lower*100)},p{int(self.winsor_upper*100)}]",
            "iqr": "IQR removal",
            "zscore": "Z-score removal",
            "none": "none",
        }.get(r.get("outlier_method", ""), r.get("outlier_method", ""))

        print("\n📋 PREPROCESSING REPORT (v2)")
        print(f"  Rows ban đầu:     {r.get('n_initial', '?'):>6,}")
        print(f"  Cells imputed:    {r.get('n_imputed_cells', 0):>6,}  ({r.get('missing_strategy', '')})")
        print(f"  Outlier affected: {r.get('n_outlier_affected', 0):>6,}  ({method_label})")
        print(f"  Rows sau xử lý:   {r.get('n_final', '?'):>6,}  ({r.get('retention_pct', 0):.1f}% giữ lại)")
        print(f"  Scaling:          {r.get('scaling', '')}")
