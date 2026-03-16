"""
src/features/builder.py  (v2 — nâng cấp)
───────────────────────
Feature Engineering nâng cấp cho Water Quality:
  1. WHO threshold binary features  — tăng F1 đáng kể
  2. Interaction features           — tương tác giữa các chỉ số nguy hiểm
  3. WQI với trọng số WHO thực tế   — hồi quy chính xác hơn
  4. Rời rạc hoá dựa trên ngưỡng WHO (thay quantile)
  5. Feature selection

Usage:
    from src.features.builder import FeatureBuilder, compute_wqi, add_who_features

    df_enhanced = add_who_features(df)          # thêm WHO + interaction features
    df["WQI"] = compute_wqi(df, weights=...)
    builder = FeatureBuilder()
    df_disc = builder.discretize(df)
    transactions = builder.to_transactions(df_disc)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import List, Dict, Optional, Any
import warnings

FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
TARGET_COL = "Potability"

# Ngưỡng WHO (lo, hi) — ngoài khoảng này là nguy hiểm
WHO_THRESHOLDS = {
    "ph":              (6.5,   8.5),
    "Hardness":        (50.0,  300.0),
    "Solids":          (0.0,   500.0),
    "Chloramines":     (0.0,   4.0),
    "Sulfate":         (0.0,   250.0),
    "Conductivity":    (0.0,   400.0),
    "Organic_carbon":  (0.0,   2.0),
    "Trihalomethanes": (0.0,   80.0),
    "Turbidity":       (0.0,   4.0),
}

# Trọng số WQI theo mức độ nguy hiểm với sức khoẻ
DEFAULT_WQI_WEIGHTS = {
    "ph":              0.18,
    "Hardness":        0.06,
    "Solids":          0.12,
    "Chloramines":     0.15,
    "Sulfate":         0.08,
    "Conductivity":    0.07,
    "Organic_carbon":  0.10,
    "Trihalomethanes": 0.14,
    "Turbidity":       0.10,
}


def compute_wqi(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Tính Water Quality Index (WQI) — chỉ số tổng hợp [0, 100].
    Trọng số mặc định dựa trên mức độ nguy hiểm WHO, không phân đều.
    """
    if weights is None:
        weights = DEFAULT_WQI_WEIGHTS

    wqi = pd.Series(100.0, index=df.index)

    for col, weight in weights.items():
        if col not in df.columns:
            continue
        lo, hi = WHO_THRESHOLDS.get(col, (0, 1))
        ideal = (lo + hi) / 2
        tolerance = (hi - lo) / 2

        deviation = np.abs(df[col] - ideal) / max(tolerance, 1e-6)
        deviation = deviation.clip(0, 1)
        wqi -= weight * deviation * 100

    return wqi.clip(0, 100).round(2)


def add_who_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm features nâng cấp dựa trên ngưỡng WHO và tương tác giữa chỉ số.

    Loại features thêm vào:
    1. Binary flags: {col}_flag = 1 nếu vượt ngưỡng WHO (0 = an toàn)
    2. Deviation ratio: {col}_dev = mức độ vượt ngưỡng (0 = trong ngưỡng)
    3. Interaction: tích của 2 chỉ số nguy hiểm nhất
    4. Aggregate: tổng số chỉ số vượt ngưỡng, điểm nguy hiểm tổng hợp

    Returns
    -------
    df : pd.DataFrame với thêm ~25 features mới
    """
    df = df.copy()

    # ── 1. Binary violation flags và deviation ratio ──────────────
    for col, (lo, hi) in WHO_THRESHOLDS.items():
        if col not in df.columns:
            continue

        # Flag: 0 = an toàn, 1 = vượt ngưỡng
        df[f"{col}_flag"] = (
            (df[col] < lo) | (df[col] > hi)
        ).astype(int)

        # Deviation ratio: seberapa jauh từ ngưỡng (0 = trong ngưỡng)
        center = (lo + hi) / 2
        half_range = max((hi - lo) / 2, 1e-6)
        df[f"{col}_dev"] = ((np.abs(df[col] - center) - half_range) / half_range).clip(lower=0)

    # ── 2. Aggregate risk features ────────────────────────────────
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    dev_cols  = [c for c in df.columns if c.endswith("_dev")]

    if flag_cols:
        df["n_violations"]      = df[flag_cols].sum(axis=1)           # đếm số vi phạm
        df["has_any_violation"] = (df["n_violations"] > 0).astype(int)
        df["has_multi_violation"] = (df["n_violations"] >= 2).astype(int)
    if dev_cols:
        df["total_dev_score"]   = df[dev_cols].sum(axis=1)            # tổng mức độ vi phạm

    # ── 3. Interaction features — các cặp chỉ số nguy hiểm nhất ──
    interactions = [
        ("Chloramines_flag", "Turbidity_flag",        "chlor_x_turb"),
        ("Trihalomethanes_flag", "Turbidity_flag",    "thm_x_turb"),
        ("ph_flag", "Solids_flag",                    "ph_x_solids"),
        ("Organic_carbon_flag", "Trihalomethanes_flag","oc_x_thm"),
        ("Sulfate_flag", "Conductivity_flag",          "sulf_x_cond"),
    ]
    for col_a, col_b, name in interactions:
        if col_a in df.columns and col_b in df.columns:
            df[name] = df[col_a] * df[col_b]

    # ── 4. Ratio features từ giá trị thực ─────────────────────────
    if "Chloramines" in df.columns and "Turbidity" in df.columns:
        df["chlor_turb_ratio"] = df["Chloramines"] / (df["Turbidity"].abs() + 1e-6)
    if "Organic_carbon" in df.columns and "Trihalomethanes" in df.columns:
        df["oc_thm_ratio"] = df["Organic_carbon"] / (df["Trihalomethanes"].abs() + 1e-6)
    if "Sulfate" in df.columns and "Solids" in df.columns:
        df["sulfate_solids_ratio"] = df["Sulfate"] / (df["Solids"].abs() + 1e-6)

    return df


def discretize_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_bins: int = 3,
    strategy: str = "who",
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Rời rạc hoá các chỉ số.
    strategy='who': dùng ngưỡng WHO thực tế thay quantile.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    if labels is None:
        label_map = {
            2: ["Low", "High"],
            3: ["Low", "Medium", "High"],
        }
        labels = label_map.get(n_bins, [f"Bin{i}" for i in range(n_bins)])

    df_disc = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            continue

        try:
            if strategy == "who" and col in WHO_THRESHOLDS and n_bins == 3:
                lo, hi = WHO_THRESHOLDS[col]
                bins_edges = [-np.inf, lo, hi, np.inf]
                bins = pd.cut(
                    df[col], bins=bins_edges,
                    labels=["Low", "Medium", "High"],
                    include_lowest=True,
                )
            elif strategy == "quantile":
                bins = pd.qcut(df[col], q=n_bins, labels=labels, duplicates="drop")
            elif strategy == "uniform":
                bins = pd.cut(df[col], bins=n_bins, labels=labels[:n_bins])
            else:
                bins = pd.qcut(df[col], q=n_bins, labels=labels, duplicates="drop")

            df_disc[f"{col}_disc"] = bins

        except Exception as e:
            warnings.warn(f"Không thể rời rạc hoá {col}: {e}")
            df_disc[f"{col}_disc"] = np.nan

    return df_disc


class FeatureBuilder:
    """Builder tổng hợp cho tất cả feature engineering tasks."""

    def __init__(
        self,
        n_bins: int = 3,
        strategy: str = "who",
        top_k: int = 15,
        random_seed: int = 42,
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.top_k = top_k
        self.random_seed = random_seed
        self._selector = None

    def discretize(
        self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        return discretize_features(df, feature_cols, self.n_bins, self.strategy)

    def to_transactions(self, df_disc: pd.DataFrame) -> List[List[str]]:
        """Chuyển DataFrame đã rời rạc hoá thành danh sách transactions cho Apriori."""
        disc_cols = [c for c in df_disc.columns if c.endswith("_disc")]
        transactions = []
        for _, row in df_disc[disc_cols].iterrows():
            items = []
            for col in disc_cols:
                val = row[col]
                if pd.notna(val):
                    items.append(f"{col.replace('_disc', '')}_{val}")
            if items:
                transactions.append(items)
        return transactions

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info",
    ) -> List[str]:
        """Chọn top-k features theo mutual information."""
        if method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=min(self.top_k, X.shape[1]))
        else:
            selector = SelectKBest(f_classif, k=min(self.top_k, X.shape[1]))

        selector.fit(X.fillna(X.median()), y)
        self._selector = selector
        mask = selector.get_support()
        return list(X.columns[mask])

    def get_importance_scores(self, feature_names: List[str]) -> pd.DataFrame:
        if self._selector is None:
            raise RuntimeError("Gọi select_features() trước.")
        scores = self._selector.scores_
        pvalues = getattr(self._selector, "pvalues_", None)
        return pd.DataFrame({
            "feature": feature_names,
            "score": scores,
            "pvalue": pvalues if pvalues is not None else [np.nan] * len(scores),
        }).sort_values("score", ascending=False)

    def add_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gọi add_who_features (backward compatible)."""
        return add_who_features(df)
