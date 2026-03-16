"""
src/visualization/plots.py
───────────────────────────
Tất cả hàm vẽ biểu đồ cho Water Quality Analysis.

Bao gồm:
  - EDA: distribution, missing values, correlation heatmap, target pie
  - Preprocessing: before/after boxplot
  - Clustering: elbow curve, silhouette plot, cluster heatmap
  - Association: lift bar chart
  - Models: confusion matrix, ROC/PR curve, feature importance
  - Semi-supervised: learning curve
  - Regression: residual plot, actual vs predicted

Usage:
    from src.visualization.plots import WaterQualityPlotter

    plotter = WaterQualityPlotter(output_dir="outputs/figures", style="seaborn-v0_8")
    plotter.plot_eda_overview(df)
    plotter.plot_confusion_matrix(metrics)
    plotter.plot_learning_curve(curve_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, List, Dict, Any

FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]


class WaterQualityPlotter:
    """
    Vẽ tất cả biểu đồ cho pipeline phân tích chất lượng nước.

    Parameters
    ----------
    output_dir : str
        Thư mục lưu ảnh biểu đồ
    style : str
        Matplotlib style sheet
    figsize_default : tuple
        Kích thước mặc định (width, height)
    dpi : int
        Độ phân giải ảnh xuất
    """

    def __init__(
        self,
        output_dir: str = "outputs/figures",
        style: str = "seaborn-v0_8-whitegrid",
        figsize_default: tuple = (12, 6),
        dpi: int = 120,
    ):
        self.output_dir = output_dir
        self.style = style
        self.figsize_default = figsize_default
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

        try:
            plt.style.use(style)
        except Exception:
            pass

        self.colors = {
            "safe": "#2196F3",
            "unsafe": "#F44336",
            "neutral": "#9E9E9E",
            "highlight": "#FF9800",
            "cluster": ["#E53935", "#1E88E5", "#43A047", "#8E24AA", "#F4511E"],
        }

    # ── EDA Plots ────────────────────────────────────────────────

    def plot_eda_overview(self, df: pd.DataFrame, save: bool = True) -> str:
        """
        Tổng quan EDA: 4 biểu đồ trong 1 figure.
        1. Distribution của pH (histogram + KDE)
        2. Missing values bar chart
        3. Target distribution pie chart
        4. Coefficient of Variation bar chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("EDA Tổng Quan — Chất Lượng Nước", fontsize=14, fontweight="bold")

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

        # 1. pH Distribution
        ax = axes[0, 0]
        if "ph" in df.columns:
            vals = df["ph"].dropna()
            ax.hist(vals, bins=40, color=self.colors["safe"], alpha=0.7, edgecolor="white")
            ax.axvline(6.5, color="red", linestyle="--", label="WHO lo (6.5)")
            ax.axvline(8.5, color="red", linestyle="--", label="WHO hi (8.5)")
            ax.set_title("Phân phối pH")
            ax.set_xlabel("pH")
            ax.legend(fontsize=8)

        # 2. Missing Values
        ax = axes[0, 1]
        missing = df[feature_cols].isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            bars = ax.bar(missing.index, missing.values, color=self.colors["highlight"])
            ax.set_title("Missing Values theo Cột")
            ax.set_ylabel("Số lượng missing")
            ax.tick_params(axis="x", rotation=45)
            for bar, val in zip(bars, missing.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f"{val:,}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "✅ Không có\nmissing values", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            ax.set_title("Missing Values")

        # 3. Target Distribution Pie
        ax = axes[1, 0]
        if "Potability" in df.columns:
            vc = df["Potability"].value_counts()
            labels = [f"Không an toàn (0)\n{vc.get(0,0):,}", f"An toàn (1)\n{vc.get(1,0):,}"]
            colors = [self.colors["unsafe"], self.colors["safe"]]
            wedges, texts, autotexts = ax.pie(
                [vc.get(0, 0), vc.get(1, 0)],
                labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, pctdistance=0.85
            )
            ax.set_title("Phân phối Target (Potability)")

        # 4. Coefficient of Variation
        ax = axes[1, 1]
        cvs = {}
        for col in feature_cols:
            vals = df[col].dropna()
            if vals.std() > 0 and vals.mean() != 0:
                cvs[col] = abs(vals.std() / vals.mean()) * 100
        if cvs:
            cv_series = pd.Series(cvs).sort_values(ascending=False)
            ax.bar(cv_series.index, cv_series.values, color=self.colors["neutral"], edgecolor="white")
            ax.set_title("Coefficient of Variation (%)")
            ax.set_ylabel("CV (%)")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = f"{self.output_dir}/01_eda_overview.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path

    def plot_correlation_heatmap(
        self, df: pd.DataFrame, save: bool = True
    ) -> str:
        """Heatmap tương quan Pearson giữa các chỉ số."""
        feature_cols = [c for c in FEATURE_COLS + ["Potability"] if c in df.columns]
        corr = df[feature_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", center=0,
            cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, annot_kws={"size": 8},
        )
        ax.set_title("Correlation Heatmap — Chỉ số Chất lượng Nước", fontsize=12)
        plt.tight_layout()

        path = f"{self.output_dir}/02_correlation_heatmap.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path

    # ── Clustering Plots ──────────────────────────────────────────

    def plot_elbow_curve(
        self, elbow_df: pd.DataFrame, best_k: int = 3, save: bool = True
    ) -> str:
        """Elbow curve + Silhouette để chọn k tối ưu."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Elbow Analysis — Chọn K tối ưu", fontsize=12)

        # Inertia
        ax1.plot(elbow_df["k"], elbow_df["inertia"], "o-", color=self.colors["safe"], linewidth=2)
        ax1.axvline(best_k, color=self.colors["highlight"], linestyle="--", label=f"k={best_k}")
        ax1.set_xlabel("Số cụm k")
        ax1.set_ylabel("Inertia (Within-SS)")
        ax1.set_title("Elbow Method")
        ax1.legend()

        # Silhouette
        ax2.bar(elbow_df["k"], elbow_df["silhouette"],
                color=[self.colors["highlight"] if k == best_k else self.colors["neutral"]
                       for k in elbow_df["k"]])
        ax2.set_xlabel("Số cụm k")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score vs k")

        plt.tight_layout()
        path = f"{self.output_dir}/03_elbow_silhouette.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path

    def plot_cluster_heatmap(
        self, profile: pd.DataFrame, save: bool = True
    ) -> str:
        """Heatmap profiling cụm — mean feature theo cụm."""
        mean_cols = [c for c in profile.columns if c.endswith("_mean")]
        if not mean_cols or "cluster" not in profile.columns:
            print("⚠ Không đủ dữ liệu để vẽ cluster heatmap.")
            return ""

        heat_data = profile.set_index("cluster")[mean_cols]
        heat_data.columns = [c.replace("_mean", "") for c in heat_data.columns]

        # Chuẩn hoá mỗi feature về [0, 1] để so sánh
        normalized = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(12, max(3, len(profile) + 1)))
        sns.heatmap(
            normalized, annot=heat_data.round(2), fmt=".2f",
            cmap="YlOrRd", ax=ax, linewidths=0.5, cbar_kws={"label": "Normalized value"},
        )

        # Thêm nhãn rủi ro nếu có
        if "risk_level" in profile.columns:
            y_labels = [f"Cụm {int(row['cluster'])} — {row['risk_level']}"
                       for _, row in profile.iterrows()]
            ax.set_yticklabels(y_labels, rotation=0)

        ax.set_title("Cluster Profiles — Chỉ số Chất lượng Nước", fontsize=12)
        ax.set_xlabel("Chỉ số")
        plt.tight_layout()

        path = f"{self.output_dir}/04_cluster_heatmap.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path

    # ── Model Evaluation Plots ────────────────────────────────────

    def plot_confusion_matrix(
        self, metrics: Dict[str, Any], labels: List[str] = None, save: bool = True
    ) -> str:
        """Confusion matrix với annotation đầy đủ."""
        if labels is None:
            labels = ["Không an toàn (0)", "An toàn (1)"]

        cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_pct, annot=False, fmt=".2%", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
            linewidths=1, linecolor="white",
        )

        # Annotation thủ công: n + %
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                color = "white" if cm_pct[i, j] > 0.5 else "black"
                ax.text(
                    j + 0.5, i + 0.45,
                    f"{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color,
                )
                ax.text(
                    j + 0.5, i + 0.6,
                    f"({cm_pct[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    fontsize=9, color=color,
                )

        ax.set_title(
            f"Confusion Matrix\n"
            f"F1={metrics.get('f1_macro', 0):.4f} | "
            f"ROC-AUC={metrics.get('roc_auc', 0):.4f}",
            fontsize=11,
        )
        ax.set_ylabel("Nhãn thực tế")
        ax.set_xlabel("Nhãn dự đoán")
        plt.tight_layout()

        path = f"{self.output_dir}/05_confusion_matrix.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Analysis — WQI",
        save: bool = True,
    ) -> str:
        """Residual plot cho regression: scatter + histogram."""
        residuals = np.array(y_true) - np.array(y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=12)

        # Scatter: Predicted vs Residuals
        ax1.scatter(y_pred, residuals, alpha=0.4, color=self.colors["safe"], s=15)
        ax1.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax1.fill_between(
            [y_pred.min(), y_pred.max()], -5, 5,
            alpha=0.1, color="green", label="±5 MAE zone"
        )
        ax1.set_xlabel("Giá trị dự báo (WQI)")
        ax1.set_ylabel("Residuals (actual - predicted)")
        ax1.set_title("Residuals vs Predicted")
        ax1.legend(fontsize=8)

        # Histogram residuals
        ax2.hist(residuals, bins=40, color=self.colors["safe"], alpha=0.7, edgecolor="white")
        ax2.axvline(residuals.mean(), color="red", linestyle="--",
                    label=f"Mean={residuals.mean():.2f}")
        ax2.axvline(0, color="green", linestyle="-", linewidth=1.5, label="Ideal")
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Phân phối Residuals")
        ax2.legend()

        plt.tight_layout()
        path = f"{self.output_dir}/06_residual_analysis.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path

    def plot_learning_curve(
        self, curve_df: pd.DataFrame, save: bool = True
    ) -> str:
        """Learning curve: Supervised vs Semi-supervised F1 theo % nhãn."""
        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(curve_df))
        labels = curve_df["labeled_pct_str"].tolist()

        ax.plot(x, curve_df["supervised_f1"], "o-", color=self.colors["unsafe"],
                linewidth=2, markersize=7, label="Supervised only")
        ax.plot(x, curve_df["semi_f1"], "s-", color=self.colors["safe"],
                linewidth=2, markersize=7, label="Semi-supervised (Label Spreading)")

        # Fill improvement area
        ax.fill_between(x, curve_df["supervised_f1"], curve_df["semi_f1"],
                        alpha=0.15, color=self.colors["safe"], label="Vùng cải thiện")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("% Nhãn có nhãn (labeled)")
        ax.set_ylabel("F1-macro Score")
        ax.set_title("Learning Curve — Semi-supervised vs Supervised\nPhân tích Chất lượng Nước")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        path = f"{self.output_dir}/07_learning_curve.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 9,
        save: bool = True,
    ) -> str:
        """Bar chart feature importance từ Random Forest."""
        if "feature" not in importance_df.columns or "score" not in importance_df.columns:
            print("⚠ importance_df cần cột 'feature' và 'score'")
            return ""

        top = importance_df.head(top_n).sort_values("score")

        fig, ax = plt.subplots(figsize=(8, 6))
        colors_bar = [self.colors["highlight"] if i < 3 else self.colors["safe"]
                      for i in range(len(top))][::-1]
        bars = ax.barh(top["feature"], top["score"], color=colors_bar)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{width:.4f}", va="center", fontsize=8)

        ax.set_title("Feature Importance (F-statistic)", fontsize=12)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()

        path = f"{self.output_dir}/08_feature_importance.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path

    def plot_association_rules(
        self, rules: pd.DataFrame, top_n: int = 10, save: bool = True
    ) -> str:
        """Scatter plot support vs confidence, kích thước = lift."""
        if rules is None or len(rules) == 0:
            print("⚠ Không có rules để vẽ.")
            return ""

        top = rules.head(top_n)

        fig, ax = plt.subplots(figsize=(9, 6))

        sc = ax.scatter(
            top["support"], top["confidence"],
            s=top["lift"] * 50,
            c=top["lift"], cmap="YlOrRd",
            alpha=0.8, edgecolors="black", linewidths=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Lift")

        # Label top 3
        for i, (_, row) in enumerate(top.head(3).iterrows()):
            ant = ", ".join(sorted(row["antecedents"]))[:25]
            ax.annotate(
                f"{ant}...\nlift={row['lift']:.2f}",
                (row["support"], row["confidence"]),
                xytext=(10, 5), textcoords="offset points", fontsize=7,
            )

        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_title("Association Rules — Support vs Confidence\n(kích thước vòng = Lift)")
        ax.axhline(0.7, color="red", linestyle="--", alpha=0.5, label="min_conf=0.7")
        ax.axvline(0.2, color="blue", linestyle="--", alpha=0.5, label="min_sup=0.2")
        ax.legend(fontsize=8)
        plt.tight_layout()

        path = f"{self.output_dir}/09_association_rules.png"
        if save:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {path}")
        return path
