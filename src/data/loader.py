"""
src/data/loader.py
──────────────────
Đọc dataset chất lượng nước, kiểm tra schema, báo cáo tổng quan.

Usage:
    from src.data.loader import load_water_data, validate_schema

    df, report = load_water_data("data/raw/water_potability.csv")
    print(report)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


# ── Cấu hình schema mong đợi ────────────────────────────────────
EXPECTED_COLUMNS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes",
    "Turbidity", "Potability",
]

WHO_THRESHOLDS = {
    "ph":               (6.5,  8.5,  "pH an toàn theo WHO"),
    "Hardness":         (0,    300,  "mg/L — độ cứng"),
    "Solids":           (0,    500,  "ppm — TDS tối đa"),
    "Chloramines":      (0,    4.0,  "ppm — clo hữu cơ"),
    "Sulfate":          (0,    250,  "mg/L — sunfat"),
    "Conductivity":     (0,    400,  "μS/cm — dẫn điện"),
    "Organic_carbon":   (0,    2.0,  "ppm — carbon hữu cơ"),
    "Trihalomethanes":  (0,    80,   "μg/L — trihalomethane"),
    "Turbidity":        (0,    4.0,  "NTU — độ đục"),
}


def load_water_data(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Đọc file CSV chất lượng nước và tạo báo cáo tổng quan.

    Parameters
    ----------
    path : str
        Đường dẫn đến file CSV (ví dụ: "data/raw/water_potability.csv")

    Returns
    -------
    df : pd.DataFrame
        DataFrame thô chưa xử lý
    report : dict
        Báo cáo tổng quan: hình dạng, missing, phân phối target, etc.

    Raises
    ------
    FileNotFoundError
        Nếu file không tồn tại
    ValueError
        Nếu schema không hợp lệ
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"❌ File không tìm thấy: {path}\n"
            "Tải dataset từ: https://www.kaggle.com/datasets/mssmartypants/water-quality\n"
            "Đặt vào: data/raw/water_potability.csv"
        )

    df = pd.read_csv(path)

    # Validate schema
    issues = validate_schema(df)
    if issues:
        print(f"⚠ Schema warnings: {issues}")

    # Tạo báo cáo tổng quan
    report = _build_report(df)
    return df, report


def validate_schema(df: pd.DataFrame) -> list:
    """Kiểm tra schema và trả về danh sách vấn đề."""
    issues = []

    # Kiểm tra cột
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        issues.append(f"Thiếu cột: {missing_cols}")

    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        issues.append(f"Cột thừa: {extra_cols}")

    # Kiểm tra kiểu dữ liệu
    for col in df.columns:
        if col in EXPECTED_COLUMNS and col != "Potability":
            if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                issues.append(f"Cột {col} có kiểu {df[col].dtype} (mong đợi số)")

    # Kiểm tra giá trị target
    if "Potability" in df.columns:
        invalid_target = df["Potability"].dropna().isin([0, 1])
        if not invalid_target.all():
            issues.append("Cột Potability chứa giá trị ngoài {0, 1}")

    return issues


def _build_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Tạo báo cáo tổng quan dataset."""
    feature_cols = [c for c in EXPECTED_COLUMNS if c != "Potability" and c in df.columns]

    # Phân tích missing
    missing_info = {}
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            missing_info[col] = {
                "count": int(n_miss),
                "pct": round(n_miss / len(df) * 100, 2)
            }

    # Thống kê mô tả
    stats = {}
    for col in feature_cols:
        vals = df[col].dropna()
        stats[col] = {
            "mean": round(float(vals.mean()), 3),
            "std": round(float(vals.std()), 3),
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
            "median": round(float(vals.median()), 3),
            "skewness": round(float(vals.skew()), 3),
        }

    # Phân phối target
    target_dist = {}
    if "Potability" in df.columns:
        vc = df["Potability"].value_counts(dropna=False)
        target_dist = {
            "safe": int(vc.get(1, 0)),
            "unsafe": int(vc.get(0, 0)),
            "missing": int(df["Potability"].isna().sum()),
            "safe_pct": round(vc.get(1, 0) / len(df) * 100, 1),
        }

    # Cảnh báo chỉ số vượt ngưỡng WHO
    who_violations = {}
    for col, (lo, hi, desc) in WHO_THRESHOLDS.items():
        if col in df.columns:
            vals = df[col].dropna()
            n_viol = int(((vals < lo) | (vals > hi)).sum())
            if n_viol > 0:
                who_violations[col] = {
                    "n_violations": n_viol,
                    "pct": round(n_viol / len(df) * 100, 1),
                    "threshold": f"[{lo}, {hi}] — {desc}",
                }

    report = {
        "shape": df.shape,
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "missing_values": missing_info,
        "total_missing_cells": df.isna().sum().sum(),
        "statistics": stats,
        "target_distribution": target_dist,
        "who_violations": who_violations,
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return report


def print_report(report: Dict[str, Any]) -> None:
    """In báo cáo tổng quan ra console."""
    print("=" * 60)
    print(f"📊 WATER QUALITY DATASET REPORT")
    print("=" * 60)
    print(f"Kích thước: {report['shape'][0]:,} hàng × {report['shape'][1]} cột")
    print(f"Missing cells: {report['total_missing_cells']:,}")
    print(f"Hàng trùng lặp: {report['duplicate_rows']}")

    print("\n🎯 Phân phối Target (Potability):")
    td = report["target_distribution"]
    print(f"  An toàn (1):    {td.get('safe', 0):>6,} ({td.get('safe_pct', 0):.1f}%)")
    print(f"  Không an toàn (0): {td.get('unsafe', 0):>6,} ({100 - td.get('safe_pct', 0):.1f}%)")

    if report["missing_values"]:
        print("\n⚠ Cột có Missing Values:")
        for col, info in report["missing_values"].items():
            print(f"  {col:25s}: {info['count']:>5} ({info['pct']:.1f}%)")

    if report["who_violations"]:
        print("\n⚠ Chỉ số vượt ngưỡng WHO:")
        for col, info in report["who_violations"].items():
            print(f"  {col:25s}: {info['n_violations']:>5} mẫu ({info['pct']:.1f}%) — {info['threshold']}")

    print("=" * 60)
