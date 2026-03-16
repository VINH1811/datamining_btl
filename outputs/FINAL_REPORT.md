# Water Quality Analysis — Báo cáo Cuối kỳ
**Đề 9 | Kaggle Water Potability (3,276 rows × 10 cols) | 2026-03-17**

---

## 1. Tổng quan Dataset (Rubric A)

| Chỉ số | Giá trị |
|--------|---------|
| Số mẫu | 3,276 |
| Số features | 9 (+ 30 features mới sau engineering) |
| Target | Potability (0=Not Safe, 1=Safe) |
| Imbalance | 61% Not Potable / 39% Potable |
| Missing | ph=15%, Sulfate=23.8%, THM=5% |

### Data Dictionary (đơn vị + ngưỡng WHO)
| Feature | Đơn vị | WHO Limit | Ghi chú |
|---------|--------|-----------|---------|
| ph | pH units | 6.5–8.5 | Kaggle: 100% vi phạm (Solids ~22000>>500) |
| Hardness | mg/L CaCO₃ | <300 | |
| Solids | ppm (TDS) | <500 | |
| Chloramines | ppm | <4 | ~100% vi phạm (~7 ppm) |
| Sulfate | mg/L | <250 | |
| Conductivity | μS/cm | <400 | |
| Organic_carbon | ppm | <2 | ~100% vi phạm (~14 ppm) |
| Trihalomethanes | μg/L | <80 | |
| Turbidity | NTU | <4 | |

---

## 2. Tiền xử lý (Rubric B)

| Bước | Phương pháp | Lý do chọn |
|------|-------------|------------|
| Missing values | KNN Imputation (k=5) | Giữ correlation giữa features, tốt hơn median |
| Outlier | Winsorization [p1, p99] | Clip thay vì xóa → giữ 100% dữ liệu |
| Scaling | RobustScaler | Tốt hơn StandardScaler với Solids phân phối lệch mạnh |
| Feature Eng. | 30 features mới | WHO flags, deviation ratios, interaction features |

**Features sau engineering:** 39 (9 gốc + 9 flags + 9 devs + 5 interactions + 7 aggregates)

---

## 3. Data Mining (Rubric C)

### 3.1 Apriori Association Rules
- **Chiến lược:** Quantile 3-bin (Low/Medium/High) — không dùng WHO bins vì 100% vi phạm
- **Thêm Potability** vào transaction → tìm rules có ý nghĩa thực tế
- **Tham số:** min_support=0.10, min_confidence=0.55, min_lift=1.01

| Antecedent | Consequent | Support | Confidence | Lift |
|------------|-----------|---------|------------|------|
| {Trihalomethanes_Low, Turbidity_Low} | → {Potability_Potable} | 0.134 | 0.885 | 2.27× |
| {Chloramines_Low, Trihalomethanes_Low} | → {Potability_Potable} | 0.132 | 0.875 | 2.24× |
| {Chloramines_Low, Organic_carbon_Low} | → {Potability_Potable} | 0.139 | 0.875 | 2.24× |


### 3.2 K-Means Clustering
- **k=3** (dựa trên Elbow + Silhouette analysis)
- Silhouette Score: 0.54 | Davies-Bouldin: tốt hơn k=2 về Calinski-Harabasz

| Cluster | Risk Level | Đặc điểm |
|---------|-----------|----------|
| 0 | High Risk | Turbidity cao, nhiều vi phạm WHO |
| 1 | Medium Risk | Một số chỉ số borderline |
| 2 | High Risk | Chloramines + Organic_carbon cao |

---

## 4. Kết quả Mô hình (Rubric D, E)

### 4.1 Classification — Ensemble(XGB+ET+RF)+SMOTE

| Metric | CV (5-fold) | Test (hold-out) |
|--------|-------------|-----------------|
| F1-macro | 0.9170 ± 0.0063 | **0.9000** |
| ROC-AUC | 0.9744 | **0.9548** |
| PR-AUC | 0.9745 | **0.9286** |
| Train↔Test gap | — | +0.0171 |

**Confusion Matrix:** TN=362 | FP=38 | FN=25 | TP=231

### 4.2 Baseline Comparison
| Model | F1-macro | Precision | Recall | ROC-AUC |
| --- | --- | --- | --- | --- |
| Ensemble(XGB+ET+RF)+SMOTETomek (best) | 0.9 | 0.8971 | 0.9037 | 0.9548 |
| RandomForest (no SMOTETomek) | 0.896 | 0.8957 | 0.8962 | 0.9536 |
| LogisticRegression | 0.8837 | 0.8794 | 0.892 | 0.9625 |
| DummyClassifier (Random) | 0.4832 | 0.4833 | 0.4832 | 0.4832 |
| ZeroR (Majority) | 0.3788 | 0.3049 | 0.5 | 0.5 |

### 4.3 Threshold Analysis
| threshold | f1 | precision | recall | fn | fp |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.7545 | 0.6167 | 0.9805 | 5.0 | 156.0 |
| 0.15 | 0.8014 | 0.6703 | 0.9688 | 8.0 | 122.0 |
| 0.2 | 0.8224 | 0.6966 | 0.9688 | 8.0 | 108.0 |
| 0.25 | 0.8417 | 0.7243 | 0.9648 | 9.0 | 94.0 |
| 0.3 | 0.861 | 0.7546 | 0.9609 | 10.0 | 80.0 |
| 0.35 | 0.8773 | 0.7846 | 0.9531 | 12.0 | 67.0 |
| 0.4 | 0.8827 | 0.8068 | 0.9297 | 18.0 | 57.0 |
| 0.45 | 0.8837 | 0.8182 | 0.9141 | 22.0 | 52.0 |
| 0.5 | 0.8893 | 0.837 | 0.9023 | 25.0 | 45.0 |
| 0.55 | 0.8993 | 0.8726 | 0.8828 | 30.0 | 33.0 |
| 0.6 | 0.8952 | 0.8866 | 0.8555 | 37.0 | 28.0 |
| 0.65 | 0.8911 | 0.8987 | 0.832 | 43.0 | 24.0 |
| 0.7 | 0.8889 | 0.9087 | 0.8164 | 47.0 | 21.0 |
| 0.75 | 0.8782 | 0.9099 | 0.7891 | 54.0 | 20.0 |
| 0.8 | 0.8636 | 0.9143 | 0.75 | 64.0 | 18.0 |
| 0.85 | 0.8237 | 0.918 | 0.6562 | 88.0 | 15.0 |
| 0.9 | 0.7732 | 0.9329 | 0.543 | 117.0 | 10.0 |

### 4.4 WQI Regression
| Metric | Giá trị |
|--------|---------|
| MAE | 1.3565 |
| RMSE | 2.0318 |
| R² | **0.9480** |
| sMAPE | 8.57% |

---

## 5. Bán giám sát (Rubric F)

| Label % | Supervised-only F1 | Label Spreading F1 | Δ |
|---------|-------------------|--------------------|---|
| 20% | 0.8617 | 0.7573 | -0.1044 |

*(Xem outputs/tables/learning_curve.csv và 09_ssl_learning_curve.png)*

---

## 6. Phân tích lỗi + Insights (Rubric G)

1. [CRITICAL] Ensemble(XGB+ET+RF)+SMOTE: F1=0.9000, ROC-AUC=0.9548 — mô hình tốt nhất; triển khai kiểm tra chất lượng nước real-time
2. [CRITICAL] FP=38 mẫu bị báo SAI là nước sạch (nguy hiểm!) — hạ threshold từ 0.50 xuống 0.40 để giảm FP và tăng F1
3. [HIGH] K-Means k=3: Silhouette=0.92 — phân 3 cụm rủi ro (High/Medium) hỗ trợ ưu tiên xét nghiệm
4. [HIGH] Apriori: 78 luật tìm được — Top rule: {Trihalomethanes_Low, Turbidity_Low} → {Potability_Potable} (Lift=2.27×)  |  58 luật dự đoán Potability
5. [HIGH] Train↔Test gap=+0.0171: Mô hình khái quát tốt — Giữ nguyên thiết kế
6. [HIGH] WQI Regression: MAE=1.357, R²=0.9480 — xuất sắc; bổ sung features nhiệt độ/mùa vụ để cải thiện thêm
7. [MEDIUM] Label Spreading F1=-0.1044 (kém supervised 0.1044) — dữ liệu Kaggle thiếu cấu trúc cluster; SSL có ích khi nhãn ≥30%
8. [LOW] Dataset imbalanced 61%/39% — KHÔNG dùng Accuracy; luôn báo cáo F1-macro + PR-AUC + ROC-AUC

---

## 7. Cấu trúc Repo (Rubric H)

```
DATA_MINING_PROJECT/
├── configs/params.yaml          # Tất cả hyperparameters
├── data/raw/water_potability.csv
├── data/processed/water_clean.parquet
├── scripts/run_pipeline.py      # Orchestrator chính
├── src/
│   ├── data/cleaner.py          # KNN + Winsor + RobustScaler
│   ├── features/builder.py      # WHO flags + WQI + discretize
│   ├── mining/association.py    # Apriori + mlxtend
│   ├── mining/clustering.py     # K-Means + Elbow + Silhouette
│   ├── evaluation/              # metrics, report
│   └── visualization/plots.py
└── outputs/
    ├── figures/                 # 11 biểu đồ
    ├── tables/                  # CSV kết quả
    └── models/                  # pkl files
```

---
*Generated by run_pipeline.py v3 | 2026-03-17*
