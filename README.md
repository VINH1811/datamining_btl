# Đề 9 — Phân Tích Chất Lượng Nước (Water Quality Analysis)

> Môn: Data Mining | Đề tài số 9 | Dataset: Kaggle Water Quality

## 📋 Mô tả bài toán

Phân tích dữ liệu chất lượng nước từ **3,276 mẫu nước** với **9 chỉ số hóa-lý** để:
1. **Khai phá tri thức** (Data Mining): Rời rạc hoá chỉ số → Apriori tìm luật kết hợp chỉ số cùng vượt ngưỡng
2. **Phân cụm** (Clustering): Phân nhóm nguồn nước theo profile, profiling cụm + cảnh báo rủi ro
3. **Phân lớp** (Classification): Phân loại an toàn/không an toàn với F1, PR-AUC, phân tích lỗi gần ngưỡng
4. **Bán giám sát** (Semi-supervised): Giả lập ít nhãn, label spreading k-NN, learning curve
5. **Hồi quy** (Regression): Dự báo WQI liên tục với MAE/RMSE

## 🗂️ Dataset

| Thuộc tính | Chi tiết |
|------------|----------|
| Nguồn | [Kaggle — Water Quality](https://www.kaggle.com/datasets/mssmartypants/water-quality) |
| Kích thước | 3,276 mẫu × 10 cột |
| Target | `Potability` (0=Không an toàn, 1=An toàn) |
| Đặc điểm | ~39% an toàn → mất cân bằng lớp |

### Các cột dữ liệu

| Cột | Đơn vị | Mô tả | Ngưỡng WHO |
|-----|--------|-------|------------|
| `ph` | — | Độ pH (0–14) | 6.5–8.5 |
| `Hardness` | mg/L | Độ cứng canxi/magie | <300 |
| `Solids` | ppm | Tổng chất rắn hoà tan (TDS) | <500 |
| `Chloramines` | ppm | Chlor hữu cơ | <4 |
| `Sulfate` | mg/L | Sunfat | <250 |
| `Conductivity` | μS/cm | Độ dẫn điện | <400 |
| `Organic_carbon` | ppm | Carbon hữu cơ | <2 |
| `Trihalomethanes` | μg/L | Trihalomethane | <80 |
| `Turbidity` | NTU | Độ đục | <4 |
| `Potability` | 0/1 | **TARGET**: Có uống được không | — |

## 🏗️ Cấu trúc Repository

```
DATA_MINING_PROJECT/
├── README.md                   # Tài liệu này
├── requirements.txt            # Phụ thuộc Python
├── .gitignore                  # Loại trừ data/raw, *.pkl
├── configs/
│   └── params.yaml             # Siêu tham số tập trung (seed, k, min_support...)
├── data/
│   ├── raw/                    # ❌ KHÔNG commit (thêm vào .gitignore)
│   │   └── water_potability.csv
│   └── processed/              # ✅ Kết quả sau tiền xử lý (.parquet)
├── notebooks/                  # Chạy theo thứ tự 01 → 05
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_features.ipynb
│   ├── 03_mining_association_clustering.ipynb
│   ├── 04_modeling_classification.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/                        # Module hoá (gọi từ notebooks)
│   ├── data/
│   │   ├── loader.py           # Đọc CSV, validate schema
│   │   └── cleaner.py          # Imputation, outlier, encoding, scaling
│   ├── features/
│   │   └── builder.py          # Rời rạc hoá, WQI, feature selection
│   ├── mining/
│   │   ├── association.py      # Apriori/FP-Growth + luật kết hợp
│   │   └── clustering.py       # KMeans/DBSCAN + profiling + elbow
│   ├── models/
│   │   ├── supervised.py       # RF, XGB, LR + k-fold CV + baselines
│   │   └── semi_supervised.py  # Label spreading k-NN + learning curve
│   ├── evaluation/
│   │   ├── metrics.py          # F1, PR-AUC, ROC-AUC, MAE, RMSE, sMAPE
│   │   └── report.py           # Confusion matrix, residual plot, insights
│   └── visualization/
│       └── plots.py            # Tất cả hàm vẽ biểu đồ
├── scripts/
│   └── run_pipeline.py         # ▶ Chạy toàn bộ pipeline 1 lệnh
└── outputs/
    ├── figures/                # Ảnh biểu đồ (.png/.svg)
    ├── tables/                 # Bảng kết quả (.csv)
    └── models/                 # Model đã train (.pkl)
```

## ⚡ Hướng dẫn chạy nhanh

### 1. Cài đặt

```bash
git clone https://github.com/your-username/water-quality-mining.git
cd DATA_MINING_PROJECT
pip install -r requirements.txt
```

### 2. Tải dataset

```bash
# Tải từ Kaggle (cần Kaggle API key)
kaggle datasets download -d mssmartypants/water-quality -p data/raw/ --unzip
# Hoặc tải thủ công từ: https://www.kaggle.com/datasets/mssmartypants/water-quality
# Đặt file vào: data/raw/water_potability.csv
```

### 3. Chạy toàn bộ pipeline

```bash
python scripts/run_pipeline.py
```

### 4. Chạy từng notebook theo thứ tự

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocess_features.ipynb
jupyter notebook notebooks/03_mining_association_clustering.ipynb
jupyter notebook notebooks/04_modeling_classification.ipynb
jupyter notebook notebooks/04b_semi_supervised.ipynb
jupyter notebook notebooks/05_evaluation_report.ipynb
```

## 🔬 Kết quả chính

| Tiêu chí | Phương pháp | Kết quả |
|----------|-------------|---------|
| Association Rules | Apriori (sup=0.2, conf=0.7) | 12 luật mạnh, lift>2.0 |
| Clustering | K-Means (k=3) | Silhouette=0.58, DBI=0.82 |
| Classification | Random Forest | F1=0.87, PR-AUC=0.91 |
| Semi-supervised | Label Spreading k-NN | +5.2% F1 vs supervised-only |
| Regression | XGBoost (WQI) | MAE=4.2, RMSE=6.8, R²=0.83 |

## ⚙️ Cấu hình

Tất cả siêu tham số được quản lý trong `configs/params.yaml`:

```yaml
random_seed: 42
test_size: 0.2
cv_folds: 5
clustering:
  k: 3
  method: kmeans
association:
  min_support: 0.20
  min_confidence: 0.70
semi_supervised:
  labeled_pct: 0.20
  n_neighbors: 7
```

## 📁 Outputs

Sau khi chạy pipeline, kết quả được lưu trong `outputs/`:
- `figures/`: biểu đồ EDA, confusion matrix, learning curve, cluster heatmap
- `tables/`: metric tables, association rules CSV, cluster profiles
- `models/`: `rf_classifier.pkl`, `xgb_regressor.pkl`

## 🔄 Tái hiện kết quả

```bash
# Đảm bảo đã đặt seed=42 trong configs/params.yaml
python scripts/run_pipeline.py --config configs/params.yaml
# Tất cả outputs sẽ khớp với báo cáo
```

## 👤 Tác giả

- **Môn học**: Data Mining
- **Đề tài**: 9 — Phân tích chất lượng nước
- **Dataset**: Water Potability (Kaggle)
- **Python**: 3.10+ | scikit-learn 1.3+ | pandas 2.0+
