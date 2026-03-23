# Hướng dẫn sử dụng Pipeline

## Tổng quan

Project này có 2 phần chính:
- **src/**: Code production (chạy tự động, có cấu trúc)
- **notebooks/**: Jupyter notebooks (khám phá, demo, trực quan hóa)

## Cách chạy

### 1. Chạy pipeline + notebooks (MẶC ĐỊNH)

```bash
python scripts/run_pipeline.py
```

Lệnh này sẽ:
1. ✅ Chạy toàn bộ pipeline (EDA → Preprocessing → Mining → Modeling → Evaluation)
2. ✅ Tạo FINAL_REPORT.md
3. ✅ Tự động chạy tất cả notebooks (01-05) để tạo output trực quan

**Thời gian:** ~2-3 phút (tùy máy)

### 2. Chỉ chạy pipeline (KHÔNG chạy notebooks)

```bash
python scripts/run_pipeline.py --skip-notebooks
```

Nhanh hơn (~40-50 giây), chỉ tạo kết quả từ src/

### 3. Chạy từng bước riêng

```bash
# Chỉ chạy EDA
python scripts/run_pipeline.py --step eda --skip-notebooks

# Chỉ chạy Mining
python scripts/run_pipeline.py --step mining --skip-notebooks

# Chỉ chạy Modeling
python scripts/run_pipeline.py --step modeling --skip-notebooks
```

### 4. Chạy với verbose mode

```bash
python scripts/run_pipeline.py --verbose
```

Hiển thị chi tiết hơn trong quá trình chạy.

## Cấu trúc outputs

Sau khi chạy, kết quả được lưu trong `outputs/`:

```
outputs/
├── FINAL_REPORT.md              # Báo cáo tổng hợp
├── figures/                     # 11 biểu đồ
│   ├── 01_eda_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_boxplot_before.png
│   ├── 04_boxplot_after.png
│   ├── 05_elbow_silhouette.png
│   ├── 06_cluster_heatmap.png
│   ├── 07_classification_results.png
│   ├── 08_feature_importance.png
│   ├── 09_ssl_learning_curve.png
│   ├── 10_wqi_regression.png
│   └── 11_radar_metrics.png
├── tables/                      # CSV/JSON kết quả
│   ├── association_rules.csv
│   ├── cluster_profiles.csv
│   ├── clustering_result.json
│   ├── baseline_comparison.csv
│   ├── threshold_analysis.csv
│   └── all_metrics.json
└── models/                      # Trained models
    ├── best_classifier.pkl
    ├── wqi_regressor.pkl
    ├── imputer.pkl
    └── scaler.pkl
```

## Notebooks

### Notebooks gốc (notebooks/*.ipynb)
- Chứa code chi tiết, có thể chạy độc lập
- Một số cell có code cũ, cần cập nhật thủ công

### Notebook mới (notebooks/03_mining_association_clustering_v2.ipynb)
- ✅ Sử dụng code từ `src/` → đồng bộ với pipeline
- ✅ Risk level tự động: High Risk / Medium Risk / Safe
- ✅ Khuyến nghị dùng notebook này thay vì bản cũ

## Thay đổi mới nhất

### Risk Level Classification
Trước đây (cố định):
- >0.65 → High Risk
- >0.45 → Medium Risk
- ≤0.45 → Low Risk

**Vấn đề:** Có thể có 2 cluster cùng Medium Risk

Bây giờ (động):
- Cluster có unsafe_ratio cao nhất → 🔴 High Risk
- Cluster có unsafe_ratio trung bình → 🟡 Medium Risk
- Cluster có unsafe_ratio thấp nhất → 🟢 Safe

**Ưu điểm:** Luôn có k mức rủi ro khác nhau khi chọn k cluster

## Troubleshooting

### Lỗi encoding (emoji không hiển thị)
```bash
$env:PYTHONIOENCODING="utf-8"; python scripts/run_pipeline.py
```

### Notebooks không chạy
Kiểm tra jupyter đã cài:
```bash
pip install jupyter nbconvert
```

### Dataset không tìm thấy
Tải từ Kaggle:
```bash
kaggle datasets download -d mssmartypants/water-quality --unzip
```
Hoặc download thủ công và đặt vào `data/raw/water_potability.csv`

## Liên hệ

Nếu có vấn đề, kiểm tra:
1. Python version ≥ 3.8
2. Đã cài đặt requirements: `pip install -r requirements.txt`
3. Dataset đã có trong `data/raw/`
