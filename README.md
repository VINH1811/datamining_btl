# 💧 Đề 9 — Phân Tích Chất Lượng Nước (Water Quality Analysis)

**"Hành trình giải mã sự sống qua từng giọt nước"**

Nước là khởi nguồn của sự sống, nhưng không phải nguồn nước nào cũng an toàn. Trong thực tế, việc xét nghiệm y tế cho từng mẫu nước đòi hỏi chi phí khổng lồ và thời gian chờ đợi lâu. Đặt trong bối cảnh đó, **Nhóm 2** chúng tôi đã tiếp nhận bài toán phân tích chất lượng nước từ bộ dữ liệu Kaggle. 

Mục tiêu của chúng tôi không chỉ là những con số khô khan, mà là xây dựng một hệ thống khai phá dữ liệu (Data Mining) và học máy (Machine Learning) có khả năng: tìm ra quy luật ẩn giấu, khoanh vùng rủi ro, và dự báo độ an toàn của nước chỉ từ những cảm biến đo lường cơ bản (như độ pH, độ đục). Đây là giải pháp hướng tới việc giám sát chất lượng nước với chi phí thấp và tốc độ theo thời gian thực.

### 👥 Đội ngũ thực hiện - Nhóm 2
* **Nguyễn Văn Vinh**
* **Đỗ Văn Vinh**
* **Bạch Ngọc Lương**
* **Lại Thành Đoàn**

> **Môn học:** Data Mining | **Đề tài:** Số 9 | **Dataset:** Kaggle Water Quality

---

## 🎯 Mô tả bài toán & Phương pháp tiếp cận

Dự án phân tích **3,276 mẫu nước** với **9 chỉ số hóa-lý** thông qua 5 chặng đường phân tích chuyên sâu:
1. **Khai phá tri thức (Data Mining):** Rời rạc hoá các chỉ số thành mức độ (Low/Medium/High) → Áp dụng thuật toán **Apriori** để tìm ra các "luật kết hợp" chết người (ví dụ: khi nào các chất độc hại cùng đồng loạt vượt ngưỡng).
2. **Phân cụm (Clustering):** Sử dụng **K-Means** để nhóm các nguồn nước theo đặc điểm hóa học, từ đó "khoanh vùng" và phát đi cảnh báo rủi ro cho từng cụm.
3. **Phân lớp (Classification):** Xây dựng ranh giới sinh tử: An toàn hay Không an toàn? Sử dụng các chỉ số khắt khe như F1-macro, PR-AUC để vượt qua thách thức mất cân bằng dữ liệu, và phân tích kỹ các lỗi sai sát ngưỡng.
4. **Học Bán giám sát (Semi-supervised):** Mô phỏng kịch bản đời thực khi "tiền cạn túi" — chỉ có một số ít mẫu được dán nhãn (xét nghiệm y tế đắt đỏ). Chúng tôi dùng **Label Spreading k-NN** để lan truyền tri thức sang các mẫu chưa biết, vẽ learning curve để tìm điểm hòa vốn.
5. **Hồi quy (Regression):** Xây dựng thang điểm WQI (Water Quality Index) liên tục và dùng Machine Learning để dự báo.

---

## 📊 Khám phá Bộ dữ liệu (Dataset)

Bộ dữ liệu là tập hợp những manh mối quý giá được thu thập từ nhiều vùng nước khác nhau.

| Tiêu chí | Chi tiết |
|------------|----------|
| **Nguồn gốc** | [Kaggle — Water Quality](https://www.kaggle.com/datasets/mssmartypants/water-quality) |
| **Kích thước** | 3,276 mẫu × 10 cột |
| **Target** | `Potability` (0 = Không an toàn, 1 = An toàn) |
| **Đặc điểm** | Chỉ có ~39% nước là an toàn → Dữ liệu mất cân bằng (Class Imbalance) |

### Từ điển Dữ liệu (Chiếu theo chuẩn WHO)

| Cột | Đơn vị | Mô tả | Ngưỡng an toàn (WHO) |
|-----|--------|-------|------------|
| `ph` | — | Độ pH của nước (0–14) | 6.5 – 8.5 |
| `Hardness` | mg/L | Độ cứng tổng (Canxi/Magie) | < 300 |
| `Solids` | ppm | Tổng chất rắn hoà tan (TDS) | < 500 |
| `Chloramines` | ppm | Lượng Chloramine dùng khử trùng | < 4.0 |
| `Sulfate` | mg/L | Nồng độ Sunfat | < 250 |
| `Conductivity` | μS/cm | Độ dẫn điện | < 400 |
| `Organic_carbon` | ppm | Tổng lượng Carbon hữu cơ | < 2.0 |
| `Trihalomethanes`| μg/L | Hợp chất THM (nguy cơ ung thư) | < 80 |
| `Turbidity` | NTU | Độ đục (chất lơ lửng) | < 4.0 |
| `Potability` | 0/1 | **TARGET**: Quyết định sinh tử - Có uống được không? | — |

---

## 🏗 Cấu trúc Trụ sở (Repository Blueprint)

Để quản lý dự án một cách khoa học và dễ dàng tái hiện lại (reproducible), nhóm đã thiết kế một kiến trúc thư mục chuẩn mực hóa:

```text
DATA_MINING_PROJECT/
├── README.md                  ← Báo cáo tổng quan dự án (Bạn đang ở đây)
├── requirements.txt           ← Danh sách "vũ khí" (Thư viện Python)
├── .gitignore                 ← Bộ lọc Git (Giấu data raw và files rác)
├── configs/
│   └── params.yaml            ← Bảng điều khiển trung tâm (seed, k, thresholds...)
├── data/
│   ├── raw/                   ← Kho lưu trữ nguyên thủy (KHÔNG commit)
│   │   └── water_potability.csv
│   └── processed/             ← Dữ liệu đã qua tinh chế (.parquet)
├── notebooks/                 ← Nhật ký nghiên cứu (Chạy theo thứ tự 01 → 05)
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_features.ipynb
│   ├── 03_mining_association_clustering.ipynb
│   ├── 04_modeling_classification.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/                       ← Lõi hệ thống (Module hóa)
│   ├── data/
│   │   ├── loader.py          ← Nạp dữ liệu
│   │   └── cleaner.py         ← Làm sạch, xử lý missing, scaling
│   ├── features/
│   │   └── builder.py         ← Kỹ nghệ đặc trưng (WQI, Rời rạc hóa)
│   ├── mining/
│   │   ├── association.py     ← Tìm quy luật (Apriori/FP-Growth)
│   │   └── clustering.py      ← Phân cụm (KMeans/DBSCAN)
│   ├── models/
│   │   ├── supervised.py      ← Học có giám sát (RF, XGB, LR)
│   │   └── semi_supervised.py ← Học bán giám sát (Label spreading)
│   ├── evaluation/
│   │   ├── metrics.py         ← Thước đo (F1, PR-AUC, MAE, RMSE...)
│   │   └── report.py          ← Xuất báo cáo, ma trận nhầm lẫn
│   └── visualization/
│       └── plots.py           ← Công cụ vẽ biểu đồ trực quan
├── scripts/
│   └── run_pipeline.py        ← Công tắc khởi động toàn bộ quy trình
└── outputs/
    ├── figures/               ← Thư viện ảnh (.png)
    ├── tables/                ← Bảng tổng hợp (.csv)
    └── models/                ← Tủ chứa mô hình đã huấn luyện (.pkl)
```

---

## 🚀 Hướng dẫn Vận hành (Quick Start)

Tham gia vào quá trình phân tích cùng Nhóm 2 chỉ với 4 bước đơn giản:

### 1. Chuẩn bị môi trường
```bash
git clone [https://github.com/your-username/water-quality-mining.git](https://github.com/your-username/water-quality-mining.git)
cd DATA_MINING_PROJECT
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu
```bash
# Sử dụng Kaggle API để lấy dữ liệu về kho chứa
kaggle datasets download -d mssmartypants/water-quality -p data/raw/ --unzip

# Hoặc tải thủ công từ Kaggle và đặt file vào: data/raw/water_potability.csv
```

### 3. Khởi động Cỗ máy tự động
Chỉ với 1 dòng lệnh, toàn bộ pipeline từ làm sạch đến đánh giá sẽ được thực thi:
```bash
python scripts/run_pipeline.py
```

### 4. Khám phá từng bước (Dành cho nhà nghiên cứu)
Đi sâu vào từng phân tích qua Jupyter Notebook:
```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocess_features.ipynb
jupyter notebook notebooks/03_mining_association_clustering.ipynb
jupyter notebook notebooks/04_modeling_classification.ipynb
jupyter notebook notebooks/04b_semi_supervised.ipynb
jupyter notebook notebooks/05_evaluation_report.ipynb
```

---

## 🏆 Thành quả Đạt được

Sau chuỗi ngày đào sâu vào dữ liệu, đây là những kết quả nổi bật nhất mà mô hình mang lại:

| Chuyên mục | Phương pháp áp dụng | Kết quả & Độ chính xác |
|:---|:---|:---|
| **Association Rules** | Apriori (sup=0.1, conf=0.55) | Bóc tách 78 luật cảnh báo (58 luật dự đoán Potability) |
| **Clustering** | K-Means (k=3, multi-metric voting) | 3 mức rủi ro rõ ràng: An toàn/Hơi nguy hiểm/Nguy hiểm (Silhouette=0.54) |
| **Classification** | Ensemble (XGB+ET+RF) + SMOTE | F1-macro=0.90, ROC-AUC=0.95 |
| **Semi-supervised** | Label Spreading k-NN | Thất bại (-10% F1) do thiếu cấu trúc cluster |
| **Regression** | RandomForest (Dự báo WQI) | Sai số thấp: MAE=1.4, RMSE=2.0, R²=0.95 |

---

## ⚙️ Bảng Điều Khiển (Config)

Chúng tôi tin vào nguyên tắc thiết kế có khả năng tái lập (Reproducibility). Mọi tinh chỉnh siêu tham số đều được tập trung duy nhất tại `configs/params.yaml`:

```yaml
random_seed: 42
test_size: 0.2
cv_folds: 5
clustering:
  k: 3  # Chọn tự động bằng multi-metric voting (k=2 vs k=3)
  method: kmeans
association:
  min_support: 0.10
  min_confidence: 0.55
semi_supervised:
  labeled_pct: 0.20
  n_neighbors: 7
```

Để tái hiện lại 100% kết quả như trong báo cáo của chúng tôi:
```bash
# Đảm bảo đã đặt seed=42 trong configs/params.yaml
python scripts/run_pipeline.py --config configs/params.yaml
```

**Outputs thu được:**
Mọi kết quả tinh túy nhất sẽ tự động được lưu vào thư mục `outputs/`:
* `figures/`: Biểu đồ EDA, Confusion Matrix, Learning Curve, Cluster Heatmap...
* `tables/`: Các bảng chỉ số Metrics, Luật kết hợp (Association Rules CSV), Cluster profiles...
* `models/`: Các mô hình tốt nhất đã đóng gói (`best_classifier.pkl`, `best_regressor.pkl`).

---
*Báo cáo được thực hiện với Python 3.10+ | scikit-learn 1.3+ | pandas 2.0+*
