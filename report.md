# PHÂN TÍCH CHI TIẾT — Đề 9: Phân Tích Chất Lượng Nước
## Nhóm 2 | Data Mining | Kaggle Water Potability

> **Thành viên:** Nguyễn Văn Vinh · Đỗ Văn Vinh · Bạch Ngọc Lương · Lại Thành Đoàn  
> **Dataset:** Kaggle Water Quality — 3,276 mẫu × 9 chỉ số hóa-lý  
> **Mục tiêu:** Phân tích và dự báo chất lượng nước uống bằng Data Mining + ML  
> **Link demo:** https://datamingnhom3.replit.app/

---

## 1. Ý TƯỞNG & BÀI TOÁN

### Vấn đề thực tiễn
Xét nghiệm y tế cho từng mẫu nước tốn kém và mất nhiều thời gian. Nhóm đặt câu hỏi: **"Liệu có thể chỉ từ vài chỉ số cảm biến đơn giản (pH, độ đục...) mà dự đoán được nước an toàn hay không?"** → Giải pháp chi phí thấp, thời gian thực.

### 5 hướng tiếp cận song song
| # | Kỹ thuật | Mục tiêu |
|---|---|---|
| 1 | **Luật kết hợp (Apriori)** | Tìm tổ hợp chỉ số nguy hiểm thường xuất hiện cùng nhau |
| 2 | **Phân cụm (K-Means)** | Nhóm nguồn nước theo mức rủi ro, "khoanh vùng" nguy hiểm |
| 3 | **Phân lớp (Ensemble ML)** | Phán xét An toàn / Không an toàn với độ chính xác cao |
| 4 | **Bán giám sát (Label Spreading)** | Giả lập tình huống chỉ có ít mẫu được xét nghiệm |
| 5 | **Hồi quy (WQI Regression)** | Tính điểm chất lượng nước liên tục 0–100 |

---

## 2. BỘ DỮ LIỆU

### Tổng quan
| Thuộc tính | Giá trị |
|---|---|
| Nguồn | Kaggle Water Quality Dataset |
| Kích thước | 3,276 mẫu × 10 cột |
| Target | `Potability` (0 = Không an toàn, 1 = An toàn) |
| Mất cân bằng | 61% Not Potable / 39% Potable |
| Missing values | ph=15%, Sulfate=23.8%, THM=5% |

### 9 chỉ số đầu vào (theo chuẩn WHO)
| Chỉ số | Đơn vị | Ngưỡng an toàn WHO | Đặc điểm dữ liệu |
|---|---|---|---|
| `ph` | pH | 6.5 – 8.5 | Missing 15% |
| `Hardness` | mg/L CaCO₃ | < 300 | Phân phối chuẩn đẹp |
| `Solids` | ppm (TDS) | < 500 | **100% vi phạm** (~22,000 ppm) |
| `Chloramines` | ppm | < 4.0 | **~100% vi phạm** (~7 ppm) |
| `Sulfate` | mg/L | < 250 | Missing 23.8% (nhiều nhất) |
| `Conductivity` | μS/cm | < 400 | Phân phối bình thường |
| `Organic_carbon` | ppm | < 2.0 | **~100% vi phạm** (~14 ppm) |
| `Trihalomethanes` | μg/L | < 80 | Missing 5% |
| `Turbidity` | NTU | < 4.0 | Phân phối nhẹ |

> **Lưu ý quan trọng:** Solids, Chloramines, Organic_carbon gần như 100% vi phạm ngưỡng WHO — đây là dataset đặc thù Kaggle, không phản ánh nước thực tế hoàn toàn, nhưng tạo ra bài toán phân lớp thú vị.

---

## 3. CẤU TRÚC DỰ ÁN & CHỨC NĂNG TỪNG FILE

```
datamining_btl/
├── configs/params.yaml          ← Trung tâm điều khiển siêu tham số
├── scripts/run_pipeline.py      ← Pipeline tổng, chạy toàn bộ từ A→Z
├── src/
│   ├── data/
│   │   ├── loader.py            ← Đọc dữ liệu CSV/Parquet, kiểm tra schema
│   │   └── cleaner.py          ← Tiền xử lý: missing, outlier, scaling
│   ├── features/
│   │   └── builder.py          ← Feature engineering 42 đặc trưng + WQI
│   ├── mining/
│   │   ├── clustering.py       ← K-Means + multi-metric voting
│   │   └── association.py      ← Apriori/FP-Growth + lọc luật
│   ├── models/
│   │   ├── supervised.py       ← Ensemble (XGB+ET+RF) + SMOTE + CV
│   │   └── semi_supervised.py  ← Label Spreading k-NN + learning curve
│   ├── evaluation/
│   │   ├── metrics.py          ← Tính F1, ROC-AUC, PR-AUC, confusion matrix
│   │   └── report.py           ← Tổng hợp và xuất báo cáo JSON/Markdown
│   └── visualization/
│       └── plots.py            ← Tất cả 11 loại biểu đồ (31 hình output)
├── notebooks/                   ← 6 notebooks tương tác
├── outputs/
│   ├── figures/                ← 31 ảnh biểu đồ PNG
│   ├── tables/                 ← all_metrics.json, clustering_result.json
│   ├── models/                 ← best_classifier.pkl, best_regressor.pkl
│   └── FINAL_REPORT.md         ← Báo cáo cuối kỳ tự động sinh
└── requirements.txt            ← Danh sách thư viện
```

---

## 4. PHÂN TÍCH CHI TIẾT TỪNG MODULE

### 4.1 `configs/params.yaml` — Trung tâm cấu hình
**Chức năng:** Tập trung 100% siêu tham số vào một file, đảm bảo tính tái lập.

```yaml
random_seed: 42          # Seed cố định → kết quả lặp lại 100%
test_size: 0.2           # 80% train, 20% test
cv_folds: 5              # 5-Fold Cross Validation

clustering:
  k_range: [2, 8]        # Thử K từ 2 đến 8
  selection: multi_metric_voting  # Bầu chọn bằng 3 chỉ số

association:
  min_support: 0.10      # Xuất hiện ≥10% mẫu
  min_confidence: 0.55   # Độ tin cậy ≥55%
  min_lift: 1.01         # Lift > 1 (có ý nghĩa)

classification:
  smote: true            # Dùng SMOTETomek cân bằng dữ liệu
  ensemble: true         # VotingClassifier (XGB+ET+RF)
  threshold: 0.52        # Ngưỡng phân lớp tối ưu

semi_supervised:
  labeled_pct_list: [0.05, 0.10, 0.15, 0.20, 0.30]
  n_neighbors: 7
```

---

### 4.2 `scripts/run_pipeline.py` — Pipeline Tổng (85 KB)
**Chức năng:** Điều phối toàn bộ pipeline từ đầu đến cuối theo thứ tự.

**Luồng thực thi:**
```
1. Load config (params.yaml)
2. Load data → Validate schema
3. EDA → Sinh 7 biểu đồ khám phá ban đầu
4. Tiền xử lý (cleaner.py) → KNN impute + Winsor + Scale
5. Feature engineering (builder.py) → 42 features + WQI
6. Phân cụm (clustering.py) → K=3 + cluster profiles
7. Luật kết hợp (association.py) → 78 luật
8. Phân lớp (supervised.py) → Ensemble + SMOTE
9. Bán giám sát (semi_supervised.py) → Learning curve
10. Hồi quy WQI (supervised.py) → RandomForest
11. Evaluation (metrics.py + report.py) → all_metrics.json
12. Lưu models (.pkl) + FINAL_REPORT.md
```

**Điểm mạnh thiết kế:** Mỗi bước có thể bật/tắt độc lập qua config — không cần chạy lại toàn bộ khi chỉnh một phần.

---

### 4.3 `src/data/loader.py` — Đọc dữ liệu
**Chức năng:** Đọc CSV/Parquet, kiểm tra schema (9 cột đúng tên), báo lỗi rõ ràng.

**Kỹ thuật:** Tự động phát hiện encoding, hỗ trợ nhiều định dạng file, kiểm tra số lượng mẫu tối thiểu.

---

### 4.4 `src/data/cleaner.py` — Tiền xử lý
**Chức năng:** 3 bước xử lý theo pipeline sklearn.

| Bước | Phương pháp | Lý do chọn |
|---|---|---|
| **Missing values** | KNN Imputation (k=5, weighted) | Giữ tương quan giữa features; tốt hơn median simple |
| **Outlier** | Winsorization [p1, p99] | Clip thay vì xóa → giữ 100% dữ liệu |
| **Scaling** | RobustScaler | Dùng median/IQR thay vì mean/std → kháng outlier |

**Tại sao KNN Imputation?** `Sulfate` bị missing 23.8% — nếu dùng median thì mất thông tin. KNN tìm 5 mẫu gần nhất (theo các chỉ số khác) rồi suy ra giá trị → chính xác hơn.

**Tại sao RobustScaler?** `Solids` có range 0–70,000 (cực lệch). StandardScaler bị ảnh hưởng bởi outlier → RobustScaler dùng median/IQR ổn định hơn.

**Tại sao Winsorization?** Xóa outlier làm mất dữ liệu quý. Clip tại p1/p99 giữ nguyên phân phối nhưng kiềm chế cực đoan.

---

### 4.5 `src/features/builder.py` — Feature Engineering (42 đặc trưng)

Từ 9 biến gốc, xây dựng thêm 33 đặc trưng mới:

#### Nhóm 1: 9 WHO Binary Flags (`{col}_flag`)
```
Chloramines_flag = 1 nếu Chloramines > 4 ppm (vượt ngưỡng WHO)
→ Binary: an toàn(0) hay nguy hiểm(1)
```

#### Nhóm 2: 9 Deviation Ratios (`{col}_dev`)
```
Chloramines_dev = 0.75 → Chloramines vượt ngưỡng 75% so với khoảng cho phép
→ Đo MỨC ĐỘ vi phạm, không chỉ có/không
```

#### Nhóm 3: 4 Aggregate Risk Features
| Feature | Ý nghĩa |
|---|---|
| `n_violations` | Tổng số chỉ số vượt ngưỡng WHO (0–9) |
| `danger_score` | Điểm nguy hiểm có trọng số theo WHO |
| `max_dev` | Mức vi phạm cao nhất trong 9 chỉ số |
| `violation_ratio` | Tỉ lệ % chỉ số vượt ngưỡng |

#### Nhóm 4: 8 Interaction Features (tích 2 chỉ số)
```
chlor_x_turb = Chloramines × Turbidity  ← Feature quan trọng nhất (41% importance)
oc_x_thm     = Organic_carbon × THM
sulf_x_cond  = Sulfate × Conductivity
```
> **Tại sao interaction?** Trong hóa học nước, các chất độc hại tương tác nhau. Chloramine + độ đục cao cùng lúc nguy hiểm hơn từng yếu tố riêng lẻ.

#### Nhóm 5: 3 Log-transform
```
log_Solids, log_OC, log_Turbidity → Xử lý phân phối lệch phải cực mạnh
```

#### Tính WQI (Water Quality Index)
```
WQI = 100 - Σ(weight_i × deviation_i × 100)
→ Trọng số theo mức độ nguy hiểm WHO: ph=18%, Chloramines=15%, THM=14%...
→ WQI = 0 (rất bẩn) đến 100 (rất sạch)
```

**Kết quả:** F1-macro tăng từ ~0.65 (không FE) lên **0.90** (có FE) — cải thiện **+38%**.

---

### 4.6 `src/mining/clustering.py` — Phân cụm K-Means

**Quy trình chọn K tối ưu — Multi-Metric Voting:**

```
Thử K = 2, 3, 4, 5, 6, 7, 8
→ Tính 3 chỉ số cho mỗi K:
   1. Silhouette (↑ cao hơn tốt hơn)     → K=2 thắng (0.554)
   2. Davies-Bouldin (↓ thấp hơn tốt hơn) → K=3 thắng (0.563)
   3. Calinski-Harabasz (↑ cao hơn tốt hơn) → K=3 thắng (7,286)
→ Bỏ phiếu: K=3 thắng 2/3 → CHỌN K=3
```

**Kết quả phân cụm K=3:**
| Cụm | Số mẫu | Tỉ lệ | Unsafe% | Phân loại |
|---|---|---|---|---|
| 0 | 812 | 24.8% | 77.7% | 🔴 Rủi ro Cao |
| 1 | 1,501 | 45.8% | 60.0% | 🟡 Rủi ro Trung bình |
| 2 | 963 | 29.4% | 48.3% | 🟢 Tương đối An toàn |

**Quyết định kỹ thuật quan trọng:** Dùng raw data (median-imputed) thay vì scaled data cho clustering → `Solids` có range 0–70,000 tạo phân tầng tự nhiên theo mức ô nhiễm, phản ánh thực tế hơn.

**Metrics cuối:** Silhouette=0.536 | Davies-Bouldin=0.563 | Calinski-Harabasz=7,285

---

### 4.7 `src/mining/association.py` — Luật kết hợp Apriori

**Chiến lược rời rạc hóa:** Quantile 3-bin (Low/Medium/High) — không dùng ngưỡng WHO vì 100% vi phạm → mất thông tin phân biệt.

**Thêm `Potability` vào transaction** → Tìm được luật có ý nghĩa thực tế ("khi nào thì an toàn/không an toàn").

**Top 3 luật quan trọng nhất:**
| Antecedent | Consequent | Confidence | Lift |
|---|---|---|---|
| {THM_Low, Turbidity_Low} | → Potable | 88.5% | 2.27× |
| {Chloramines_Low, THM_Low} | → Potable | 87.5% | 2.24× |
| {Chloramines_Low, OC_Low} | → Potable | 87.5% | 2.24× |

**Đọc hiểu:** "Khi cả Trihalomethane và Turbidity đều thấp (nhóm Low) → xác suất nước an toàn là 88.5%, gấp 2.27 lần so với ngẫu nhiên."

**Kết quả:** 78 luật tổng cộng (58 luật dự đoán Potability trực tiếp).

---

### 4.8 `src/models/supervised.py` — Phân lớp Ensemble + Hồi quy WQI

#### Phân lớp
**Thách thức:** Dữ liệu mất cân bằng 61%/39% → Nếu dùng Accuracy thì mô hình đoán "tất cả không an toàn" đạt 61% mà không học gì. → Phải dùng F1-macro, PR-AUC.

**Giải pháp SMOTETomek:**
```
SMOTE: Tổng hợp mẫu thiểu số (Safe) mới bằng nội suy k-NN
Tomek Links: Xóa các cặp mẫu biên giới mơ hồ
→ Cân bằng dataset trước khi train
```

**Kiến trúc Ensemble VotingClassifier:**
```
XGBoost      → Gradient boosting, mạnh với dữ liệu phi tuyến
ExtraTrees   → Cực kỳ nhanh, randomized splits
RandomForest → Bagging, ổn định, kháng overfitting
→ Soft Voting: trung bình xác suất của 3 mô hình
→ Threshold tuning: 0.52 (thay vì 0.5 mặc định)
```

**Tối ưu ngưỡng phân lớp:**
| Ngưỡng | F1 | FN (bỏ sót nguy hiểm) | FP (cảnh báo nhầm) |
|---|---|---|---|
| 0.1 | 0.755 | 5 | 156 |
| 0.52 | **0.900** | **25** | **38** |
| 0.9 | 0.824 | 110 | 3 |

> Ngưỡng 0.52 cân bằng tốt nhất giữa bỏ sót nguy hiểm và cảnh báo thừa.

**Kết quả phân lớp:**
| Chỉ số | CV 5-fold | Test hold-out |
|---|---|---|
| **F1-macro** | 0.917 ± 0.006 | **0.900** |
| **ROC-AUC** | 0.974 | **0.955** |
| **PR-AUC** | 0.975 | **0.929** |
| Train↔Test gap | — | 0.017 (không overfit) |

**Confusion Matrix:**
```
Dự đoán:    Not Safe    Safe
Thực tế Not Safe:  362 (TN)   38 (FP)
Thực tế Safe:       25 (FN)  231 (TP)
```
→ Chỉ bỏ sót 25 mẫu nguy hiểm (FN=25) — rất an toàn cho ứng dụng thực tế.

**So sánh baselines:**
| Mô hình | F1-macro | ROC-AUC |
|---|---|---|
| **Ensemble (XGB+ET+RF) + SMOTETomek** | **0.900** | **0.955** |
| RandomForest (không SMOTE) | 0.896 | 0.954 |
| LogisticRegression | 0.884 | 0.963 |
| DummyClassifier Random | 0.483 | 0.483 |
| ZeroR (majority) | 0.379 | 0.500 |

#### Hồi quy WQI
**Mục tiêu:** Thay vì chỉ phán "an toàn/không", tính điểm cụ thể 0–100.

| Chỉ số | Giá trị |
|---|---|
| MAE | 1.357 (sai số trung bình 1.36 điểm) |
| RMSE | 2.032 |
| **R²** | **0.948** (giải thích 94.8% phương sai) |
| SMAPE | 8.57% |

---

### 4.9 `src/models/semi_supervised.py` — Học Bán giám sát

**Bối cảnh:** Giả lập tình huống chỉ có 20% mẫu được xét nghiệm y tế (524/2,620 mẫu training có nhãn).

**Thuật toán Label Spreading k-NN:**
```
1. Xây đồ thị k-NN (k=7) giữa tất cả mẫu
2. "Lan truyền" nhãn từ mẫu có nhãn → mẫu chưa nhãn
3. Mẫu gần nhau về đặc trưng → được gán nhãn giống nhau
```

**Kết quả:** SSL F1=0.757 vs Supervised F1=0.862 → **Thua -10.4%**

**Phân tích lý do thất bại:**
- Dataset nước không có cấu trúc cluster rõ ràng (Silhouette chỉ 0.54)
- An toàn/Không an toàn không tạo vùng tách biệt trong không gian đặc trưng
- Label Spreading hoạt động tốt khi dữ liệu có cluster rõ (ảnh, văn bản...)
- Đây là **kết quả thực tế và trung thực** — không "làm đẹp" số liệu

> **Bài học:** Semi-supervised không phải lúc nào cũng tốt hơn supervised. Dataset phải có cấu trúc manifold rõ ràng.

---

### 4.10 `src/evaluation/` — Đánh giá & Báo cáo

**metrics.py:** Tính toán đầy đủ F1, Precision, Recall, ROC-AUC, PR-AUC, confusion matrix, threshold analysis theo từng mức ngưỡng.

**report.py:** Tổng hợp tất cả kết quả → xuất `all_metrics.json` + `FINAL_REPORT.md` tự động.

---

### 4.11 `src/visualization/plots.py` — 11 loại biểu đồ

| # | Tên file | Nội dung |
|---|---|---|
| 01a | `01a_missing_values.png` | Heatmap vị trí missing values |
| 01b | `01b_target_distribution.png` | Pie chart: 61% unsafe / 39% safe |
| 01c | `01c_histograms_who.png` | 9 histogram + đường WHO |
| 01d | `01d_correlation_heatmap.png` | Heatmap tương quan giữa features |
| 01e | `01e_coefficient_variation.png` | CV của từng feature (Solids cao nhất) |
| 01f | `01f_boxplot_safe_unsafe.png` | Boxplot so sánh an toàn vs không an toàn |
| 02b | `02b_wqi_distribution.png` | Phân phối điểm WQI |
| 02c | `02c_discretization.png` | Rời rạc hóa Low/Medium/High |
| 02d | `02d_feature_importance.png` | Top features (chlor_x_turb = 41%) |
| 03b | `03b_elbow_analysis.png` | Elbow curve + 3 chỉ số chọn K |
| 03c | `03c_cluster_heatmap.png` | Heatmap profile 3 cụm |
| 04b | `04b_best_model_eval.png` | Evaluation model tốt nhất |
| 04c | `04c_roc_pr_curve.png` | ROC curve + PR curve |
| 05a | `05a_confusion_matrix.png` | Ma trận nhầm lẫn |
| 05b | `05b_threshold_analysis.png` | F1/Precision/Recall vs ngưỡng |
| 09 | `09_ssl_learning_curve.png` | SSL learning curve (F1 vs % nhãn) |
| 10 | `10_wqi_regression.png` | Actual vs Predicted WQI |
| 11 | `11_radar_metrics.png` | Radar chart tổng hợp tất cả metrics |

---

## 5. KẾT QUẢ TỔNG HỢP

### Bảng thành tích cuối cùng
| Kỹ thuật | Chỉ số chính | Kết quả | Đánh giá |
|---|---|---|---|
| **Apriori** | Số luật tìm được | 78 luật | ✅ Tốt — 58 luật dự đoán Potability |
| **K-Means** | Silhouette | 0.536 | ✅ Tốt — 3 cụm rõ ràng |
| **Ensemble** | F1-macro | **0.900** | ✅ Xuất sắc |
| **Ensemble** | ROC-AUC | **0.955** | ✅ Xuất sắc |
| **Semi-SSL** | F1-macro | 0.757 | ⚠️ Thấp hơn supervised |
| **WQI Regr.** | R² | **0.948** | ✅ Xuất sắc |

### Models đã lưu (outputs/models/)
| File | Nội dung | Kích thước |
|---|---|---|
| `best_classifier.pkl` | VotingClassifier (XGB+ET+RF) đã train | ~50 MB |
| `best_regressor.pkl` | RandomForest hồi quy WQI | ~30 MB |
| `imputer.pkl` | KNN Imputer (fit trên train) | nhỏ |
| `scaler.pkl` | RobustScaler (fit trên train) | nhỏ |
| `winsor_bounds.pkl` | Giới hạn Winsorization | nhỏ |
| `wqi_regressor.pkl` | Bản sao regressor WQI | ~30 MB |

---

## 6. CÁC ĐIỂM NỔI BẬT ĐỂ THUYẾT TRÌNH

### A. Feature Engineering là chìa khóa (+38% F1)
Đây là đóng góp kỹ thuật quan trọng nhất. Feature `chlor_x_turb` (tích Chloramines × Turbidity) trở thành feature quan trọng nhất với 41% importance — điều này không thể phát hiện nếu chỉ dùng 9 feature gốc.

### B. Multi-metric Voting cho K-Means
Thay vì chỉ dùng Silhouette (chọn K=2), nhóm dùng 3 chỉ số bình chọn → K=3 tốt hơn theo 2/3 chỉ số. Đây là cách tiếp cận khoa học và thuyết phục hơn.

### C. Transparency về Semi-supervised
Kết quả SSL thua supervised (-10.4%) được báo cáo trung thực kèm phân tích lý do. Đây thể hiện tính học thuật nghiêm túc — không "làm đẹp" kết quả.

### D. Threshold tuning thực tế
Ngưỡng 0.52 được chọn thay vì 0.5 mặc định sau khi phân tích trade-off FN/FP. Trong y tế/môi trường, bỏ sót mẫu nguy hiểm (FN) tệ hơn cảnh báo thừa (FP).

### E. Hệ thống pipeline hoàn chỉnh
Toàn bộ từ raw data → trained model → report chạy được bằng 1 lệnh:
```bash
python scripts/run_pipeline.py --config configs/params.yaml
```

### F. WQI — chỉ số sáng tạo
Thay vì chỉ phán "an toàn/không", nhóm tạo thêm WQI 0–100 có trọng số WHO → người dùng biết nước tệ đến mức nào, không chỉ biết có/không.

---

## 7. LUỒNG THUYẾT TRÌNH GỢI Ý

```
1. BÀI TOÁN (2 phút)
   → "Chi phí xét nghiệm cao → cần giải pháp từ cảm biến"
   → Giới thiệu 9 chỉ số hóa-lý

2. DỮ LIỆU & TIỀN XỬ LÝ (3 phút)
   → Mất cân bằng 61/39, missing 23.8%
   → Tại sao KNN Imputation, RobustScaler, Winsorization

3. FEATURE ENGINEERING (3 phút)
   → 42 features từ 9 gốc → +38% F1
   → WHO flags, interaction features (chlor_x_turb = 41%)

4. DATA MINING (5 phút)
   → Apriori: 78 luật, top 3 luật cảnh báo (lift 2.27×)
   → K-Means: multi-metric voting → K=3 → 3 mức rủi ro

5. MÔ HÌNH ML (5 phút)
   → SMOTETomek cân bằng dữ liệu
   → Ensemble XGB+ET+RF → F1=0.90, ROC=0.955
   → Threshold tuning: 0.52, FN=25 (rất ít bỏ sót)

6. BÁN GIÁM SÁT (2 phút)
   → Kịch bản chỉ 20% nhãn
   → Kết quả thực: -10.4% → phân tích lý do

7. WQI REGRESSION (2 phút)
   → R²=0.948, MAE=1.36
   → Biểu đồ Actual vs Predicted

8. KẾT LUẬN (2 phút)
   → Hệ thống pipeline hoàn chỉnh, 1 lệnh chạy
   → Ứng dụng thực tế: giám sát nước chi phí thấp
```

---

## 8. CÂU HỎI THƯỜNG GẶP KHI THUYẾT TRÌNH

**Q: Tại sao không dùng Deep Learning?**  
A: Dataset chỉ 3,276 mẫu × 9 features — quá nhỏ cho DL. Ensemble trees hoạt động tốt hơn và giải thích được.

**Q: Tại sao K=3 mà không phải K=2?**  
A: Multi-metric voting: K=3 thắng Davies-Bouldin và Calinski-Harabasz, chỉ thua Silhouette. 2/3 chỉ số → K=3.

**Q: SMOTETomek là gì?**  
A: SMOTE tạo mẫu giả cho nhóm thiểu số bằng nội suy, Tomek Links xóa các cặp mẫu biên nhập nhằng → cân bằng và làm rõ ranh giới phân lớp.

**Q: Tại sao SSL thua?**  
A: Label Spreading cần cluster rõ ràng để lan truyền nhãn. Dữ liệu nước không có cụm tách biệt sắc nét → nhãn lan truyền sai → F1 thấp.

**Q: Feature `chlor_x_turb` có ý nghĩa gì?**  
A: Trong hóa học xử lý nước, Chloramine tác dụng mạnh hơn khi độ đục cao (nhiều hạt vẩn) — tương tác này nguy hiểm. Model tự học được điều này qua feature tích.

---

*Phân tích được tổng hợp từ toàn bộ source code, notebooks, metrics JSON, và FINAL_REPORT.md của dự án.*  
*Ngày: 2026-03-24*
