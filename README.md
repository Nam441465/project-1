# Hệ Thống Phát Hiện Xâm Nhập Mạng (NIDS) trên Lưu Lượng NetFlow NF-UNSW-NB15

Dự án nghiên cứu, thực nghiệm và phát triển hệ thống phát hiện xâm nhập mạng (Network Intrusion Detection System - NIDS) dựa trên phương pháp luận của bài báo:  
**"Random Forest-Based Network Intrusion Detection with Feature Selection and Class Balancing on UNSW-NB15 Traffic"** (*Information*, MDPI, 2026, 17(9), 836), kết hợp các cải tiến chuyên sâu về **Gradient Boosting**, **Tối ưu Ngưỡng Mục tiêu Recall**, **Mô hình Nhạy Chi phí (Cost-Sensitive NIDS)** và **Giải phẫu Mẫu Bỏ sót (False Negative Forensics)**.

---

## 1. Tổng quan Đề tài & Đóng góp Nghiên cứu

### 1.1. Bối cảnh & Dữ liệu Nghiên cứu
- **Tập dữ liệu:** `NF-UNSW-NB15.csv` (1,623,118 bản ghi luồng mạng NetFlow v1, gồm 14 trường thông tin chuẩn L3/L4/L7).
- **Phân chia dữ liệu:** Phân tầng theo nhãn tấn công (`random_state=10`):
  - **Tập Huấn luyện (Train):** 1,104,497 dòng (68.05%).
  - **Tập Hiệu chỉnh (Validation):** 155,586 dòng (9.59%).
  - **Tập Kiểm thử Độc lập (Held-out Test):** 363,035 dòng (22.37% - khi lọc nhị phân Benign + 1 Attack là ~347,200 đến 352,300 mẫu mỗi bài toán).
- **5 Loại tấn công trọng điểm:** `DoS`, `Exploits`, `Backdoor`, `Analysis`, `Reconnaissance` (tỷ lệ mất cân bằng cực đoan từ 0.06% đến 1.58% so với lưu lượng bình thường Benign).
- **10 Đặc trưng NetFlow phi địa chỉ IP:** Loại bỏ `IPV4_SRC_ADDR` và `IPV4_DST_ADDR` để chống hiện tượng mô hình học vẹt (overfitting) địa chỉ IP cố định của phòng thí nghiệm:
  - `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`, `L7_PROTO`, `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`, `TCP_FLAGS`, `FLOW_DURATION_MILLISECONDS`.

### 1.2. Các Giai đoạn Nghiên cứu
1. **Giai đoạn 1 (Tái lập & Thẩm định Độc lập Nghiên cứu Cơ sở):**
   - Lựa chọn đặc trưng tự động bằng `SelectKBest(score_func=mutual_info_classif)` qua dải $k \in \{3, 5, 7, 10\}$.
   - So sánh 2 điều kiện: **Without SMOTE** (nguyên bản) và **With SMOTE** (cân bằng 56,000 mẫu thiểu số).
   - Tối ưu tham số qua 3-fold `RandomizedSearchCV` chống rò rỉ dữ liệu bằng `imblearn.pipeline.Pipeline`, dò ngưỡng $t^*$ tối ưu $F_1$ trên Validation và đánh giá trên Held-out Test.
   - **Xác nhận luận điểm của bài báo MDPI:** SMOTE không phải giải pháp vạn năng; 4/5 loại tấn công đạt $F_1$ cao hơn khi KHÔNG dùng SMOTE, và SMOTE làm gia tăng đáng kể số ca báo động giả (False Positives).
2. **Giai đoạn 2 (Cải tiến Toàn diện Hệ thống NIDS):**
   - **HistGradientBoosting:** Tăng tốc độ huấn luyện từ **12x đến 53x lần** so với Random Forest, F1 bám sát chỉ cách biệt từ 0.004 đến 0.045.
   - **Recall-Targeted Thresholding:** Điều chỉnh ngưỡng để đạt mục tiêu độ phủ $R \ge 95\%, 98\%, 99\%$, giảm thiểu tối đa số ca lọt lưới (Zero-Trust SOC).
   - **Cost-Sensitive Optimization:** Tối ưu hóa hàm chi phí an ninh mạng $\text{Cost}(t) = c \times \text{FN}(t) + 1 \times \text{FP}(t)$ ($c \in \{1, 5, 10, 20\}$), tiết kiệm tới 38.1% chi phí vận hành tại $c=10$.
   - **False Negative Forensics:** Phân lập mẫu bỏ sót thành 2 nhóm: **Near-Miss** (sát ngưỡng) và **Deep Stealth** (ẩn mình sâu); chứng minh nguyên nhân vật lý mạng và đề xuất kiến trúc **2-Tier Hybrid NIDS**.

---

## 2. Cấu trúc Thư mục Dự án

```
project/
├── data/                               # Dữ liệu luồng mạng NetFlow
│   ├── NF-UNSW-NB15.csv                # Dữ liệu gốc 1.62M dòng (112MB)
│   ├── NetFlow_v1_Features.csv         # Bảng mô tả 14 đặc trưng NetFlow v1
│   └── processed/                      # Phân vùng dữ liệu chuẩn hóa
│       ├── train.parquet               # Tập huấn luyện nạp nhanh (16MB)
│       ├── val.parquet                 # Tập validation nạp nhanh (2.5MB)
│       ├── heldout_test.parquet        # Tập test độc lập (5.5MB)
│       ├── split_summary.csv           # Bảng thống kê phân bố lớp
│       └── split_metadata.json         # Metadata tỷ lệ phân chia
│
├── docs/                               # Tài liệu nghiên cứu & Kế hoạch
│   ├── information-17-00836.pdf        # Bài báo gốc MDPI Information 2026
│   ├── ke-hoach-UNSW-NB15.pdf          # Đề cương & kế hoạch nghiên cứu 15 buổi
│   └── Kiến thức mạng cần học...pdf   # Tài liệu bổ trợ kiến thức giao thức mạng
│
├── src/                                # Mã nguồn thực thi chuẩn hóa module (11 scripts)
│   ├── split_data.py                   # 1. Phân chia tập dữ liệu train/val/test phân tầng
│   ├── preprocessor.py                 # 2. Xử lý dữ liệu, trích xuất 10 đặc trưng NetFlow
│   ├── feature_selection.py            # 3. Chọn đặc trưng Mutual Info & lấy mẫu SMOTE
│   ├── train_rf.py                     # 4. Huấn luyện Random Forest, dò ngưỡng F1-Opt
│   ├── run_experiments.py              # 5. Điều phối thực nghiệm cơ sở RF cho 5 attacks
│   ├── export_rf_models.py             # 6. Tái xuất chuẩn hóa 10 mô hình RF phục vụ Web Demo
│   ├── generate_eda_and_baseline_plots.py # 7. Xuất 8 biểu đồ EDA & Baseline (Hình 1 - 8)
│   ├── train_gb.py                     # 8. Huấn luyện HistGradientBoosting (Tối ưu tốc độ)
│   ├── plot_rf_vs_gb.py                # 9. Trực quan hóa đối sánh RF vs GB (Hình 9 - 10)
│   ├── advanced_evaluation.py          # 10. Tối ưu Recall, Cost-Sensitive & Forensics (Hình 11 - 13)
│   └── test_pipeline.py                # 11. Bộ kiểm thử tự động toàn diện (7 test suites)
│
├── results/                            # Bảng số liệu và báo cáo khoa học
│   ├── baseline/                       # Kết quả thực nghiệm cơ sở Random Forest
│   │   ├── summary_rf_comparison.csv   # Bảng tổng kết đối sánh Without vs With SMOTE
│   │   ├── mi_scores_5attacks.csv      # Điểm số Mutual Information của 10 đặc trưng
│   │   ├── *_metrics.csv               # Bảng chi tiết từng mốc k cho 5 loại tấn công
│   │   ├── *_best_proba_*.npy          # Mảng xác suất dự đoán tối ưu trên Val và Test
│   │   ├── eda_and_baseline_findings.md# Báo cáo phân tích EDA và Baseline
│   │   └── models/                     # Mô hình Random Forest đã huấn luyện (.joblib)
│   │
│   └── improved/                       # Kết quả cải tiến Giai đoạn 2
│       ├── summary_gb_comparison.csv   # Bảng tổng kết HistGradientBoosting
│       ├── rf_vs_gb_comparison.csv     # Bảng đối sánh trực tiếp Random Forest vs GB
│       ├── recall_targeted_thresholds.csv # Kết quả tối ưu ngưỡng theo Recall 95%, 98%, 99%
│       ├── cost_sensitive_evaluation.csv # Kết quả tối ưu chi phí vận hành SOC (c = 1..20)
│       ├── fn_forensic_analysis.csv    # Thống kê giải phẫu đặc trưng mẫu bỏ sót FN
│       ├── fn_and_cost_findings.md     # Báo cáo tổng hợp phát hiện chuyên sâu Bước 5
│       └── models/                     # Mô hình Gradient Boosting đã huấn luyện (.joblib)
│
├── figures/                            # 13 Biểu đồ khoa học chất lượng cao (300 DPI)
│   ├── fig01_class_imbalance.png       # Hình 1: Phân bố lớp và tỷ lệ mất cân bằng dữ liệu NF-UNSW-NB15
│   ├── fig02_feature_distributions.png # Hình 2: Phân phối đặc trưng lưu lượng mạng theo từng loại tấn công (Boxplot)
│   ├── fig03_correlation_heatmap.png   # Hình 3: Ma trận tương quan Pearson giữa 10 đặc trưng NetFlow
│   ├── fig04_mutual_information_heatmap.png # Hình 4: Điểm số và xếp hạng Mutual Information cho 5 loại tấn công
│   ├── fig05_train_val_test_consistency.png # Hình 5: Kiểm chứng tính nhất quán phân phối Stratified giữa Train, Val và Held-out Test
│   ├── fig06_rf_smote_vs_nosmote_comparison.png # Hình 6: Đối sánh hiệu năng Random Forest giữa Without SMOTE và With SMOTE (56,000 mẫu)
│   ├── fig07_rf_k_sensitivity.png      # Hình 7: Độ nhạy hiệu năng F1 theo số lượng đặc trưng k in {3, 5, 7, 10} và đường cong dò ngưỡng Validation
│   ├── fig08_rf_confusion_matrices_table8.png # Hình 8: Ma trận nhầm lẫn của Random Forest trên tập Held-out Test (Without SMOTE vs With SMOTE)
│   ├── fig09_rf_vs_gb_comparison.png   # Hình 9: Đối sánh F1, Precision, Recall và Speedup RF vs GB
│   ├── fig10_precision_recall_curves_rf_vs_gb.png # Hình 10: So sánh đường cong PR giữa RF và GB
│   ├── fig11_recall_threshold_tradeoff.png # Hình 11: Đánh đổi P-R-F1 và mức cắt giảm FN theo Recall
│   ├── fig12_cost_sensitive_analysis.png # Hình 12: Tối ưu chi phí SOC và tỷ lệ tiết kiệm tại c=10
│   └── fig13_false_negative_forensics.png # Hình 13: Giải phẫu Near-Miss vs Deep Stealth & 2-Tier NIDS
│
├── web_app/                            # Ứng dụng Web Demo tương tác (SOC NIDS Live Console)
│   ├── app.py                          # Giao diện Dashboard Streamlit 3 Tabs chuyên nghiệp
│   └── inference_engine.py             # Lõi suy luận NIDS đa mô hình (Auto-Discovery Registry)
│
├── run_web.sh                          # Script khởi động nhanh Web Demo trên cổng 8501
├── requirements.txt                    # Danh sách thư viện và phiên bản chuẩn hóa
├── .gitignore                          # Cấu hình loại trừ file rác, cache và binary lớn
├── log.md                              # Nhật ký nghiên cứu chi tiết theo từng ngày
└── README.md                           # Tài liệu tổng quan dự án
```

---

## 3. Bảng Tổng Hợp Kết Quả Thực Nghiệm

### 3.1. Bảng 1: Kết quả Thực nghiệm Cơ sở Random Forest (Without SMOTE vs With SMOTE)
Đánh giá độc lập trên tập **Held-out Test (363,035 mẫu tổng - ~347,200 đến 352,300 mẫu mỗi bài toán nhị phân)**:

| Loại Tấn công | Điều kiện | $k^*$ Tối ưu | Ngưỡng $t^*$ | Test Precision | Test Recall | **Test $F_1$** | Báo động giả (FP) | Bỏ sót (FN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | **Without SMOTE** | 10 | 0.76 | **0.8244** | 0.9097 | **0.8650** | **219** | 102 |
| | With SMOTE | 10 | 0.58 | 0.7922 | **0.9177** | 0.8503 | 272 (+53) | **93** (-9) |
| **Exploits** | **Without SMOTE** | 10 | 0.46 | **0.9283** | 0.9664 | **0.9470** | **413** | 186 |
| | With SMOTE | 10 | 0.56 | 0.9222 | **0.9689** | 0.9450 | 452 (+39) | **172** (-14) |
| **Backdoor** | **Without SMOTE** | 10 | 0.30 | **0.9547** | **0.9523** | **0.9535** | **18** | **19** |
| | With SMOTE | 10 | 0.59 | 0.9474 | 0.9497 | 0.9486 | 21 (+3) | 20 (+1) |
| **Analysis** | **Without SMOTE** | 3 | 0.55 | **0.9915** | 0.7803 | **0.8733** | **3** | 98 |
| | With SMOTE | 10 | 0.94 | 0.9751 | **0.7892** | 0.8724 | 9 (+6) | **94** (-4) |
| **Reconnaissance** | **Without SMOTE** | 10 | 0.44 | 0.9464 | **0.9891** | **0.9673** | 154 | **30** |
| | With SMOTE | 10 | 0.76 | **0.9486** | 0.9804 | 0.9642 | **146** (-8) | 54 (+24) |

> **Kết luận:** Trạng thái **Without SMOTE** chiến thắng trên 4/5 loại tấn công. SMOTE làm phát sinh từ 39 đến 53 cảnh báo giả thêm trên `DoS` và `Exploits`, trong khi chỉ cứu được một lượng nhỏ mẫu bỏ sót.

---

### 3.2. Bảng 2: Đối sánh Random Forest vs HistGradientBoosting (Điều kiện Chuẩn Without SMOTE)

| Loại Tấn công | Mô hình | Test Precision | Test Recall | Test $F_1$ | PR-AUC | Thời gian Fit ($k^*$) | Tốc độ Tăng tốc (**Speedup**) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | Random Forest | **0.8244** | **0.9097** | **0.8650** | **0.9331** | 106.12s | Gốc (1.0x) |
| | HistGradientBoosting | 0.7639 | 0.8850 | 0.8200 | 0.8899 | **8.06s** | **Nhanh hơn 13.17x** |
| **Exploits** | Random Forest | **0.9283** | **0.9664** | **0.9470** | **0.9604** | 123.94s | Gốc (1.0x) |
| | HistGradientBoosting | 0.9208 | 0.9451 | 0.9328 | 0.9501 | **2.34s** | **Nhanh hơn 52.97x** |
| **Backdoor** | Random Forest | 0.9547 | **0.9523** | **0.9535** | 0.9827 | 90.05s | Gốc (1.0x) |
| | HistGradientBoosting | **0.9782** | 0.9020 | 0.9386 | **0.9848** | **2.03s** | **Nhanh hơn 44.36x** |
| **Analysis** | Random Forest | **0.9915** | **0.7803** | **0.8733** | **0.8965** | 61.27s | Gốc (1.0x) |
| | HistGradientBoosting | 0.9914 | 0.7735 | 0.8690 | 0.8701 | **1.34s** | **Nhanh hơn 45.72x** |
| **Reconnaissance** | Random Forest | **0.9464** | **0.9891** | **0.9673** | **0.9651** | 84.44s | Gốc (1.0x) |
| | HistGradientBoosting | 0.9353 | 0.9884 | 0.9611 | 0.9624 | **2.22s** | **Nhanh hơn 38.04x** |

> **Kết luận:** `HistGradientBoosting` đạt hiệu năng phát hiện tiệm cận Random Forest (chỉ kém 0.004 đến 0.045 F1) nhưng mang lại **tốc độ huấn luyện vượt bậc (nhanh hơn từ 13.17x đến 52.97x lần)**, cực kỳ phù hợp cho môi trường luồng mạng lớn và tái huấn luyện định kỳ (Continuous Retraining).
>
> *Lưu ý phương pháp luận (`train_gb.py`):* Ở nhánh `Without SMOTE`, `HistGradientBoosting` áp dụng **Hybrid Minority-Preserving Sampling** (giữ nguyên 100% mẫu Attack + lấy mẫu phân tầng 150,000 mẫu Benign thay vì toàn bộ ~1.055M Benign) nhằm bảo vệ độ phân giải chia giỏ 255 quantile bins không bị lấn át bởi lưu lượng lành tính. Thiết kế này cùng thuật toán histogram binning đóng góp vào mức tăng tốc huấn luyện vượt trội, đồng thời làm phân phối xác suất thô dịch chuyển lên vùng cao (khiến ngưỡng $t^*$ tối ưu đạt tới `0.99` trên Validation).

---

### 3.3. Bảng 3: Tối ưu Chi phí Vận hành SOC ($c=10$) & Giải phẫu Mẫu Bỏ sót (False Negatives)

Hàm chi phí vận hành: $\text{Cost}(t) = 10 \times \text{FN}(t) + 1 \times \text{FP}(t)$ (1 cuộc tấn công lọt lưới nguy hiểm gấp 10 lần 1 báo động giả):

| Loại Tấn công | FN tại F1-Opt | FN tại $t^*_{\text{cost}}$ | FP tại F1-Opt | FP tại $t^*_{\text{cost}}$ | **Tiết kiệm Chi phí SOC** | Mẫu Sát ngưỡng (**Near-Miss**) | Mẫu Ẩn mình (**Deep Stealth**) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | 102 | **34** | 219 | 427 | **Tiết kiệm 38.10%** | **100 / 102 (98.0%)** | 2 / 102 (2.0%) |
| **Exploits** | 186 | **54** | 413 | 903 | **Tiết kiệm 36.52%** | **152 / 186 (81.7%)** | 34 / 186 (18.3%) |
| **Reconnaissance**| 30 | **15** | 154 | 189 | **Tiết kiệm 25.33%** | **18 / 30 (60.0%)** | 12 / 30 (40.0%) |
| **Backdoor** | 19 | **12** | 18 | 54 | **Tiết kiệm 16.35%** | **11 / 19 (57.9%)** | 8 / 19 (42.1%) |
| **Analysis** | 98 | **14** | 3 | 715 | **Tiết kiệm 13.02%** | 48 / 98 (49.0%) | **50 / 98 (51.0%)** |

#### Phát hiện Vật lý Mạng & Kiến trúc 2-Tier NIDS:
1. **Dạng tấn công Slow-Rate DoS (`DoS`):** 98.0% mẫu FN là dạng **Near-Miss** ($0.10 \le p < 0.76$). Kẻ tấn công duy trì luồng dữ liệu thấp (Slowloris) để tránh ngưỡng cảnh báo mặc định. Hạ ngưỡng nhạy chi phí ($c=10$) cứu được **68 / 102 cuộc tấn công lọt lưới**.
2. **Dạng tấn công Web Scanners / Probes (`Analysis`):** 51.0% mẫu FN là dạng **Deep Stealth** ($p < 0.10$). 88% cuộc tấn công đánh vào cổng 80 (HTTP) với kích thước gói tin bình thường (`IN_BYTES` trung vị 814B, `OUT_BYTES` 1,155B) hoàn toàn giống lưu lượng duyệt web thông thường của Benign. Ở tầng NetFlow (L3/L4) không có thông tin URI/payload tầng 7 nên không thể phân biệt được bằng thống kê luồng đơn thuần.
3. **Kiến trúc Khuyến nghị (2-Tier Hybrid NIDS):**
   - **Tầng 1 (Line-rate Filtering):** Sử dụng `HistGradientBoosting` với ngưỡng chi phí $c=10$ để thanh lọc 99.8% lưu lượng Benign ở tốc độ hàng triệu gói tin/giây.
   - **Tầng 2 (Deep Inspection):** Chuyển các luồng nghi vấn trong vùng tranh chấp ($0.10 \le p < t^*$) sang hệ thống `Random Forest` kết hợp Deep Packet Inspection (DPI) để giải mã payload.

---

## 4. Hướng dẫn Chạy & Tái Lập Thực Nghiệm

### 4.1. Thiết lập Môi trường
Yêu cầu Python 3.10+ trên Linux/WSL2 hoặc Windows (khuyến nghị cài đặt bên trong môi trường ảo `venv`):
```bash
# Cài đặt đồng bộ toàn bộ thư viện theo phiên bản chuẩn hóa
pip install -r requirements.txt

# Hoặc cài đặt trực tiếp các gói thành phần:
# pip install numpy pandas scikit-learn==1.9.1 imbalanced-learn==0.14.2 matplotlib pyarrow joblib streamlit
```

### 4.2. Kiểm thử Tự động Pipeline
Chạy bộ kiểm thử tự động 7 test suites để xác nhận toàn bộ tính toàn vẹn của dữ liệu và kết quả:
```bash
python3 src/test_pipeline.py
```

### 4.3. Các Lệnh Thực thi Từng Phần Dự án
```bash
# 1. Phân chia tập dữ liệu train/val/test và lưu cache Parquet (nếu chạy lại từ đầu)
python3 src/split_data.py

# 2. Huấn luyện chuỗi thực nghiệm cơ sở Random Forest cho 5 loại tấn công
# (Tự động bỏ qua mô hình đã hoàn tất; thêm --force nếu muốn huấn luyện lại từ đầu)
python3 src/run_experiments.py --all

# 3. Xuất 8 biểu đồ EDA và Baseline chất lượng cao (300 DPI)
python3 src/generate_eda_and_baseline_plots.py

# 4. Huấn luyện chuỗi thực nghiệm Gradient Boosting (5 loại tấn công)
python3 src/train_gb.py

# 5. Xuất biểu đồ đối sánh RF vs GB (Hình 9 và Hình 10)
python3 src/plot_rf_vs_gb.py

# 6. Chạy đánh giá nâng cao (Recall mục tiêu, Chi phí Cost-Sensitive, Forensic FN và Hình 11, 12, 13)
python3 src/advanced_evaluation.py
```

### 4.4. Khởi chạy Giao diện Web Demo (SOC NIDS Live Console)
Hệ thống cung cấp giao diện trực quan hóa Dashboard Streamlit hỗ trợ demo thời gian thực:
- **Cách 1 (Khởi động nhanh qua script Bash):**
  ```bash
  ./run_web.sh
  ```
- **Cách 2 (Khởi động trực tiếp bằng Streamlit):**
  ```bash
  streamlit run web_app/app.py --server.port 8501
  ```
- **Truy cập Giao diện Web:** Mở trình duyệt trên máy tính tại địa chỉ: **`http://localhost:8501`**
  - **Tab 1 (Live Single-Flow):** Nạp nhanh các mẫu thử thực tế (Benign, Typical, Near-Miss FN) hoặc tự chỉnh sửa 10 thông số NetFlow để xem phản ứng trực tiếp của Hệ thống 2 Tầng.
  - **Tab 2 (Batch Scanner):** Quét hàng loạt hàng nghìn luồng từ Held-out Test hoặc file CSV tải lên, tự động đo Throughput và tính ma trận nhầm lẫn (Confusion Matrix).
  - **Tab 3 (Model Registry & Findings):** Quản lý tự động 20 mô hình trực tuyến, xem trực tiếp 13 biểu đồ 300 DPI và toàn văn các báo cáo khoa học.

---

## 5. Danh mục 13 Biểu đồ Khoa học (Figures)

Tất cả các biểu đồ đều được xuất tại thư mục `figures/` với độ phân giải chuẩn in ấn khoa học **300 DPI**, sử dụng font chữ `DejaVu Sans`:

1. `fig01_class_imbalance.png`: Phân bố lớp và tỷ lệ mất cân bằng dữ liệu NF-UNSW-NB15.
2. `fig02_feature_distributions.png`: Phân phối đặc trưng lưu lượng mạng theo từng loại tấn công (Boxplot).
3. `fig03_correlation_heatmap.png`: Ma trận tương quan Pearson giữa 10 đặc trưng NetFlow.
4. `fig04_mutual_information_heatmap.png`: Điểm số và xếp hạng Mutual Information cho 5 loại tấn công.
5. `fig05_train_val_test_consistency.png`: Kiểm chứng tính nhất quán phân phối Stratified giữa Train, Val và Held-out Test.
6. `fig06_rf_smote_vs_nosmote_comparison.png`: Đối sánh hiệu năng Random Forest giữa Without SMOTE và With SMOTE (56,000 mẫu).
7. `fig07_rf_k_sensitivity.png`: Độ nhạy hiệu năng F1 theo số lượng đặc trưng k in {3, 5, 7, 10} và đường cong dò ngưỡng Validation.
8. `fig08_rf_confusion_matrices_table8.png`: Ma trận nhầm lẫn của Random Forest trên tập Held-out Test (Without SMOTE vs With SMOTE).
9. `fig09_rf_vs_gb_comparison.png`: Biểu đồ so sánh trực tiếp $F_1$, Precision, Recall và Tốc độ huấn luyện giữa RF và GB.
10. `fig10_precision_recall_curves_rf_vs_gb.png`: Đối sánh đường cong PR giữa Random Forest và Gradient Boosting.
11. `fig11_recall_threshold_tradeoff.png`: Đánh đổi Precision-Recall-F1 và mức cắt giảm mẫu bỏ sót FN tại $R \ge 95\%, 98\%$.
12. `fig12_cost_sensitive_analysis.png`: Tối ưu hóa hàm chi phí SOC và tỷ lệ tiết kiệm chi phí tại $c=10$.
13. `fig13_false_negative_forensics.png`: Phân loại Near-Miss vs Deep Stealth, boxplot đặc trưng và ma trận phòng thủ 2-Tier NIDS.

---
*Dự án hoàn thành đầy đủ các tiêu chuẩn học thuật và sẵn sàng phục vụ báo cáo thuyết trình.*
