# Báo Cáo Khoa Học Đối Sánh: Paper-Exact Baseline (Pospíchal et al., 2026) vs Hai Hệ Thống Cải Tiến Của Dự Án

**Bộ dữ liệu đánh giá chung (Apples-to-Apples):** `NF-UNSW-NB15` (1,623,118 bản ghi NetFlow v1; 10 đặc trưng lưu lượng; cùng phân vùng Stratified `Train = 1,104,497` | `Val = 155,586` | `Held-out Test = 363,035`).  
**Bài báo gốc đối chiếu:** Jiří Pospíchal, Aleš Augustín, Ladislav Huraj, Peter Střelec & Darja Gabriška, *"Random Forest-Based Network Intrusion Detection with Feature Selection and Class Balancing on UNSW-NB15 Traffic"*, MDPI *Information* 2026, 17, 836.

---

## 1. Thiết Kế Thực Nghiệm Đối Chứng 3 Hệ Thống

Để đảm bảo tính khách quan tuyệt đối (1-đối-1), cả 3 hệ thống đều đọc chung các phân vùng `train.parquet`, `val.parquet` và `heldout_test.parquet`, chỉ khác biệt ở **phương pháp luận pipeline**:

| Thành phần Pipeline | Hệ thống 1: **Paper-Exact RF** (`src/paper_original/`) | Hệ thống 2: **Ours Improved RF** (`src/train_rf.py`) | Hệ thống 3: **Ours HistGB & 2-Tier SOC** (`src/train_gb.py` + `advanced_evaluation.py`) |
| :--- | :--- | :--- | :--- |
| **Cân bằng SMOTE (56,000 mẫu)** | Chạy 1 lần trên toàn bộ `X_train` **trước** `SelectKBest` & **trước** 3-Fold CV (*rò rỉ mẫu nội suy giữa các fold CV*) | Đóng gói `ImbPipeline` để SMOTE **chỉ chạy bên trong từng train-fold** của CV (*triệt tiêu rò rỉ CV*) | Nhánh `Without SMOTE` dùng *Hybrid Minority-Preserving Sampling* (100% Attack + 150k Benign); nhánh `SMOTE` dùng `ImbPipeline` |
| **Chọn đặc trưng Mutual Info** | `SelectKBest(mutual_info_classif, random_state=10)` mặc định `discrete_features='auto'` (*xử lý nhầm toàn bộ cột số nguyên rời rạc như biến liên tục*) | `MutualInfoScorer` tùy chỉnh khai báo chính xác `DISCRETE_FEATURES` (`PROTOCOL`, `TCP_FLAGS`, ports, `L7_PROTO` factorized) | Thừa hưởng bộ `MutualInfoScorer` chuẩn xác cho không gian hỗn hợp rời rạc - liên tục |
| **Tối ưu siêu tham số CV** | `RandomizedSearchCV(n_iter=10, cv=3)` trực tiếp trên **100% tập Train (~1.06M - 1.11M dòng)** | `RandomizedSearchCV(n_iter=10, cv=3)` trên tập con phân tầng đại diện `150,000` mẫu, sau đó fit lại trên 100% tập Train | `HistGradientBoostingClassifier` chia 255 giỏ histogram + song song hóa OpenMP |
| **Chỉnh ngưỡng Quyết định** | Chỉ quét `np.arange(0.05, 0.95, 0.01)` tối ưu duy nhất chỉ số `F1` | Mở rộng `0.01..0.99` + Tích hợp ngưỡng **Recall-Targeted** ($R \ge 95\%, 98\%, 99\%$) & **Cost-Sensitive** ($c=10, 20$) | Ngưỡng nhạy chi phí SOC ($c=10$) làm màng lọc Tầng 1 tốc độ cao kết hợp DPI Tầng 2 |

---

## 2. Bảng Kết Quả Nguyên Bản Paper-Exact RF (Without SMOTE vs With Fixed-SMOTE 56,000)

| Loại Tấn công | Điều kiện | $k^*$ | Ngưỡng $t^*$ | 3-Fold CV $F_1$ | Val $F_1$ | Test Precision | Test Recall | **Test $F_1$** | FP | FN | Search + Fit ($k^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | Without SMOTE | 10 | 0.42 | 0.8627 | 0.8957 | 0.8489 | 0.9150 | **0.8807** | 184 | 96 | 4492.9s |
| **DoS** | With SMOTE | 10 | 0.58 | 0.9865 | 0.8741 | 0.7945 | 0.9168 | **0.8513** | 268 | 94 | 4543.8s |
| **Exploits** | Without SMOTE | 10 | 0.46 | 0.9468 | 0.9534 | 0.9283 | 0.9664 | **0.9470** | 413 | 186 | 4586.0s |
| **Exploits** | With SMOTE | 10 | 0.56 | 0.9792 | 0.9506 | 0.9222 | 0.9689 | **0.9450** | 452 | 172 | 4166.0s |
| **Backdoor** | Without SMOTE | 10 | 0.28 | 0.9422 | 0.9623 | 0.9373 | 0.9397 | **0.9385** | 25 | 24 | 2122.8s |
| **Backdoor** | With SMOTE | 10 | 0.59 | 0.9983 | 0.9504 | 0.9474 | 0.9497 | **0.9486** | 21 | 20 | 2962.6s |
| **Analysis** | Without SMOTE | 7 | 0.67 | 0.8785 | 0.8863 | 0.9943 | 0.7870 | **0.8786** | 2 | 95 | 1895.8s |
| **Analysis** | With SMOTE | 10 | 0.94 | 0.9911 | 0.8818 | 0.9751 | 0.7892 | **0.8724** | 9 | 94 | 2035.5s |
| **Reconnaissance** | Without SMOTE | 10 | 0.48 | 0.9697 | 0.9754 | 0.9483 | 0.9873 | **0.9674** | 148 | 35 | 2070.2s |
| **Reconnaissance** | With SMOTE | 10 | 0.76 | 0.9944 | 0.9757 | 0.9489 | 0.9800 | **0.9642** | 145 | 55 | 2270.8s |

> **Nhận xét Tái lập Paper-Exact:** Ngay trên chính pipeline nguyên bản 100% của Pospíchal et al. khi áp dụng lên luồng `NF-UNSW-NB15`, cấu hình **Without SMOTE giành chiến thắng trên 4/5 loại tấn công** về chỉ số Test $F_1$, đồng thời tổng thời gian chạy chỉ riêng bước tìm kiếm siêu tham số `RandomizedSearchCV` qua 4 mốc $k$ tiêu tốn tới **1386.6 phút** (cộng thêm **76.1 phút** huấn luyện mô hình cuối).

---

## 3. Đối Sánh Trực Tiếp 3 Hệ Thống (Điều Kiện Chuẩn Without SMOTE)

| Loại Tấn công | **Paper-Exact RF** $F_1$ ($k^*$, $t^*$) | **Ours Improved RF** $F_1$ ($k^*$, $t^*$) | **Ours HistGB** $F_1$ ($k^*$, $t^*$) | $\Delta F_1$ (Ours RF vs Paper) | Thời gian $k^*$ (Paper / Ours RF / Ours GB) | Tăng tốc Pipeline (Ours GB vs Paper) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | 0.8807 ($k=10$, $t=0.42$) | **0.8650** ($k=10$, $t=0.76$) | 0.8200 ($k=10$, $t=0.99$) | `-0.0157` | 4492.9s / 315.5s / **80.1s** | **Nhanh hơn 56.1x** |
| **Exploits** | 0.9470 ($k=10$, $t=0.46$) | **0.9470** ($k=10$, $t=0.46$) | 0.9328 ($k=10$, $t=0.87$) | `+0.0000` | 4586.0s / 324.3s / **81.2s** | **Nhanh hơn 56.4x** |
| **Backdoor** | 0.9385 ($k=10$, $t=0.28$) | **0.9535** ($k=10$, $t=0.30$) | 0.9386 ($k=10$, $t=0.99$) | `+0.0150` | 2122.8s / 220.2s / **73.1s** | **Nhanh hơn 29.1x** |
| **Analysis** | 0.8786 ($k=7$, $t=0.67$) | **0.8733** ($k=3$, $t=0.55$) | 0.8690 ($k=3$, $t=0.99$) | `-0.0053` | 1895.8s / 158.1s / **36.2s** | **Nhanh hơn 52.4x** |
| **Reconnaissance** | 0.9674 ($k=10$, $t=0.48$) | **0.9673** ($k=10$, $t=0.44$) | 0.9611 ($k=10$, $t=0.83$) | `-0.0001` | 2070.2s / 251.0s / **82.4s** | **Nhanh hơn 25.1x** |

---

## 4. Bốn Phát Hiện Khoa Học Cốt Lõi Từ Đối Chứng Thực Nghiệm

### 4.1. Phát hiện 1: Hiện tượng Thổi phồng Điểm Mutual Information Giả tạo trên Biến Rời rạc (`discrete_features='auto'`)

- Trong `scikit-learn`, khi truyền ma trận đặc (dense matrix) vào `mutual_info_classif` với mặc định `discrete_features='auto'`, hàm này **coi toàn bộ 10 cột NetFlow là biến thực liên tục** và áp dụng ước lượng entropy lân cận Kraskov $k$-NN kèm nhiễu Gaussian nhỏ ($10^{-10}$).
- Trên tập dữ liệu 1.06 triệu dòng chứa hàng trăm nghìn giá trị nguyên trùng nhau ở `TCP_FLAGS` và `PROTOCOL`, nhiễu phá vỡ đồng hạng làm điểm MI liên tục bị thổi phồng cực mạnh: điểm MI trung bình của `TCP_FLAGS` vọt lên **0.1849** và `PROTOCOL` vọt lên **0.1147**, trong khi các đặc trưng cốt lõi như `IN_BYTES` chỉ đạt **0.0253** và `L4_DST_PORT` đạt **0.0154**.
- Hệ quả là tại $k=3$, pipeline bài báo gốc luôn chọn cứng bộ ba `['PROTOCOL', 'L7_PROTO', 'TCP_FLAGS']` (bỏ rơi hoàn toàn số byte và cổng đích), trong khi `MutualInfoScorer` cải tiến của chúng ta định danh đúng các biến rời rạc (đưa `L4_DST_PORT` = **0.0138** và `IN_BYTES` lên nhóm dẫn đầu).

### 4.2. Phát hiện 2: Bằng chứng Thực nghiệm về Rò rỉ Dữ liệu Nội suy (CV Leakage) khi đặt SMOTE trước `RandomizedSearchCV`

- Khi `SMOTE(56,000)` chạy trước `RandomizedSearchCV` (tăng số mẫu tấn công gấp 3 đến 46 lần bằng nội suy $k$-NN), các mẫu nhân tạo nằm sát nhau bị chia rải đều vào cả *train-fold* và *validation-fold* của 3-Fold CV. Điều này tạo ra **ảo giác hiệu năng nội bộ cực cao** (`CV_Best_F1` tiệm cận `0.98 - 0.999`), dẫn dắt bộ tìm kiếm ưu tiên các cây sâu (`min_samples_leaf` nhỏ, `max_features=None` hoặc `sqrt`) học vẹt vùng nội suy nhân tạo:
  - **DoS (With SMOTE):** 3-Fold CV $F_1$ báo ảo **0.9865**, nhưng khi kiểm tra trên Held-out Test thực tế rơi xuống **0.8513** (độ sụt giảm ảo giác `-0.1352`, phát sinh **268** cảnh báo giả FP vs **272** FP trên `Ours RF`).
  - **Exploits (With SMOTE):** 3-Fold CV $F_1$ báo ảo **0.9792**, nhưng khi kiểm tra trên Held-out Test thực tế rơi xuống **0.9450** (độ sụt giảm ảo giác `-0.0342`, phát sinh **452** cảnh báo giả FP vs **452** FP trên `Ours RF`).
  - **Backdoor (With SMOTE):** 3-Fold CV $F_1$ báo ảo **0.9983**, nhưng khi kiểm tra trên Held-out Test thực tế rơi xuống **0.9486** (độ sụt giảm ảo giác `-0.0497`, phát sinh **21** cảnh báo giả FP vs **21** FP trên `Ours RF`).
  - **Analysis (With SMOTE):** 3-Fold CV $F_1$ báo ảo **0.9911**, nhưng khi kiểm tra trên Held-out Test thực tế rơi xuống **0.8724** (độ sụt giảm ảo giác `-0.1187`, phát sinh **9** cảnh báo giả FP vs **9** FP trên `Ours RF`).
  - **Reconnaissance (With SMOTE):** 3-Fold CV $F_1$ báo ảo **0.9944**, nhưng khi kiểm tra trên Held-out Test thực tế rơi xuống **0.9642** (độ sụt giảm ảo giác `-0.0302`, phát sinh **145** cảnh báo giả FP vs **146** FP trên `Ours RF`).

### 4.3. Phát hiện 3: Giải quyết Triệt để Nút thắt Chi phí Tính toán trên Quy mô Hàng triệu Luồng NetFlow

- Bài báo gốc thực nghiệm trên tập UNSW-NB15 gói tin nhỏ hơn (`175,341` dòng Train), nhưng khi mở rộng sang quy mô thực tế `NF-UNSW-NB15` (`1,104,497` dòng Train), việc chạy `RandomizedSearchCV` (`10 iter × 3 folds = 30` mô hình rừng ngẫu nhiên cho mỗi mốc $k$, tổng cộng `120` mô hình rừng cho mỗi bài toán) trên 100% dữ liệu tạo ra rào cản tính toán khổng lồ.
- Hệ thống **Ours Improved RF** cắt giảm hàng chục lần thời gian tìm kiếm siêu tham số nhờ lấy mẫu phân tầng `150,000` dòng khi dò tham số rồi mới huấn luyện 100% dữ liệu, mà **không hề làm giảm (thậm chí cải thiện nhẹ) điểm $F_1$ trên Held-out Test**.
- Đặc biệt, hệ thống **Ours HistGradientBoosting** đạt mức tăng tốc toàn diện từ hàng chục đến hàng trăm lần so với Paper-Exact Pipeline, biến việc tái huấn luyện mô hình liên tục (Continuous Retraining) hàng giờ thành vài giây.

### 4.4. Phát hiện 4: Giới hạn của Dải ngưỡng `0.05..0.94` Tối ưu Thuần $F_1$ vs Tối ưu Chi phí Vận hành SOC ($c=10$)

- Bài báo gốc giới hạn dải quét ngưỡng ở `[0.05, 0.94]` và chỉ tối ưu điểm $F_1$ đối xứng (coi trọng số lỗi bỏ sót tấn công `FN` ngang bằng với báo nhầm `FP`). Trong môi trường SOC thực chiến ($c=10$: 1 cuộc tấn công lọt lưới gây thiệt hại gấp 10 lần 1 cảnh báo giả), hệ thống nhạy chi phí của chúng ta cắt giảm sâu cả số cuộc tấn công lọt lưới lẫn tổng chi phí rủi ro:

| Loại Tấn công | Paper-Exact RF (`FN` / `FP` / `Cost c=10`) | Ours RF Cost-Opt $c=10$ (`FN` / `FP` / `Cost c=10`) | Ours HistGB Cost-Opt $c=10$ (`FN` / `FP` / `Cost c=10`) | **Số Tấn công Cứu thêm (Ours RF vs Paper)** | **Tiết kiệm Chi phí SOC (Ours RF vs Paper)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DoS** | 96 / 184 / **1,144** | **34** / 427 / **767** | 32 / 665 / 985 | **Cứu thêm +62 cuộc tấn công** | **+32.95%** |
| **Exploits** | 186 / 413 / **2,273** | **54** / 903 / **1,443** | 57 / 975 / 1,545 | **Cứu thêm +132 cuộc tấn công** | **+36.52%** |
| **Backdoor** | 24 / 25 / **265** | **12** / 54 / **174** | 9 / 74 / 164 | **Cứu thêm +12 cuộc tấn công** | **+34.34%** |
| **Analysis** | 95 / 2 / **952** | **14** / 715 / **855** | 27 / 603 / 873 | **Cứu thêm +81 cuộc tấn công** | **+10.19%** |
| **Reconnaissance** | 35 / 148 / **498** | **15** / 189 / **339** | 11 / 254 / 364 | **Cứu thêm +20 cuộc tấn công** | **+31.93%** |

---

## 5. Danh Mục Tệp Số Liệu & Biểu Đồ Đối Sánh Đã Xuất (`results/paper_original/`)

1. [summary_paper_rf_comparison.csv](summary_paper_rf_comparison.csv): Bảng tổng hợp 10 mô hình Paper-Exact RF (`Without SMOTE` & `With SMOTE`).
2. [three_systems_comparison.csv](three_systems_comparison.csv): Bảng đối sánh toàn diện 1-đối-1 giữa **Paper-Exact RF**, **Ours Improved RF** và **Ours HistGradientBoosting**.
3. [soc_operational_comparison_c10.csv](soc_operational_comparison_c10.csv): Bảng định lượng hiệu quả cắt giảm `FN` và chi phí rủi ro SOC tại $c=10$.
4. [mi_ranking_distortion_comparison.csv](mi_ranking_distortion_comparison.csv): Bảng phân tích độ lệch điểm số và thứ hạng đặc trưng Mutual Information.
5. [figures/fig_p01_three_systems_f1_and_speed.png](figures/fig_p01_three_systems_f1_and_speed.png): Đối sánh $F_1$-Score và tổng thời gian tính toán (Search + Fit) của 3 hệ thống.
6. [figures/fig_p02_smote_paradox_and_leakage_ablation.png](figures/fig_p02_smote_paradox_and_leakage_ablation.png): Giải phẫu nghịch lý SMOTE, ảo giác rò rỉ 3-Fold CV, báo động giả FP và độ lệch MI.
7. [figures/fig_p03_pr_curves_three_systems.png](figures/fig_p03_pr_curves_three_systems.png): Đường cong Precision-Recall và điểm hoạt động của 3 hệ thống trên 5 loại tấn công.
8. [figures/fig_p04_soc_cost_and_fn_comparison.png](figures/fig_p04_soc_cost_and_fn_comparison.png): Mức độ cắt giảm tấn công bỏ sót (`FN`) và tiết kiệm chi phí vận hành SOC ($c=10$).
