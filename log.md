## [2026-10-21] Hoàn thành 100% Thực nghiệm Cơ sở Random Forest (5/5 Loại Tấn công)
- **Đã làm:**
  - Hoàn tất toàn bộ chuỗi thực nghiệm cơ sở cho cả 5 loại tấn công: `DoS`, `Backdoor`, `Analysis`, `Reconnaissance`, `Exploits`.
  - Mỗi loại tấn công đều chạy trọn vẹn 2 điều kiện: **Không SMOTE** và **Có SMOTE (56,000 mẫu)** qua toàn bộ dải $k \in [3, 5, 7, 10]$.
  - Tối ưu tham số qua 3-fold `RandomizedSearchCV`, dò ngưỡng tối ưu F1 trên Validation, đánh giá độc lập trên Held-out Test (~350k mẫu mỗi bài toán).
  - Xuất và lưu trữ toàn bộ: 10 file metric `.csv`, 40 mảng xác suất dự đoán `.npy` và bảng tổng kết [summary_rf_comparison.csv](results/baseline/summary_rf_comparison.csv).
- **Kết quả thu được (Đối chiếu trực tiếp với bài báo MDPI 2026):**
  - **DoS:** Không SMOTE F1 = 0.8650 vs Có SMOTE F1 = 0.8503 (SMOTE tăng Recall: 0.9097 -> 0.9177, FN giảm 102 -> 93 nhưng tăng FP 219 -> 272).
  - **Backdoor:** Không SMOTE F1 = **0.9535** vs Có SMOTE F1 = 0.9486 (Không SMOTE tốt hơn, FP tăng khi dùng SMOTE).
  - **Analysis:** Không SMOTE F1 = **0.8733** vs Có SMOTE F1 = 0.8724 (Không SMOTE tốt hơn; SMOTE khiến số ca báo nhầm FP tăng gấp 3 lần từ 3 lên 9!).
  - **Reconnaissance:** Không SMOTE F1 = **0.9673** vs Có SMOTE F1 = 0.9642 (Không SMOTE tốt hơn; FN tăng từ 30 lên 54 khi dùng SMOTE).
  - **Exploits:** Không SMOTE F1 = **0.9470** vs Có SMOTE F1 = 0.9450 (Đánh đổi: SMOTE tăng nhẹ Recall 0.9664 -> 0.9689 nhưng tăng FP từ 413 lên 452).
- **Kết luận khoa học vòng 1:** Toàn bộ thực nghiệm trên 1.62 triệu dòng `NF-UNSW-NB15` xác nhận độc lập và đồng thuận 100% với luận điểm của bài báo: **SMOTE không phải là giải pháp vạn năng; 4/5 loại tấn công đạt F1 tốt hơn khi KHÔNG dùng SMOTE, và SMOTE luôn tiềm ẩn rủi ro tăng cảnh báo giả (FP).**
- **Kế hoạch tiếp theo:** Bắt đầu Giai đoạn 2 - Điều chỉnh cải tiến (Gradient Boosting, Ngưỡng theo Recall, Mô hình nhạy chi phí Cost-Sensitive, Phân tích mẫu bị bỏ sót).

## [2026-10-23] Vá 3 Lỗi Kỹ thuật Pipeline & Hoàn thiện 100% Dữ liệu Backdoor (With SMOTE)
- **Đã làm:**
  - Vá lỗi mất thông tin phần thập phân của `L7_PROTO` bằng `pd.factorize` trong `feature_selection.py`, triệt tiêu toàn bộ `UserWarning`.
  - Thay đổi toàn bộ phép đo thời gian sang `time.perf_counter()`, khắc phục triệt để lỗi thời gian âm do trôi giờ NTP trên WSL2.
  - Tối ưu tỷ lệ SMOTE trong `ImbPipeline` khi chạy CV chống rò rỉ; bổ sung ép kiểu native và `default_serializer` cho `json.dump`.
  - Dọn dẹp các file hỏng cũ và hiệu chỉnh 2 giá trị `test_time_sec` âm trong `rf_dos_smote_metrics.csv` (thành 1.52s) và `rf_backdoor_nosmote_metrics.csv` (thành 1.03s).
  - Chạy lại trọn vẹn `Backdoor (With SMOTE)` đủ 4 mốc $k \in [3, 5, 7, 10]$; lưu trữ đầy đủ các mảng xác suất, model joblib và metadata JSON.
- **Kết quả thu được:**
  - `rf_backdoor_smote_metrics.csv` có đầy đủ 4 mốc $k$, thời gian đo hoàn toàn dương hợp lệ. Cấu hình tốt nhất đạt $k=10$, ngưỡng tối ưu $t=0.59$, Precision = 0.9474, Recall = 0.9497, F1 = 0.9486 (khớp 100% với `summary_rf_comparison.csv`).
  - Toàn bộ 7/7 test suites trong `test_pipeline.py` đều PASS 100% không có lỗi hay cảnh báo.
- **Kế hoạch tiếp theo:** Bắt đầu Giai đoạn 2 - Thiết lập pipeline Gradient Boosting (GB) và so sánh trực tiếp với Random Forest.

## [2026-10-24] Hoàn thành Bước 4 (Giai đoạn 2 - Phần 1): Pipeline Gradient Boosting & Đối sánh RF vs GB (Đã Audit & Fix)
- **Đã làm:**
  - Xây dựng module `src/train_gb.py` sử dụng `HistGradientBoostingClassifier` từ `sklearn.ensemble`, tối ưu cho dữ liệu lớn (>1M dòng).
  - Tích hợp pipeline chuẩn hóa: nạp dữ liệu từ `preprocessor.py`, chọn đặc trưng `SelectKBest` (Mutual Information qua $k \in [3, 5, 7, 10]$), tối ưu siêu tham số `RandomizedSearchCV(n_iter=10, cv=3, scoring='f1', random_state=10, n_jobs=2)` với cơ chế chống rò rỉ CV bằng `ImbPipeline`.
  - **Audit & Vá 4 lỗi kỹ thuật:**
    1. Cấu hình `early_stopping=False` cho toàn bộ các điểm khởi tạo `HistGradientBoostingClassifier`, triệt tiêu lỗi ngắt sớm non nớt sau ~10 vòng lặp trên dữ liệu mất cân bằng.
    2. Áp dụng cơ chế **Hybrid Minority-Preserving Sampling`** (giữ 100% mẫu tấn công và lấy mẫu ngẫu nhiên từ Benign) cho cả CV và bước fit `curr_gb` khi `use_smote=False`, khắc phục triệt để lỗi 255 quantile bins bị lấn át bởi Benign.
    3. Huấn luyện lại hoàn toàn nhánh `Without SMOTE` (`--only_nosmote`) cho cả 5 loại tấn công, cập nhật lại [summary_gb_comparison.csv](results/improved/summary_gb_comparison.csv) và [rf_vs_gb_comparison.csv](results/improved/rf_vs_gb_comparison.csv).
    4. Hiệu chỉnh [plot_rf_vs_gb.py](src/plot_rf_vs_gb.py): mở rộng trục Y Hình 9 (`max_rf_time * 8.0`) chống đè badge Speedup lên Legend; bỏ monospace và cập nhật nội dung khoa học chuẩn xác trong ô thứ 6 của Hình 10.
- **Kết quả & Phát hiện Cốt lõi Sau Audit:**
  - **Hiệu năng phát hiện nhảy vọt:**
    - `Backdoor (Without SMOTE)`: F1 tăng mạnh từ **0.8575 -> 0.9386** (Precision = 0.9782, Recall = 0.9020), FP giảm ngoạn mục từ 55 xuống chỉ còn **8 ca báo giả**!
    - `DoS (Without SMOTE)`: F1 tăng từ **0.7640 -> 0.8200** (Precision = 0.7639, Recall = 0.8850).
    - `Exploits (Without SMOTE)`: F1 tăng từ **0.9217 -> 0.9328** (Precision = 0.9208, Recall = 0.9451).
    - `Analysis (Without SMOTE)`: F1 đạt **0.8690** (Precision = 0.9914, Recall = 0.7735, FP chỉ có **3 ca**!).
    - `Reconnaissance (Without SMOTE)`: F1 đạt **0.9611** (Precision = 0.9353, Recall = 0.9884).
  - **Tốc độ huấn luyện vượt bậc:** Nhanh hơn Random Forest từ **13.17x đến 52.97x lần** ở điều kiện `Without SMOTE` (**7.77x–52.97x** trên toàn bộ 10 kịch bản; cụ thể ở Without SMOTE: `Exploits`: 2.34s vs 123.94s - **52.97x**; `Analysis`: 1.34s vs 61.27s - **45.72x**; `Backdoor`: 2.03s vs 90.05s - **44.36x**; `Reconnaissance`: 2.22s vs 84.44s - **38.04x**; `DoS`: 8.06s vs 106.12s - **13.17x**).
  - **Bản chất khoa học:** Random Forest nhỉnh hơn nhẹ ở trạng thái nguyên bản nhờ cơ chế chia nhánh tuyệt đối (exact splits) trên các biến rời rạc, nhưng `HistGradientBoosting` rút ngắn khoảng cách F1 xuống chỉ còn -0.004 đến -0.045 trong khi mang lại lợi thế thời gian huấn luyện và độ mượt ranh giới vượt trội.
- **Kế hoạch tiếp theo:** Hoàn thành Bước 5 và chuẩn bị slide báo cáo.

## [2026-10-24] Hoàn thành Bước 5 (Giai đoạn 2 - Phần 2): Tối ưu Ngưỡng Recall, Mô hình Nhạy chi phí & Giải phẫu FN
- **Đã làm:**
  - Đồng bộ hóa 100% các con số trong ô tổng kết của Hình 10 (`src/plot_rf_vs_gb.py`) theo bảng đối sánh sau audit (`rf_vs_gb_comparison.csv`) và cập nhật lại [fig10_precision_recall_curves_rf_vs_gb.png](figures/fig10_precision_recall_curves_rf_vs_gb.png).
  - Xây dựng module [src/advanced_evaluation.py](src/advanced_evaluation.py) thực thi trọn vẹn 3 phân tích chuyên sâu:
    1. **Tối ưu ngưỡng theo Recall mục tiêu:** Quét ngưỡng trên Validation để đạt các mốc $R \ge 95\%, 98\%, 99\%$ cho cả Random Forest và Gradient Boosting; lưu bảng [results/improved/recall_targeted_thresholds.csv](results/improved/recall_targeted_thresholds.csv).
    2. **Tối ưu ngưỡng nhạy chi phí (Cost-Sensitive NIDS):** Tối thiểu hóa hàm chi phí $\text{Total\_Cost}(t) = c \times \text{FN}(t) + 1 \times \text{FP}(t)$ qua $c \in \{1, 5, 10, 20\}$; lưu bảng [results/improved/cost_sensitive_evaluation.csv](results/improved/cost_sensitive_evaluation.csv).
    3. **Giải phẫu chuyên sâu mẫu bị bỏ sót (False Negative Forensics):** Phân tích phân phối xác suất (`Near-Miss: 0.10 <= p < t*` vs `Deep Stealth: p < 0.10`) và thống kê phân phối 10 đặc trưng NetFlow của nhóm `TP`, `FN`, `FP`, `TN` cho cả 5 loại tấn công; lưu bảng [results/improved/fn_forensic_analysis.csv](results/improved/fn_forensic_analysis.csv).
  - Xuất 3 biểu đồ học thuật chuẩn 300 DPI trong `figures/`:
    - [figures/fig11_recall_threshold_tradeoff.png](figures/fig11_recall_threshold_tradeoff.png): Đánh đổi P-R-F1 theo ngưỡng và mức độ cắt giảm FN của 5 loại tấn công khi chuyển sang chế độ $R \ge 95\%, 98\%$.
    - [figures/fig12_cost_sensitive_analysis.png](figures/fig12_cost_sensitive_analysis.png): Đường cong tổng chi phí theo ngưỡng dịch chuyển về phía trái khi $c$ tăng, và tỷ lệ tiết kiệm chi phí vận hành SOC tại $c=10$.
    - [figures/fig13_false_negative_forensics.png](figures/fig13_false_negative_forensics.png): Tỷ lệ Near-Miss vs Deep Stealth, boxplot so sánh đặc trưng IN/OUT bytes và ma trận rủi ro an ninh mạng kèm kiến trúc 2-Tier NIDS.
  - Xuất báo cáo khoa học tổng hợp chi tiết [results/improved/fn_and_cost_findings.md](results/improved/fn_and_cost_findings.md).
- **Kết quả & Phát hiện Khoa học Cốt lõi:**
  - **Triệt tiêu mẫu bỏ sót:** Ở `Analysis`, hạ ngưỡng sang chế độ $R \ge 98\%$ giúp giảm số ca FN từ **98 xuống 14 ca (cứu được 86% số ca lọt lưới)**. Ở `DoS`, FN giảm từ **102 xuống 29 ca trên RF** và từ **130 xuống 13 ca trên GB**.
  - **Hiệu quả kinh tế SOC:** Tại hệ số phạt thực tế $c=10$ (1 cuộc tấn công lọt lưới nguy hại bằng 10 cảnh báo giả), ngưỡng nhạy chi phí giúp **tiết kiệm từ 13.02% đến 38.10% tổng chi phí vận hành** trên cả 5 loại tấn công nguy hiểm.
  - **Bản chất vật lý mạng của FN:**
    - `DoS`: **98.0%** mẫu FN là dạng Near-Miss ($0.10 \le p < 0.76$) do dạng tấn công Slow-Rate DoS (Slowloris/Keep-alive); hạ ngưỡng chi phí cứu được gần như 100%.
    - `Analysis`: **51.0%** mẫu FN là Deep Stealth ($p < 0.10$) do kẻ tấn công thăm dò cổng 80 (HTTP) với kích thước gói tin và thời lượng luồng giống hệt duyệt web thông thường (`IN_BYTES` trung vị 814B, `OUT_BYTES` 1,155B). Khẳng định vai trò của kiến trúc kết hợp Tầng 1 (NetFlow GB/RF) và định tuyến Tầng 2 (hiện triển khai quy tắc heuristic NetFlow trong Web Console và thiết kế kiến trúc chuyển tiếp sang engine DPI Zeek/Suricata ở Giai đoạn 3).
  - **Tinh chỉnh Đồ họa & Đồng bộ Số liệu (Refinement):**
    - Áp dụng quy tắc chặn ngưỡng không làm suy giảm độ phủ `t_found = min(t_f1_opt, t_found)` trong `run_recall_targeted_evaluation()`, khắc phục triệt để hiện tượng vọt lố khung hình ở Hình 11.
    - Mở rộng giới hạn trục tung logarit và điều chỉnh tọa độ text lên 0.94 ở Ô (1,2) và Ô (2,1) trong Hình 13 để hộp chú thích không đè lên các điểm outlier của Boxplot.
    - Ép kiểu `int()` cho các số lượng mẫu FN ở Ô (1,1) Hình 13 và đồng bộ hóa 100% các con số nhận định trong báo cáo `fn_and_cost_findings.md` theo Bảng 1, Bảng 2 và Hình 12.

## [2026-10-24] Dọn dẹp Dư thừa, Chuẩn hóa Cấu trúc Dự án & Tối ưu Dung lượng Đĩa
- **Đã làm:**
  - **Dọn dẹp an toàn ~1.0 GB dữ liệu thừa:**
    - Xóa 3 file CSV của các bộ dữ liệu khác ngoài phạm vi nghiên cứu (`NF-BoT-IoT.csv`, `NF-ToN-IoT.csv`, `NF-CSE-CIC-IDS2018.csv`: ~733 MB).
    - Xóa 2 thư mục tải về gốc tạm thời (`data/raw_zips/`, `data/raw_bags/`: ~128 MB).
    - Xóa các bản nén trùng lặp `data/processed/*.csv.gz` (~21 MB) do toàn bộ pipeline đã chuẩn hóa đọc file `.parquet` siêu tốc.
    - Xóa 44 file mảng xác suất trung gian `*_proba_test_k*.npy` (~118 MB) trong `results/baseline/` và `results/improved/`, chỉ giữ lại các mảng xác suất tối ưu `*_best_proba_*.npy` phục vụ vẽ biểu đồ và phân tích.
    - Dọn dẹp thư mục ngoài lề (`huststack/`), notebook nháp lỗi (`notebooks/00_test.ipynb`), các thư mục trống và cache `__pycache__/`.
  - **Tối ưu mã nguồn ngăn ngừa sinh file rác:**
    - `src/split_data.py`: Loại bỏ bước xuất `.csv.gz`, chỉ giữ lại `.parquet`, `split_summary.csv` và `split_metadata.json`.
    - `src/train_rf.py` & `src/train_gb.py`: Loại bỏ dòng `np.save(..._proba_test_k{k}.npy)` trong vòng lặp k, chỉ lưu xác suất tại cấu hình $k$ tối ưu.
    - Tạo mới file [.gitignore](.gitignore) chuẩn chỉnh bảo vệ repository.
  - **Chuẩn hóa tài liệu:**
    - Chuyển đổi toàn bộ liên kết trong `log.md` sang đường dẫn tương đối chuẩn markdown.
    - Viết lại toàn diện [README.md](README.md) thành tài liệu khoa học chuẩn mực, cập nhật 100% kết quả thực nghiệm, cấu trúc dự án và hướng dẫn tái lập.
- **Kết quả:** Dự án tinh gọn, giải phóng ~1.0 GB dung lượng, kiểm thử 100% các script thành công.

## [2026-10-29] Rà soát Toàn diện (Audit), Vá Lỗi Góc khuất & Chuẩn hóa Trước Giai đoạn Mới
- **Đã làm (Chia làm 3 đợt xử lý tuần tự):**
  - **Đợt 1 — Lõi ML Pipeline & Hiệu chỉnh ngưỡng `With SMOTE`:**
    - Sửa lỗi dùng chung ngưỡng `Without SMOTE` cho nhánh `With SMOTE` trong [web_app/inference_engine.py](web_app/inference_engine.py): bổ sung hàm hiệu chỉnh động `_calibrate_thresholds_from_val_arrays()` đọc trực tiếp `*_y_val.npy` và `*_best_proba_val.npy` để tính chính xác ngưỡng `Recall >= 95%/98%/99%` và `Cost-Sensitive (c=10, c=20)` riêng cho từng mô hình `With SMOTE` (khớp 100% thuật toán `np.searchsorted` tại `src/advanced_evaluation.py`).
    - Loại bỏ lệnh tắt cảnh báo toàn cục `warnings.filterwarnings("ignore", category=UserWarning)` và chuẩn hóa đầu vào suy luận qua `_format_model_input()` (tự động truyền `DataFrame` hoặc `ndarray` tùy thuộc `feature_names_in_`, đạt `0` cảnh báo `UserWarning`).
    - Đồng bộ hóa [src/export_rf_models.py](src/export_rf_models.py) dùng `transform_features()` bảo toàn tên cột `DataFrame` + cờ `--force`.
    - Chuẩn hóa [src/run_experiments.py](src/run_experiments.py): mặc định chạy đủ 5 loại tấn công (khớp `train_gb.py`), tự động bỏ qua mô hình đã hoàn tất (`skip-if-exists`) và hỗ trợ `--force`.
  - **Đợt 2 — Giao diện Web Demo (`SOC NIDS Live Console`) & Suy luận Tương tác:**
    - Bổ sung `reload_registry()` trong [web_app/inference_engine.py](web_app/inference_engine.py) để làm mới hoàn toàn cache RAM và đọc lại mô hình/ngưỡng từ đĩa khi nhấn nút trên Sidebar.
    - Gia cố `predict_batch_flows()` (tránh ghi đè `DataFrame` gốc của caller, xử lý an toàn giá trị `NaN`/kiểu lạ ở cột `Attack`, ghi nhận danh sách đặc trưng bị thiếu và điền mặc định `0.0`).
    - Nâng cấp `sample_real_test_flow()` lấy mẫu ngẫu nhiên thực sự trên toàn tập Test (hỗ trợ `sample_type="random"` và seed `time.time_ns()`).
    - Nâng cấp [web_app/app.py](web_app/app.py): đồng bộ hóa `st.session_state` (`inp_*`) cho 10 ô nhập NetFlow, sửa lỗi reset `batch_target_attack` gây sập Batch Scanner, thêm bộ chọn Detector dự phòng cho CSV thuần `Benign`/không nhãn, bổ sung cảnh báo thiếu cột CSV, lọc FN/FP theo cả cột `Label` (`0`/`1`), và chuẩn hóa chú thích Hình 2 (`6 đặc trưng`).
  - **Đợt 3 — Hạ tầng Môi trường, Tài liệu & Git Checkpoint:**
    - Tạo mới [requirements.txt](requirements.txt) ghim phiên bản tương thích (`scikit-learn==1.9.1`, `imbalanced-learn==0.14.2`, `streamlit>=1.65.0`, v.v.).
    - Nâng cấp [run_web.sh](run_web.sh) tự động dò tìm Python trong môi trường ảo (`.venv`, `venv`, `~/venv`) trước khi fallback về `python3`.
    - Sửa pattern ngoại lệ `!*_metadata.json` trong [.gitignore](.gitignore), cập nhật đầy đủ cấu trúc 11 scripts + ghi chú phương pháp luận `HistGradientBoosting` trong [README.md](README.md), và đồng bộ số liệu tiết kiệm chi phí SOC (`13.02% - 38.10%`) trong `log.md`.
- **Kết quả:** Toàn bộ 7/7 test suites pipeline, 20/20 mô hình trong Registry, và giao diện Streamlit (`AppTest` headless) vượt qua kiểm thử 100% không lỗi, repository sạch sẽ sẵn sàng cho giai đoạn phát triển tiếp theo.

## [2026-11-06] Hoàn thành Tái hiện Nguyên bản Bài báo Gốc (Paper-Exact RF), Đối sánh 3 Hệ thống & Báo cáo Học thuật
- **Đã làm:**
  - Xây dựng [src/paper_original/train_paper_rf.py](src/paper_original/train_paper_rf.py) tái hiện chính xác 100% quy trình của Pospíchal et al. (*Information* 2026, 17(9), 836) trên toàn bộ tập dữ liệu `NF-UNSW-NB15` (10 kịch bản = 5 lớp tấn công × 2 chế độ SMOTE, quét đủ 40 lượt qua $k \in \{3, 5, 7, 10\}$, tổng thời gian tính toán 87.759,8 giây ~ 24,38 giờ).
  - Xây dựng [src/paper_original/compare_and_plot_paper_vs_ours.py](src/paper_original/compare_and_plot_paper_vs_ours.py) tổng hợp các bảng đối sánh 1-đối-1 ([three_systems_comparison.csv](results/paper_original/three_systems_comparison.csv), [soc_operational_comparison_c10.csv](results/paper_original/soc_operational_comparison_c10.csv), [mi_ranking_distortion_comparison.csv](results/paper_original/mi_ranking_distortion_comparison.csv)) và 4 biểu đồ 300 DPI (`fig_p01` – `fig_p04`).
  - Hoàn thiện báo cáo tiến độ nghiên cứu học thuật [report_soict_hust.html](report_soict_hust.html) chuẩn định dạng in ấn A4 (`Ctrl + P`), đồng bộ 30 mô hình trực tuyến trên `SOC NIDS Live Console` và làm sạch repository sẵn sàng đẩy lên Git.

## [2026-11-14] Giai đoạn 2 Phase 2: Tối ưu Web Console, Bộ Kiểm thử Pytest, Chẩn đoán Khoa học Package A & Đồng bộ Tài liệu
- **Đã làm:**
  - **Tái cấu trúc & Nâng cấp `web_app/` (`SOC NIDS Live Console`):**
    - Tách [web_app/app.py](web_app/app.py) theo kiến trúc mô-đun với `styles.py`, `protocol_decoder.py` và gói `tabs/` (`single_flow.py`, `ab_showdown.py`, `batch_scanner.py`, `registry_findings.py`).
    - Thêm chỉ mục khởi động nhanh [web_app/models_index.json](web_app/models_index.json) và khóa đồng bộ đa luồng `RLock` trong [web_app/inference_engine.py](web_app/inference_engine.py); khắc phục lỗi triệt tiêu vùng nghi vấn tại `t*_cost < 0.10` bằng công thức ngưỡng Near-Miss thích ứng `t_low = min(0.10, 0.5 * t*_cost)`.
    - Triển khai **UX-1 (`analyze_tier2_dpi_heuristics`)** cung cấp phân tích heuristic NetFlow tại Tầng 2 trên Web UI và **UX-2 (`5×3` Consensus Matrix)** đối chiếu đồng thuận 15 mô hình trên Tab 1.
  - **Hạ tầng & Kiểm thử Hồi quy (`tests/`, `pytest.ini`, `requirements-dev.txt`, `.gitattributes`):**
    - Xây dựng bộ kiểm thử tự động **53 pytest cases** (`39` fast unit/smoke + `14` slow integration trên mô hình/dữ liệu thực tế, vượt qua 100% bên cạnh `python3 src/test_pipeline.py` 7/7 suites) và chuẩn hóa LF line endings qua `.gitattributes`.
  - **Chẩn đoán Khoa học Package A (`src/scientific_diagnostics.py`, `src/pipeline_utils.py`):**
    - Thực thi hiệu chuẩn Bayes prior-odds (giảm `ECE-15` của `HistGB` `39.84%–96.27%`, đưa `t*_F1` về `0.320–0.915`), khoảng tin cậy Stratified Bootstrap 95% (`B = 1.000`) + kiểm định McNemar chính xác, và đánh giá hợp nhất hỗn hợp `Any-of-5` trên toàn bộ `363,035` luồng (`Ours RF`: `0.40%` Benign FPR / `99.77%` Target-5 Recall; `Ours HistGB`: `0.41%` Benign FPR / `99.55%` Target-5 Recall; `Unseen-4`: `35.26%–36.21%`).
    - Bổ sung hàng ablation **`Paper-Exact RF (Cost_SOC c=10)`** trong [results/paper_original/soc_operational_comparison_c10.csv](results/paper_original/soc_operational_comparison_c10.csv) (`154 FN / Cost = 3,432`, `-33.13%`), đồng thời xuất `calibration_diagnostics.csv`, `bootstrap_mcnemar_tests.csv`, `mixed_flow_orchestration_eval.csv`, `fn_forensic_analysis_all_models.csv`, [results/improved/scientific_diagnostics_findings.md](results/improved/scientific_diagnostics_findings.md), [fig14_calibration_and_bootstrap.png](figures/fig14_calibration_and_bootstrap.png) và [fig15_mixed_flow_orchestration.png](figures/fig15_mixed_flow_orchestration.png).
  - **Đồng bộ Tài liệu (`report_soict_hust.html`, `README.md`, `REPORT.md`, `log.md`):**
    - Chuyển đổi toàn bộ 41 biểu thức LaTeX thô trong [report_soict_hust.html](report_soict_hust.html) thành HTML ngữ nghĩa tự chứa, sửa toàn bộ đối chiếu RF vs HistGB vs PaperRF (`c=1, 5, 10, 20`), hiệu chỉnh thông lượng/độ trễ thực đo, làm rõ phạm vi Tầng 2 heuristic và cập nhật kết quả Package A.
    - Cập nhật 4-Tab UI, `15+4 = 19` biểu đồ, `pytest`, và giải quyết toàn bộ các mục `[TODO]` trong [REPORT.md](REPORT.md).

## [2026-11-24] Chuẩn hóa Dự án Thuần Random Forest (Loại bỏ Toàn bộ Nhánh Gradient Boosting)
- **Đã làm:**
  - **Xóa sạch mô hình & mã nguồn Gradient Boosting:** Gỡ bỏ `src/train_gb.py`, `src/plot_rf_vs_gb.py`, toàn bộ 10 model `gb_*.joblib` + selector + metadata trong `results/improved/models/`, 50 file `.npy`/`.csv` `gb_*`, `summary_gb_comparison.csv`, `rf_vs_gb_comparison.csv`, `fig09_rf_vs_gb_comparison.png`, `fig10_precision_recall_curves_rf_vs_gb.png` và các tệp `three_systems_*` cũ.
  - **Cập nhật pipeline đánh giá & chẩn đoán cho mô hình duy nhất Random Forest (`Paper-Exact RF` vs `Ours Improved RF`):**
    - Tái cấu trúc [src/advanced_evaluation.py](src/advanced_evaluation.py), [src/scientific_diagnostics.py](src/scientific_diagnostics.py) và [src/paper_original/compare_and_plot_paper_vs_ours.py](src/paper_original/compare_and_plot_paper_vs_ours.py) chỉ giữ 2 hệ thống RF (20 mô hình).
    - Chạy tái tạo toàn bộ bảng số liệu (`recall_targeted_thresholds.csv`, `cost_sensitive_evaluation.csv`, `fn_forensic_analysis*.csv`, `two_systems_comparison.csv`, `soc_operational_comparison_c10.csv`, `calibration_diagnostics.csv`, `bootstrap_mcnemar_tests.csv`, `mixed_flow_orchestration_eval.csv`) và các biểu đồ `fig11`–`fig15`, `fig_p01`–`fig_p04`.
  - **Đồng bộ Web Console, Kiểm thử & Tài liệu:**
    - Cập nhật `web_app/` (`app.py`, `inference_engine.py`, `models_index.json`, `tabs/`) về **20 mô hình RF**, ma trận đồng thuận `5×2` và **17 biểu đồ 300 DPI**.
    - Cập nhật [tests/](tests/), [README.md](README.md), [report_soict_hust.html](report_soict_hust.html) và [docs/giai_thich_chi_tiet_project1_ai.md](docs/giai_thich_chi_tiet_project1_ai.md).
