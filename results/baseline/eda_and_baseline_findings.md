# Báo cáo Tổng hợp Phân tích Dữ liệu (EDA) & Thực nghiệm Cơ sở Random Forest

Dự án Nghiên cứu & Tái lập Hệ thống NIDS trên tập dữ liệu **NF-UNSW-NB15** (1.62M bản ghi)
*Bài báo tham chiếu: "Random Forest-Based Network Intrusion Detection with Feature Selection and Class Balancing on UNSW-NB15 Traffic" (MDPI Information 2026).*

---

## 6 Phát hiện Khoa học Cốt lõi từ Dữ liệu & Thực nghiệm Baseline

### Phát hiện 1: Đặc tính Mất cân bằng Lớp Cực đoan (Extreme Class Imbalance)
- **Số liệu thực tế:** Trong tổng số **1,104,497** mẫu của tập huấn luyện (`Train Set`), lưu lượng bình thường `Benign` chiếm áp đảo tuyệt đối **1,055,226 mẫu (95.54%)**, trong khi toàn bộ 9 loại tấn công gộp lại chỉ chiếm **49,271 mẫu (4.46%)** (tỷ lệ mất cân bằng chung **1 : 21.4**).
- **Phân hóa nội bộ các lớp tấn công:** Khi phân tách nhị phân theo phương pháp bài báo, tỷ lệ mất cân bằng tăng vọt lên mức cực đoan:
  - `Exploits`: 16,832 mẫu $\rightarrow$ **1 : 63**
  - `Reconnaissance`: 8,364 mẫu $\rightarrow$ **1 : 126**
  - `DoS`: 3,437 mẫu $\rightarrow$ **1 : 307**
  - `Analysis`: 1,358 mẫu $\rightarrow$ **1 : 777**
  - `Backdoor`: 1,213 mẫu $\rightarrow$ **1 : 870**
- **Ý nghĩa kỹ thuật:** Tỷ lệ mất cân bằng lên tới hàng trăm lần là nguyên nhân chính khiến chỉ số *Accuracy* trở nên vô nghĩa (mô hình đoán bừa 100% Benign vẫn đạt >99.8% Accuracy). Mọi tối ưu bắt buộc phải căn cứ theo **F1-Score, Precision, Recall và Ma trận Nhầm lẫn**.

---

### Phát hiện 2: Tính Đa cộng tuyến & Cặp Đặc trưng Tương quan Cực cao
- **Số liệu thực tế:** Ma trận tương quan Pearson trên 10 đặc trưng lưu lượng mạng (`FEATURE_COLS`) phát hiện cặp biến đồng biến gần như hoàn hảo:
  - **`OUT_BYTES`** và **`OUT_PKTS`**: Hệ số tương quan $r = \mathbf{0.9725}$.
  - Ngoài ra, cặp `IN_PKTS` và `OUT_PKTS` đạt $r = 0.7444$, cặp `IN_BYTES` và `IN_PKTS` đạt $r = 0.6985$.
- **Ý nghĩa kỹ thuật:** Trong thuật toán Random Forest, việc đưa cả hai biến tương quan > 0.95 vào mô hình không làm giảm chất lượng dự đoán (do cây phân nhánh ngẫu nhiên `max_features`), nhưng làm **phân tán điểm số quan trọng (Feature Importance)** và tăng chi phí tính toán. Thuật toán `SelectKBest` với Mutual Information giúp tinh lọc và loại bỏ sự dư thừa này ở các mức $k=3, 5, 7$.

---

### Phát hiện 3: "Dấu vân tay Mạng" Phân hóa Sâu sắc qua Điểm Mutual Information
Bảng tính điểm Mutual Information (không dùng SMOTE) chỉ ra mỗi loại tấn công mang một cấu trúc hành vi mạng hoàn toàn riêng biệt:
- **`DoS`**: Điểm MI cao nhất tập trung ở `IN_BYTES` (0.0119), `OUT_BYTES` (0.0099) và `L4_DST_PORT` (0.0088) (đặc trưng ngập lụt băng thông và tấn công cổng đích).
- **`Backdoor`**: Điểm MI cao nhất thuộc về `IN_BYTES` (0.0067), `PROTOCOL` (0.0065) và `L4_DST_PORT` (0.0062) (hành vi kết nối ngược độc hại qua cổng chuyên biệt và giao thức đặc thù).
- **`Analysis`**: Phụ thuộc mạnh vào `IN_BYTES` (0.0084), `PROTOCOL` (0.0074) và `L4_DST_PORT` (0.0070).
- **`Reconnaissance`**: Tập trung vào `IN_BYTES` (0.0388), `OUT_BYTES` (0.0310) (hành vi quét cổng diện rộng).
- **`Exploits`**: Phân tán đều trên các thuộc tính gói tin và luồng dữ liệu `IN_BYTES` (0.0517), `OUT_BYTES` (0.0444).

---

### Phát hiện 4: Tính Nhất quán Tuyệt đối của Phân chia Dữ liệu (Zero Data Drift)
- Quá trình phân tầng có kiểm soát (`StratifiedShuffleSplit`, `random_state=10`) bảo toàn tỷ lệ từng loại tấn công qua cả 3 tập độc lập:
  - `Train Set` (1,104,497 mẫu - **68.05%**)
  - `Validation Set` (155,586 mẫu - **9.59%**)
  - `Held-out Test Set` (363,035 mẫu - **22.37%**)
- Đường phân phối tích lũy thực nghiệm (Empirical CDF) của các đặc trưng `IN_BYTES`, `OUT_BYTES`, `FLOW_DURATION_MILLISECONDS` trên cả 3 tập trùng khớp 100%, chứng minh **không hề có hiện tượng trôi dữ liệu (Data Drift) hay rò rỉ thông tin (Data Leakage)** giữa các tập đánh giá.

---

### Phát hiện 5: "Nghịch lý SMOTE" (SMOTE Paradox) trong Không gian NetFlow
- **Thực tế thực nghiệm:** Dữ liệu đối chiếu trực tiếp giữa `Without SMOTE` và `With SMOTE (56,000)` khẳng định: **4/5 loại tấn công đạt F1-Score cao hơn khi KHÔNG dùng SMOTE!**
  - `Backdoor`: F1 không SMOTE = **0.9535** vs có SMOTE = **0.9486** (FP tăng từ 18 lên 21).
  - `Analysis`: F1 không SMOTE = **0.8733** vs có SMOTE = **0.8724** (FP tăng gấp 3 lần từ 3 lên 9!).
  - `Reconnaissance`: F1 không SMOTE = **0.9673** vs có SMOTE = **0.9642** (FN tăng từ 30 lên 54).
  - `Exploits`: F1 không SMOTE = **0.9470** vs có SMOTE = **0.9450** (FP tăng từ 413 lên 452).
  - `DoS`: SMOTE giúp tăng nhẹ Recall (0.9097 $\rightarrow$ 0.9177, FN giảm 102 $\rightarrow$ 93), nhưng làm F1 giảm (0.8650 $\rightarrow$ 0.8503) do FP tăng mạnh từ 219 lên 272.
- **Bản chất khoa học:** Kỹ thuật nội suy lân cận $k$-NN của SMOTE hoạt động kém trên không gian đặc trưng mạng chứa nhiều biến rời rạc (`discrete`) và phân cụm hẹp. SMOTE vô tình sinh ra các điểm dữ liệu nhân tạo nằm xâm lấn vào vùng ranh giới quyết định của `Benign`, dẫn đến **sự bùng nổ cảnh báo giả (False Positive Alarm Explosion)**.

---

### Phát hiện 6: Bài toán Đánh đổi Sinh tử trong Vận hành SOC (Table 8 Replication)
- Trong Trung tâm Điều hành An ninh Mạng (SOC), cảnh báo giả (FP) làm quá tải chuyên viên ("alert fatigue"), nhưng cảnh báo bỏ sót (FN) lại trực tiếp dẫn tới thảm họa an ninh khi mã độc xâm nhập trót lọt.
- Phân tích Table 8 trực quan hóa chỉ rõ:
  1. Với `Analysis`: Mô hình không SMOTE có độ chính xác cực cao (**99.15% Precision**, chỉ 3 ca báo nhầm trên 346k mẫu Benign), nhưng lại **bỏ sót tới 98 cuộc tấn công** (Recall chỉ đạt 78.03%).
  2. Với `DoS`: Muốn giảm 9 ca bỏ sót (từ 102 xuống 93), hệ thống phải trả giá bằng **53 ca báo giả tăng thêm** (219 lên 272).
  3. Với `Backdoor`: Cả hai cấu hình đều đạt khả năng phát hiện vượt trội (>95% TP), nhưng Không SMOTE vẫn giữ mức cảnh báo giả thấp nhất (18 FP).
- **Kết luận định hướng Giai đoạn 2:** Không thể giải quyết bài toán bỏ sót FN bằng việc lạm dụng SMOTE. Cần chuyển hướng sang:
  1. Mô hình Gradient Boosting (`HistGradientBoosting` / `XGBoost` / `LightGBM`) với khả năng học ranh giới phi tuyến tốt hơn.
  2. Kỹ thuật **Dò ngưỡng theo Recall mục tiêu** ($R \ge 0.95, 0.98, 0.99$).
  3. **Mô hình nhạy chi phí (Cost-Sensitive Learning)** gán trọng số thiệt hại $c \times FN$ cao hơn $FP$.

---
*Báo cáo được tự động khởi tạo bởi script `src/generate_eda_and_baseline_plots.py`.*
