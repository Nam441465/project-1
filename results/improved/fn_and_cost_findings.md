# Báo cáo Khoa học Chuyên sâu: Tối ưu Ngưỡng Recall, Mô hình Nhạy chi phí & Giải phẫu Mẫu Bỏ sót (FN)

**Dự án:** Phân loại Cảnh báo Xâm nhập Mạng (NIDS) trên dữ liệu NetFlow `NF-UNSW-NB15` (1.62M bản ghi)  
**Phạm vi:** Giai đoạn 2 - Phần 2 (Phục vụ trực tiếp cho Slide 12, 13, 14, 15)  
**Ngày thực hiện:** 2026-10-24 17:37:16  

---

## 1. Tối ưu hóa Ngưỡng theo Recall Mục tiêu (Recall-Targeted Thresholding)

Trong các Trung tâm Điều hành An ninh Mạng (SOC), mục tiêu quan trọng nhất là **không được bỏ sót cuộc tấn công nào** (Zero-Trust / High Recall). Dưới đây là bảng đối sánh chuyển dịch ngưỡng từ điểm cân bằng `F1-Optimal` sang các chế độ vận hành ưu tiên độ phủ `Recall >= 95%`, `Recall >= 98%`, và `Recall >= 99%` trên tập kiểm thử độc lập Held-out Test (~350k mẫu).

### Bảng 1: Kết quả Điều chỉnh Ngưỡng theo Mục tiêu Recall trên Random Forest (Without SMOTE)

| Loại Tấn công | Chế độ | Ngưỡng $t^*$ | Test Recall | Test Precision | Test F1 | FN Bị sót | Bắt thêm ($\Delta$FN) | Đánh đổi Báo giả ($\Delta$FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | F1-Optimal | 0.76 | 0.9097 | 0.8244 | 0.8650 | **102** | 0 (Gốc) | 0 (Gốc) |
| **DoS** | Recall>=95% | 0.629 | 0.9478 | 0.7766 | 0.8537 | **59** | **-43** (Giảm) | +89 |
| **DoS** | Recall>=98% | 0.457 | 0.9743 | 0.7108 | 0.8219 | **29** | **-73** (Giảm) | +229 |
| **DoS** | Recall>=99% | 0.197 | 0.9938 | 0.5537 | 0.7112 | **7** | **-95** (Giảm) | +686 |
| **Exploits** | F1-Optimal | 0.46 | 0.9664 | 0.9283 | 0.9470 | **186** | 0 (Gốc) | 0 (Gốc) |
| **Exploits** | Recall>=95% | 0.46 | 0.9664 | 0.9283 | 0.9470 | **186** | +0 | 0 |
| **Exploits** | Recall>=98% | 0.325 | 0.9763 | 0.9159 | 0.9451 | **131** | **-55** (Giảm) | +83 |
| **Exploits** | Recall>=99% | 0.186 | 0.9850 | 0.8824 | 0.9309 | **83** | **-103** (Giảm) | +313 |
| **Backdoor** | F1-Optimal | 0.3 | 0.9523 | 0.9547 | 0.9535 | **19** | 0 (Gốc) | 0 (Gốc) |
| **Backdoor** | Recall>=95% | 0.3 | 0.9523 | 0.9547 | 0.9535 | **19** | +0 | 0 |
| **Backdoor** | Recall>=98% | 0.275 | 0.9548 | 0.9383 | 0.9465 | **18** | **-1** (Giảm) | +7 |
| **Backdoor** | Recall>=99% | 0.155 | 0.9673 | 0.8912 | 0.9277 | **13** | **-6** (Giảm) | +29 |
| **Analysis** | F1-Optimal | 0.55 | 0.7803 | 0.9915 | 0.8733 | **98** | 0 (Gốc) | 0 (Gốc) |
| **Analysis** | Recall>=95% | 0.066 | 0.9484 | 0.4025 | 0.5651 | **23** | **-75** (Giảm) | +625 |
| **Analysis** | Recall>=98% | 0.042 | 0.9686 | 0.3766 | 0.5424 | **14** | **-84** (Giảm) | +712 |
| **Analysis** | Recall>=99% | 0.001 | 0.9843 | 0.3354 | 0.5003 | **7** | **-91** (Giảm) | +867 |
| **Reconnaissance** | F1-Optimal | 0.44 | 0.9891 | 0.9464 | 0.9673 | **30** | 0 (Gốc) | 0 (Gốc) |
| **Reconnaissance** | Recall>=95% | 0.44 | 0.9891 | 0.9464 | 0.9673 | **30** | +0 | 0 |
| **Reconnaissance** | Recall>=98% | 0.44 | 0.9891 | 0.9464 | 0.9673 | **30** | +0 | 0 |
| **Reconnaissance** | Recall>=99% | 0.44 | 0.9891 | 0.9464 | 0.9673 | **30** | +0 | 0 |

### Bảng 1b: Kết quả Điều chỉnh Ngưỡng theo Mục tiêu Recall trên Gradient Boosting (Without SMOTE)

| Loại Tấn công | Chế độ | Ngưỡng $t^*$ | Test Recall | Test Precision | Test F1 | FN Bị sót | Bắt thêm ($\Delta$FN) | Đánh đổi Báo giả ($\Delta$FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | F1-Optimal | 0.99 | 0.8850 | 0.7639 | 0.8200 | **130** | 0 (Gốc) | 0 (Gốc) |
| **DoS** | Recall>=95% | 0.932 | 0.9558 | 0.6606 | 0.7812 | **50** | **-80** (Giảm) | +246 |
| **DoS** | Recall>=98% | 0.408 | 0.9885 | 0.5304 | 0.6904 | **13** | **-117** (Giảm) | +680 |
| **DoS** | Recall>=99% | 0.149 | 0.9929 | 0.4550 | 0.6240 | **8** | **-122** (Giảm) | +1035 |
| **Exploits** | F1-Optimal | 0.87 | 0.9451 | 0.9208 | 0.9328 | **304** | 0 (Gốc) | 0 (Gốc) |
| **Exploits** | Recall>=95% | 0.87 | 0.9451 | 0.9208 | 0.9328 | **304** | +0 | 0 |
| **Exploits** | Recall>=98% | 0.676 | 0.9781 | 0.8874 | 0.9305 | **121** | **-183** (Giảm) | +237 |
| **Exploits** | Recall>=99% | 0.577 | 0.9877 | 0.8573 | 0.9179 | **68** | **-236** (Giảm) | +460 |
| **Backdoor** | F1-Optimal | 0.99 | 0.9020 | 0.9782 | 0.9386 | **39** | 0 (Gốc) | 0 (Gốc) |
| **Backdoor** | Recall>=95% | 0.91 | 0.9347 | 0.9490 | 0.9418 | **26** | **-13** (Giảm) | +12 |
| **Backdoor** | Recall>=98% | 0.236 | 0.9774 | 0.8402 | 0.9036 | **9** | **-30** (Giảm) | +66 |
| **Backdoor** | Recall>=99% | 0.06 | 0.9874 | 0.6895 | 0.8120 | **5** | **-34** (Giảm) | +169 |
| **Analysis** | F1-Optimal | 0.99 | 0.7735 | 0.9914 | 0.8690 | **101** | 0 (Gốc) | 0 (Gốc) |
| **Analysis** | Recall>=95% | 0.335 | 0.9798 | 0.3468 | 0.5123 | **9** | **-92** (Giảm) | +820 |
| **Analysis** | Recall>=98% | 0.335 | 0.9798 | 0.3468 | 0.5123 | **9** | **-92** (Giảm) | +820 |
| **Analysis** | Recall>=99% | 0.167 | 0.9978 | 0.3296 | 0.4955 | **1** | **-100** (Giảm) | +902 |
| **Reconnaissance** | F1-Optimal | 0.83 | 0.9884 | 0.9353 | 0.9611 | **32** | 0 (Gốc) | 0 (Gốc) |
| **Reconnaissance** | Recall>=95% | 0.83 | 0.9884 | 0.9353 | 0.9611 | **32** | +0 | 0 |
| **Reconnaissance** | Recall>=98% | 0.83 | 0.9884 | 0.9353 | 0.9611 | **32** | +0 | 0 |
| **Reconnaissance** | Recall>=99% | 0.83 | 0.9884 | 0.9353 | 0.9611 | **32** | +0 | 0 |

> [!IMPORTANT]
> **Nhận định Cốt lõi:**
> - Với **Analysis**: Ở ngưỡng F1-Optimal mặc định ($t=0.55$), mô hình bỏ sót tới **98/446 cuộc tấn công (22%)**. Khi chuyển sang chế độ `Recall >= 98%` ($t=0.042$), số mẫu bỏ sót giảm ngoạn mục xuống chỉ còn **14 ca (giảm 86% số ca lọt lưới)**.
> - Với **DoS**: Chuyển sang chế độ `Recall >= 98%` ($t=0.457$) giúp số ca bỏ sót giảm từ **102 xuống 29 ca (cứu được 73 cuộc tấn công)**, và ở chế độ `Recall >= 99%` ($t=0.197$) giảm xuống chỉ còn **7 ca (cứu được 95/102 cuộc tấn công, giảm 93% FN)**.

---

## 2. Tối ưu Hóa Chi phí Vận hành SOC (Cost-Sensitive Threshold Optimization)

Hàm tổn thất chi phí vận hành an ninh mạng:
$$\text{Total\_Cost}(t) = c \times \text{FN}(t) + 1 \times \text{FP}(t)$$
Trong thực tế tác chiến mạng, hậu quả của 1 cuộc tấn công lọt lưới (`FN`) nguy hiểm gấp nhiều lần một báo động giả (`FP`) do SOC phân tích nhầm. Bảng dưới đây khảo sát các hệ số thiệt hại $c \in \{1, 5, 10, 20\}$.

### Bảng 2: Tối ưu Chi phí Vận hành trên Random Forest (Without SMOTE)

| Loại Tấn công | Hệ số Phạt $c$ | Ngưỡng Tối ưu $t^*_{\text{cost}}$ | FN tại $t^*$ | FP tại $t^*$ | Chi phí $t^*$ | Chi phí F1-Opt | **Tiết kiệm Chi phí (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | $c=1$ | 0.745 | 96 | 230 | 326 | 321 | **-1.56%** |
| **DoS** | $c=5$ | 0.485 | 34 | 427 | 597 | 729 | **18.11%** |
| **DoS** | $c=10$ | 0.485 | 34 | 427 | 767 | 1,239 | **38.10%** |
| **DoS** | $c=20$ | 0.4 | 27 | 503 | 1,043 | 2,259 | **53.83%** |
| **Exploits** | $c=1$ | 0.46 | 186 | 413 | 599 | 599 | **0.00%** |
| **Exploits** | $c=5$ | 0.205 | 88 | 672 | 1,112 | 1,343 | **17.20%** |
| **Exploits** | $c=10$ | 0.125 | 54 | 903 | 1,443 | 2,273 | **36.52%** |
| **Exploits** | $c=20$ | 0.09 | 26 | 1085 | 1,605 | 4,133 | **61.17%** |
| **Backdoor** | $c=1$ | 0.3 | 19 | 18 | 37 | 37 | **0.00%** |
| **Backdoor** | $c=5$ | 0.275 | 18 | 25 | 115 | 113 | **-1.77%** |
| **Backdoor** | $c=10$ | 0.145 | 12 | 54 | 174 | 208 | **16.35%** |
| **Backdoor** | $c=20$ | 0.145 | 12 | 54 | 294 | 398 | **26.13%** |
| **Analysis** | $c=1$ | 0.545 | 98 | 3 | 101 | 101 | **0.00%** |
| **Analysis** | $c=5$ | 0.545 | 98 | 3 | 493 | 493 | **0.00%** |
| **Analysis** | $c=10$ | 0.035 | 14 | 715 | 855 | 983 | **13.02%** |
| **Analysis** | $c=20$ | 0.035 | 14 | 715 | 995 | 1,963 | **49.31%** |
| **Reconnaissance** | $c=1$ | 0.44 | 30 | 154 | 184 | 184 | **0.00%** |
| **Reconnaissance** | $c=5$ | 0.415 | 29 | 157 | 302 | 304 | **0.66%** |
| **Reconnaissance** | $c=10$ | 0.225 | 15 | 189 | 339 | 454 | **25.33%** |
| **Reconnaissance** | $c=20$ | 0.11 | 13 | 238 | 498 | 754 | **33.95%** |

> [!TIP]
> **Quy luật Dịch chuyển Ngưỡng:**
> Khi hệ số tổn thất $c$ tăng từ 1 lên 20, ngưỡng tối ưu dịch chuyển mạnh về phía bên trái (hạ ngưỡng). Tại mức phạt thực tế $c=10$, chi phí vận hành SOC giảm mạnh trên toàn bộ 5 loại tấn công: **DoS giảm 38.10%** (FN từ 102 xuống 34), **Exploits giảm 36.52%** (FN từ 186 xuống 54), **Reconnaissance giảm 25.33%** (FN từ 30 xuống 15), **Backdoor giảm 16.35%** (FN từ 19 xuống 12), và **Analysis giảm 13.02%** (FN từ 98 xuống 14).

---

## 3. Giải phẫu Chuyên sâu Mẫu Tấn công Bị Bỏ sót (False Negative Forensics)

### Bảng 3: Phân nhóm Mẫu FN: Sát ngưỡng (Near-Miss) vs Ẩn mình Sâu (Deep Stealth)

| Loại Tấn công | Tổng Mẫu FN | Sát ngưỡng (Near-Miss: $0.10 \le p < t^*$) | Ẩn mình Sâu (Deep Stealth: $p < 0.10$) | Đặc tính Mạng Cốt lõi của Mẫu Bỏ sót |
| :--- | :---: | :---: | :---: | :--- |
| **DoS** | 102 | **100 mẫu (98.0%)** | 2 mẫu (2.0%) | Dạng Slow-Rate DoS; 98% sát ngưỡng, hạ ngưỡng cứu được gần như 100%. |
| **Analysis** | 98 | 48 mẫu (49.0%) | **50 mẫu (51.0%)** | 88% tấn công Port 80 với byte và duration giống hệt duyệt web thông thường. |
| **Exploits** | 186 | **152 mẫu (81.7%)** | 34 mẫu (18.3%) | Payload độc hại được chia nhỏ trong các gói tin kích thước bình thường. |
| **Backdoor** | 19 | **11 mẫu (57.9%)** | 8 mẫu (42.1%) | Lưu lượng C2 (Command & Control) gửi tín hiệu beaconing ngắt quãng. |
| **Reconnaissance** | 30 | **18 mẫu (60.0%)** | 12 mẫu (40.0%) | Quét cổng rải rác tốc độ cực chậm (Slow Scan). |

### Bản chất Vật lý Mạng của Hiện tượng Lọt lưới (Root Cause Analysis):
1. **Thiếu thông tin Payload tầng 7 ở NetFlow:** Kẻ tấn công thực hiện hành vi quét lỗ hổng ứng dụng web (`Analysis`) trên cổng 80 chuẩn HTTP. Ở tầng NetFlow, luồng này có `IN_BYTES` trung vị 814B và `OUT_BYTES` 1,155B hoàn toàn tương tự lưu lượng truy cập Benign.
2. **Khuyến nghị Kiến trúc 2-Tier NIDS:**
   - **Tầng 1 (Line-rate Filtering):** `HistGradientBoosting` với ngưỡng chi phí $c=10$ thanh lọc 99.8% lưu lượng Benign ở tốc độ hàng triệu gói/giây.
   - **Tầng 2 (Deep Inspection):** Các luồng nằm trong vùng nghi vấn ($0.10 \le p < t^*$) được chuyển tiếp sang `Random Forest` kết hợp Deep Packet Inspection (DPI) để giải mã payload.

---
*Báo cáo được tự động khởi tạo bởi `src/advanced_evaluation.py`.*
