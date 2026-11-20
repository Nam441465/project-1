# Báo cáo Khoa học Chuyên sâu: Tối ưu Ngưỡng Recall, Mô hình Nhạy chi phí & Giải phẫu Mẫu Bỏ sót (FN)

**Dự án:** Phân loại Cảnh báo Xâm nhập Mạng (NIDS) trên dữ liệu NetFlow `NF-UNSW-NB15` (1.62M bản ghi)  
**Phạm vi:** Giai đoạn 2 - Phần 2 (Đối chiếu RF, HistGB và Paper-Exact RF trên Held-out Test)  
**Ngày thực hiện:** 2026-11-14 22:49:09  

---

## 1. Tối ưu hóa Ngưỡng theo Recall Mục tiêu (Recall-Targeted Thresholding)

Trong các Trung tâm Điều hành An ninh Mạng (SOC), mục tiêu quan trọng nhất là **không được bỏ sót cuộc tấn công nào** (Zero-Trust / High Recall). Dưới đây là bảng đối sánh chuyển dịch ngưỡng từ điểm cân bằng `F1-Optimal` sang các chế độ vận hành ưu tiên độ phủ `Recall >= 95%`, `Recall >= 98%`, và `Recall >= 99%` trên tập kiểm thử độc lập Held-out Test.

### Bảng 1: Kết quả Điều chỉnh Ngưỡng theo Mục tiêu Recall trên Random Forest (Without SMOTE)

| Loại Tấn công | Chế độ | Ngưỡng $t^*$ | Near-Miss $t_{low}$ | Test Recall | Test Precision | Test F1 | FN Bị sót | Bắt thêm ($\Delta$FN) | Đánh đổi Báo giả ($\Delta$FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | F1-Optimal | 0.76 | 0.1 | 0.9097 | 0.8244 | 0.8650 | **102** | 0 (Gốc) | 0 (Gốc) |
| **DoS** | Recall>=95% | 0.629 | 0.1 | 0.9478 | 0.7766 | 0.8537 | **59** | **-43** (Giảm) | +89 |
| **DoS** | Recall>=98% | 0.457 | 0.1 | 0.9743 | 0.7108 | 0.8219 | **29** | **-73** (Giảm) | +229 |
| **DoS** | Recall>=99% | 0.197 | 0.0985 | 0.9938 | 0.5537 | 0.7112 | **7** | **-95** (Giảm) | +686 |
| **Exploits** | F1-Optimal | 0.46 | 0.1 | 0.9664 | 0.9283 | 0.9470 | **186** | 0 (Gốc) | 0 (Gốc) |
| **Exploits** | Recall>=95% | 0.46 | 0.1 | 0.9664 | 0.9283 | 0.9470 | **186** | +0 | 0 |
| **Exploits** | Recall>=98% | 0.325 | 0.1 | 0.9763 | 0.9159 | 0.9451 | **131** | **-55** (Giảm) | +83 |
| **Exploits** | Recall>=99% | 0.186 | 0.093 | 0.9850 | 0.8824 | 0.9309 | **83** | **-103** (Giảm) | +313 |
| **Backdoor** | F1-Optimal | 0.3 | 0.1 | 0.9523 | 0.9547 | 0.9535 | **19** | 0 (Gốc) | 0 (Gốc) |
| **Backdoor** | Recall>=95% | 0.3 | 0.1 | 0.9523 | 0.9547 | 0.9535 | **19** | +0 | 0 |
| **Backdoor** | Recall>=98% | 0.275 | 0.1 | 0.9548 | 0.9383 | 0.9465 | **18** | **-1** (Giảm) | +7 |
| **Backdoor** | Recall>=99% | 0.155 | 0.0775 | 0.9673 | 0.8912 | 0.9277 | **13** | **-6** (Giảm) | +29 |
| **Analysis** | F1-Optimal | 0.55 | 0.1 | 0.7803 | 0.9915 | 0.8733 | **98** | 0 (Gốc) | 0 (Gốc) |
| **Analysis** | Recall>=95% | 0.066 | 0.033 | 0.9484 | 0.4025 | 0.5651 | **23** | **-75** (Giảm) | +625 |
| **Analysis** | Recall>=98% | 0.042 | 0.021 | 0.9686 | 0.3766 | 0.5424 | **14** | **-84** (Giảm) | +712 |
| **Analysis** | Recall>=99% | 0.001 | 0.0005 | 0.9843 | 0.3354 | 0.5003 | **7** | **-91** (Giảm) | +867 |
| **Reconnaissance** | F1-Optimal | 0.44 | 0.1 | 0.9891 | 0.9464 | 0.9673 | **30** | 0 (Gốc) | 0 (Gốc) |
| **Reconnaissance** | Recall>=95% | 0.44 | 0.1 | 0.9891 | 0.9464 | 0.9673 | **30** | +0 | 0 |
| **Reconnaissance** | Recall>=98% | 0.44 | 0.1 | 0.9891 | 0.9464 | 0.9673 | **30** | +0 | 0 |
| **Reconnaissance** | Recall>=99% | 0.44 | 0.1 | 0.9891 | 0.9464 | 0.9673 | **30** | +0 | 0 |

### Bảng 1b: Kết quả Điều chỉnh Ngưỡng theo Mục tiêu Recall trên Gradient Boosting (Without SMOTE)

| Loại Tấn công | Chế độ | Ngưỡng $t^*$ | Near-Miss $t_{low}$ | Test Recall | Test Precision | Test F1 | FN Bị sót | Bắt thêm ($\Delta$FN) | Đánh đổi Báo giả ($\Delta$FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | F1-Optimal | 0.99 | 0.1 | 0.8850 | 0.7639 | 0.8200 | **130** | 0 (Gốc) | 0 (Gốc) |
| **DoS** | Recall>=95% | 0.932 | 0.1 | 0.9558 | 0.6606 | 0.7812 | **50** | **-80** (Giảm) | +246 |
| **DoS** | Recall>=98% | 0.408 | 0.1 | 0.9885 | 0.5304 | 0.6904 | **13** | **-117** (Giảm) | +680 |
| **DoS** | Recall>=99% | 0.149 | 0.0745 | 0.9929 | 0.4550 | 0.6240 | **8** | **-122** (Giảm) | +1035 |
| **Exploits** | F1-Optimal | 0.87 | 0.1 | 0.9451 | 0.9208 | 0.9328 | **304** | 0 (Gốc) | 0 (Gốc) |
| **Exploits** | Recall>=95% | 0.87 | 0.1 | 0.9451 | 0.9208 | 0.9328 | **304** | +0 | 0 |
| **Exploits** | Recall>=98% | 0.676 | 0.1 | 0.9781 | 0.8874 | 0.9305 | **121** | **-183** (Giảm) | +237 |
| **Exploits** | Recall>=99% | 0.577 | 0.1 | 0.9877 | 0.8573 | 0.9179 | **68** | **-236** (Giảm) | +460 |
| **Backdoor** | F1-Optimal | 0.99 | 0.1 | 0.9020 | 0.9782 | 0.9386 | **39** | 0 (Gốc) | 0 (Gốc) |
| **Backdoor** | Recall>=95% | 0.91 | 0.1 | 0.9347 | 0.9490 | 0.9418 | **26** | **-13** (Giảm) | +12 |
| **Backdoor** | Recall>=98% | 0.236 | 0.1 | 0.9774 | 0.8402 | 0.9036 | **9** | **-30** (Giảm) | +66 |
| **Backdoor** | Recall>=99% | 0.06 | 0.03 | 0.9874 | 0.6895 | 0.8120 | **5** | **-34** (Giảm) | +169 |
| **Analysis** | F1-Optimal | 0.99 | 0.1 | 0.7735 | 0.9914 | 0.8690 | **101** | 0 (Gốc) | 0 (Gốc) |
| **Analysis** | Recall>=95% | 0.335 | 0.1 | 0.9798 | 0.3468 | 0.5123 | **9** | **-92** (Giảm) | +820 |
| **Analysis** | Recall>=98% | 0.335 | 0.1 | 0.9798 | 0.3468 | 0.5123 | **9** | **-92** (Giảm) | +820 |
| **Analysis** | Recall>=99% | 0.167 | 0.0835 | 0.9978 | 0.3296 | 0.4955 | **1** | **-100** (Giảm) | +902 |
| **Reconnaissance** | F1-Optimal | 0.83 | 0.1 | 0.9884 | 0.9353 | 0.9611 | **32** | 0 (Gốc) | 0 (Gốc) |
| **Reconnaissance** | Recall>=95% | 0.83 | 0.1 | 0.9884 | 0.9353 | 0.9611 | **32** | +0 | 0 |
| **Reconnaissance** | Recall>=98% | 0.83 | 0.1 | 0.9884 | 0.9353 | 0.9611 | **32** | +0 | 0 |
| **Reconnaissance** | Recall>=99% | 0.83 | 0.1 | 0.9884 | 0.9353 | 0.9611 | **32** | +0 | 0 |

> [!IMPORTANT]
> **Nhận định Cốt lõi (Từ dữ liệu đo kiểm RF):**
> - Với **Analysis**: Ở ngưỡng F1-Optimal mặc định ($t=0.55$), mô hình bỏ sót **98/446 cuộc tấn công (22.0%)**. Khi chuyển sang chế độ `Recall >= 98%` ($t=0.042$, vùng Near-Miss thích ứng $t_{low}=0.0210$), số mẫu bỏ sót giảm xuống còn **14 ca (giảm 85.7% số ca lọt lưới)**.
> - Với **DoS**: Chuyển sang chế độ `Recall >= 98%` ($t=0.457$) giúp số ca bỏ sót giảm từ **102 xuống 29 ca (cứu được 73 cuộc tấn công)**, và ở chế độ `Recall >= 99%` ($t=0.197$) giảm xuống chỉ còn **7 ca (cứu được 95/102 ca bỏ sót, giảm 93.1% FN trên 1,130 mẫu DoS)**.

---

## 2. Tối ưu Hóa Chi phí Vận hành SOC (Cost-Sensitive Threshold Optimization)

Hàm tổn thất chi phí vận hành an ninh mạng:
$$\text{Total\_Cost}(t) = c \times \text{FN}(t) + 1 \times \text{FP}(t)$$
Trong thực tế tác chiến mạng, hậu quả của 1 cuộc tấn công lọt lưới (`FN`) nguy hiểm gấp nhiều lần một báo động giả (`FP`) do SOC phân tích nhầm. Bảng dưới đây khảo sát các hệ số thiệt hại $c \in \{1, 5, 10, 20\}$ cho cả **Random Forest** và **HistGradientBoosting**, đồng thời áp dụng cận dưới Near-Miss thích ứng $t_{low} = \min(0.10, 0.5 \cdot t^*)$ để không làm sụp đổ vùng chuyển tiếp Tầng 2 khi $t^*_{\text{cost}} < 0.10$.

### Bảng 2: Tối ưu Chi phí Vận hành trên Random Forest (Without SMOTE)

| Loại Tấn công | Hệ số Phạt $c$ | Ngưỡng Tối ưu $t^*_{\text{cost}}$ | Near-Miss $t_{low}$ | FN tại $t^*$ | FP tại $t^*$ | Chi phí $t^*$ | Chi phí F1-Opt | **Tiết kiệm Chi phí (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | $c=1$ | 0.745 | 0.1000 | 96 | 230 | 326 | 321 | **-1.56%** |
| **DoS** | $c=5$ | 0.485 | 0.1000 | 34 | 427 | 597 | 729 | **18.11%** |
| **DoS** | $c=10$ | 0.485 | 0.1000 | 34 | 427 | 767 | 1,239 | **38.10%** |
| **DoS** | $c=20$ | 0.400 | 0.1000 | 27 | 503 | 1,043 | 2,259 | **53.83%** |
| **Exploits** | $c=1$ | 0.460 | 0.1000 | 186 | 413 | 599 | 599 | **0.00%** |
| **Exploits** | $c=5$ | 0.205 | 0.1000 | 88 | 672 | 1,112 | 1,343 | **17.20%** |
| **Exploits** | $c=10$ | 0.125 | 0.0625 | 54 | 903 | 1,443 | 2,273 | **36.52%** |
| **Exploits** | $c=20$ | 0.090 | 0.0450 | 26 | 1085 | 1,605 | 4,133 | **61.17%** |
| **Backdoor** | $c=1$ | 0.300 | 0.1000 | 19 | 18 | 37 | 37 | **0.00%** |
| **Backdoor** | $c=5$ | 0.275 | 0.1000 | 18 | 25 | 115 | 113 | **-1.77%** |
| **Backdoor** | $c=10$ | 0.145 | 0.0725 | 12 | 54 | 174 | 208 | **16.35%** |
| **Backdoor** | $c=20$ | 0.145 | 0.0725 | 12 | 54 | 294 | 398 | **26.13%** |
| **Analysis** | $c=1$ | 0.545 | 0.1000 | 98 | 3 | 101 | 101 | **0.00%** |
| **Analysis** | $c=5$ | 0.545 | 0.1000 | 98 | 3 | 493 | 493 | **0.00%** |
| **Analysis** | $c=10$ | 0.035 | 0.0175 | 14 | 715 | 855 | 983 | **13.02%** |
| **Analysis** | $c=20$ | 0.035 | 0.0175 | 14 | 715 | 995 | 1,963 | **49.31%** |
| **Reconnaissance** | $c=1$ | 0.440 | 0.1000 | 30 | 154 | 184 | 184 | **0.00%** |
| **Reconnaissance** | $c=5$ | 0.415 | 0.1000 | 29 | 157 | 302 | 304 | **0.66%** |
| **Reconnaissance** | $c=10$ | 0.225 | 0.1000 | 15 | 189 | 339 | 454 | **25.33%** |
| **Reconnaissance** | $c=20$ | 0.110 | 0.0550 | 13 | 238 | 498 | 754 | **33.95%** |

### Bảng 2b: Tối ưu Chi phí Vận hành trên HistGradientBoosting (Without SMOTE)

| Loại Tấn công | Hệ số Phạt $c$ | Ngưỡng Tối ưu $t^*_{\text{cost}}$ | Near-Miss $t_{low}$ | FN tại $t^*$ | FP tại $t^*$ | Chi phí $t^*$ | Chi phí F1-Opt | **Tiết kiệm Chi phí (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | $c=1$ | 0.995 | 0.1000 | 186 | 242 | 428 | 439 | **2.51%** |
| **DoS** | $c=5$ | 0.860 | 0.1000 | 34 | 643 | 813 | 959 | **15.22%** |
| **DoS** | $c=10$ | 0.840 | 0.1000 | 32 | 665 | 985 | 1,609 | **38.78%** |
| **DoS** | $c=20$ | 0.840 | 0.1000 | 32 | 665 | 1,305 | 2,909 | **55.14%** |
| **Exploits** | $c=1$ | 0.870 | 0.1000 | 304 | 450 | 754 | 754 | **0.00%** |
| **Exploits** | $c=5$ | 0.610 | 0.1000 | 85 | 822 | 1,247 | 1,970 | **36.70%** |
| **Exploits** | $c=10$ | 0.535 | 0.1000 | 57 | 975 | 1,545 | 3,490 | **55.73%** |
| **Exploits** | $c=20$ | 0.305 | 0.1000 | 18 | 1386 | 1,746 | 6,530 | **73.26%** |
| **Backdoor** | $c=1$ | 0.990 | 0.1000 | 39 | 8 | 47 | 47 | **0.00%** |
| **Backdoor** | $c=5$ | 0.235 | 0.1000 | 9 | 74 | 119 | 203 | **41.38%** |
| **Backdoor** | $c=10$ | 0.235 | 0.1000 | 9 | 74 | 164 | 398 | **58.79%** |
| **Backdoor** | $c=20$ | 0.035 | 0.0175 | 4 | 188 | 268 | 788 | **65.99%** |
| **Analysis** | $c=1$ | 0.990 | 0.1000 | 101 | 3 | 104 | 104 | **0.00%** |
| **Analysis** | $c=5$ | 0.755 | 0.1000 | 99 | 7 | 502 | 508 | **1.18%** |
| **Analysis** | $c=10$ | 0.340 | 0.1000 | 27 | 603 | 873 | 1,013 | **13.82%** |
| **Analysis** | $c=20$ | 0.155 | 0.0775 | 1 | 905 | 925 | 2,023 | **54.28%** |
| **Reconnaissance** | $c=1$ | 0.825 | 0.1000 | 32 | 192 | 224 | 220 | **-1.82%** |
| **Reconnaissance** | $c=5$ | 0.755 | 0.1000 | 28 | 204 | 344 | 348 | **1.15%** |
| **Reconnaissance** | $c=10$ | 0.220 | 0.1000 | 11 | 254 | 364 | 508 | **28.35%** |
| **Reconnaissance** | $c=20$ | 0.220 | 0.1000 | 11 | 254 | 474 | 828 | **42.75%** |

> [!TIP]
> **Quy luật Dịch chuyển Ngưỡng ($c=10$):**
> - **Random Forest ($c=10$):** **DoS giảm 38.10%** (FN: 102 -> 34), **Exploits giảm 36.52%** (FN: 186 -> 54), **Backdoor giảm 16.35%** (FN: 19 -> 12), **Analysis giảm 13.02%** (FN: 98 -> 14), **Reconnaissance giảm 25.33%** (FN: 30 -> 15).
> - **HistGradientBoosting ($c=10$):** **DoS giảm 38.78%** (FN: 130 -> 32), **Exploits giảm 55.73%** (FN: 304 -> 57), **Backdoor giảm 58.79%** (FN: 39 -> 9), **Analysis giảm 13.82%** (FN: 101 -> 27), **Reconnaissance giảm 28.35%** (FN: 32 -> 11).
> - **Xử lý trường hợp $t^*_{\text{cost}} < 0.10$:** Khi ngưỡng chi phí giảm sâu dưới `0.10` (`Exploits RF c=20` ($t^*=0.090 \rightarrow t_{low}=0.0450$), `Backdoor GB c=20` ($t^*=0.035 \rightarrow t_{low}=0.0175$), `Analysis RF c=10` ($t^*=0.035 \rightarrow t_{low}=0.0175$), `Analysis RF c=20` ($t^*=0.035 \rightarrow t_{low}=0.0175$)), quy tắc thích ứng $t_{low} = \min(0.10, 0.5 \cdot t^*)$ giữ vùng Near-Miss $[t_{low}, t^*)$ luôn hợp lệ và không bị triệt tiêu về rỗng.

---

## 3. Giải phẫu Chuyên sâu Mẫu Tấn công Bị Bỏ sót (False Negative Forensics)

### Bảng 3: Đối sánh FN Forensics Đa mô hình tại F1-Optimal (`RF` vs `HistGB` vs `PaperRF`)

| Loại Tấn công | Mô hình | Ngưỡng $t^*$ | Near-Miss $t_{low}$ | Tổng Mẫu FN | Near-Miss ($t_{low} \le p < t^*$) | Deep Stealth ($p < t_{low}$) | `IN_BYTES` med (FN) | `OUT_BYTES` med (FN) | Zero-OutBytes (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | RF | 0.76 | 0.1000 | **102** | 100 (98.0%) | 2 (2.0%) | 821.0 | 932.0 | 18.63% |
| **DoS** | GB | 0.99 | 0.1000 | **130** | 123 (94.6%) | 7 (5.4%) | 819.0 | 878.0 | 18.46% |
| **DoS** | PaperRF | 0.42 | 0.1000 | **94** | 72 (76.6%) | 22 (23.4%) | 821.0 | 932.0 | 21.28% |
| **Exploits** | RF | 0.46 | 0.1000 | **186** | 152 (81.7%) | 34 (18.3%) | 800.0 | 1116.0 | 8.60% |
| **Exploits** | GB | 0.87 | 0.1000 | **304** | 296 (97.4%) | 8 (2.6%) | 810.0 | 1061.0 | 11.84% |
| **Exploits** | PaperRF | 0.46 | 0.1000 | **186** | 152 (81.7%) | 34 (18.3%) | 800.0 | 1116.0 | 8.60% |
| **Backdoor** | RF | 0.30 | 0.1000 | **19** | 11 (57.9%) | 8 (42.1%) | 770.0 | 256.0 | 42.11% |
| **Backdoor** | GB | 0.99 | 0.1000 | **39** | 33 (84.6%) | 6 (15.4%) | 460.0 | 256.0 | 25.64% |
| **Backdoor** | PaperRF | 0.28 | 0.1000 | **24** | 12 (50.0%) | 12 (50.0%) | 687.0 | 173.0 | 37.50% |
| **Analysis** | RF | 0.55 | 0.1000 | **98** | 48 (49.0%) | 50 (51.0%) | 814.0 | 1155.0 | 11.22% |
| **Analysis** | GB | 0.99 | 0.1000 | **101** | 100 (99.0%) | 1 (1.0%) | 812.0 | 1146.0 | 14.85% |
| **Analysis** | PaperRF | 0.67 | 0.1000 | **95** | 55 (57.9%) | 40 (42.1%) | 812.0 | 1154.0 | 10.53% |
| **Reconnaissance** | RF | 0.44 | 0.1000 | **30** | 18 (60.0%) | 12 (40.0%) | 849.0 | 0.0 | 53.33% |
| **Reconnaissance** | GB | 0.83 | 0.1000 | **32** | 22 (68.8%) | 10 (31.2%) | 838.0 | 336.0 | 34.38% |
| **Reconnaissance** | PaperRF | 0.48 | 0.1000 | **35** | 22 (62.9%) | 13 (37.1%) | 850.0 | 0.0 | 51.43% |

### Bảng 3b: Phân rã Near-Miss tại Chế độ Vận hành Cost-Opt $c=10$ (Đảm bảo Vùng Tầng 2 Không Bị Sập)

| Loại Tấn công | Mô hình | Ngưỡng $t^*_{c=10}$ | Near-Miss $t_{low}$ | Tổng Mẫu FN | Near-Miss ($t_{low} \le p < t^*$) | Deep Stealth ($p < t_{low}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | RF | 0.485 | 0.1000 | **34** | 32 (94.1%) | 2 (5.9%) |
| **DoS** | GB | 0.840 | 0.1000 | **32** | 25 (78.1%) | 7 (21.9%) |
| **Exploits** | RF | 0.125 | 0.0625 | **54** | 39 (72.2%) | 15 (27.8%) |
| **Exploits** | GB | 0.535 | 0.1000 | **57** | 49 (86.0%) | 8 (14.0%) |
| **Backdoor** | RF | 0.145 | 0.0725 | **12** | 6 (50.0%) | 6 (50.0%) |
| **Backdoor** | GB | 0.235 | 0.1000 | **9** | 3 (33.3%) | 6 (66.7%) |
| **Analysis** | RF | 0.035 | 0.0175 | **14** | 4 (28.6%) | 10 (71.4%) |
| **Analysis** | GB | 0.340 | 0.1000 | **27** | 26 (96.3%) | 1 (3.7%) |
| **Reconnaissance** | RF | 0.225 | 0.1000 | **15** | 3 (20.0%) | 12 (80.0%) |
| **Reconnaissance** | GB | 0.220 | 0.1000 | **11** | 1 (9.1%) | 10 (90.9%) |

### Bản chất Vật lý Mạng của Hiện tượng Lọt lưới (Root Cause Analysis):
1. **Thiếu thông tin Payload tầng 7 ở NetFlow:** Kẻ tấn công thực hiện hành vi dò quét/khai thác ứng dụng web (`Analysis`) trên cổng phổ thông (`Top_Dst_Ports` của FN RF: `{80: 86, 0: 11, 25: 1}`). Ở tầng NetFlow, nhóm FN này có `IN_BYTES` trung vị **814.0B** và `OUT_BYTES` trung vị **1155.0B**, trùng khớp với lưu lượng web `Benign` thông thường.
2. **Tỷ lệ Near-Miss cao ở F1-Optimal:** Với `DoS`, **98.04%** mẫu FN của RF (100/102) nằm sát ngưỡng (`0.10 <= p < 0.76`); trên `HistGB`, hiện tượng bão hòa ngưỡng F1-Opt (`t=0.99` ở `DoS`, `Backdoor`, `Analysis`) càng đẩy tỷ lệ Near-Miss lên cao hơn nữa trước khi được giải phóng bằng tối ưu chi phí $c=10$ hoặc hiệu chuẩn xác suất.
3. **Kiến trúc 2-Tier Hybrid NIDS Đề xuất:**
   - **Tầng 1 (Suy luận tốc độ cao):** `HistGradientBoosting` với ngưỡng chi phí $c=10$ cho qua đúng **99.72%–99.98%** lưu lượng `Benign` (tốc độ đo kiểm batch **172.7k–416.6k flows/s** trên vector hóa, độ trễ suy luận đơn luồng **~4.1–5.8 ms**).
   - **Tầng 2 (Kiến trúc Chuyển tiếp DPI & Heuristic NetFlow):** Các luồng nằm trong vùng nghi vấn $[t_{low}, t^*)$ (với $t_{low} = \min(0.10, 0.5 \cdot t^*)$) được thiết kế chuyển tiếp sang nhánh đối chứng `Random Forest` kết hợp kiểm tra sâu DPI (trong Web App demo hiện thực sẵn bộ phân tích heuristic NetFlow tầng L4/L7/TCP-flags làm lớp hỗ trợ quyết định).

---
*Báo cáo được tự động khởi tạo bởi `src/advanced_evaluation.py`.*
