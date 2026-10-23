# Dự án Phân loại Cảnh báo Xâm nhập Mạng (NIDS) - UNSW-NB15

Dự án nghiên cứu và thực nghiệm hệ thống phát hiện xâm nhập mạng dựa trên phương pháp của bài báo:
**"Random Forest-Based Network Intrusion Detection with Feature Selection and Class Balancing on UNSW-NB15 Traffic"** (*Information*, MDPI, 2026, 17(9), 836).

---

## Cấu trúc Thư mục Dự án

```
project/
├── data/                      # Quản lý toàn bộ tập dữ liệu
│   ├── NF-UNSW-NB15.csv       # Dữ liệu NetFlow chính (1.62M dòng, 14 cột)
│   ├── NF-BoT-IoT.csv         # Dữ liệu NetFlow mở rộng
│   ├── NF-ToN-IoT.csv         # Dữ liệu NetFlow mở rộng
│   ├── NF-CSE-CIC-IDS2018.csv # Dữ liệu NetFlow mở rộng
│   ├── NetFlow_v1_Features.csv# Bảng mô tả 14 đặc trưng NetFlow
│   ├── processed/             # Lưu trữ dữ liệu sau khi chia train/val/test hoặc cache parquet
│   ├── raw_zips/              # File nén zip dữ liệu gốc
│   └── raw_bags/              # Thư mục metadata định dạng BagIt khi tải về
│
├── docs/                      # Tài liệu nghiên cứu, bài báo và tài liệu nền tảng
│   ├── information-17-00836.pdf
│   ├── ke-hoach-UNSW-NB15.pdf
│   └── Kiến thức mạng cần học cho dự án phân loại cảnh báo (UNSW-NB15).pdf
│
├── notebooks/                 # Jupyter Notebooks phục vụ EDA và trực quan hóa
│   └── 00_test.ipynb          # Notebook nháp thử nghiệm ban đầu
│
├── src/                       # Mã nguồn thực thi chuẩn hóa theo module
│   ├── data_loader.py         # Nạp và chia dữ liệu (Train 70%, Test 30% -> Val 30%, Held-out 70%)
│   ├── preprocessor.py        # Mã hóa biến, lọc nhị phân từng loại tấn công
│   ├── feature_selection.py   # Lựa chọn đặc trưng SelectKBest (Mutual Information)
│   ├── models.py              # Xây dựng và tinh chỉnh Random Forest, Gradient Boosting
│   └── evaluation.py          # Tính toán ma trận nhầm lẫn, PR-curve, ngưỡng recall, chi phí
│
├── results/                   # Lưu trữ toàn bộ bảng số liệu thực nghiệm
│   ├── baseline/              # Kết quả theo đúng phương pháp tác giả (Có/Không SMOTE)
│   └── improved/              # Kết quả sau khi cải tiến (GBM, ngưỡng recall, cost-sensitive)
│
├── figures/                   # Biểu đồ phân phối, ma trận nhầm lẫn, đường cong PR
├── slides/                    # Bộ slide thuyết trình và báo cáo
├── log.md                     # Nhật ký ghi nhận tiến trình từng buổi
└── README.md                  # Tài liệu tổng quan dự án
```
