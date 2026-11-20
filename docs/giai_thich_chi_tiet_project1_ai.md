# Giải Thích Chi Tiết Từng Bước Project 1 (Phần AI / Machine Learning)

> **Phạm vi tài liệu:** Giải thích đầy đủ, chính xác 100% theo mã nguồn thực tế của toàn bộ phần AI/Học máy trong Project 1 (bộ dữ liệu `NF-UNSW-NB15`), viết theo lối dễ đọc, đi từng bước từ đầu đến cuối để ôn tập và bảo vệ đồ án.

---

## 0. Bức Tranh Tổng Quan: Project 1 Làm Gì?

Về bản chất, toàn bộ Project 1 xoay quanh **3 hệ thống mô hình** (tổng cộng **30 mô hình nhị phân**) để phát hiện xâm nhập mạng trên lưu lượng NetFlow:

1. **Hệ thống 1 — `Paper-Exact RF` (10 mô hình):** Tái hiện nguyên bản 100% pipeline Random Forest của bài báo nền *Pospíchal et al. (MDPI Information, 2026)* trên tập dữ liệu mới `NF-UNSW-NB15`, vừa để đối chứng công bằng vừa để chỉ ra các điểm yếu của bài báo gốc.
2. **Hệ thống 2 — `Ours Improved RF` (10 mô hình):** Vẫn dùng thuật toán Random Forest, nhưng sửa các lỗi phương pháp luận trong pipeline của bài báo gốc (chống rò rỉ SMOTE trong Cross-Validation, sửa lỗi tính Mutual Information cho biến rời rạc, tăng tốc dò siêu tham số).
3. **Hệ thống 3 — `Ours HistGB` (10 mô hình):** Đổi sang thuật toán thứ hai là **HistGradientBoosting** (`HistGradientBoostingClassifier`) để tăng tốc độ huấn luyện lên hàng chục đến hơn một trăm lần.

### Tại sao mỗi hệ thống đều có đúng 10 mô hình?
- Thay vì gom tất cả các loại tấn công vào 1 mô hình đa lớp, dự án làm đúng theo thiết kế bài báo: tách thành **5 bài toán phân loại nhị phân riêng biệt** (**`Benign`** = 0 vs **1 loại tấn công mục tiêu** = 1).
- **5 lớp tấn công mục tiêu:** `DoS`, `Exploits`, `Backdoor`, `Analysis`, `Reconnaissance`. *(Các loại tấn công khác bị lọc bỏ khỏi tập Train/Val/Test của từng bài toán nhị phân để tránh nhiễu liên tấn công; riêng 4 lớp `Fuzzers`, `Generic`, `Shellcode`, `Worms` được giữ để đánh giá ngoài phạm vi ở bước kiểm thử hỗn hợp cuối).*
- Mỗi loại tấn công lại chạy dưới **2 điều kiện dữ liệu**:
  1. `Without SMOTE` (`nosmote`): Giữ nguyên dữ liệu mất cân bằng thật.
  2. `With SMOTE` (`smote`): Sinh thêm mẫu tổng hợp cho lớp tấn công trên tập Train lên mục tiêu `56.000` mẫu.
- Suy ra: $5 \text{ lớp} \times 2 \text{ điều kiện} = 10 \text{ mô hình}$ cho mỗi hệ thống $\rightarrow 30 \text{ mô hình}$ tổng cộng.

---

## Bước 1: Làm Sạch Dữ Liệu, Bỏ IP & Chia Tập Dữ Liệu (`split_data.py`, `preprocessor.py`)

Bước này **dùng chung cho cả 3 hệ thống** để đảm bảo so sánh công bằng (apples-to-apples):

1. **Dữ liệu đầu vào:** `NF-UNSW-NB15.csv` gồm **1.623.118 bản ghi** luồng mạng NetFlow v1 (14 cột gốc, không có giá trị `null`).
2. **Loại bỏ địa chỉ IP (`drop_ip=True`):**
   - Loại bỏ 2 cột `IPV4_SRC_ADDR` và `IPV4_DST_ADDR` (cùng các cột nhãn `Label`, `Attack`).
   - **Lý do:** Tránh mô hình "học vẹt" địa chỉ IP cố định trong môi trường phòng thí nghiệm (vì thực tế hacker đổi IP liên tục).
   - **10 đặc trưng NetFlow còn lại (`FEATURE_COLS`):**
     `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`, `L7_PROTO`, `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`, `TCP_FLAGS`, `FLOW_DURATION_MILLISECONDS`.
3. **Chia 3 phân vùng phân tầng (`stratify=Attack`, `random_state=10`):**
   - Theo đúng tỷ lệ bài báo gốc (`82.332 / 257.673 ≈ 31,95%` Test, còn lại `68,05%` Train), sau đó tập Test lại chia tiếp `30%` làm Validation và `70%` làm Held-out Test:
     - **Train:** `1.104.497` dòng (`68,05%`) — dùng để học mô hình và dò siêu tham số.
     - **Validation:** `155.586` dòng (`9,59%`) — dùng để chọn ngưỡng quyết định và chọn mốc số cột $k^*$ tốt nhất.
     - **Held-out Test:** `363.035` dòng (`22,37%`) — khóa kín, chỉ dùng đánh giá cuối cùng.
   - *(Khi lọc nhị phân `Benign` + 1 loại tấn công, mỗi bài toán còn khoảng 347 nghìn mẫu Test, trừ các tấn công khác ra).*

---

## Bước 2: Hệ Thống 1 — Tái Hiện Nguyên Bản Bài Báo (`Paper-Exact RF`)

Cài đặt trong [train_paper_rf.py](../src/paper_original/train_paper_rf.py). Với mỗi cặp `(Attack, Condition)`, quy trình chạy như sau:

### 2.1. Cân bằng dữ liệu SMOTE (nếu là nhánh `With SMOTE`)
- Chạy `SMOTE(sampling_strategy={1: 56000}, random_state=10)` một lần duy nhất trên toàn bộ `(X_train, y_train)` để nội suy $k$-NN nâng lớp tấn công lên `56.000` mẫu.
- **Lưu ý cực kỳ quan trọng:** Tập `Val` và `Held-out Test` **đã tách riêng từ Bước 1**, nên SMOTE **không** rò rỉ sang tập Test. Điểm yếu của bước này nằm ở chỗ nó chạy SMOTE *trước* bước Cross-Validation (CV) nên làm rò rỉ dữ liệu giữa các fold bên trong CV (sẽ giải thích kỹ ở phần cải tiến).

### 2.2. Vòng lặp lớn qua 4 mốc số lượng cột: `k ∈ {3, 5, 7, 10}`
Bài báo không mặc định dùng cả 10 cột ngay mà thử lần lượt giữ lại **3, 5, 7, hoặc 10 cột tốt nhất** thông qua `SelectKBest(mutual_info_classif)`:

- **Mutual Information (MI) hoạt động thế nào?**
  - MI đo mức độ phụ thuộc thống kê giữa đặc trưng $X$ và nhãn $Y$:
    $$I(X; Y) = H(Y) - H(Y \mid X)$$
    (biết được giá trị cột $X$ thì độ bất định khi đoán xem luồng là sạch hay tấn công giảm đi bao nhiêu).
  - **Điểm MI càng CAO thì cột đó càng quan trọng và được giữ lại** (không phải tính độ nhiễu để giữ cột nhỏ).
- **Điểm yếu của bài báo gốc ở bước này (`discrete_features='auto'`):**
  - Khi truyền bảng số vào `mutual_info_classif` với mặc định `discrete_features='auto'`, `scikit-learn` coi **tất cả 10 cột đều là biến liên tục** và dùng thuật toán ước lượng **Kraskov $k$-NN với $k=3$ láng giềng gần nhất** (dựa trên hàm digamma $\psi(\cdot)$, cộng thêm một lượng nhiễu Gaussian cực nhỏ $10^{-10}$ để phá đồng hạng các giá trị bằng nhau).
  - Với các cột rời rạc có hàng trăm nghìn dòng trùng đúng 1 giá trị (như `TCP_FLAGS`, `PROTOCOL`), khoảng cách đến 3 láng giềng gần nhất gần như bằng `0`, làm ước lượng Kraskov bị **thổi phồng điểm MI giả tạo**.
  - Hệ quả: ở mốc $k=3$, bài báo gốc luôn chọn cứng bộ `['PROTOCOL', 'L7_PROTO', 'TCP_FLAGS']`, bỏ rơi các cột cực kỳ quan trọng như `IN_BYTES` hay `L4_DST_PORT`.

### 2.3. Vòng lặp con bên trong: Tìm kiếm siêu tham số (`RandomizedSearchCV`)
Tại mỗi mốc $k$, bài báo tối ưu **4 siêu tham số** của `RandomForestClassifier`:
1. **`n_estimators`**: Số cây trong rừng — `[100, 200, 300, 400]`.
2. **`min_samples_leaf`**: Số mẫu tối thiểu ở mỗi nút lá để dừng không tách tiếp — `[1, 2, 3]`.
3. **`max_features`**: Số cột ứng viên xét tại mỗi nút chia của cây — `["sqrt", None]` (hoặc $\sqrt{k}$ cột ngẫu nhiên, hoặc xét cả $k$ cột).
4. **`class_weight`**: Trọng số phạt giữa lớp tấn công và lớp sạch — `[None, "balanced"]`.

- Tổng số tổ hợp là $4 \times 3 \times 2 \times 2 = 48$ tổ hợp.
- `RandomizedSearchCV` bốc ngẫu nhiên **`n_iter = 10`** tổ hợp, mỗi tổ hợp chạy **3-Fold Cross-Validation (`cv = 3`)** theo điểm `F1` $\rightarrow$ tức là phải fit **30 rừng cây** cho mỗi mốc $k$ (và **120 rừng cây** cho 4 mốc $k$ của 1 kịch bản).
- **Điểm yếu của bài báo gốc:** Chạy toàn bộ 30 lần fit đó trên **100% tập Train (~1,06–1,11 triệu dòng)** nên cực kỳ chậm (10 kịch bản mất tổng cộng **87.759,8 giây ≈ 24,38 giờ**).

### 2.4. Bản chất thuật toán Cây Quyết Định & Random Forest
- **Dựng 1 Cây Quyết Định (CART với chỉ số Gini):**
  - Cây **không** xếp sẵn thứ tự cột cố định từ gốc đến ngọn. Tại **mỗi nút cần chia**, thuật toán thử các cột ứng viên (toàn bộ $k$ cột nếu `max_features=None`, hoặc bốc ngẫu nhiên $\sqrt{k}$ cột nếu `max_features="sqrt"`) và thử mọi ngưỡng cắt trên từng cột đó.
  - Công thức Gini của 1 nút chứa $y$ mẫu tấn công và $n$ mẫu sạch:
    $$G = 1 - \left(\frac{y}{y+n}\right)^2 - \left(\frac{n}{y+n}\right)^2$$
  - Khi tách thành 2 nhánh con (trái $N_1 = y_1+n_1$, phải $N_2 = y_2+n_2$, tổng $N = N_1+N_2$), Gini có trọng số sau chia là:
    $$G_{\text{split}} = \frac{N_1}{N} G_1 + \frac{N_2}{N} G_2$$
  - Nút sẽ chọn cặp `(cột, ngưỡng)` làm cho $G_{\text{split}}$ **nhỏ nhất** (tức các nhánh con sạch/thuần nhất), và cứ tách tiếp cho đến khi thuần nhất hoặc chạm `min_samples_leaf`.
- **Ghép thành 1 Rừng Cây (Random Forest = Bagging + Feature Randomness):**
  - Bốc ngẫu nhiên **có hoàn lại (Bootstrap)** đúng $N$ dòng từ tập Train để tạo một tập dữ liệu mới cho mỗi cây (một số dòng bị trùng, khoảng ~36,8% dòng không được bốc).
  - Lặp lại `n_estimators` lần (100, 200, 300 hoặc 400 cây) và kết hợp thêm việc chọn ngẫu nhiên tập con cột tại mỗi nút chia (nếu `max_features="sqrt"`). Khi dự đoán, rừng lấy trung bình xác suất dự đoán của tất cả các cây.

### 2.5. Huấn luyện (Fit) mô hình cuối & Quét ngưỡng trên Validation
- Vì `RandomizedSearchCV` đặt `refit=False` (chỉ dò ra bộ `best_params` tốt nhất chứ chưa giữ lại mô hình cuối), ngay sau đó code phải khởi tạo một `RandomForestClassifier(**best_params)` mới và gọi `.fit(X_train_k, y_train)` trên **100% tập Train**.
- Tiếp theo, đưa mô hình vừa fit đi dự đoán xác suất trên **tập Validation**, quét ngưỡng từ `0.05` đến `0.94` (bước `0.01`, dùng `np.arange(0.05, 0.95, 0.01)`) để tìm ngưỡng $t^*$ cho `Val F1` cao nhất.
- Lặp hết 4 mốc `k ∈ {3, 5, 7, 10}`, lấy mốc $k^*$ có `Val F1` cao nhất (hòa điểm ưu tiên $k$ nhỏ hơn), rồi mới đem đi chấm điểm một lần duy nhất trên **Held-out Test**.

---

## Bước 3: Hệ Thống 2 — Random Forest Cải Tiến (`Ours Improved RF`)

Cài đặt trong [train_rf.py](../src/train_rf.py) và [feature_selection.py](../src/feature_selection.py). Thuật toán cốt lõi vẫn là `RandomForestClassifier`, nhưng sửa **4 lỗ hổng lớn** trong pipeline bài báo gốc:

1. **Triệt tiêu rò rỉ dữ liệu SMOTE trong 3-Fold CV (`prevent_cv_leakage=True`):**
   - *Vấn đề cũ:* Bài báo chạy SMOTE trên toàn bộ `X_train` **trước** khi vào 3-Fold CV. Các điểm nội suy $k$-NN nằm sát nhau bị chia vào cả *train-fold* lẫn *val-fold* của CV $\rightarrow$ `CV F1` bị ảo lên tới `0,9792 – 0,9983`, nhưng ra `Held-out Test` thực tế lại sụt mạnh.
   - *Cách sửa:* Giữ bản `X_train_raw` chưa SMOTE; bọc `SMOTE` và `RandomForestClassifier` vào trong **`ImbPipeline`** khi chạy `RandomizedSearchCV`. Nhờ đó **SMOTE chỉ chạy độc lập bên trong từng train-fold** của CV, tuyệt đối không rò rỉ sang validation-fold.
2. **Sửa lỗi ước lượng Mutual Information cho biến rời rạc (`MutualInfoScorer`):**
   - Trong [feature_selection.py](../src/feature_selection.py#L44-L89), khai báo tường minh 3 biến rời rạc:
     `DISCRETE_FEATURES = {"PROTOCOL", "L7_PROTO", "TCP_FLAGS"}` *(lưu ý: cổng `L4_SRC_PORT` và `L4_DST_PORT` vẫn để liên tục vì có hàng chục nghìn giá trị cổng)*.
   - Dùng `pd.factorize` mã hóa 3 cột rời rạc này và truyền `discrete_features=discrete_mask` vào `mutual_info_classif` (lấy mẫu phân tầng tối đa **`max_samples_for_mi = 200.000`** dòng). Kết quả hết bị thổi phồng MI giả tạo (ví dụ `Analysis Without SMOTE` đạt tối ưu ngay ở $k^*=3$ thay vì $k^*=7$).
3. **Tăng tốc bước dò siêu tham số (`subsample_minority_preserving`):**
   - Ở bước `RandomizedSearchCV` (30 lần fit), thay vì chạy trên toàn bộ ~1,06–1,11 triệu dòng, chỉ lấy mẫu con **`search_sample_limit = 150.000` dòng**: **giữ 100% mẫu tấn công (`y == 1`)** và lấy ngẫu nhiên lớp `Benign` (`y == 0`) cho đủ 150.000 dòng.
   - Sau khi tìm ra `best_params`, **bước Fit mô hình cuối (`curr_rf.fit`) vẫn huấn luyện trên 100% tập Train**!
   - **Hiệu quả:** Toàn pipeline nhanh hơn **`6,55x – 14,24x`** so với `Paper-Exact RF` mà `Test F1` hầu như giữ nguyên hoặc tăng ở lớp khó (`Backdoor Without SMOTE` tăng từ `0,9385` lên `0,9535`).
4. **Mở rộng lưới quét ngưỡng Validation:**
   - Quét rộng `0.01..0.99` (bằng `np.linspace`, tránh lỗi trôi dấu phẩy động của `np.arange`) với thuật toán vector hóa $O(N \log N)$.

---

## Bước 4: Hệ Thống 3 — `Ours HistGB` (`HistGradientBoostingClassifier`)

Cài đặt trong [train_gb.py](../src/train_gb.py). Đây là thuật toán thứ hai nhằm tối ưu tốc độ huấn luyện lại mô hình.

### 4.1. Phân biệt rõ: HistGradientBoosting KHÔNG PHẢI AdaBoost
- **AdaBoost** (không dùng trong dự án này): Dựng cây đơn giản, **tăng trọng số mẫu đoán sai** bằng $e^\alpha$ với $\alpha = \frac{1}{2}\ln\frac{1-\epsilon}{\epsilon}$ để cây sau tập trung học lại các dòng sai đó.
- **HistGradientBoostingClassifier (thuật toán dự án dùng):**
  1. **Học tuần tự bằng Gradient âm (Pseudo-residuals) của hàm Log-Loss:** Không đổi trọng số mẫu và không bốc lại dòng theo $e^\alpha$. Mỗi cây hồi quy mới được xây để dự đoán **phần dư sai số** của tổ hợp các cây trước:
     $$r_{i, m} = y_i - p_{i, m-1}$$
     (hiệu giữa nhãn thật $y_i \in \{0,1\}$ và xác suất dự đoán hiện tại $p_{i,m-1} = \sigma(F_{m-1}(x_i))$), rồi cộng dần vào điểm dự đoán tổng:
     $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
     với $\eta$ là `learning_rate`.
  2. **Histogram Binning (chữ `Hist`):** Trước khi chia nhánh, mọi đặc trưng số liên tục được gom vào tối đa **255 giỏ (bins)** (số nguyên 8-bit) và song song hóa OpenMP. Nhờ tìm điểm chia trên 255 giỏ thay vì sắp xếp hàng triệu giá trị thực, tốc độ tăng vọt.
  3. **Không gian siêu tham số của HistGB:** `max_iter` (`[100, 200, 300]`), `learning_rate` (`[0.05, 0.1, 0.2]`), `max_depth` (`[3, 5, None]`), `min_samples_leaf` (`[20, 50, 100]`), `class_weight` (`[None, "balanced"]`), tắt `early_stopping=False`.

### 4.2. Khác biệt rất quan trọng trong bước Fit mô hình cuối (`hybrid_downsample`)
- Ở bước MI và `RandomizedSearchCV`, `HistGB` dùng chung `MutualInfoScorer` (200k), `ImbPipeline` (nếu có SMOTE) và `subsample_minority_preserving` (150k) như `Ours Improved RF`.
- Nhưng ở **bước Fit mô hình cuối nhánh `Without SMOTE`**, khác với hai bản RF (fit trên 100% ~1,06M dòng), `Ours HistGB` chạy mặc định `balance_mode="hybrid_downsample"`:
  - Giữ **100% mẫu tấn công**, nhưng **cắt giảm lớp `Benign` xuống tối đa `150.000` mẫu** ngay cả khi fit mô hình cuối!
- **Kết quả & Đánh đổi:**
  - Thời gian fit cuối (`Without SMOTE`) chỉ còn **`1,34 – 8,06` giây** — nhanh hơn `Ours RF` **`13,17x – 52,97x`** và nhanh hơn bước fit của `Paper-Exact RF` **`29,51x – 119,95x`** (toàn pipeline nhanh hơn `25,11x – 56,44x`).
  - Đổi lại, `Test F1` của `HistGB` thấp hơn `Ours RF` nhẹ (`-0,0043` đến `-0,0450`), và do cắt giảm `Benign` xuống 150k (làm tỷ lệ tấn công tăng giả tạo `~7,0348x` so với tập gốc), xác suất thô bị đẩy cao khiến ngưỡng tối ưu $F_1$ nhảy lên **`0,83 – 0,99`**.

---

## Bước 5: Tối Ưu Ngưỡng Quyết Định Hậu Huấn Luyện & Chỉ Số Phạt `Cost_SOC` (`advanced_evaluation.py`)

Sau khi đã có xác suất dự đoán từ các mô hình nhánh `Without SMOTE`, [advanced_evaluation.py](../src/advanced_evaluation.py) không giữ nguyên ngưỡng $t^*_{\text{F1}}$ mặc định của bài báo nữa (vì tối ưu $F_1$ coi lỗi bỏ sót tấn công `FN` nhẹ ngang bằng lỗi báo nhầm `FP`), mà cải tiến thành **2 chiến lược tìm ngưỡng trên tập Validation**:

1. **Recall-Targeted Thresholding:** Hạ ngưỡng $t$ (quét 990 điểm `0.99..0.001`) để ép độ phủ đạt tối thiểu **`Recall >= 95%`, `98%`, hoặc `99%`**.
2. **Cost-Sensitive Thresholding (Tối ưu chỉ số phạt SOC):**
   - Quét `199` ngưỡng từ `0.005` đến `0.995` trên tập Validation để cực tiểu hóa hàm chi phí phạt:
     $$\text{Cost}_{\text{SOC}}(t; c) = c \times \text{FN}(t) + 1 \times \text{FP}(t)$$
   - Trong đó `FN` (bỏ lọt 1 vụ tấn công thật) bị phạt nặng gấp **$c$ lần** so với `FP` (1 cảnh báo nhầm cho chuyên viên SOC kiểm tra). Dự án khảo sát $c \in \{1, 5, 10, 20\}$ và lấy **$c = 10$ làm mốc chính**.
   - **Kết quả tại $c = 10$ (tổng 5 lớp trên Held-out Test):**
     - `Paper-Exact RF (F1-Opt)`: `436 FN`, `772 FP`, `Cost_SOC = 5.132`.
     - `Ours RF (Cost-Opt c=10)`: **`129 FN`** (-70,41%), `2.288 FP`, **`Cost_SOC = 3.578`** (-30,28%).
     - `Ours HistGB (Cost-Opt c=10)`: **`136 FN`** (-68,81%), `2.571 FP`, **`Cost_SOC = 3.931`** (-23,40%).
   - **Phát hiện Ablation quan trọng (cần nhớ khi phản biện):** Nếu đem chính xác suất của `Paper-Exact RF` đi tối ưu ngưỡng theo $c=10$, nó cũng giảm xuống còn **`154 FN`**, `1.892 FP`, `Cost_SOC = 3.432`. Tức là phần lớn mức giảm `FN` đến từ **kỹ thuật tối ưu ngưỡng bất đối xứng $c=10$**, còn pipeline mới đóng góp tốc độ, chống rò rỉ CV/MI và giảm tiếp `FN` từ `154` xuống `129`/`136`.

3. **Giải phẫu lỗi bỏ sót (`FN Forensics`) & Vùng Nghi Vấn thích ứng:**
   - Các mẫu bị bỏ sót (`FN`, tức $P < t^*_{\text{cost}}$) được chia làm 2 nhóm bằng cận dưới thích ứng `t_low = min(0.10, 0.5 * t*_cost)`:
     - **Near-Miss (`t_low <= P < t*_cost`):** Nằm sát dưới ngưỡng chặn (ứng với vùng nghi vấn chuyển sang Tầng 2 phân tích sâu).
     - **Deep Stealth (`P < t_low`):** Xác suất rất thấp, mô hình coi như giống hệt lưu lượng sạch.

---

## Bước 6: Những Phát Hiện & Kiểm Chứng Khoa Học Bổ Sung (`scientific_diagnostics.py`)

Đây là 4 điểm cực kỳ giá trị ở phần AI trong dự án (đặc biệt hay gặp khi thầy cô hỏi phản biện):

1. **Nghịch lý SMOTE (SMOTE Paradox):**
   - Trên dữ liệu lớn `NF-UNSW-NB15`, cấu hình **`Without SMOTE` lại cho `Test F1` cao hơn `With SMOTE` ở `5/5` lớp trên `Ours RF` và `4/5` lớp trên `Paper-Exact RF`**. Nội suy SMOTE quá nhiều trên vùng biên làm mô hình lấn sang vùng `Benign` và tăng báo nhầm `FP`.
2. **Hiệu chuẩn xác suất Bayes Prior-Odds (`ECE-15`):**
   - Vì `HistGB (Without SMOTE)` downsample `Benign` xuống `150.000` mẫu (tăng prior lớp tấn công `7,0348x`), xác suất thô bị lệch cao.
   - `scientific_diagnostics.py` áp dụng phép dịch log-odds đóng (closed-form): $\text{logit}(p_{\text{cal}}) = \text{logit}(p_{\text{raw}}) - \ln(\text{odds\_ratio})$, giúp giảm sai số hiệu chuẩn **`ECE-15` đi `39,84% – 96,27%`** và đưa ngưỡng $t^*_{\text{F1}}$ về vùng hợp lý hơn (`0,320 – 0,915`) mà giữ nguyên `PR-AUC`.
3. **Kiểm định ý nghĩa thống kê (`Bootstrap 95% CI` & `McNemar`):**
   - Chạy Stratified Bootstrap `B = 1.000` lần và kiểm định McNemar chính xác giữa từng cặp mô hình (`RF vs GB`, `RF vs PaperRF`, `GB vs PaperRF`) để khẳng định độ tin cậy thống kê chứ không chỉ so sánh một con số `F1` đơn lẻ.
4. **Đánh giá luồng hỗn hợp `Any-of-5` & 4 lớp ngoài phạm vi (`Unseen-4`):**
   - Hợp nhất cả 5 bộ dò nhị phân (cơ chế *Any-of-5*, chỉ cần ít nhất 1 bộ dò vượt ngưỡng là cảnh báo) và chạy trên toàn bộ **`363.035` luồng Held-out Test** ở mốc $c=10$:
     - `Benign FPR` vẫn rất thấp: **`0,40%`** (`Ours RF`) và **`0,41%`** (`Ours HistGB`), với độ phủ 5 lớp mục tiêu đạt **`99,77%`** (`RF`) và **`99,55%`** (`HistGB`).
     - Trên **4 lớp tấn công chưa từng học lúc train** (`Fuzzers`, `Generic`, `Shellcode`, `Worms` — tổng `5.938` luồng), hệ thống chỉ phát hiện ké được **`35,26% – 36,21%`** (đặc biệt `Fuzzers` rất thấp `16,22% – 17,14%`). Do đó, kết luận đúng mực của dự án là **không suy diễn hệ thống này thành bộ phát hiện zero-day tổng quát**.
