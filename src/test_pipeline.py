import sys
import os
import time
import numpy as np
import pandas as pd

from pathlib import Path

# Thêm đường dẫn src vào path
SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))

from preprocessor import (
    load_split,
    filter_binary_attack,
    prepare_features_and_labels,
    get_all_splits_for_attack,
    FEATURE_COLS,
    normalize_attack_name
)
from feature_selection import (
    apply_smote,
    fit_feature_selector,
    transform_features
)

def run_all_tests():
    print("=" * 65)
    print(" BẮT ĐẦU KIỂM THỬ TOÀN DIỆN PIPELINE TIỀN XỬ LÝ & ĐẶC TRƯNG")
    print("=" * 65)
    
    # -------------------------------------------------------------
    # TEST 1: Kiểm tra nạp 3 tập dữ liệu
    # -------------------------------------------------------------
    print("\n[TEST 1] Kiểm tra nạp các phân vùng (Train, Val, Held-out Test)...")
    t0 = time.perf_counter()
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("heldout_test")
    
    assert len(train_df) == 1104497, f"Sai số dòng train: {len(train_df)}"
    assert len(val_df) == 155586, f"Sai số dòng val: {len(val_df)}"
    assert len(test_df) == 363035, f"Sai số dòng held-out: {len(test_df)}"
    assert train_df.isnull().sum().sum() == 0, "Train có giá trị null!"
    assert val_df.isnull().sum().sum() == 0, "Val có giá trị null!"
    assert test_df.isnull().sum().sum() == 0, "Held-out test có giá trị null!"
    print(f"  -> PASS: 3 tập nạp chuẩn xác 100% trong {time.perf_counter()-t0:.2f}s.")

    # -------------------------------------------------------------
    # TEST 2: Kiểm tra lọc nhị phân trên cả 5 loại tấn công
    # -------------------------------------------------------------
    print("\n[TEST 2] Kiểm tra lọc nhị phân cho 5 loại tấn công mục tiêu...")
    attacks = ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]
    expected_train_counts = {
        "DoS": 3437,
        "Exploits": 16832,
        "Backdoor": 1213,
        "Analysis": 1358,
        "Reconnaissance": 8364
    }
    
    for att in attacks:
        filtered = filter_binary_attack(train_df, att)
        norm_name = normalize_attack_name(att)
        
        # Chỉ được phép chứa Benign và loại tấn công mục tiêu
        unique_attacks = set(filtered["Attack"].unique())
        assert unique_attacks == {"Benign", norm_name}, f"Lọc sai cho {att}: {unique_attacks}"
        
        # Kiểm tra nhãn target_label
        counts = filtered["target_label"].value_counts().to_dict()
        assert counts[0] == 1055226, f"Số Benign sai cho {att}: {counts[0]}"
        assert counts[1] == expected_train_counts[norm_name], f"Số mẫu tấn công sai cho {att}: {counts[1]}"
        print(f"  -> PASS [{norm_name}]: Benign = {counts[0]:,} | Attack = {counts[1]:,} | Nhãn nhị phân chuẩn 0/1.")

    # -------------------------------------------------------------
    # TEST 3: Kiểm tra tách ma trận X và nhãn y (Feature Separation)
    # -------------------------------------------------------------
    print("\n[TEST 3] Kiểm tra tách ma trận X và nhãn y (loại bỏ IP để chống rò rỉ)...")
    sample_filtered = filter_binary_attack(train_df, "Backdoor")
    X, y = prepare_features_and_labels(sample_filtered, drop_ip=True)
    
    assert X.shape[1] == 10, f"Số đặc trưng phải là 10, thực tế: {X.shape[1]}"
    assert list(X.columns) == FEATURE_COLS, f"Tên cột không khớp: {list(X.columns)}"
    assert np.isnan(X.to_numpy()).sum() == 0, "X chứa giá trị NaN!"
    assert np.isinf(X.to_numpy()).sum() == 0, "X chứa giá trị vô cùng (inf)!"
    assert len(y) == len(X), "Độ dài X và y không khớp!"
    assert set(np.unique(y)) == {0, 1}, f"Giá trị y không phải {0, 1}: {np.unique(y)}"
    print(f"  -> PASS: X có kích thước {X.shape}, 10 đặc trưng số sạch 100%, y gồm {np.bincount(y)}.")

    # -------------------------------------------------------------
    # TEST 4: Kiểm tra SMOTE nâng lớp thiểu số lên 56,000 mẫu
    # -------------------------------------------------------------
    print("\n[TEST 4] Kiểm tra thuật toán SMOTE (Fixed target = 56,000 mẫu theo bài báo)...")
    # Lấy mẫu nhỏ đại diện để kiểm tra SMOTE nhanh
    # Dùng lớp Backdoor (1,213 mẫu tấn công, 10,000 Benign để test hàm)
    mask = (sample_filtered["target_label"] == 1) | (sample_filtered["target_label"] == 0)
    # Lấy 1,213 Backdoor + 5,000 Benign test
    sub_df = pd.concat([
        sample_filtered[sample_filtered["target_label"] == 0].iloc[:10000],
        sample_filtered[sample_filtered["target_label"] == 1]
    ])
    X_sub, y_sub = prepare_features_and_labels(sub_df, drop_ip=True)
    
    # Thử SMOTE nâng lên 5,000 mẫu
    X_sm, y_sm = apply_smote(X_sub, y_sub, target_samples=5000, random_state=10)
    counts_sm = np.bincount(y_sm)
    assert counts_sm[0] == 10000, f"Lớp đa số bị thay đổi: {counts_sm[0]}"
    assert counts_sm[1] == 5000, f"Lớp thiểu số sau SMOTE không đúng 5,000: {counts_sm[1]}"
    print(f"  -> PASS: SMOTE nâng đúng mục tiêu ({counts_sm[1]:,} mẫu), lớp đa số được bảo toàn ({counts_sm[0]:,} mẫu).")

    # -------------------------------------------------------------
    # TEST 5: Kiểm tra SelectKBest với Mutual Information & L7_PROTO
    # -------------------------------------------------------------
    print("\n[TEST 5] Kiểm tra SelectKBest (Mutual Information) với k=5 và k=10...")
    import warnings
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        selector_k5, sel_feats_5, scores_df = fit_feature_selector(
            X_sub, y_sub, k=5, random_state=10, max_samples_for_mi=5000
        )
        user_warnings = [w for w in recorded_warnings if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0, f"Phát hiện UserWarning trong MutualInfoScorer: {[str(w.message) for w in user_warnings]}"
        
    assert len(sel_feats_5) == 5, f"Số đặc trưng chọn được phải là 5, thực tế: {len(sel_feats_5)}"
    
    X_trans_5 = transform_features(selector_k5, X_sub)
    assert X_trans_5.shape == (len(X_sub), 5), f"Kích thước X biến đổi sai: {X_trans_5.shape}"
    assert list(X_trans_5.columns) == sel_feats_5, "Tên cột sau transform không khớp với selected_features"
    assert (scores_df["MI_Score"] >= 0).all(), "Điểm Mutual Information có giá trị âm!"
    
    # Kiểm tra bảo toàn 100% phần thập phân của L7_PROTO qua transform_features
    selector_k10, sel_feats_10, _ = fit_feature_selector(
        X_sub, y_sub, k=10, random_state=10, max_samples_for_mi=5000
    )
    X_trans_10 = transform_features(selector_k10, X_sub)
    assert "L7_PROTO" in X_trans_10.columns, "L7_PROTO không có trong X_trans_10"
    assert (X_trans_10["L7_PROTO"] == X_sub["L7_PROTO"]).all(), "Cột L7_PROTO bị thay đổi giá trị sau transform_features!"
    assert (X_trans_10["L7_PROTO"] % 1 != 0).any(), "Dữ liệu L7_PROTO không chứa giá trị thập phân nào để kiểm tra!"
    
    print(f"  -> PASS: Top 5 đặc trưng theo MI: {sel_feats_5}")
    print(f"  -> PASS: L7_PROTO bảo toàn 100% giá trị thập phân gốc; MutualInfoScorer sạch hoàn toàn UserWarning.")
    print("  -> Điểm số MI top 3 đặc trưng đầu:")
    for idx, row in scores_df.head(3).iterrows():
        print(f"     * {row['Feature']:<25}: {row['MI_Score']:.5f}")

    # -------------------------------------------------------------
    # TEST 6: Kiểm tra hàm tích hợp nạp toàn bộ splits
    # -------------------------------------------------------------
    print("\n[TEST 6] Kiểm tra hàm get_all_splits_for_attack cho 'Analysis'...")
    X_tr, y_tr, X_v, y_v, X_te, y_te = get_all_splits_for_attack("Analysis")
    assert len(X_tr) == 1055226 + 1358, "Train size không khớp"
    assert len(X_v) == 148645 + 191, "Val size không khớp"
    assert len(X_te) == 346841 + 446, "Heldout size không khớp"
    assert (np.bincount(y_tr) == [1055226, 1358]).all(), "Phân bố y_tr sai"
    assert (np.bincount(y_v) == [148645, 191]).all(), "Phân bố y_v sai"
    assert (np.bincount(y_te) == [346841, 446]).all(), "Phân bố y_te sai"
    print(f"  -> PASS: get_all_splits_for_attack nạp và tách nhãn đồng bộ hoàn hảo trên cả 3 tập.")

    # -------------------------------------------------------------
    # TEST 7: Kiểm tra xử lý biến rời rạc & cơ chế lưu/tải mô hình (Joblib)
    # -------------------------------------------------------------
    print("\n[TEST 7] Kiểm tra xử lý biến rời rạc & cơ chế lưu mô hình (Joblib)...")
    import tempfile
    import joblib
    from feature_selection import DISCRETE_FEATURES
    from sklearn.ensemble import RandomForestClassifier
    
    assert len(DISCRETE_FEATURES) == 3, f"Số đặc trưng rời rạc phải là 3: {DISCRETE_FEATURES}"
    
    # Huấn luyện mô hình thử nghiệm nhỏ
    rf_toy = RandomForestClassifier(n_estimators=10, random_state=10)
    rf_toy.fit(X_trans_5, y_sub)
    
    # Test lưu và tải lại
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        joblib.dump(rf_toy, tmp_path)
        loaded_rf = joblib.load(tmp_path)
        pred_orig = rf_toy.predict(X_trans_5)
        pred_loaded = loaded_rf.predict(X_trans_5)
        assert (pred_orig == pred_loaded).all(), "Mô hình sau khi tải lại dự đoán không khớp!"
        
        # Kiểm tra serialize cả selector đặc trưng
        joblib.dump(selector_k5, tmp_path)
        loaded_selector = joblib.load(tmp_path)
        assert (loaded_selector.get_support() == selector_k5.get_support()).all(), "Selector sau khi tải lại không khớp!"
        print("  -> PASS: DISCRETE_FEATURES khai báo chuẩn xác; Model & Selector Persistence hoạt động 100%.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n" + "=" * 65)
    print(" TẤT CẢ 7 TEST SUITE ĐÃ VƯỢT QUA THÀNH CÔNG (ZERO ERRORS)!")
    print("=" * 65)

if __name__ == "__main__":
    run_all_tests()
