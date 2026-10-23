import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

def apply_smote(
    X: pd.DataFrame, 
    y: np.ndarray, 
    target_samples: int = 56000, 
    random_state: int = 10
):
    """
    Áp dụng SMOTE đúng theo bài báo (MDPI Information 2026, 17(9), 836 - Section 3.6):
    - Chỉ áp dụng trên tập Train.
    - Nâng lớp tấn công thiểu số lên mục tiêu cố định 56,000 mẫu.
    - Nếu target_samples='auto', nâng lớp thiểu số lên bằng số mẫu lớp đa số.
    """
    counts = np.bincount(y)
    minority_count = counts[1]
    majority_count = counts[0]
    
    if target_samples == "auto":
        strategy = "auto"
        print(f"[*] Áp dụng SMOTE 'auto': Nâng lớp thiểu số từ {minority_count:,} lên {majority_count:,} mẫu.")
    else:
        target_count = int(target_samples)
        if minority_count >= target_count:
            print(f"[!] Cảnh báo: Lớp thiểu số ({minority_count:,}) đã >= target ({target_count:,}). Bỏ qua SMOTE.")
            return X, y
        strategy = {1: target_count}
        print(f"[*] Áp dụng SMOTE: Nâng lớp thiểu số từ {minority_count:,} lên {target_count:,} mẫu (lớp đa số = {majority_count:,}).")
        
    smote = SMOTE(sampling_strategy=strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Giữ lại DataFrame với đúng tên cột ban đầu
    if isinstance(X, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)
        
    new_counts = np.bincount(y_res)
    print(f"[+] Hoàn tất SMOTE: Lớp 0 = {new_counts[0]:,} mẫu, Lớp 1 = {new_counts[1]:,} mẫu.")
    return X_res, y_res

DISCRETE_FEATURES = {
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "TCP_FLAGS"
}

class MutualInfoScorer:
    """Callable class ở cấp module để SelectKBest có thể serialize / lưu bằng joblib mà không bị PicklingError."""
    def __init__(self, discrete_mask=None, random_state=10, max_samples_for_mi=200000):
        self.discrete_mask = discrete_mask
        self.random_state = random_state
        self.max_samples_for_mi = max_samples_for_mi

    def __call__(self, X_in, y_in):
        kwargs = {"random_state": self.random_state}
        if self.discrete_mask is not None and len(self.discrete_mask) == X_in.shape[1]:
            kwargs["discrete_features"] = self.discrete_mask

        if self.max_samples_for_mi and len(X_in) > self.max_samples_for_mi:
            # Lấy mẫu phân tầng để tính MI chính xác và tiết kiệm thời gian
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.max_samples_for_mi, random_state=self.random_state)
            idx, _ = next(sss.split(X_in, y_in))
            X_sample = X_in[idx] if isinstance(X_in, np.ndarray) else X_in.iloc[idx]
            y_sample = y_in[idx]
            return mutual_info_classif(X_sample, y_sample, **kwargs)
        return mutual_info_classif(X_in, y_in, **kwargs)

def fit_feature_selector(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    k: int, 
    random_state: int = 10,
    max_samples_for_mi: int = 200000
):
    """
    Chọn k đặc trưng tốt nhất bằng Mutual Information (SelectKBest):
    - Khớp chính xác phương pháp bài báo (Section 3.3).
    - Có hỗ trợ max_samples_for_mi để tính toán MI siêu nhanh trên tập lớn mà vẫn đảm bảo độ tin cậy.
    - Hỗ trợ discrete_features để xử lý chính xác các biến rời rạc mạng (cổng, giao thức, cờ TCP).
    """
    total_samples = len(X_train)
    k = min(k, X_train.shape[1])
    
    # Xác định mặt nạ các đặc trưng rời rạc (discrete) và đảm bảo kiểu dữ liệu int
    discrete_mask = None
    X_train_fit = X_train
    if isinstance(X_train, pd.DataFrame):
        discrete_mask = [col in DISCRETE_FEATURES for col in X_train.columns]
        # Ép kiểu int cho các cột rời rạc (như L7_PROTO dạng float) để loại bỏ UserWarning
        float_discrete = [c for c in DISCRETE_FEATURES if c in X_train.columns and pd.api.types.is_float_dtype(X_train[c])]
        if float_discrete:
            X_train_fit = X_train.copy()
            for c in float_discrete:
                X_train_fit[c] = X_train_fit[c].fillna(0).astype(np.int64)
    
    scorer = MutualInfoScorer(discrete_mask=discrete_mask, random_state=random_state, max_samples_for_mi=max_samples_for_mi)
    selector = SelectKBest(score_func=scorer, k=k)
    selector.fit(X_train_fit, y_train)
    
    # Lấy danh sách tên đặc trưng được chọn và điểm số tương ứng
    feature_names = np.array(X_train.columns)
    selected_mask = selector.get_support()
    selected_features = feature_names[selected_mask].tolist()
    
    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "MI_Score": selector.scores_
    }).sort_values(by="MI_Score", ascending=False).reset_index(drop=True)
    
    return selector, selected_features, scores_df

def transform_features(selector: SelectKBest, X: pd.DataFrame) -> pd.DataFrame:
    """Biến đổi ma trận X theo selector đã fit, trả về DataFrame giữ nguyên tên cột."""
    selected_cols = np.array(X.columns)[selector.get_support()]
    X_trans = selector.transform(X)
    return pd.DataFrame(X_trans, columns=selected_cols, index=X.index)
