from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ATTACK_MAP = {
    "dos": "DoS",
    "exploit": "Exploits",
    "exploits": "Exploits",
    "backdoor": "Backdoor",
    "analysis": "Analysis",
    "reconnaissance": "Reconnaissance",
    "recon": "Reconnaissance"
}

FEATURE_COLS = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS"
]

def normalize_attack_name(attack_name: str) -> str:
    """Chuẩn hóa tên tấn công thành dạng chuẩn trong dataset."""
    key = attack_name.strip().lower()
    if key in ATTACK_MAP:
        return ATTACK_MAP[key]
    valid_attacks = {"DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance", "Fuzzers", "Generic", "Shellcode", "Worms"}
    for va in valid_attacks:
        if va.lower() == key:
            return va
    raise ValueError(f"Tên tấn công không hợp lệ: '{attack_name}'. Các loại hỗ trợ: {list(ATTACK_MAP.values())}")

def load_split(split: str = "train") -> pd.DataFrame:
    """Nạp một phân vùng dữ liệu từ cache parquet."""
    valid_splits = {"train", "val", "heldout_test"}
    if split not in valid_splits:
        raise ValueError(f"Split '{split}' không hợp lệ. Phải thuộc: {valid_splits}")
    
    file_path = PROCESSED_DIR / f"{split}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    
    return pd.read_parquet(file_path)

def filter_binary_attack(df: pd.DataFrame, attack_name: str) -> pd.DataFrame:
    """
    Lọc nhị phân theo đúng phương pháp bài báo:
    Chỉ giữ lại lưu lượng Benign (Normal) và loại tấn công mục tiêu (Target Attack).
    Tất cả các loại tấn công khác bị loại bỏ để tránh nhiễu liên tấn công (attack-to-attack confusion).
    """
    target_attack = normalize_attack_name(attack_name)
    
    mask = (df["Attack"] == "Benign") | (df["Attack"] == target_attack)
    filtered_df = df[mask].copy()
    
    filtered_df["target_label"] = (filtered_df["Attack"] == target_attack).astype(int)
    
    return filtered_df

def prepare_features_and_labels(df: pd.DataFrame, drop_ip: bool = True):
    """
    Tách ma trận đặc trưng X và nhãn mục tiêu y.
    Mặc định drop_ip=True loại bỏ IP nguồn/đích để tránh mô hình học vẹt môi trường lab.
    """
    if "target_label" in df.columns:
        y = df["target_label"].to_numpy(dtype=np.int32)
    elif "Label" in df.columns:
        y = df["Label"].to_numpy(dtype=np.int32)
    else:
        raise ValueError("Không tìm thấy cột nhãn trong DataFrame.")
        
    if drop_ip:
        X = df[FEATURE_COLS].copy()
    else:
        cols = [c for c in df.columns if c not in ["Label", "Attack", "target_label"]]
        X = df[cols].copy()
        
    return X, y

def get_binary_data(attack_name: str, split: str = "train", drop_ip: bool = True):
    """Nạp và lọc dữ liệu nhị phân cho 1 phân vùng duy nhất."""
    df = load_split(split)
    filtered = filter_binary_attack(df, attack_name)
    X, y = prepare_features_and_labels(filtered, drop_ip=drop_ip)
    return X, y

def get_all_splits_for_attack(attack_name: str, drop_ip: bool = True):
    """
    Nạp toàn bộ 3 phân vùng (Train, Val, Held-out Test) cho 1 loại tấn công.
    Trả về: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    norm_name = normalize_attack_name(attack_name)
    
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("heldout_test")
    
    train_f = filter_binary_attack(train_df, norm_name)
    val_f = filter_binary_attack(val_df, norm_name)
    test_f = filter_binary_attack(test_df, norm_name)
    
    X_train, y_train = prepare_features_and_labels(train_f, drop_ip=drop_ip)
    X_val, y_val = prepare_features_and_labels(val_f, drop_ip=drop_ip)
    X_test, y_test = prepare_features_and_labels(test_f, drop_ip=drop_ip)
    
    return X_train, y_train, X_val, y_val, X_test, y_test
