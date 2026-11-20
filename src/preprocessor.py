import gc
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

def load_split(
    split: str = "train",
    columns: list[str] | None = None,
    filters: list | None = None,
) -> pd.DataFrame:
    """Nạp một phân vùng dữ liệu từ cache parquet (hỗ trợ chiếu cột và lọc PyArrow)."""
    valid_splits = {"train", "val", "heldout_test"}
    if split not in valid_splits:
        raise ValueError(f"Split '{split}' không hợp lệ. Phải thuộc: {valid_splits}")
    
    file_path = PROCESSED_DIR / f"{split}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    
    read_kwargs = {}
    if columns is not None:
        read_kwargs["columns"] = columns
    if filters is not None:
        read_kwargs["filters"] = filters
    return pd.read_parquet(file_path, **read_kwargs)

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
    columns = FEATURE_COLS + ["Attack"] if drop_ip else None
    df = load_split(split, columns=columns)
    filtered = filter_binary_attack(df, attack_name)
    del df
    X, y = prepare_features_and_labels(filtered, drop_ip=drop_ip)
    del filtered
    return X, y

def get_all_splits_for_attack(attack_name: str, drop_ip: bool = True):
    """
    Nạp tuần tự và giải phóng bộ nhớ trung gian cho toàn bộ 3 phân vùng
    (Train, Val, Held-out Test) của 1 loại tấn công.
    Trả về: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    norm_name = normalize_attack_name(attack_name)
    columns = FEATURE_COLS + ["Attack"] if drop_ip else None

    results = []
    for split_name in ("train", "val", "heldout_test"):
        df_split = load_split(split_name, columns=columns)
        df_filtered = filter_binary_attack(df_split, norm_name)
        del df_split
        X_split, y_split = prepare_features_and_labels(df_filtered, drop_ip=drop_ip)
        del df_filtered
        results.extend((X_split, y_split))

    gc.collect()
    return tuple(results)
