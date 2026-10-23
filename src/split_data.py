from pathlib import Path
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "NF-UNSW-NB15.csv"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Đang nạp dữ liệu từ: {data_path}")
    start_time = time.time()
    df = pd.read_csv(data_path)
    load_time = time.time() - start_time
    print(f"[+] Nạp thành công: {len(df):,} dòng, {len(df.columns)} cột trong {load_time:.2f}s")
    
    # Kiểm tra giá trị null
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("[!] Phát hiện giá trị null:")
        print(null_counts[null_counts > 0])
    else:
        print("[+] Dữ liệu sạch, không có giá trị null.")
        
    print("\n[*] Phân bố toàn bộ nhãn tấn công (Attack):")
    total_dist = df["Attack"].value_counts()
    print(total_dist)
    
    # Tỷ lệ phân chia theo bài báo (MDPI Information 2026, 17(9), 836 - Table 1):
    # Tập gốc của tác giả: Train = 175,341; Test = 82,332 -> Tổng = 257,673
    # Tỷ lệ Test / Tổng = 82,332 / 257,673 ≈ 0.31952125 (Train ≈ 68.048%)
    # Tập Test lại chia tiếp: 30% Validation, 70% Held-out Test
    test_ratio = 82332 / 257673
    val_within_test_ratio = 0.30
    heldout_within_test_ratio = 0.70
    
    print(f"\n[*] Bắt đầu phân chia dữ liệu (random_state=10, stratify=Attack):")
    print(f"    - Tỷ lệ Train/Test ban đầu: {1-test_ratio:.4%} / {test_ratio:.4%}")
    
    # Bước 1: Chia Train và Test partition
    train_df, test_partition_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=10,
        stratify=df["Attack"]
    )
    
    # Bước 2: Chia Test partition thành Validation (30%) và Held-out Test (70%)
    # test_size=0.70 nghĩa là Held-out = 70%, phần còn lại là Validation = 30%
    val_df, heldout_test_df = train_test_split(
        test_partition_df,
        test_size=heldout_within_test_ratio,
        random_state=10,
        stratify=test_partition_df["Attack"]
    )
    
    total_rows = len(df)
    print("\n[+] Kết quả phân chia số lượng mẫu:")
    print(f"    - Tổng dữ liệu:       {total_rows:>9,} mẫu (100.00%)")
    print(f"    - Train set:          {len(train_df):>9,} mẫu ({len(train_df)/total_rows:.2%})")
    print(f"    - Validation set:     {len(val_df):>9,} mẫu ({len(val_df)/total_rows:.2%}) [30% của Test]")
    print(f"    - Held-out Test set:  {len(heldout_test_df):>9,} mẫu ({len(heldout_test_df)/total_rows:.2%}) [70% của Test]")
    
    # Bảng phân bố từng loại nhãn qua các tập
    summary_df = pd.DataFrame({
        "Total": total_dist,
        "Train": train_df["Attack"].value_counts(),
        "Val": val_df["Attack"].value_counts(),
        "Heldout_Test": heldout_test_df["Attack"].value_counts()
    }).fillna(0).astype(int)
    
    summary_df["Train_%"] = (summary_df["Train"] / summary_df["Total"] * 100).round(2)
    summary_df["Val_%"] = (summary_df["Val"] / summary_df["Total"] * 100).round(2)
    summary_df["Heldout_%"] = (summary_df["Heldout_Test"] / summary_df["Total"] * 100).round(2)
    
    print("\n[*] Chi tiết số lượng từng lớp tấn công qua các tập:")
    print(summary_df.to_string())
    
    # Lưu ra đĩa (cả Parquet để nạp siêu nhanh và CSV để tiện xem)
    print(f"\n[*] Đang lưu các tập dữ liệu vào {output_dir}...")
    save_start = time.time()
    
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    heldout_test_df.to_parquet(output_dir / "heldout_test.parquet", index=False)
    
    # Lưu CSV bản nén gzip để tiết kiệm dung lượng đĩa
    train_df.to_csv(output_dir / "train.csv.gz", index=False, compression="gzip")
    val_df.to_csv(output_dir / "val.csv.gz", index=False, compression="gzip")
    heldout_test_df.to_csv(output_dir / "heldout_test.csv.gz", index=False, compression="gzip")
    
    summary_df.to_csv(output_dir / "split_summary.csv")
    
    metadata = {
        "dataset_name": "NF-UNSW-NB15",
        "random_state": 10,
        "total_records": total_rows,
        "train_records": len(train_df),
        "val_records": len(val_df),
        "heldout_test_records": len(heldout_test_df),
        "ratios": {
            "train_ratio_of_total": len(train_df) / total_rows,
            "test_partition_ratio_of_total": test_ratio,
            "val_ratio_of_test": val_within_test_ratio,
            "heldout_ratio_of_test": heldout_within_test_ratio
        }
    }
    with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[+] Đã lưu toàn bộ file thành công trong {time.time() - save_start:.2f}s!")

if __name__ == "__main__":
    split_dataset()
