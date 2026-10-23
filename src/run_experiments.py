from pathlib import Path
import sys
import argparse
import time
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))
from train_rf import train_and_evaluate_rf

BASE_DIR = SRC_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATTACKS = ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]

def run_experiment_pipeline(
    attacks=None,
    run_nosmote=True,
    run_smote=True,
    k_values=None,
    n_iter=10,
    cv=3,
    sample_limit=150000,
    smote_target=56000,
    prevent_cv_leakage=True
):
    if attacks is None:
        attacks = ["DoS"]
    if k_values is None:
        k_values = [3, 5, 7, 10]
        
    start_total = time.time()
    print("=" * 75)
    print(" TIẾN TRÌNH THỰC NGHIỆM RANDOM FOREST - UNSW-NB15 BASELINE REPLICATION")
    print("=" * 75)
    print(f" Danh sách tấn công: {attacks}")
    print(f" Giá trị k:         {k_values}")
    print(f" Chạy Không SMOTE:  {run_nosmote}")
    print(f" Chạy Có SMOTE:     {run_smote} (Target = {smote_target:,})")
    print(f" Siêu tham số:      n_iter={n_iter}, cv={cv}, sample_limit={sample_limit:,}")
    print(f" Chống rò rỉ CV:    {prevent_cv_leakage}")
    print("=" * 75)
    
    summary_records = []
    out_summary_file = RESULTS_DIR / "summary_rf_comparison.csv"
    
    for att in attacks:
        # 1. Chạy điều kiện KHÔNG SMOTE
        if run_nosmote:
            print(f"\n>>> [1/2] BẮT ĐẦU CHẠY: {att} - KHÔNG SMOTE...")
            df_no = train_and_evaluate_rf(
                attack_name=att,
                use_smote=False,
                k_values=k_values,
                n_iter=n_iter,
                cv=cv,
                search_sample_limit=sample_limit,
                prevent_cv_leakage=prevent_cv_leakage
            )
            # Lấy dòng tốt nhất theo F1 trên Validation
            best_no = df_no.loc[df_no["val_f1"].idxmax()]
            summary_records.append({
                "Attack": att,
                "Condition": "Without SMOTE",
                "Best_k": best_no["k"],
                "Threshold": best_no["val_threshold"],
                "Precision": best_no["test_precision"],
                "Recall": best_no["test_recall"],
                "F1_Score": best_no["test_f1"],
                "FP": int(best_no["FP"]),
                "FN": int(best_no["FN"])
            })
            
        # 2. Chạy điều kiện CÓ SMOTE (Mục tiêu smote_target mẫu)
        if run_smote:
            print(f"\n>>> [2/2] BẮT ĐẦU CHẠY: {att} - CÓ SMOTE (Target = {smote_target:,})...")
            df_sm = train_and_evaluate_rf(
                attack_name=att,
                use_smote=True,
                smote_target=smote_target,
                k_values=k_values,
                n_iter=n_iter,
                cv=cv,
                search_sample_limit=sample_limit,
                prevent_cv_leakage=prevent_cv_leakage
            )
            best_sm = df_sm.loc[df_sm["val_f1"].idxmax()]
            summary_records.append({
                "Attack": att,
                "Condition": "With SMOTE",
                "Best_k": best_sm["k"],
                "Threshold": best_sm["val_threshold"],
                "Precision": best_sm["test_precision"],
                "Recall": best_sm["test_recall"],
                "F1_Score": best_sm["test_f1"],
                "FP": int(best_sm["FP"]),
                "FN": int(best_sm["FN"])
            })
            
        # Cập nhật ngay file summary sau mỗi loại tấn công để không bao giờ mất kết quả
        current_summary = pd.DataFrame(summary_records)
        if out_summary_file.exists():
            try:
                existing_df = pd.read_csv(out_summary_file)
                merged_df = pd.concat([existing_df, current_summary], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=["Attack", "Condition"], keep="last")
            except Exception:
                merged_df = current_summary
        else:
            merged_df = current_summary
            
        merged_df.to_csv(out_summary_file, index=False)
        print(f"\n[+] Đã cập nhật kết quả cho {att} vào: {out_summary_file}")

    print("\n" + "=" * 80)
    print(" BẢNG TỔNG KẾT TOÀN DIỆN RANDOM FOREST (KHÔNG SMOTE vs CÓ SMOTE)")
    print("=" * 80)
    if out_summary_file.exists():
        final_summary = pd.read_csv(out_summary_file)
        print(final_summary.to_string(index=False))
    print(f"\n[+] Tổng thời gian thực nghiệm: {(time.time() - start_total)/60:.2f} phút.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy toàn bộ chuỗi thực nghiệm Random Forest cho các bài toán tấn công")
    parser.add_argument("--attacks", nargs="+", default=["DoS"], help="Danh sách các loại tấn công cần chạy")
    parser.add_argument("--all", action="store_true", help="Chạy tất cả 5 loại tấn công")
    parser.add_argument("--k_values", nargs="+", type=int, default=[3, 5, 7, 10])
    parser.add_argument("--only_nosmote", action="store_true")
    parser.add_argument("--only_smote", action="store_true")
    parser.add_argument("--n_iter", type=int, default=10, help="Số lần ngẫu nhiên của RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Số fold cross-validation")
    parser.add_argument("--sample_limit", type=int, default=150000, help="Giới hạn mẫu tìm kiếm siêu tham số")
    parser.add_argument("--smote_target", type=int, default=56000, help="Mục tiêu số mẫu thiểu số sau SMOTE")
    parser.add_argument("--no_leakage_fix", action="store_true", help="Tắt cơ chế chống rò rỉ dữ liệu trong CV")
    args = parser.parse_args()
    
    target_attacks = ATTACKS if args.all else args.attacks
    run_no = not args.only_smote
    run_sm = not args.only_nosmote
    
    run_experiment_pipeline(
        attacks=target_attacks,
        run_nosmote=run_no,
        run_smote=run_sm,
        k_values=args.k_values,
        n_iter=args.n_iter,
        cv=args.cv,
        sample_limit=args.sample_limit,
        smote_target=args.smote_target,
        prevent_cv_leakage=not args.no_leakage_fix
    )
