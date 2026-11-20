import argparse
import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    TARGET_ATTACKS,
    is_experiment_complete,
    select_best_row,
    trim_memory,
)
from preprocessor import normalize_attack_name  # noqa: E402
from train_rf import train_and_evaluate_rf  # noqa: E402

BASE_DIR = SRC_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "baseline"
MODELS_DIR = RESULTS_DIR / "models"

ATTACKS = list(TARGET_ATTACKS)


def _summary_record_from_df(att: str, condition_label: str, df_metrics: pd.DataFrame) -> dict:
    """Build a summary comparison row from a detailed metrics DataFrame."""
    best_row = select_best_row(df_metrics)
    return {
        "Attack": att,
        "Condition": condition_label,
        "Best_k": int(best_row["k"]),
        "Threshold": float(best_row["val_threshold"]),
        "Precision": float(best_row["test_precision"]),
        "Recall": float(best_row["test_recall"]),
        "F1_Score": float(best_row["test_f1"]),
        "FP": int(best_row["FP"]),
        "FN": int(best_row["FN"]),
    }


def _write_summary_table(out_summary_file: Path, summary_records: list[dict]) -> None:
    """Merge new summary records with existing summary_rf_comparison.csv and persist once."""
    if not summary_records:
        return
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
    print(f"\n[+] Đã cập nhật bảng tổng kết RF tại: {out_summary_file}")


def run_experiment_pipeline(
    attacks=None,
    run_nosmote=True,
    run_smote=True,
    k_values=None,
    n_iter=10,
    cv=3,
    random_state=10,
    sample_limit=150000,
    smote_target=56000,
    prevent_cv_leakage=True,
    force_retrain=False,
    n_jobs=4,
    cv_jobs=1,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if attacks is None:
        attacks = list(TARGET_ATTACKS)
    if k_values is None:
        k_values = list(DEFAULT_K_VALUES)
    if not k_values:
        raise ValueError("k_values must not be empty.")

    start_total = time.perf_counter()
    print("=" * 75)
    print(" TIẾN TRÌNH THỰC NGHIỆM RANDOM FOREST - UNSW-NB15 BASELINE REPLICATION")
    print("=" * 75)
    print(f" Danh sách tấn công: {attacks}")
    print(f" Giá trị k:         {k_values}")
    print(f" Chạy Không SMOTE:  {run_nosmote}")
    print(f" Chạy Có SMOTE:     {run_smote} (Target = {smote_target:,})")
    print(
        f" Siêu tham số:      n_iter={n_iter}, cv={cv}, sample_limit={sample_limit:,}, "
        f"seed={random_state}, n_jobs={n_jobs}, cv_jobs={cv_jobs}"
    )
    print(f" Chống rò rỉ CV:    {prevent_cv_leakage}")
    print(f" Ép huấn luyện lại: {force_retrain}")
    print("=" * 75)

    summary_records = []
    out_summary_file = RESULTS_DIR / "summary_rf_comparison.csv"
    any_executed = False

    for att in attacks:
        norm_att = normalize_attack_name(att)
        # 1. Chạy điều kiện KHÔNG SMOTE
        if run_nosmote:
            prefix_no = f"rf_{norm_att.lower()}_nosmote"
            csv_no = RESULTS_DIR / f"{prefix_no}_metrics.csv"
            if (not force_retrain) and is_experiment_complete(RESULTS_DIR, MODELS_DIR, prefix_no, k_values):
                print(
                    f"\n[+] Đã tồn tại kết quả hoàn chỉnh cho {norm_att} [NOSMOTE], bỏ qua huấn luyện lại (dùng --force để chạy lại)."
                )
                df_no = pd.read_csv(csv_no)
            else:
                print(f"\n>>> [1/2] BẮT ĐẦU CHẠY: {norm_att} - KHÔNG SMOTE...")
                df_no = train_and_evaluate_rf(
                    attack_name=norm_att,
                    use_smote=False,
                    k_values=k_values,
                    n_iter=n_iter,
                    cv=cv,
                    random_state=random_state,
                    search_sample_limit=sample_limit,
                    prevent_cv_leakage=prevent_cv_leakage,
                    n_jobs=n_jobs,
                    cv_jobs=cv_jobs,
                )
                any_executed = True
                trim_memory()
            summary_records.append(_summary_record_from_df(norm_att, "Without SMOTE", df_no))

        # 2. Chạy điều kiện CÓ SMOTE (Mục tiêu smote_target mẫu)
        if run_smote:
            prefix_sm = f"rf_{norm_att.lower()}_smote"
            csv_sm = RESULTS_DIR / f"{prefix_sm}_metrics.csv"
            if (not force_retrain) and is_experiment_complete(RESULTS_DIR, MODELS_DIR, prefix_sm, k_values):
                print(
                    f"\n[+] Đã tồn tại kết quả hoàn chỉnh cho {norm_att} [SMOTE], bỏ qua huấn luyện lại (dùng --force để chạy lại)."
                )
                df_sm = pd.read_csv(csv_sm)
            else:
                print(
                    f"\n>>> [2/2] BẮT ĐẦU CHẠY: {norm_att} - CÓ SMOTE (Target = {smote_target:,})..."
                )
                df_sm = train_and_evaluate_rf(
                    attack_name=norm_att,
                    use_smote=True,
                    smote_target=smote_target,
                    k_values=k_values,
                    n_iter=n_iter,
                    cv=cv,
                    random_state=random_state,
                    search_sample_limit=sample_limit,
                    prevent_cv_leakage=prevent_cv_leakage,
                    n_jobs=n_jobs,
                    cv_jobs=cv_jobs,
                )
                any_executed = True
                trim_memory()
            summary_records.append(_summary_record_from_df(norm_att, "With SMOTE", df_sm))

    if any_executed or not out_summary_file.exists():
        _write_summary_table(out_summary_file, summary_records)

    print("\n" + "=" * 80)
    print(" BẢNG TỔNG KẾT TOÀN DIỆN RANDOM FOREST (KHÔNG SMOTE vs CÓ SMOTE)")
    print("=" * 80)
    if out_summary_file.exists():
        final_summary = pd.read_csv(out_summary_file)
        print(final_summary.to_string(index=False))
    print(
        f"\n[+] Tổng thời gian thực nghiệm: {(time.perf_counter() - start_total) / 60:.2f} phút."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chạy toàn bộ chuỗi thực nghiệm Random Forest cho các bài toán tấn công"
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=None,
        help="Danh sách các loại tấn công cần chạy (mặc định: cả 5 loại)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Chạy tất cả 5 loại tấn công"
    )
    parser.add_argument("--k_values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--only_nosmote", action="store_true")
    parser.add_argument("--only_smote", action="store_true")
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="Số lần ngẫu nhiên của RandomizedSearchCV",
    )
    parser.add_argument("--cv", type=int, default=3, help="Số fold cross-validation")
    parser.add_argument("--random_state", type=int, default=10, help="Random seed (mặc định: 10)")
    parser.add_argument("--n_jobs", type=int, default=4, help="Số luồng cho RandomForest (mặc định: 4)")
    parser.add_argument("--cv_jobs", type=int, default=1, help="Số tiến trình song song cho fold CV (mặc định: 1)")
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=150000,
        help="Giới hạn mẫu tìm kiếm siêu tham số",
    )
    parser.add_argument(
        "--smote_target",
        type=int,
        default=56000,
        help="Mục tiêu số mẫu thiểu số sau SMOTE",
    )
    parser.add_argument(
        "--no_leakage_fix",
        action="store_true",
        help="Tắt cơ chế chống rò rỉ dữ liệu trong CV",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ép huấn luyện lại ngay cả khi đã có kết quả lưu sẵn",
    )
    args = parser.parse_args()

    target_attacks = list(TARGET_ATTACKS) if (args.all or args.attacks is None) else args.attacks
    run_no = not args.only_smote
    run_sm = not args.only_nosmote

    run_experiment_pipeline(
        attacks=target_attacks,
        run_nosmote=run_no,
        run_smote=run_sm,
        k_values=args.k_values,
        n_iter=args.n_iter,
        cv=args.cv,
        random_state=args.random_state,
        sample_limit=args.sample_limit,
        smote_target=args.smote_target,
        prevent_cv_leakage=not args.no_leakage_fix,
        force_retrain=args.force,
        n_jobs=args.n_jobs,
        cv_jobs=args.cv_jobs,
    )
