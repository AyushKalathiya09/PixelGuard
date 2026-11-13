"""
evaluate_split.py  -  Rigorous Train/Test Split Evaluation on MICC-F220 Benchmark.

Implements Fix E:
1. Takes a balanced 40-image subset from the MICC-F220 dataset (20 authentic, 20 tampered).
2. Performs a stratified 50/50 split into Tuning Set (20 images) and Test Set (20 images).
3. Selects optimal detection parameters using ONLY the tuning set.
4. Evaluates final performance strictly on the unseen test set to ensure generalizability without data leakage.
"""

import os
import random
import csv
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from main import analyze_image

def main():
    random.seed(42)
    np.random.seed(42)

    base = os.path.dirname(os.path.abspath(__file__))
    micc_dir = os.path.join(base, "dataset", "micc_f220", "MICC-F220")
    gt_file = os.path.join(micc_dir, "groundtruthDB_220.txt")
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isfile(gt_file):
        print(f"[ERROR] Ground truth file not found at {gt_file}")
        return

    # Parse ground truth
    items = []
    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                fname, lbl = parts[0], int(parts[1])
                fpath = os.path.join(micc_dir, fname)
                if os.path.isfile(fpath):
                    items.append((fpath, fname, lbl))

    auth_all = [it for it in items if it[2] == 0]
    tamp_all = [it for it in items if it[2] == 1]
    print(f"Total MICC-F220 images found: {len(items)} (Authentic: {len(auth_all)}, Tampered: {len(tamp_all)})")

    # Sample 20 authentic and 20 tampered
    random.shuffle(auth_all)
    random.shuffle(tamp_all)
    auth_subset = auth_all[:20]
    tamp_subset = tamp_all[:20]

    # Stratified 50/50 split
    tune_set = auth_subset[:10] + tamp_subset[:10]
    test_set = auth_subset[10:] + tamp_subset[10:]

    print(f"\nDataset Split:")
    print(f"  Tuning Set: {len(tune_set)} images (10 authentic, 10 tampered)")
    print(f"  Test Set:   {len(test_set)} images (10 authentic, 10 tampered) [HELD-OUT]")

    # ── Step 1: Feature Extraction on Tuning Set ────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 1: TUNING SET PARAMETER SWEEP")
    print("=" * 65)

    tune_records = []
    for fpath, fname, true_label in tune_set:
        label_str = "TAMPERED" if true_label == 1 else "AUTHENTIC"
        t0 = time.time()
        res = analyze_image(fpath)
        dur = time.time() - t0
        tune_records.append({
            "fpath": fpath,
            "filename": fname,
            "true_label": true_label,
            "tampered_ratio": res["tampered_ratio"],
            "num_matches": res["num_matches"],
            "duration": dur,
        })
        print(f"  [Tune] {fname:25s} ({label_str:9s}) -> ratio={res['tampered_ratio']:.4f}, matches={res['num_matches']:3d} ({dur:.2f}s)")

    # Parameter grid sweep
    ratio_thresholds = [0.03, 0.05, 0.08, 0.10]
    min_matches_opts = [3, 4, 5, 8]

    best_acc = -1
    best_params = (0.08, 5)

    for rt in ratio_thresholds:
        for mm in min_matches_opts:
            correct = 0
            for r in tune_records:
                pred = 1 if (r["tampered_ratio"] > rt or r["num_matches"] >= mm) else 0
                if pred == r["true_label"]:
                    correct += 1
            acc = correct / len(tune_records)
            if acc > best_acc:
                best_acc = acc
                best_params = (rt, mm)

    print(f"\n  Optimal Tuned Parameters on Tuning Set:")
    print(f"    ratio_threshold = {best_params[0]}")
    print(f"    min_matches     = {best_params[1]}")
    print(f"    Tuning Accuracy = {best_acc:.2%}")

    # ── Step 2: Evaluation on Unseen Test Set ────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2: HELD-OUT TEST SET EVALUATION (Unseen Data)")
    print("=" * 65)

    best_rt, best_mm = best_params
    y_true = []
    y_pred = []
    test_rows = []
    test_times = []

    for fpath, fname, true_label in test_set:
        label_str = "TAMPERED" if true_label == 1 else "AUTHENTIC"
        t0 = time.time()
        res = analyze_image(fpath, ratio_threshold=best_rt, min_matches=best_mm)
        dur = time.time() - t0
        test_times.append(dur)

        pred_label = 1 if res["verdict"] == "TAMPERED" else 0
        y_true.append(true_label)
        y_pred.append(pred_label)

        status = "OK" if pred_label == true_label else "WRONG"
        print(f"  [Test] {fname:25s} ({label_str:9s}) -> {res['verdict']:9s} [{status:5s}] (matches={res['num_matches']:2d}, ratio={res['tampered_ratio']:.4f}, {dur:.2f}s)")

        test_rows.append({
            "filename": fname,
            "true_label": true_label,
            "predicted": pred_label,
            "verdict": res["verdict"],
            "tampered_ratio": round(res["tampered_ratio"], 4),
            "num_matches": res["num_matches"],
            "status": status,
        })

    # Metrics computation
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 65)
    print("  HELD-OUT TEST SET PERFORMANCE (Examiner-Credible Metrics)")
    print("=" * 65)
    print(f"  Total Test Images: {len(test_set)} (Balanced 10 Auth / 10 Tamp)")
    print(f"  Accuracy:          {acc:.2%}")
    print(f"  Precision:         {prec:.2%}")
    print(f"  Recall:            {rec:.2%}")
    print(f"  F1 Score:          {f1:.2%}")
    print(f"  False Pos. Rate:   {fp_rate:.2%} ({fp}/{fp+tn})")
    print(f"  Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Avg Time / Image:  {np.mean(test_times):.2f}s")
    print("=" * 65)

    # Save CSV
    csv_path = os.path.join(results_dir, "test_set_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "predicted", "verdict", "tampered_ratio", "num_matches", "status"])
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"\n  Saved: {csv_path}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0C0C0C")
    ax.set_facecolor("#0C0C0C")

    orange_cmap = LinearSegmentedColormap.from_list("black_orange", ["#0C0C0C", "#FF6F32"], N=256)
    disp = ConfusionMatrixDisplay(cm, display_labels=["AUTHENTIC", "TAMPERED"])
    disp.plot(ax=ax, cmap=orange_cmap, colorbar=False)

    ax.set_title("Test Set Confusion Matrix (MICC-F220)", color="#FFFFFF", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", color="#FFFFFF", fontsize=12)
    ax.set_ylabel("True", color="#FFFFFF", fontsize=12)
    ax.tick_params(colors="#FFFFFF")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#FFFFFF")
    for text in disp.text_.ravel():
        text.set_color("#FFFFFF")
        text.set_fontsize(16)
        text.set_fontweight("bold")

    plt.tight_layout()
    cm_path = os.path.join(results_dir, "test_set_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, facecolor="#0C0C0C", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {cm_path}")

if __name__ == "__main__":
    main()

