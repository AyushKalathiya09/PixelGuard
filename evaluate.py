"""
evaluate.py  -  Evaluation metrics for the forgery-detection pipeline.

Computes image-level accuracy / precision / recall / F1 and a confusion
matrix, plus optional pixel-level metrics when ground-truth masks exist.

Outputs:
    results/results.csv           - per-image results
    results/confusion_matrix.png  - visual confusion matrix

Usage:
    python evaluate.py
"""

import os
import glob
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

from main import analyze_image


# ──────────────────────────────────────────────────────────────────────
# Pixel-level metrics
# ──────────────────────────────────────────────────────────────────────
def pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute pixel-level precision, recall, F1, and IoU.

    Parameters
    ----------
    pred_mask : np.ndarray  – Predicted binary mask (0 or 255).
    gt_mask   : np.ndarray  – Ground-truth binary mask (0 or 255).

    Returns
    -------
    dict with keys: px_precision, px_recall, px_f1, px_iou
    """
    # Resize if needed
    if pred_mask.shape[:2] != gt_mask.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    # Convert to boolean
    pred = pred_mask > 127
    gt   = gt_mask > 127

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "px_precision": round(precision, 4),
        "px_recall":    round(recall, 4),
        "px_f1":        round(f1, 4),
        "px_iou":       round(iou, 4),
    }


# ──────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────
def evaluate_dataset():
    BASE        = os.path.dirname(__file__)
    DATASET_DIR = os.path.join(BASE, "dataset")
    GT_DIR      = os.path.join(DATASET_DIR, "ground_truth")
    RESULTS_DIR = os.path.join(BASE, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    # Collect images
    images = []
    for subfolder, label in [("authentic", 0), ("tampered", 1)]:
        folder = os.path.join(DATASET_DIR, subfolder)
        if not os.path.isdir(folder):
            continue
        for ext in EXTS:
            for path in sorted(glob.glob(os.path.join(folder, ext))):
                images.append((path, label))

    if not images:
        print("[EVAL] No images found.")
        return

    print(f"[EVAL] Evaluating {len(images)} images\n")

    # Run pipeline on each image
    y_true = []
    y_pred = []
    rows = []

    for path, true_label in images:
        name = os.path.basename(path)
        label_str = "TAMPERED" if true_label == 1 else "AUTHENTIC"
        print(f"  {name} ({label_str}) ... ", end="", flush=True)

        result = analyze_image(path)

        pred_label = 1 if result["verdict"] == "TAMPERED" else 0
        y_true.append(true_label)
        y_pred.append(pred_label)

        row = {
            "filename":       name,
            "true_label":     true_label,
            "predicted":      pred_label,
            "verdict":        result["verdict"],
            "tampered_ratio": round(result["tampered_ratio"], 4),
            "num_matches":    result["num_matches"],
        }

        # Check for ground-truth mask
        stem = os.path.splitext(name)[0]
        gt_candidates = [
            os.path.join(GT_DIR, stem + ext)
            for ext in (".png", ".jpg", ".bmp", ".tif")
        ]
        gt_path = None
        for gp in gt_candidates:
            if os.path.isfile(gp):
                gt_path = gp
                break

        if gt_path is not None:
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                px = pixel_metrics(result["final_mask"], gt_mask)
                row.update(px)
                print(f"IoU={px['px_iou']:.3f} ", end="")

        correct = "OK" if pred_label == true_label else "WRONG"
        print(f"-> {result['verdict']} [{correct}]")
        rows.append(row)

    # ── Image-level metrics ──────────────────────────────────────────
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc  = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    rec  = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1   = f1_score(y_true_arr, y_pred_arr, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  IMAGE-LEVEL METRICS")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  Precision: {prec:.2%}")
    print(f"  Recall:    {rec:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print(f"{'='*50}")

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    fieldnames = [
        "filename", "true_label", "predicted", "verdict",
        "tampered_ratio", "num_matches",
        "px_precision", "px_recall", "px_f1", "px_iou",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    # ── Confusion Matrix Plot (dark theme, orange accent) ────────────
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0C0C0C")
    ax.set_facecolor("#0C0C0C")

    # Custom colormap: black to orange
    from matplotlib.colors import LinearSegmentedColormap
    orange_cmap = LinearSegmentedColormap.from_list(
        "black_orange", ["#0C0C0C", "#FF6F32"], N=256
    )

    disp = ConfusionMatrixDisplay(cm, display_labels=["AUTHENTIC", "TAMPERED"])
    disp.plot(ax=ax, cmap=orange_cmap, colorbar=False)

    # Style text and labels
    ax.set_title("Confusion Matrix", color="#FFFFFF", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", color="#FFFFFF", fontsize=12)
    ax.set_ylabel("True", color="#FFFFFF", fontsize=12)
    ax.tick_params(colors="#FFFFFF")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#FFFFFF")

    # Make cell text white
    for text in disp.text_.ravel():
        text.set_color("#FFFFFF")
        text.set_fontsize(16)
        text.set_fontweight("bold")

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, facecolor="#0C0C0C", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {cm_path}")


if __name__ == "__main__":
    evaluate_dataset()
