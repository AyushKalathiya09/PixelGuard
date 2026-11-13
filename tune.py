"""
tune.py  -  Find the best ratio_threshold and min_matches for the pipeline.

Runs analyze_image on all dataset images and sweeps parameter combinations
to find the highest accuracy.

Usage:
    python tune.py
"""

import os
import glob
import sys

from main import analyze_image


def collect_images(dataset_dir):
    """Collect all images with their true labels."""
    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    images = []

    for subfolder, label in [("authentic", "AUTHENTIC"), ("tampered", "TAMPERED")]:
        folder = os.path.join(dataset_dir, subfolder)
        if not os.path.isdir(folder):
            continue
        for ext in EXTS:
            for path in sorted(glob.glob(os.path.join(folder, ext))):
                images.append((path, label))

    return images


def run_sweep():
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
    images = collect_images(DATASET_DIR)

    if not images:
        print("[TUNE] No images found in dataset/")
        return

    print(f"[TUNE] Found {len(images)} images\n")

    # ── Step 1: Run analyze_image on all images (cache results) ──────
    # We only need tampered_ratio and num_matches per image; no need to
    # re-run the full pipeline for every threshold combo.
    print("Running pipeline on all images...\n")

    records = []
    for path, true_label in images:
        name = os.path.basename(path)
        print(f"  {name} ({true_label}) ... ", end="", flush=True)

        result = analyze_image(path)

        rec = {
            "filename":       name,
            "true_label":     true_label,
            "tampered_ratio": result["tampered_ratio"],
            "num_matches":    result["num_matches"],
        }
        records.append(rec)
        print(f"ratio={rec['tampered_ratio']:.4f}, matches={rec['num_matches']}")

    # ── Step 2: Print summary table ──────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Filename':<30} {'True Label':<12} {'Ratio':>8} {'Matches':>8}")
    print(f"{'-'*80}")
    for r in records:
        print(f"{r['filename']:<30} {r['true_label']:<12} "
              f"{r['tampered_ratio']:>8.4f} {r['num_matches']:>8}")
    print(f"{'='*80}\n")

    # ── Step 3: Sweep parameters ─────────────────────────────────────
    ratio_thresholds = [0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    min_matches_list = [5, 10, 15, 20]

    print(f"{'ratio_thresh':>14} {'min_match':>10} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print(f"{'-'*56}")

    best_acc = 0
    best_params = (0.01, 10)

    for rt in ratio_thresholds:
        for mm in min_matches_list:
            correct = 0
            total = len(records)

            for r in records:
                # Predict verdict with these thresholds
                if r["tampered_ratio"] > rt or r["num_matches"] >= mm:
                    predicted = "TAMPERED"
                else:
                    predicted = "AUTHENTIC"

                if predicted == r["true_label"]:
                    correct += 1

            acc = correct / total if total > 0 else 0
            marker = ""
            if acc > best_acc:
                best_acc = acc
                best_params = (rt, mm)
                marker = " <-- best"

            print(f"{rt:>14.3f} {mm:>10} {acc:>10.2%} {correct:>10} {total:>8}{marker}")

    # ── Step 4: Report best ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BEST PARAMETERS:")
    print(f"    ratio_threshold = {best_params[0]}")
    print(f"    min_matches     = {best_params[1]}")
    print(f"    accuracy        = {best_acc:.2%}")
    print(f"{'='*60}")
    print(f"\n  Update main.py defaults with these values.")

    return best_params, best_acc


if __name__ == "__main__":
    best_params, best_acc = run_sweep()

