"""
run_known_answer_test.py  –  Known-answer test for forgery detection.

1. Runs analyze_image on all synthetic forgeries AND original authentic images
2. Reports:
   - How many forgeries were detected (verdict == TAMPERED)
   - How many originals were wrongly flagged (false positives)
   - IoU between predicted mask and ground-truth mask for each forgery
3. Prints a clear table with results
"""

import os
import glob
import sys

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import analyze_image
from evaluate import pixel_metrics


BASE     = os.path.dirname(os.path.abspath(__file__))
AUTH_DIR = os.path.join(BASE, "dataset", "authentic")
SYN_DIR  = os.path.join(BASE, "dataset", "synthetic")
GT_DIR   = os.path.join(BASE, "dataset", "synthetic_gt")


def find_images(folder, exts=("*.jpg", "*.jpeg", "*.png", "*.bmp")):
    """Collect all image paths in a folder."""
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def compute_iou(pred_mask, gt_mask):
    """Compute IoU between predicted and ground-truth masks."""
    # Resize pred to match gt if needed
    if pred_mask.shape[:2] != gt_mask.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    pred = pred_mask > 127
    gt   = gt_mask > 127
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    return intersection / union if union > 0 else 0.0


def main():
    print("=" * 78)
    print("  KNOWN-ANSWER TEST  –  Forgery Detection Pipeline")
    print("=" * 78)

    # ── 1. Test authentic images (expect AUTHENTIC, no false positives) ──
    print("\n" + "─" * 78)
    print("  PART 1: AUTHENTIC IMAGES (expect AUTHENTIC verdict)")
    print("─" * 78)

    auth_paths = find_images(AUTH_DIR)
    if not auth_paths:
        print("  [ERROR] No authentic images found!")
        return

    auth_results = []
    for path in auth_paths:
        name = os.path.basename(path)
        print(f"\n  Analyzing: {name} ... ", end="", flush=True)
        result = analyze_image(path)
        verdict = result["verdict"]
        is_fp = verdict == "TAMPERED"
        status = "FALSE POSITIVE!" if is_fp else "OK"
        print(f"{verdict}  [{status}]")
        auth_results.append({
            "filename": name,
            "verdict": verdict,
            "tampered_ratio": result["tampered_ratio"],
            "num_matches": result["num_matches"],
            "false_positive": is_fp,
        })

    # ── 2. Test synthetic forgeries (expect TAMPERED) ──
    print("\n" + "─" * 78)
    print("  PART 2: SYNTHETIC FORGERIES (expect TAMPERED verdict)")
    print("─" * 78)

    syn_paths = find_images(SYN_DIR)
    if not syn_paths:
        print("  [ERROR] No synthetic forgeries found! Run make_test_forgeries.py first.")
        return

    forgery_results = []
    for path in syn_paths:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        print(f"\n  Analyzing: {name} ... ", end="", flush=True)

        result = analyze_image(path)
        verdict = result["verdict"]
        detected = verdict == "TAMPERED"

        # Load ground-truth mask
        gt_path = os.path.join(GT_DIR, stem + ".png")
        iou = None
        px_prec = None
        px_rec = None
        px_f1 = None

        if os.path.isfile(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                pred_mask = result["final_mask"]
                iou = compute_iou(pred_mask, gt_mask)
                px = pixel_metrics(pred_mask, gt_mask)
                px_prec = px["px_precision"]
                px_rec = px["px_recall"]
                px_f1 = px["px_f1"]

        status = "DETECTED" if detected else "MISSED"
        iou_str = f"IoU={iou:.4f}" if iou is not None else "IoU=N/A"
        print(f"{verdict}  [{status}]  {iou_str}")

        forgery_results.append({
            "filename": name,
            "verdict": verdict,
            "detected": detected,
            "tampered_ratio": result["tampered_ratio"],
            "num_matches": result["num_matches"],
            "iou": iou,
            "px_precision": px_prec,
            "px_recall": px_rec,
            "px_f1": px_f1,
        })

    # ── 3. Print summary table ──
    print("\n\n")
    print("=" * 78)
    print("  RESULTS SUMMARY")
    print("=" * 78)

    # Authentic table
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  AUTHENTIC IMAGES                                          │")
    print("  ├───────────────────────┬──────────┬───────────┬─────────────┤")
    print("  │ Filename              │ Verdict  │ Tamp.Rat  │ CM Matches  │")
    print("  ├───────────────────────┼──────────┼───────────┼─────────────┤")
    for r in auth_results:
        fn = r["filename"][:21].ljust(21)
        vd = r["verdict"].ljust(8)
        tr = f"{r['tampered_ratio']:.4f}".ljust(9)
        cm = str(r["num_matches"]).ljust(11)
        print(f"  │ {fn} │ {vd} │ {tr} │ {cm} │")
    print("  └───────────────────────┴──────────┴───────────┴─────────────┘")

    n_fp = sum(1 for r in auth_results if r["false_positive"])
    print(f"\n  False Positives: {n_fp} / {len(auth_results)}")

    # Forgery table
    print("\n  ┌────────────────────────────────────────────────────────────────────────────┐")
    print("  │  SYNTHETIC FORGERIES                                                       │")
    print("  ├──────────────────────────────┬──────────┬──────────┬────────┬───────┬───────┤")
    print("  │ Filename                     │ Verdict  │ Detected │  IoU   │ Prec  │ Rec   │")
    print("  ├──────────────────────────────┼──────────┼──────────┼────────┼───────┼───────┤")
    for r in forgery_results:
        fn = r["filename"][:28].ljust(28)
        vd = r["verdict"].ljust(8)
        det = ("YES" if r["detected"] else "NO").ljust(8)
        iou_s = f"{r['iou']:.4f}" if r["iou"] is not None else "N/A   "
        prec_s = f"{r['px_precision']:.3f}" if r["px_precision"] is not None else "N/A  "
        rec_s = f"{r['px_recall']:.3f}" if r["px_recall"] is not None else "N/A  "
        print(f"  │ {fn} │ {vd} │ {det} │ {iou_s} │ {prec_s} │ {rec_s} │")
    print("  └──────────────────────────────┴──────────┴──────────┴────────┴───────┴───────┘")

    n_detected = sum(1 for r in forgery_results if r["detected"])
    n_total = len(forgery_results)
    print(f"\n  Detected: {n_detected} / {n_total} forgeries")

    # IoU breakdown by type
    print("\n  IoU by forgery type:")
    for ftype in ["copymove", "rotated30", "scaled12", "spliced"]:
        ious = [r["iou"] for r in forgery_results
                if ftype in r["filename"] and r["iou"] is not None]
        if ious:
            avg_iou = np.mean(ious)
            print(f"    {ftype:12s}:  avg IoU = {avg_iou:.4f}  "
                  f"(range: {min(ious):.4f} – {max(ious):.4f})")

    # Overall verdict
    print(f"\n{'='*78}")
    all_ious = [r["iou"] for r in forgery_results if r["iou"] is not None]
    avg_iou = np.mean(all_ious) if all_ious else 0
    cm_ious = [r["iou"] for r in forgery_results
               if "copymove" in r["filename"] and r["iou"] is not None]
    avg_cm_iou = np.mean(cm_ious) if cm_ious else 0

    detection_rate = n_detected / n_total if n_total > 0 else 0
    fp_rate = n_fp / len(auth_results) if auth_results else 0

    print(f"  Detection Rate:     {detection_rate:.0%}  ({n_detected}/{n_total})")
    print(f"  False Positive Rate:{fp_rate:.0%}  ({n_fp}/{len(auth_results)})")
    print(f"  Average IoU (all):  {avg_iou:.4f}")
    print(f"  Average IoU (copy-move): {avg_cm_iou:.4f}")

    # Pass/Fail criteria
    print(f"\n  PASS CRITERIA:")
    pass_detection = detection_rate >= 0.5
    pass_fp = fp_rate <= 0.25
    pass_iou = avg_cm_iou >= 0.3

    checks = [
        ("Most forgeries detected (≥50%)", pass_detection),
        ("Originals rarely flagged (FP ≤ 25%)", pass_fp),
        ("Copy-move IoU ≥ 0.3", pass_iou),
    ]
    all_pass = True
    for desc, passed in checks:
        icon = "PASS ✓" if passed else "FAIL ✗"
        print(f"    [{icon}]  {desc}")
        if not passed:
            all_pass = False

    print(f"\n  OVERALL: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
