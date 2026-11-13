"""
main.py  -  Full forgery-detection pipeline and CLI entry point.

Chains ELA -> Copy-Move -> Segmentation -> Decision (TAMPERED / AUTHENTIC).

Functions
---------
analyze_image : Run the complete pipeline on a single image.

CLI usage:
    python main.py --image path/to/image.jpg
"""

import os
import argparse

import cv2
import numpy as np

from utils import load_image, save_image, show_images
from ela import compute_ela, ela_to_gray
from copy_move import detect_copy_move
from segmentation import segment_ela, combine_masks, overlay_mask, draw_bboxes


# ──────────────────────────────────────────────────────────────────────
# analyze_image  –  the master function
# ──────────────────────────────────────────────────────────────────────
def analyze_image(
    path: str,
    ela_quality: int = 90,
    ratio: float = 0.6,
    min_matches: int = 5,
    ratio_threshold: float = 0.075,
) -> dict:
    """
    Run the full forgery-detection pipeline on a single image.

    Parameters
    ----------
    path             : str   – Path to the image file.
    ela_quality      : int   – JPEG re-save quality for ELA (1-100).
    ratio            : float – Lowe's ratio for copy-move matching.
    min_matches      : int   – Min copy-move matches to flag tampered.
    ratio_threshold  : float – Min tampered-pixel ratio to flag tampered (default 0.075).

    Returns
    -------
    dict with keys:
        original, ela_image, ela_mask, cm_vis, cm_mask,
        final_mask, overlay, bbox_image, verdict, tampered_ratio, num_matches
    """
    # ── 1. ELA pipeline ──────────────────────────────────────────────
    ela_image = compute_ela(path, quality=ela_quality, scale=15)
    ela_gray  = ela_to_gray(ela_image)
    ela_mask  = segment_ela(ela_gray, min_area=500)

    # Flood guard: if ELA mask covers > 50% of the image, texture flooded the ELA residual
    ela_total = ela_mask.shape[0] * ela_mask.shape[1]
    if ela_total > 0 and (cv2.countNonZero(ela_mask) / ela_total) > 0.50:
        ela_mask = np.zeros_like(ela_mask)

    # ── 1b. Cold-spot ELA (inverse anomaly) ──────────────────────────
    # In highly textured images the ELA heatmap is globally elevated.
    # A spliced-in smooth patch appears as a region with LOWER ELA than its
    # surroundings (a "cold spot").  Detect this by inverting the ELA gray
    # and segmenting below-mean regions — but only when the image has a
    # high baseline ELA (mean > 60) so we don't fire on clean low-ELA images.
    ela_mean = float(np.mean(ela_gray))
    cold_spot_detected = False
    if ela_mean > 60.0 and cv2.countNonZero(ela_mask) == 0:
        ela_inv = cv2.bitwise_not(ela_gray)   # 255 where ELA is low
        cold_mask = segment_ela(ela_inv, min_area=500)
        cold_total = cold_mask.shape[0] * cold_mask.shape[1]
        # Accept cold-spot mask only if it covers a plausible tampered fraction
        cold_ratio = cv2.countNonZero(cold_mask) / cold_total if cold_total > 0 else 0.0
        if 0.01 < cold_ratio < 0.40:
            ela_mask = cv2.bitwise_or(ela_mask, cold_mask)
            cold_spot_detected = True   # bypass ratio threshold in decision

    # ── 2. Copy-Move pipeline ────────────────────────────────────────
    matched_pairs, cm_mask, cm_vis = detect_copy_move(
        path, ratio=ratio, min_matches=min_matches
    )

    # ── 3. Combine masks ─────────────────────────────────────────────
    final_mask = combine_masks(ela_mask, cm_mask)

    # ── 4. Compute tampered ratio ────────────────────────────────────
    total_pixels = final_mask.shape[0] * final_mask.shape[1]
    white_pixels = cv2.countNonZero(final_mask)
    tampered_ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0

    # ── 5. Decision ──────────────────────────────────────────────────
    num_matches = len(matched_pairs)
    if tampered_ratio > ratio_threshold or num_matches >= min_matches or cold_spot_detected:
        verdict = "TAMPERED"
    else:
        verdict = "AUTHENTIC"

    # ── 6. Overlay & bounding boxes ──────────────────────────────────
    original_bgr = load_image(path, color="bgr")
    original_rgb = load_image(path, color="rgb")

    # Orange overlay in BGR: #FF6F32 = (50, 111, 255) BGR
    overlay_bgr  = overlay_mask(original_bgr, final_mask,
                                color=(50, 111, 255), alpha=0.4)
    overlay_rgb  = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    bbox_bgr     = draw_bboxes(original_bgr, final_mask,
                               color=(0, 255, 0), thickness=2)
    bbox_rgb     = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB)

    return {
        "original":       original_rgb,
        "ela_image":      ela_image,        # RGB
        "ela_gray":       ela_gray,         # grayscale
        "ela_mask":       ela_mask,         # binary
        "cm_vis":         cm_vis,           # RGB
        "cm_mask":        cm_mask,          # binary
        "final_mask":     final_mask,       # binary
        "overlay":        overlay_rgb,      # RGB
        "bbox_image":     bbox_rgb,         # RGB
        "verdict":        verdict,
        "tampered_ratio": tampered_ratio,
        "num_matches":    num_matches,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Digital Image Tampering & Forgery Detection"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--ela-quality", type=int, default=90)
    parser.add_argument("--ratio", type=float, default=0.6)
    parser.add_argument("--min-matches", type=int, default=5)
    parser.add_argument("--ratio-threshold", type=float, default=0.075)
    args = parser.parse_args()

    # Run analysis
    print(f"\n{'='*60}")
    print(f"  Analyzing: {args.image}")
    print(f"{'='*60}\n")

    result = analyze_image(
        args.image,
        ela_quality=args.ela_quality,
        ratio=args.ratio,
        min_matches=args.min_matches,
        ratio_threshold=args.ratio_threshold,
    )

    # Print verdict
    v = result["verdict"]
    r = result["tampered_ratio"]
    m = result["num_matches"]
    print(f"\n{'='*60}")
    print(f"  VERDICT:         {v}")
    print(f"  Tampered Ratio:  {r:.4f} ({r*100:.2f}%)")
    print(f"  Copy-Move Matches: {m}")
    print(f"{'='*60}\n")

    # Save outputs
    name = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(os.path.dirname(__file__), "results", name)
    os.makedirs(out_dir, exist_ok=True)

    show_images(
        [result["original"], result["ela_image"], result["ela_gray"]],
        ["Original", "ELA (Color)", "ELA (Grayscale)"],
        save_path=os.path.join(out_dir, "ela.png"),
    )
    show_images(
        [result["original"], result["cm_vis"], result["cm_mask"]],
        ["Original", f"Copy-Move ({m} matches)", "CM Mask"],
        save_path=os.path.join(out_dir, "copy_move.png"),
    )
    show_images(
        [result["ela_mask"], result["cm_mask"], result["final_mask"]],
        ["ELA Mask", "CM Mask", "Final Mask"],
        save_path=os.path.join(out_dir, "masks.png"),
    )
    show_images(
        [result["original"], result["overlay"], result["bbox_image"]],
        ["Original", f"Overlay [{v}]", "Bounding Boxes"],
        save_path=os.path.join(out_dir, "result.png"),
    )

    print(f"  Outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
