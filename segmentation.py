"""
segmentation.py  -  Segment tampered regions from ELA / copy-move masks.

Converts the fuzzy ELA grayscale heatmap into clean binary masks, combines
with copy-move masks, and provides overlay / bounding-box visualization.

Functions
---------
segment_ela    : Otsu + morphology + contour filtering -> binary mask.
combine_masks  : Bitwise OR of ELA mask and copy-move mask.
overlay_mask   : Semi-transparent colored overlay on the original image.
draw_bboxes    : Green bounding boxes around each detected region.
"""

import os
import glob

import cv2
import numpy as np

from utils import load_image, show_images
from ela import compute_ela, ela_to_gray


# ──────────────────────────────────────────────────────────────────────
# 1.  segment_ela
# ──────────────────────────────────────────────────────────────────────
def segment_ela(ela_gray: np.ndarray, min_area: int = 500, use_adaptive: bool = True) -> np.ndarray:
    """
    Convert an ELA grayscale heatmap into a clean binary mask.

    Steps
    -----
    1. Thresholding: Adaptive (mean + 2.5*std) balanced sigma thresholding to
       suppress JPEG texture noise on authentic images while catching genuine
       tampered regions, falling back to Otsu if uniform.
    2. Morphological opening (5x5) - removes small noise dots.
    3. Morphological closing (15x15) - fills holes inside regions.
    4. Contour filtering - removes regions smaller than min_area pixels.

    Parameters
    ----------
    ela_gray     : np.ndarray
        Single-channel grayscale ELA map (uint8, 0-255).
    min_area     : int
        Minimum contour area in pixels. Smaller regions are removed.
    use_adaptive : bool
        If True, use mean + 2*std adaptive thresholding; else Otsu.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255).
    """
    if use_adaptive:
        mean_val = float(np.mean(ela_gray))
        std_val = float(np.std(ela_gray))
        # 2.5-sigma threshold: conservative enough to suppress JPEG texture noise
        # on clean images, while still catching genuine tampered regions.
        thresh = min(max(mean_val + 2.5 * std_val, 30.0), 220.0)
        _, binary = cv2.threshold(ela_gray, int(thresh), 255, cv2.THRESH_BINARY)
    else:
        # Otsu threshold
        _, binary = cv2.threshold(ela_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Morphological opening (erode then dilate) to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Step 3: Morphological closing (dilate then erode) to fill holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Step 4: Find contours and remove small ones
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a clean mask with only large-enough regions
    clean_mask = np.zeros_like(binary)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)

    return clean_mask


# ──────────────────────────────────────────────────────────────────────
# 2.  combine_masks
# ──────────────────────────────────────────────────────────────────────
def combine_masks(ela_mask: np.ndarray, cm_mask: np.ndarray) -> np.ndarray:
    """
    Combine ELA and copy-move masks using bitwise OR.

    If the masks have different sizes, cm_mask is resized to match ela_mask.

    Parameters
    ----------
    ela_mask : np.ndarray
        Binary mask from segment_ela (uint8, 0 or 255).
    cm_mask  : np.ndarray
        Binary mask from copy-move detection (uint8, 0 or 255).

    Returns
    -------
    np.ndarray
        Combined binary mask (uint8, 0 or 255).
    """
    # Resize cm_mask if dimensions differ
    if ela_mask.shape[:2] != cm_mask.shape[:2]:
        cm_mask = cv2.resize(cm_mask, (ela_mask.shape[1], ela_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    return cv2.bitwise_or(ela_mask, cm_mask)


# ──────────────────────────────────────────────────────────────────────
# 3.  overlay_mask
# ──────────────────────────────────────────────────────────────────────
def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 111, 50),   # #FF6F32 orange accent
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw a semi-transparent colored overlay on the detected regions.

    Parameters
    ----------
    image : np.ndarray
        Original image (BGR, uint8).
    mask  : np.ndarray
        Binary mask (uint8, 0 or 255).
    color : tuple of int
        BGR color for the overlay. Default is orange #FF6F32 = (50,111,255) BGR.
    alpha : float
        Transparency (0 = invisible, 1 = fully opaque). Default 0.4.

    Returns
    -------
    np.ndarray
        BGR image with overlay applied.
    """
    # Resize mask if needed
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    result = image.copy()

    # Create a solid color layer
    overlay_layer = np.full_like(image, color, dtype=np.uint8)

    # Apply overlay only where mask is white (255)
    mask_bool = mask > 0
    if np.any(mask_bool):
        blended = cv2.addWeighted(
            image[mask_bool].reshape(-1, 3), 1 - alpha,
            overlay_layer[mask_bool].reshape(-1, 3), alpha,
            0,
        )
        if blended is not None:
            result[mask_bool] = blended.reshape(-1, 3)

    return result


# ──────────────────────────────────────────────────────────────────────
# 4.  draw_bboxes
# ──────────────────────────────────────────────────────────────────────
def draw_bboxes(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),   # green
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes around each contour in the mask.

    Parameters
    ----------
    image     : np.ndarray
        Original image (BGR, uint8).
    mask      : np.ndarray
        Binary mask (uint8, 0 or 255).
    color     : tuple of int
        BGR color for the boxes. Default green.
    thickness : int
        Line thickness. Default 2.

    Returns
    -------
    np.ndarray
        BGR image with bounding boxes drawn.
    """
    # Resize mask if needed
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    result = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), color, thickness)

    return result


# ──────────────────────────────────────────────────────────────────────
# Self-test: ELA -> segment -> overlay pipeline
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "seg_test")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    # Test on tampered images
    folder = os.path.join(DATASET_DIR, "tampered")
    image_paths = []
    for ext in EXTS:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))

    if not image_paths:
        print("[SEG] No tampered images found.")
    else:
        print(f"[SEG] Processing {len(image_paths)} tampered image(s)\n")

        for img_path in sorted(image_paths):
            name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"  Image: {name}")

            # Load original as BGR for overlays, and RGB for display
            img_bgr = load_image(img_path, color="bgr")
            img_rgb = load_image(img_path, color="rgb")

            # ELA pipeline
            ela_img = compute_ela(img_path, quality=90, scale=15)
            gray_map = ela_to_gray(ela_img)

            # Segment
            binary_mask = segment_ela(gray_map, min_area=500)

            # Count white pixels in mask
            white_px = cv2.countNonZero(binary_mask)
            total_px = binary_mask.shape[0] * binary_mask.shape[1]
            pct = 100.0 * white_px / total_px
            print(f"    Mask coverage: {white_px} px ({pct:.1f}%)")

            # Overlay (orange, using BGR for overlay function)
            overlay_bgr = overlay_mask(img_bgr, binary_mask,
                                       color=(50, 111, 255), alpha=0.4)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            # Bounding boxes
            bbox_bgr = draw_bboxes(img_bgr, binary_mask)
            bbox_rgb = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB)

            # Save 4-panel figure
            save_path = os.path.join(RESULTS_DIR, f"{name}.png")
            show_images(
                images=[img_rgb, gray_map, binary_mask, overlay_rgb],
                titles=["Original", "ELA Grayscale", "Binary Mask", "Overlay"],
                save_path=save_path,
            )

    print(f"\n[SEG] Done - results saved to {RESULTS_DIR}")
