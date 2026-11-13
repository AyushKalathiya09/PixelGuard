"""
ela.py  –  Error Level Analysis (ELA) for image forgery detection.

ELA works by re-saving an image as JPEG and comparing it to the original.
Regions that were tampered (pasted from another image) have different
compression artefacts, so they appear *brighter* in the difference image.

Functions
---------
compute_ela  : Produce a colour ELA image (amplified difference).
ela_to_gray  : Convert the ELA image to a blurred grayscale heatmap.
"""

import io
import os
import glob

import cv2
import numpy as np
from PIL import Image

# Import our shared helpers
from utils import load_image, show_images


# ──────────────────────────────────────────────────────────────────────
# 1.  compute_ela
# ──────────────────────────────────────────────────────────────────────
def compute_ela(
    image_path: str,
    quality: int = 90,
    scale: int = 15,
) -> np.ndarray:
    """
    Compute the Error Level Analysis (ELA) of an image.

    Steps
    -----
    1. Open the original image with Pillow.
    2. Re-save it as JPEG into an in-memory buffer at *quality*.
    3. Re-open the buffer → this is the "re-compressed" version.
    4. diff = |original − re-compressed|   (per pixel, per channel)
    5. Amplify:  ela = clip(diff × scale, 0, 255)

    Parameters
    ----------
    image_path : str
        Path to the source image file.
    quality    : int, optional
        JPEG quality for the re-save step (1–100). Default 90.
    scale      : int, optional
        Multiplication factor to make small differences visible. Default 15.

    Returns
    -------
    np.ndarray
        ELA image in **RGB** colour space, dtype uint8, same size as input.
    """
    # --- Step 1: Open original with Pillow ---
    original = Image.open(image_path).convert("RGB")

    # --- Step 2: Re-save as JPEG in memory ---
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)                              # rewind to the start

    # --- Step 3: Re-open the re-compressed image ---
    resaved = Image.open(buffer).convert("RGB")

    # --- Step 4: Absolute difference ---
    orig_arr   = np.array(original, dtype=np.float32)
    resav_arr  = np.array(resaved,  dtype=np.float32)
    diff       = np.abs(orig_arr - resav_arr)

    # --- Step 5: Amplify & clip to 0-255 ---
    ela_img = np.clip(diff * scale, 0, 255).astype(np.uint8)

    return ela_img                              # RGB, uint8


# ──────────────────────────────────────────────────────────────────────
# 2.  ela_to_gray
# ──────────────────────────────────────────────────────────────────────
def ela_to_gray(ela_img: np.ndarray) -> np.ndarray:
    """
    Convert a colour ELA image to a blurred, normalised grayscale map.

    Steps
    -----
    1. Convert RGB → grayscale.
    2. Gaussian blur (5×5 kernel) to reduce noise.
    3. Normalize pixel range to 0-255.

    Parameters
    ----------
    ela_img : np.ndarray
        Colour ELA image (RGB, uint8) from compute_ela().

    Returns
    -------
    np.ndarray
        Single-channel grayscale heatmap (uint8, 0–255).
    """
    # Step 1: RGB → gray
    gray = cv2.cvtColor(ela_img, cv2.COLOR_RGB2GRAY)

    # Step 2: Gaussian blur to smooth out pixel-level noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Normalize to use the full 0-255 range
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    return gray.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# Self-test: run ELA on every image in dataset/
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "ela_test")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Supported image extensions
    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    # Scan both authentic/ and tampered/ subfolders
    for subfolder in ("authentic", "tampered"):
        folder = os.path.join(DATASET_DIR, subfolder)
        if not os.path.isdir(folder):
            print(f"[ELA] Skipping (folder not found): {folder}")
            continue

        # Collect all image paths in this subfolder
        image_paths = []
        for ext in EXTS:
            image_paths.extend(glob.glob(os.path.join(folder, ext)))

        if not image_paths:
            print(f"[ELA] No images found in {folder}")
            continue

        print(f"\n[ELA] Processing {len(image_paths)} image(s) in '{subfolder}/'")

        for img_path in sorted(image_paths):
            name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"  -> {name}")

            # Load original as RGB for display
            original = load_image(img_path, color="rgb")

            # Compute ELA
            ela_img  = compute_ela(img_path, quality=90, scale=15)

            # Convert to grayscale heatmap
            gray_map = ela_to_gray(ela_img)

            # Save a 3-panel comparison figure
            save_name = f"{subfolder}_{name}.png"
            save_path = os.path.join(RESULTS_DIR, save_name)
            show_images(
                images=[original, ela_img, gray_map],
                titles=["Original", "ELA (colour)", "ELA (grayscale)"],
                save_path=save_path,
            )

    print(f"\n[ELA] Done — results saved to {RESULTS_DIR}")
