"""
robustness.py  –  Robustness evaluation of the forgery-detection pipeline.

Creates adversarial "attacked" versions of every tampered image and checks
whether the pipeline still flags them as TAMPERED.

Attacks
-------
1. Gaussian noise   (sigma = 10)
2. JPEG re-compression (quality = 70)
3. Gaussian blur    (5×5 kernel)
4. Downscale        (0.8× in both dimensions)

Usage
-----
    python robustness.py
"""

import os
import sys
import glob
import tempfile
import traceback

import cv2
import numpy as np

from main import analyze_image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Attack functions – each returns a BGR numpy array
# ──────────────────────────────────────────────────────────────────────

def attack_gaussian_noise(img: np.ndarray, sigma: float = 10) -> np.ndarray:
    """Add Gaussian noise with the given standard deviation."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float64)
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def attack_jpeg_recompress(img: np.ndarray, quality: int = 70) -> np.ndarray:
    """Re-compress through JPEG at the given quality level."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed during re-compression attack")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def attack_gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply Gaussian blur with a ksize×ksize kernel."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def attack_downscale(img: np.ndarray, factor: float = 0.8) -> np.ndarray:
    """Resize to factor× of original dimensions (e.g. 0.8 = 80 %)."""
    h, w = img.shape[:2]
    new_w, new_h = int(w * factor), int(h * factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# Registry: (display_name, function)
ATTACKS = [
    ("Gaussian noise σ=10",     attack_gaussian_noise),
    ("JPEG Q=70 recompress",    attack_jpeg_recompress),
    ("Gaussian blur 5×5",       attack_gaussian_blur),
    ("Downscale 0.8×",          attack_downscale),
]


# ──────────────────────────────────────────────────────────────────────
# Helper – save a numpy image to a temp JPEG and return the path
# ──────────────────────────────────────────────────────────────────────

def _save_temp_jpeg(img: np.ndarray, prefix: str = "robustness_") -> str:
    """Write *img* (BGR) to a temporary JPEG file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".jpg", prefix=prefix)
    os.close(fd)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


# ──────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────

def run_robustness_check():
    # Locate tampered images
    tampered_dir = os.path.join(os.path.dirname(__file__), "dataset", "tampered")
    images = sorted(glob.glob(os.path.join(tampered_dir, "*.jpg")))
    if not images:
        # Also try png
        images = sorted(glob.glob(os.path.join(tampered_dir, "*.*")))
    if not images:
        print(f"[ERROR] No images found in {tampered_dir}")
        sys.exit(1)

    n_images = len(images)
    basenames = [os.path.basename(p) for p in images]

    print(f"\n{'='*72}")
    print(f"  ROBUSTNESS CHECK  –  {n_images} tampered image(s)")
    print(f"{'='*72}\n")

    # results[attack_name] = list of (basename, verdict, tampered_ratio)
    results: dict[str, list[tuple[str, str, float]]] = {}

    # Also run the baseline (no attack) for comparison
    all_conditions = [("No attack (baseline)", None)] + ATTACKS

    for attack_name, attack_fn in all_conditions:
        results[attack_name] = []
        print(f"  ▸ Running: {attack_name} …")

        for img_path, bname in zip(images, basenames):
            tmp_path = None
            try:
                original = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if original is None:
                    raise FileNotFoundError(f"Cannot read {img_path}")

                if attack_fn is not None:
                    attacked = attack_fn(original)
                    tmp_path = _save_temp_jpeg(attacked, prefix=f"rob_{bname}_")
                    test_path = tmp_path
                else:
                    test_path = img_path

                res = analyze_image(test_path)
                verdict = res["verdict"]
                t_ratio = res["tampered_ratio"]

            except Exception as exc:
                print(f"    [WARN] {bname}: {exc}")
                traceback.print_exc()
                verdict = "ERROR"
                t_ratio = 0.0

            finally:
                # Clean up temp file
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            results[attack_name].append((bname, verdict, t_ratio))

    # ── Print results table ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  RESULTS TABLE")
    print(f"{'='*72}\n")

    # Header
    col_w = 26
    img_w = 18
    header = f"{'Attack':<{col_w}}"
    for bname in basenames:
        short = bname[:img_w - 2]
        header += f" | {short:<{img_w}}"
    header += f" | {'Det. Rate':>10}"
    print(header)
    print("-" * len(header))

    # Detection rate summary for final analysis
    detection_rates: dict[str, float] = {}

    for attack_name, _ in all_conditions:
        row_data = results[attack_name]
        detected = sum(1 for _, v, _ in row_data if v == "TAMPERED")
        rate = detected / n_images if n_images > 0 else 0.0
        detection_rates[attack_name] = rate

        row = f"{attack_name:<{col_w}}"
        for bname, verdict, t_ratio in row_data:
            cell = f"{verdict[:3]} ({t_ratio:.2%})"
            row += f" | {cell:<{img_w}}"
        row += f" | {rate:>9.0%} "
        print(row)

    print("-" * len(header))

    # ── Detailed per-image breakdown ─────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PER-IMAGE DETAIL")
    print(f"{'='*72}\n")

    for attack_name, _ in all_conditions:
        print(f"  {attack_name}:")
        for bname, verdict, t_ratio in results[attack_name]:
            status = "✓" if verdict == "TAMPERED" else "✗"
            print(f"    {status}  {bname:<20s}  →  {verdict:<10s}  (ratio {t_ratio:.4f})")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}\n")

    baseline_rate = detection_rates.get("No attack (baseline)", 1.0)
    print(f"  Baseline detection rate : {baseline_rate:.0%}\n")

    # Find worst attack (excluding baseline)
    attack_only = {k: v for k, v in detection_rates.items() if k != "No attack (baseline)"}
    if attack_only:
        worst_name = min(attack_only, key=attack_only.get)
        worst_rate = attack_only[worst_name]
        best_name  = max(attack_only, key=attack_only.get)
        best_rate  = attack_only[best_name]

        print(f"  Most robust against     : {best_name}  ({best_rate:.0%} detection)")
        print(f"  Most vulnerable to      : {worst_name}  ({worst_rate:.0%} detection)")
        print()

        # Explain why the worst attack is damaging
        drop = baseline_rate - worst_rate
        if drop > 0:
            print(f"  ⚠  '{worst_name}' drops detection by {drop:.0%}.")
            if "blur" in worst_name.lower():
                print("     Blurring smooths out ELA artefacts, reducing high-frequency")
                print("     residuals that the pipeline relies on to spot tampering.")
            elif "noise" in worst_name.lower():
                print("     Additive noise overwhelms the subtle ELA differences left by")
                print("     JPEG double-compression, making tampered regions blend in.")
            elif "jpeg" in worst_name.lower() or "recompress" in worst_name.lower():
                print("     Re-compressing at a lower quality uniformises compression")
                print("     artefacts across the image, masking the double-compression")
                print("     signatures that ELA is designed to detect.")
            elif "downscale" in worst_name.lower() or "scale" in worst_name.lower():
                print("     Downscaling reduces spatial resolution, destroying fine-grained")
                print("     copy-move keypoints and shrinking tampered regions below the")
                print("     minimum-area threshold used for segmentation.")
            else:
                print("     The attack likely disrupts the features the pipeline uses.")
        else:
            print("  ✓  All attacks still achieve baseline detection – pipeline is robust!")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    run_robustness_check()
