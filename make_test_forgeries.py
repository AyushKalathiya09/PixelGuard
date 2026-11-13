"""
make_test_forgeries.py  –  Generate synthetic forgeries for known-answer testing.

For each authentic image, creates:
  1. Copy-move forgery (120×120 patch pasted at different location)
  2. Rotated copy-move (patch rotated 30°)
  3. Scaled copy-move (patch scaled 1.2×)
  4. Splicing forgery (patch from a DIFFERENT image)

Saves forged images as JPEG in dataset/synthetic/
Saves ground-truth masks as PNG  in dataset/synthetic_gt/
"""

import os
import random
import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
AUTH_DIR = os.path.join(BASE, "dataset", "authentic")
SYN_DIR  = os.path.join(BASE, "dataset", "synthetic")
GT_DIR   = os.path.join(BASE, "dataset", "synthetic_gt")

os.makedirs(SYN_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

AUTH_IMAGES = [
    "bricks.jpg",
    "scene_a.jpg",
    "scene_b.jpg",
    "textured_scene.jpg",
]

PATCH_SIZE = 120


def _safe_random_pos(h, w, patch_h, patch_w, exclude=None, min_dist=60):
    """Return a random (y, x) position for a patch, far enough from *exclude*."""
    for _ in range(500):
        y = random.randint(0, h - patch_h - 1)
        x = random.randint(0, w - patch_w - 1)
        if exclude is not None:
            ey, ex = exclude
            if abs(y - ey) + abs(x - ex) < min_dist + PATCH_SIZE:
                continue
        return y, x
    # Fallback: opposite corner
    y = max(0, h - patch_h - 10)
    x = max(0, w - patch_w - 10)
    return y, x


def rotate_patch(patch, angle_deg):
    """Rotate a square patch around its centre."""
    h, w = patch.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    rotated = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated


def scale_patch(patch, factor):
    """Scale a patch by *factor*, keeping the result cropped to original size."""
    h, w = patch.shape[:2]
    new_w = int(w * factor)
    new_h = int(h * factor)
    scaled = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Crop from centre back to original size
    cy, cx = new_h // 2, new_w // 2
    y1 = max(cy - h // 2, 0)
    x1 = max(cx - w // 2, 0)
    cropped = scaled[y1:y1+h, x1:x1+w]
    if cropped.shape[0] < h or cropped.shape[1] < w:
        canvas = np.zeros_like(patch)
        canvas[:cropped.shape[0], :cropped.shape[1]] = cropped
        cropped = canvas
    return cropped


def paste_patch(image, patch, y, x):
    """Paste *patch* into *image* at (y, x). Returns modified image."""
    ph, pw = patch.shape[:2]
    img = image.copy()
    img[y:y+ph, x:x+pw] = patch
    return img


def make_mask(shape, y, x, ph, pw):
    """Create a binary mask with white rectangle at the paste location."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[y:y+ph, x:x+pw] = 255
    return mask


def save_forgery(img, mask, name):
    """Save forged JPEG and ground-truth PNG."""
    img_path = os.path.join(SYN_DIR, name + ".jpg")
    gt_path  = os.path.join(GT_DIR,  name + ".png")
    cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(gt_path, mask)
    print(f"  Saved: {img_path}")
    print(f"         {gt_path}")


def main():
    # Load all authentic images (BGR)
    auth_images = {}
    for fname in AUTH_IMAGES:
        path = os.path.join(AUTH_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot load {path}, skipping.")
            continue
        auth_images[fname] = img
        print(f"Loaded: {fname}  shape={img.shape}")

    if not auth_images:
        print("No authentic images found. Aborting.")
        return

    sift = cv2.SIFT_create(contrastThreshold=0.005)
    names = list(auth_images.keys())
    count = 0

    for fname in names:
        img = auth_images[fname]
        stem = os.path.splitext(fname)[0]
        h, w = img.shape[:2]
        ps = PATCH_SIZE

        if h < ps + 20 or w < ps + 20:
            print(f"[WARN] {fname} too small ({h}x{w}), skipping.")
            continue

        print(f"\n--- Generating forgeries for: {fname} ---")

        # Select patch in region with highest feature keypoint density
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = sift.detect(gray, None)
        best_c, best_y, best_x = -1, 10, 10
        for y in range(10, h - ps - 10, 8):
            for x in range(10, w - ps - 10, 8):
                c = sum(1 for kp in kps if x <= kp.pt[0] < x + ps and y <= kp.pt[1] < y + ps)
                if c > best_c:
                    best_c, best_y, best_x = c, y, x

        src_y, src_x = best_y, best_x
        patch = img[src_y:src_y+ps, src_x:src_x+ps].copy()
        print(f"  Best patch at ({src_y},{src_x}) with {best_c} keypoints")

        # ── 1. Plain copy-move ──────────────────────────────────────
        dst_y, dst_x = _safe_random_pos(h, w, ps, ps, exclude=(src_y, src_x))
        forged = paste_patch(img, patch, dst_y, dst_x)
        mask   = make_mask(img.shape, dst_y, dst_x, ps, ps)
        save_forgery(forged, mask, f"{stem}_copymove")
        count += 1

        # ── 2. Rotated copy-move (30°) ──────────────────────────────
        rot_patch = rotate_patch(patch, 30)
        dst_y2, dst_x2 = _safe_random_pos(h, w, ps, ps, exclude=(src_y, src_x))
        forged2 = paste_patch(img, rot_patch, dst_y2, dst_x2)
        mask2   = make_mask(img.shape, dst_y2, dst_x2, ps, ps)
        save_forgery(forged2, mask2, f"{stem}_rotated30")
        count += 1

        # ── 3. Scaled copy-move (1.2×) ──────────────────────────────
        sc_patch = scale_patch(patch, 1.2)
        dst_y3, dst_x3 = _safe_random_pos(h, w, ps, ps, exclude=(src_y, src_x))
        forged3 = paste_patch(img, sc_patch, dst_y3, dst_x3)
        mask3   = make_mask(img.shape, dst_y3, dst_x3, ps, ps)
        save_forgery(forged3, mask3, f"{stem}_scaled12")
        count += 1

        # ── 4. Splicing (patch from a DIFFERENT image) ──────────────
        other_names = [n for n in names if n != fname]
        if other_names:
            donor_name = other_names[0]
            donor = auth_images[donor_name]
            donor_gray = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
            donor_kps = sift.detect(donor_gray, None)
            dh, dw = donor.shape[:2]
            best_dc, d_y, d_x = -1, 10, 10
            for y in range(10, dh - ps - 10, 8):
                for x in range(10, dw - ps - 10, 8):
                    c = sum(1 for kp in donor_kps if x <= kp.pt[0] < x + ps and y <= kp.pt[1] < y + ps)
                    if c > best_dc:
                        best_dc, d_y, d_x = c, y, x
            actual_ph = min(ps, dh - d_y)
            actual_pw = min(ps, dw - d_x)
            donor_patch = donor[d_y:d_y+actual_ph, d_x:d_x+actual_pw].copy()
            if actual_ph < ps or actual_pw < ps:
                donor_patch = cv2.resize(donor_patch, (ps, ps))

            dst_y4, dst_x4 = _safe_random_pos(h, w, ps, ps)
            forged4 = paste_patch(img, donor_patch, dst_y4, dst_x4)
            mask4   = make_mask(img.shape, dst_y4, dst_x4, ps, ps)
            save_forgery(forged4, mask4, f"{stem}_spliced")
            count += 1

    print(f"\n{'='*50}")
    print(f"  Generated {count} synthetic forgeries")
    print(f"  Images:       {SYN_DIR}")
    print(f"  Ground truth: {GT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
