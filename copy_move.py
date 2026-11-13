"""
copy_move.py  -  Copy-Move forgery detection using SIFT + DBSCAN + RANSAC.

Copy-Move forgery is when someone copies a region of an image and pastes it
somewhere else in the SAME image. We detect this by finding keypoints that
match suspiciously well at distant locations.

Pipeline:  SIFT keypoints -> BFMatcher(k=3) -> ratio test -> min-distance
           filter -> de-duplicate mirrors -> DBSCAN cluster -> RANSAC verify
           -> convex-hull mask.

Functions
---------
detect_copy_move : Full copy-move detection pipeline.
"""

import os
import glob

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from utils import load_image, show_images


# ──────────────────────────────────────────────────────────────────────
# detect_copy_move
# ──────────────────────────────────────────────────────────────────────
def detect_copy_move(
    image_path: str,
    ratio: float = 0.6,
    min_dist: int = 40,
    min_matches: int = 10,
) -> tuple:
    """
    Detect copy-move forgery in a single image.

    Parameters
    ----------
    image_path  : str
        Path to the image file.
    ratio       : float
        Lowe's ratio test threshold (lower = stricter). Default 0.6.
    min_dist    : int
        Minimum pixel distance between matched keypoints. Default 40.
    min_matches : int
        Minimum number of valid matches to declare copy-move. Default 10.

    Returns
    -------
    matched_pairs : list of ((x1,y1), (x2,y2))
        Verified matched keypoint coordinate pairs.
    cm_mask       : np.ndarray, uint8
        Binary mask (0 or 255) with detected regions filled.
        Same size as the (possibly resized) image.
    vis_image     : np.ndarray, RGB uint8
        Visualization image with matches drawn.
    """

    # ── Step 1: Load and optionally resize ───────────────────────────
    img_bgr = load_image(image_path, color="bgr")
    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    scale = 1.0

    if max_side > 1000:
        scale = 1000.0 / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Prepare empty results (returned if nothing found)
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    empty_vis = img_rgb.copy()

    # ── Step 2: SIFT keypoints and descriptors ───────────────────────
    # Use contrastThreshold=0.005 to retain keypoints in smooth/low-contrast regions
    sift = cv2.SIFT_create(contrastThreshold=0.005)
    kps, des = sift.detectAndCompute(gray, None)

    # ── Step 2b: ORB fallback for smooth images ──────────────────────
    # If SIFT finds fewer than 10 keypoints, use ORB (more robust on uniform regions)
    use_orb = False
    if des is None or len(kps) < 10:
        print("    [CM] SIFT found < 10 keypoints; falling back to ORB.")
        orb = cv2.ORB_create(nfeatures=2000)
        kps, des = orb.detectAndCompute(gray, None)
        use_orb = True

    if des is None or len(kps) < 2:
        print("    [CM] Too few keypoints found.")
        return [], empty_mask, empty_vis

    # ── Step 3: BFMatcher with k=3 ──────────────────────────────────
    # k=3 because match[0] is the point itself (distance ~ 0),
    # match[1] is the best real match, match[2] is second-best.
    # Use Hamming norm for ORB (binary descriptor); L2 for SIFT (float descriptor).
    norm = cv2.NORM_HAMMING if use_orb else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    raw_matches = bf.knnMatch(des, des, k=3)

    # ── Step 4: Ratio test on matches[1] vs matches[2] ──────────────
    good_pairs = []
    for match_group in raw_matches:
        if len(match_group) < 3:
            continue
        m_self, m_best, m_second = match_group[0], match_group[1], match_group[2]
        # Ratio test: best real match must be significantly better than second-best
        if m_best.distance < ratio * m_second.distance:
            idx_a = m_best.queryIdx
            idx_b = m_best.trainIdx

            pt_a = kps[idx_a].pt  # (x, y) float
            pt_b = kps[idx_b].pt

            # ── Step 5: Minimum distance filter ─────────────────────
            dist = np.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)
            if dist < min_dist:
                continue

            good_pairs.append((idx_a, idx_b, pt_a, pt_b))

    # ── Step 6: Remove duplicate mirrored pairs (A->B == B->A) ──────
    seen = set()
    unique_pairs = []
    for idx_a, idx_b, pt_a, pt_b in good_pairs:
        key = (min(idx_a, idx_b), max(idx_a, idx_b))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((pt_a, pt_b))

    print(f"    [CM] Matches after filtering: {len(unique_pairs)}")

    # ── Step 7: Check minimum matches threshold ─────────────────────
    if len(unique_pairs) < min_matches:
        print("    [CM] Not enough matches -> no copy-move detected.")
        return [], empty_mask, empty_vis

    # ── Step 8: DBSCAN clustering ────────────────────────────────────
    # Cluster based on the SOURCE keypoints (pt_a side) to group spatially contiguous patches
    src_pts = np.array([p[0] for p in unique_pairs], dtype=np.float32)
    dst_pts = np.array([p[1] for p in unique_pairs], dtype=np.float32)

    clustering = DBSCAN(eps=60, min_samples=3).fit(src_pts)
    labels = clustering.labels_
    unique_labels = [l for l in set(labels) if l != -1]

    # ── Step 9: RANSAC verification per cluster ──────────────────────
    verified_clusters = []
    verified_pairs = []

    if unique_labels:
        # Run RANSAC separately for each DBSCAN spatial cluster
        for lbl in unique_labels:
            c_indices = np.where(labels == lbl)[0]
            if len(c_indices) >= 4:
                c_src = src_pts[c_indices].reshape(-1, 1, 2)
                c_dst = dst_pts[c_indices].reshape(-1, 1, 2)
                _, inliers = cv2.estimateAffinePartial2D(
                    c_src, c_dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                if inliers is not None:
                    inl_idx = c_indices[inliers.ravel().astype(bool)]
                    # Coherent copy-move cluster requires at least 4 inliers within the same cluster
                    if len(inl_idx) >= 4:
                        cluster_inliers = [unique_pairs[idx] for idx in inl_idx]
                        verified_clusters.append(cluster_inliers)
                        verified_pairs.extend(cluster_inliers)
        print(f"    [CM] Inliers after cluster RANSAC: {len(verified_pairs)} across {len(verified_clusters)} cluster(s)")
    else:
        # Fallback to global RANSAC across all pairs if no distinct cluster formed
        if len(unique_pairs) >= 4:
            src_arr = src_pts.reshape(-1, 1, 2)
            dst_arr = dst_pts.reshape(-1, 1, 2)
            _, inliers = cv2.estimateAffinePartial2D(
                src_arr, dst_arr, method=cv2.RANSAC, ransacReprojThreshold=4.0
            )
            if inliers is not None:
                global_inliers = []
                for i, is_inlier in enumerate(inliers.ravel()):
                    if is_inlier:
                        global_inliers.append(unique_pairs[i])
                if global_inliers:
                    verified_clusters.append(global_inliers)
                    verified_pairs.extend(global_inliers)
            print(f"    [CM] Inliers after global RANSAC: {len(verified_pairs)}")
        else:
            verified_clusters.append(unique_pairs)
            verified_pairs = list(unique_pairs)

    if len(verified_pairs) == 0:
        print("    [CM] No inliers survived RANSAC.")
        return [], empty_mask, empty_vis

    # ── Step 10: Build mask using Fix G (Spatial DBSCAN on all inlier endpoints) ─
    cm_mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Collect ALL inlier endpoints (both points of every verified pair)
    all_endpoints = []
    for pt_a, pt_b in verified_pairs:
        all_endpoints.append(pt_a)
        all_endpoints.append(pt_b)

    if len(all_endpoints) > 0:
        all_endpoints = np.array(all_endpoints, dtype=np.float32)

        # 2. Cluster all endpoints spatially with DBSCAN (eps=60, min_samples=3)
        endpoint_clustering = DBSCAN(eps=60, min_samples=3).fit(all_endpoints)
        ep_labels = endpoint_clustering.labels_
        unique_groups = [l for l in set(ep_labels) if l != -1]

        # 3. Fill one convex hull per group. Never create a hull that spans two groups.
        if unique_groups:
            for grp_lbl in unique_groups:
                grp_pts = all_endpoints[ep_labels == grp_lbl]
                if len(grp_pts) >= 3:
                    hull = cv2.convexHull(grp_pts.reshape(-1, 1, 2).astype(np.int32))
                    cv2.fillConvexPoly(cm_mask, hull, 255)
                else:
                    for pt in grp_pts:
                        cv2.circle(cm_mask, (int(pt[0]), int(pt[1])), 8, 255, -1)
        else:
            # Fallback if no cluster formed: draw small markers around endpoints
            for pt in all_endpoints:
                cv2.circle(cm_mask, (int(pt[0]), int(pt[1])), 8, 255, -1)

    # ── Step 11: Draw visualization ──────────────────────────────────
    vis_image = img_rgb.copy()
    # Orange accent color for matches (RGB)
    line_color = (255, 111, 50)   # #FF6F32
    circle_color = (255, 255, 255)  # white

    for pt_a, pt_b in verified_pairs:
        a = (int(pt_a[0]), int(pt_a[1]))
        b = (int(pt_b[0]), int(pt_b[1]))
        cv2.line(vis_image, a, b, line_color, 1, cv2.LINE_AA)
        cv2.circle(vis_image, a, 4, circle_color, 1, cv2.LINE_AA)
        cv2.circle(vis_image, b, 4, circle_color, 1, cv2.LINE_AA)

    # Format matched_pairs as list of ((x1,y1), (x2,y2))
    matched_pairs = [
        ((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
        for a, b in verified_pairs
    ]

    return matched_pairs, cm_mask, vis_image


# ──────────────────────────────────────────────────────────────────────
# Self-test: run on all dataset images
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "copy_move_test")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    for subfolder in ("authentic", "tampered"):
        folder = os.path.join(DATASET_DIR, subfolder)
        if not os.path.isdir(folder):
            print(f"[CM] Skipping (folder not found): {folder}")
            continue

        image_paths = []
        for ext in EXTS:
            image_paths.extend(glob.glob(os.path.join(folder, ext)))

        if not image_paths:
            print(f"[CM] No images found in {folder}")
            continue

        print(f"\n[CM] Processing {len(image_paths)} image(s) in '{subfolder}/'")

        for img_path in sorted(image_paths):
            name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"  Image: {name}")

            original = load_image(img_path, color="rgb")
            matched_pairs, cm_mask, vis_image = detect_copy_move(img_path)

            print(f"    Verified matches: {len(matched_pairs)}")

            # Save 3-panel figure: Original | Matches | Mask
            save_name = f"{subfolder}_{name}.png"
            save_path = os.path.join(RESULTS_DIR, save_name)
            show_images(
                images=[original, vis_image, cm_mask],
                titles=["Original", f"Matches ({len(matched_pairs)})", "CM Mask"],
                save_path=save_path,
            )

    print(f"\n[CM] Done - results saved to {RESULTS_DIR}")
