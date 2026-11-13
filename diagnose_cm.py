import os
import cv2
import numpy as np

def diagnose_image(img_name):
    base = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base, "dataset", "synthetic", img_name + ".jpg")
    gt_path = os.path.join(base, "dataset", "synthetic_gt", img_name + ".png")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read {img_path}")
        return
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    h, w = img.shape[:2]
    max_side = max(h, w)
    scale = 1.0
    if max_side > 1000:
        scale = 1000.0 / max_side
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if gt is not None:
            gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        h, w = new_h, new_w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(gray, None)
    
    num_kps = len(kps) if kps else 0
    kps_in_gt = 0
    if gt is not None and kps:
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < h and 0 <= x < w and gt[y, x] > 127:
                kps_in_gt += 1
                
    if des is None or len(kps) < 2:
        print(f"\n--- {img_name} ---")
        print(f"Total SIFT keypoints: {num_kps}")
        print(f"Keypoints in GT patch: {kps_in_gt}")
        print("FAILED: Too few keypoints (< 2)")
        return

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des, des, k=3)
    
    # Try different ratios to see impact
    for ratio in [0.6, 0.7, 0.75, 0.8]:
        good_pairs = []
        for match_group in raw_matches:
            if len(match_group) < 3:
                continue
            m_self, m_best, m_second = match_group[0], match_group[1], match_group[2]
            if m_best.distance < ratio * m_second.distance:
                idx_a = m_best.queryIdx
                idx_b = m_best.trainIdx
                pt_a = kps[idx_a].pt
                pt_b = kps[idx_b].pt
                dist = np.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)
                good_pairs.append((idx_a, idx_b, pt_a, pt_b, dist))
        
        # min_dist filter
        pairs_min_dist_40 = [p for p in good_pairs if p[4] >= 40]
        pairs_min_dist_20 = [p for p in good_pairs if p[4] >= 20]
        
        # Deduplication
        seen = set()
        unique_40 = []
        for idx_a, idx_b, pt_a, pt_b, d in pairs_min_dist_40:
            key = (min(idx_a, idx_b), max(idx_a, idx_b))
            if key not in seen:
                seen.add(key)
                unique_40.append((pt_a, pt_b))

        # RANSAC on unique_40
        inliers_count = 0
        if len(unique_40) >= 4:
            src = np.array([p[0] for p in unique_40], dtype=np.float32).reshape(-1, 1, 2)
            dst = np.array([p[1] for p in unique_40], dtype=np.float32).reshape(-1, 1, 2)
            _, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if inliers is not None:
                inliers_count = int(np.sum(inliers))

        if ratio == 0.6:
            print(f"\n=======================================================")
            print(f"  Diagnosis for: {img_name}")
            print(f"=======================================================")
            print(f"Total SIFT keypoints in image:        {num_kps}")
            print(f"Keypoints inside GT copied patch:     {kps_in_gt}")
            print(f"Total raw knnMatch groups:            {len(raw_matches)}")
            print(f"Matches after ratio=0.6 test:         {len(good_pairs)}")
            print(f"Matches after min_dist>=40 filter:    {len(pairs_min_dist_40)}")
            print(f"Unique pairs (de-duplicated):         {len(unique_40)}")
            print(f"Inliers after RANSAC:                 {inliers_count}")
            print(f"-------------------------------------------------------")
            print(f"Ratio sensitivity comparison:")
        print(f"  ratio={ratio:.2f} -> ratio_test: {len(good_pairs):3d} | dist>=40: {len(pairs_min_dist_40):3d} | unique: {len(unique_40):3d} | inliers: {inliers_count:3d}")

for name in ["scene_a_copymove", "scene_b_copymove", "textured_scene_copymove"]:
    diagnose_image(name)

