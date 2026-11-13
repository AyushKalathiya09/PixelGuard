"""
test_modules.py – Unit tests for the forgery-detection pipeline modules.

Tests cover:
  1. ELA (compute_ela, ela_to_gray)
  2. Copy-move detection (detect_copy_move)
  3. Segmentation (segment_ela, contour filtering)
  4. combine_masks (bitwise OR)
  5. Decision rule (TAMPERED / AUTHENTIC verdicts)
"""

import sys
import os
import tempfile

import cv2
import numpy as np
import pytest

# ── Make the project root importable ──────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ela import compute_ela, ela_to_gray
from copy_move import detect_copy_move
from segmentation import segment_ela, combine_masks


# ======================================================================
# Helper: save a NumPy RGB image as a JPEG temp file
# ======================================================================
def _save_temp_jpeg(image_rgb: np.ndarray, suffix: str = ".jpg") -> str:
    """Write *image_rgb* (uint8, RGB) to a temporary JPEG file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


# ======================================================================
# 1.  ELA TESTS
# ======================================================================
class TestELA:
    """Tests for compute_ela and ela_to_gray."""

    def test_output_shape_and_dtype(self):
        """ELA output must have the same HxW as input and dtype uint8."""
        # Create a small random colour image
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        path = _save_temp_jpeg(img)
        try:
            ela = compute_ela(path, quality=90, scale=15)
            assert ela.shape == img.shape, (
                f"Shape mismatch: ELA {ela.shape} vs input {img.shape}"
            )
            assert ela.dtype == np.uint8, f"Expected uint8, got {ela.dtype}"
        finally:
            os.remove(path)

    def test_uniform_image_low_ela(self):
        """A pure-white (uniform) image should produce very low ELA values."""
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        path = _save_temp_jpeg(white)
        try:
            ela = compute_ela(path, quality=90, scale=15)
            mean_val = ela.mean()
            assert mean_val < 10, (
                f"Uniform image ELA mean should be < 10, got {mean_val:.2f}"
            )
        finally:
            os.remove(path)

    def test_ela_to_gray_shape(self):
        """ela_to_gray must return a single-channel uint8 image."""
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
        path = _save_temp_jpeg(img)
        try:
            ela = compute_ela(path)
            gray = ela_to_gray(ela)
            assert gray.ndim == 2, f"Expected 2-D, got {gray.ndim}-D"
            assert gray.shape == ela.shape[:2], (
                f"HxW mismatch: gray {gray.shape} vs ELA {ela.shape[:2]}"
            )
            assert gray.dtype == np.uint8
        finally:
            os.remove(path)


# ======================================================================
# 2.  COPY-MOVE TESTS
# ======================================================================
class TestCopyMove:
    """Tests for detect_copy_move."""

    def test_synthetic_copy_finds_matches(self):
        """
        Build a synthetic image with a copied textured patch.
        detect_copy_move must find verified matches.
        """
        rng = np.random.default_rng(123)

        # Start with a mildly noisy canvas (adds keypoint variety)
        canvas = rng.integers(100, 180, (500, 500, 3), dtype=np.uint8)

        # Create a detailed textured patch (64x64) with strong features
        patch = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        # Add some structure to make SIFT find good keypoints
        for i in range(0, 64, 8):
            patch[i : i + 4, :, :] = rng.integers(0, 256, (4, 64, 3), dtype=np.uint8)

        # Paste the SAME patch in two well-separated locations
        canvas[50:114, 50:114] = patch
        canvas[300:364, 300:364] = patch

        path = _save_temp_jpeg(canvas)
        try:
            matched_pairs, cm_mask, vis = detect_copy_move(
                path, ratio=0.75, min_dist=30, min_matches=4
            )
            assert len(matched_pairs) > 0, (
                "Expected copy-move matches on synthetic duplicated patch, got 0"
            )
            # The mask should have some white pixels
            assert cm_mask.max() == 255, "cm_mask should contain 255 pixels"
        finally:
            os.remove(path)

    def test_plain_image_no_matches(self):
        """A single plain-colour image should yield zero verified matches."""
        plain = np.full((200, 200, 3), 128, dtype=np.uint8)
        path = _save_temp_jpeg(plain)
        try:
            matched_pairs, cm_mask, vis = detect_copy_move(path)
            assert len(matched_pairs) == 0, (
                f"Expected 0 matches on plain image, got {len(matched_pairs)}"
            )
        finally:
            os.remove(path)


# ======================================================================
# 3.  SEGMENTATION TESTS
# ======================================================================
class TestSegmentation:
    """Tests for segment_ela contour filtering."""

    def test_single_large_region_kept(self):
        """
        A 500x500 black mask with a 100x100 white square should produce
        exactly one contour region after segment_ela-style filtering.
        """
        mask = np.zeros((500, 500), dtype=np.uint8)
        mask[200:300, 200:300] = 255  # 100x100 white square (area = 10 000)

        result = segment_ela(mask, min_area=500)
        # The big square should survive
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1, (
            f"Expected 1 region, got {len(contours)}"
        )

    def test_tiny_specks_removed(self):
        """
        Small specks (area < min_area) must be removed by segment_ela.
        """
        mask = np.zeros((500, 500), dtype=np.uint8)
        # Add a tiny 3x3 speck (area = 9)
        mask[10:13, 10:13] = 255
        # Add another tiny 5x5 speck (area = 25)
        mask[400:405, 400:405] = 255

        result = segment_ela(mask, min_area=500)
        assert cv2.countNonZero(result) == 0, (
            "Tiny specks should be removed; mask should be all black"
        )


# ======================================================================
# 4.  combine_masks TEST
# ======================================================================
class TestCombineMasks:
    """Tests for combine_masks (bitwise OR)."""

    def test_bitwise_or(self):
        """combine_masks must equal the bitwise OR of two input masks."""
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_b = np.zeros((100, 100), dtype=np.uint8)

        mask_a[10:40, 10:40] = 255   # top-left block
        mask_b[60:90, 60:90] = 255   # bottom-right block

        combined = combine_masks(mask_a, mask_b)
        expected = cv2.bitwise_or(mask_a, mask_b)

        assert np.array_equal(combined, expected), (
            "combine_masks result does not match cv2.bitwise_or"
        )

    def test_overlapping_regions(self):
        """Overlapping white regions should remain white (OR logic)."""
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_b = np.zeros((100, 100), dtype=np.uint8)

        mask_a[30:70, 30:70] = 255
        mask_b[50:90, 50:90] = 255

        combined = combine_masks(mask_a, mask_b)
        expected = cv2.bitwise_or(mask_a, mask_b)

        assert np.array_equal(combined, expected)
        # The overlap region should be 255
        assert combined[60, 60] == 255


# ======================================================================
# 5.  DECISION RULE TESTS
# ======================================================================
class TestDecisionRule:
    """Tests for the TAMPERED / AUTHENTIC decision logic from analyze_image."""

    @staticmethod
    def _decide(tampered_ratio: float, num_matches: int,
                ratio_threshold: float = 0.08, min_matches: int = 5) -> str:
        """
        Replicate the decision rule from main.py analyze_image (lines 72-76).
        """
        if tampered_ratio > ratio_threshold or num_matches >= min_matches:
            return "TAMPERED"
        else:
            return "AUTHENTIC"

    def test_high_ratio_gives_tampered(self):
        """If tampered_ratio > threshold, verdict must be TAMPERED."""
        assert self._decide(tampered_ratio=0.15, num_matches=0) == "TAMPERED"

    def test_many_matches_gives_tampered(self):
        """If num_matches >= min_matches, verdict must be TAMPERED."""
        assert self._decide(tampered_ratio=0.0, num_matches=10) == "TAMPERED"

    def test_both_triggers_gives_tampered(self):
        """Both conditions true → TAMPERED."""
        assert self._decide(tampered_ratio=0.20, num_matches=15) == "TAMPERED"

    def test_zero_everything_gives_authentic(self):
        """Zero matches and small ratio → AUTHENTIC."""
        assert self._decide(tampered_ratio=0.0, num_matches=0) == "AUTHENTIC"

    def test_below_threshold_gives_authentic(self):
        """Ratio just below threshold with few matches → AUTHENTIC."""
        assert self._decide(tampered_ratio=0.07, num_matches=3) == "AUTHENTIC"

    def test_exact_threshold_ratio_gives_authentic(self):
        """Ratio exactly at threshold (not >) → AUTHENTIC."""
        assert self._decide(tampered_ratio=0.08, num_matches=0) == "AUTHENTIC"

    def test_exact_min_matches_gives_tampered(self):
        """num_matches exactly at min_matches (>=) → TAMPERED."""
        assert self._decide(tampered_ratio=0.0, num_matches=5) == "TAMPERED"
