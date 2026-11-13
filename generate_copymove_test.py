"""
Generate a textured copy-move test image where SIFT can find keypoints.
We create a noisy textured scene, then copy a rectangular region to
another location — this simulates a real copy-move forgery.
"""
import os
import numpy as np
from PIL import Image, ImageFilter

BASE = os.path.dirname(__file__)
TAMP_DIR = os.path.join(BASE, "dataset", "tampered")
GT_DIR   = os.path.join(BASE, "dataset", "ground_truth")

# Create a richly textured image (random noise + blur = cloud-like texture)
np.random.seed(42)
w, h = 640, 480
# Layer 1: coarse blobs
coarse = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
img = Image.fromarray(coarse).filter(ImageFilter.GaussianBlur(6))

# Layer 2: fine detail
fine = np.random.randint(0, 60, (h, w, 3), dtype=np.uint8)
fine_img = Image.fromarray(fine)

# Blend them
arr = np.clip(np.array(img, dtype=np.int16) + np.array(fine_img, dtype=np.int16),
              0, 255).astype(np.uint8)
base_img = Image.fromarray(arr)

# Save the authentic version
base_img.save(os.path.join(os.path.join(BASE, "dataset", "authentic"),
              "textured_scene.jpg"), "JPEG", quality=95)

# Now create copy-move: copy a 150x120 patch from one place to another
tampered = base_img.copy()
patch = base_img.crop((50, 50, 200, 170))     # source region
tampered.paste(patch, (400, 280))              # paste at distant location
tampered.save(os.path.join(TAMP_DIR, "copymove_test.jpg"), "JPEG", quality=92)

# Ground truth mask
mask = Image.new("L", (w, h), 0)
from PIL import ImageDraw
draw = ImageDraw.Draw(mask)
draw.rectangle([50, 50, 200, 170], fill=255)    # source
draw.rectangle([400, 280, 550, 400], fill=255)   # destination
mask.save(os.path.join(GT_DIR, "copymove_test.png"))

print("[OK] Created textured copy-move test image")
print("  authentic/textured_scene.jpg")
print("  tampered/copymove_test.jpg")
print("  ground_truth/copymove_test.png")

