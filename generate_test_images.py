"""
generate_test_images.py  –  Create synthetic test images for the project.

Creates:
  - dataset/authentic/  : clean images (landscape, gradient)
  - dataset/tampered/   : images with obvious copy-paste tampering
  - dataset/ground_truth/ : binary masks showing tampered regions
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.dirname(__file__)
AUTH_DIR = os.path.join(BASE, "dataset", "authentic")
TAMP_DIR = os.path.join(BASE, "dataset", "tampered")
GT_DIR   = os.path.join(BASE, "dataset", "ground_truth")

for d in (AUTH_DIR, TAMP_DIR, GT_DIR):
    os.makedirs(d, exist_ok=True)


def save_jpeg(img, path, quality=95):
    """Save a PIL image as JPEG."""
    img.convert("RGB").save(path, "JPEG", quality=quality)


# ── Authentic image 1: colourful gradient with shapes ────────────────
w, h = 640, 480
arr = np.zeros((h, w, 3), dtype=np.uint8)
for y in range(h):
    for x in range(w):
        arr[y, x] = [int(255 * x / w), int(255 * y / h), 128]
img1 = Image.fromarray(arr)
draw = ImageDraw.Draw(img1)
draw.ellipse([200, 100, 350, 250], fill=(255, 80, 80))
draw.rectangle([400, 300, 550, 420], fill=(80, 80, 255))
save_jpeg(img1, os.path.join(AUTH_DIR, "scene_a.jpg"))
print("[OK] authentic/scene_a.jpg")

# ── Authentic image 2: noisy natural-ish texture ─────────────────────
noise = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
img2 = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(3))
draw2 = ImageDraw.Draw(img2)
draw2.polygon([(320, 50), (500, 400), (140, 400)], fill=(40, 180, 80))
save_jpeg(img2, os.path.join(AUTH_DIR, "scene_b.jpg"))
print("[OK] authentic/scene_b.jpg")


# ── Tampered image 1: take scene_a and paste a foreign block ─────────
tampered1 = img1.copy()
# Paste a block from img2 (foreign source, different compression history)
foreign_patch = img2.crop((50, 50, 220, 200))
tampered1.paste(foreign_patch, (350, 200))
# Save at a DIFFERENT quality to simulate re-saving
save_jpeg(tampered1, os.path.join(TAMP_DIR, "tampered_a.jpg"), quality=92)
print("[OK] tampered/tampered_a.jpg")

# Ground-truth mask for tampered_a: white where the paste is
mask1 = Image.new("L", (w, h), 0)
draw_m1 = ImageDraw.Draw(mask1)
draw_m1.rectangle([350, 200, 350 + 170, 200 + 150], fill=255)
mask1.save(os.path.join(GT_DIR, "tampered_a.png"))
print("[OK] ground_truth/tampered_a.png")


# ── Tampered image 2: clone a region within scene_b ──────────────────
tampered2 = img2.copy()
# Copy the triangle region as a rectangle crop and paste it elsewhere
cloned_patch = img2.crop((140, 50, 400, 300))
tampered2.paste(cloned_patch, (300, 180))
save_jpeg(tampered2, os.path.join(TAMP_DIR, "tampered_b.jpg"), quality=88)
print("[OK] tampered/tampered_b.jpg")

# Ground-truth mask for tampered_b
mask2 = Image.new("L", (w, h), 0)
draw_m2 = ImageDraw.Draw(mask2)
draw_m2.rectangle([300, 180, 300 + 260, 180 + 250], fill=255)
mask2.save(os.path.join(GT_DIR, "tampered_b.png"))
print("[OK] ground_truth/tampered_b.png")

print("\n[OK] All test images generated!")
