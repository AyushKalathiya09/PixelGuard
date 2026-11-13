"""
Generate a copy-move test image with strong, structured texture that SIFT
can reliably detect keypoints on. Uses a brick-like pattern.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.dirname(__file__)
TAMP_DIR = os.path.join(BASE, "dataset", "tampered")
GT_DIR   = os.path.join(BASE, "dataset", "ground_truth")

w, h = 800, 600
np.random.seed(123)

# Create a structured texture: brick-like pattern with noise
img_arr = np.random.randint(100, 160, (h, w, 3), dtype=np.uint8)
img = Image.fromarray(img_arr)

# Add structured elements: grid of small rectangles (like bricks)
draw = ImageDraw.Draw(img)
for y in range(0, h, 30):
    offset = 20 if (y // 30) % 2 else 0
    for x in range(-20 + offset, w, 40):
        r = np.random.randint(140, 220)
        g = np.random.randint(80, 140)
        b = np.random.randint(60, 100)
        draw.rectangle([x, y, x+38, y+28], fill=(r, g, b),
                        outline=(80, 50, 30), width=1)

# Add some distinctive features
draw.ellipse([100, 100, 200, 200], fill=(50, 120, 200), outline=(30, 80, 160), width=2)
draw.ellipse([150, 250, 230, 330], fill=(200, 50, 50), outline=(150, 30, 30), width=2)
draw.polygon([(300, 80), (370, 180), (230, 180)], fill=(50, 180, 50))

# Apply slight blur to make it more photo-like
img = img.filter(ImageFilter.GaussianBlur(1))
img.save(os.path.join(os.path.join(BASE, "dataset", "authentic"),
         "bricks.jpg"), "JPEG", quality=95)

# Copy-move: copy a 200x150 region and paste it elsewhere
tampered = img.copy()
# Copy the blue circle area
patch = img.crop((80, 80, 280, 230))
# Paste it at a distant location
tampered.paste(patch, (480, 350))
tampered.save(os.path.join(TAMP_DIR, "bricks_copymove.jpg"), "JPEG", quality=93)

# Ground truth mask
mask = Image.new("L", (w, h), 0)
dm = ImageDraw.Draw(mask)
dm.rectangle([80, 80, 280, 230], fill=255)
dm.rectangle([480, 350, 680, 500], fill=255)
mask.save(os.path.join(GT_DIR, "bricks_copymove.png"))

print("[OK] Brick-texture copy-move test images created")

