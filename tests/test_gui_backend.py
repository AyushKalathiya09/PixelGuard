import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from main import analyze_image

cases = {}

# 1. Valid JPEG (authentic)
try:
    res = analyze_image('dataset/authentic/scene_a.jpg')
    v = res['verdict']
    cases['1. Valid JPEG (authentic)'] = f'PASS: Verdict={v}'
except Exception as e:
    cases['1. Valid JPEG (authentic)'] = f'FAIL: {e}'

# 2. Copy-move image
try:
    res = analyze_image('dataset/tampered/bricks_copymove.jpg')
    v = res['verdict']
    m = res['num_matches']
    cases['2. Copy-move image'] = f'PASS: Verdict={v}, Matches={m}'
except Exception as e:
    cases['2. Copy-move image'] = f'FAIL: {e}'

# 3. Splicing image
try:
    res = analyze_image('dataset/synthetic/scene_a_spliced.jpg')
    v = res['verdict']
    r = res['tampered_ratio']
    cases['3. Splicing image'] = f'PASS: Verdict={v}, Ratio={r:.4f}'
except Exception as e:
    cases['3. Splicing image'] = f'FAIL: {e}'

# 4. PNG file
tmp_png = tempfile.mktemp(suffix='.png')
img = cv2.imread('dataset/authentic/scene_a.jpg')
cv2.imwrite(tmp_png, img)
try:
    res = analyze_image(tmp_png)
    v = res['verdict']
    cases['4. PNG file'] = f'PROCESSED (Verdict={v}) - Needs explicit warning in app UI'
except Exception as e:
    cases['4. PNG file'] = f'FAIL: {e}'
finally:
    if os.path.exists(tmp_png):
        os.remove(tmp_png)

# 5. Corrupted / non-image file
tmp_corrupt = tempfile.mktemp(suffix='.jpg')
with open(tmp_corrupt, 'wb') as f:
    f.write(b'THIS_IS_NOT_AN_IMAGE_FILE_JUST_CORRUPT_BYTES')
try:
    res = analyze_image(tmp_corrupt)
    cases['5. Corrupted file'] = 'FAIL: Did not raise expected error'
except Exception as e:
    cases['5. Corrupted file'] = f'RAISES EXCEPTION ({type(e).__name__}) - Needs try/except in app.py'
finally:
    if os.path.exists(tmp_corrupt):
        os.remove(tmp_corrupt)

# 6. Very large image (>4000 px)
tmp_large = tempfile.mktemp(suffix='.jpg')
large_img = np.zeros((4200, 4200, 3), dtype=np.uint8)
cv2.circle(large_img, (2100, 2100), 500, (255, 200, 150), -1)
cv2.imwrite(tmp_large, large_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
try:
    res = analyze_image(tmp_large)
    v = res['verdict']
    cases['6. Very large image (4200px)'] = f'PASS: Finished successfully! Verdict={v}'
except Exception as e:
    cases['6. Very large image (4200px)'] = f'FAIL: {e}'
finally:
    if os.path.exists(tmp_large):
        os.remove(tmp_large)

print('='*75)
print('GUI PIPELINE TEST CASES')
print('='*75)
for k, val in cases.items():
    print(f'{k:<35}: {val}')
