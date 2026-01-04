import cv2
import os
import re
import numpy as np

# ===== paths =====
IMG_DIR = "photo"          # or "photo_resized"
GT_DIR  = "GT"

# ===== parse GT =====
def parse_gt_file(txt_path):
    polys = []
    with open(txt_path, "r") as f:
        for line in f:
            nums = list(map(int, re.findall(r"\d+", line)))
            if len(nums) >= 8:
                poly = np.array(nums[:8]).reshape(4, 2)
                polys.append(poly)
    return polys

# ===== main =====
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    name = os.path.splitext(fname)[0]
    img_path = os.path.join(IMG_DIR, fname)
    gt_path  = os.path.join(GT_DIR, f"{name}.txt")

    if not os.path.exists(gt_path):
        print(f"[WARN] No GT for {fname}")
        continue

    img = cv2.imread(img_path)
    gt_polys = parse_gt_file(gt_path)

    vis = img.copy()

    for poly in gt_polys:
        cv2.polylines(
            vis,
            [poly.astype(np.int32)],
            isClosed=True,
            color=(0, 0, 255),   # 🔴 GT in red
            thickness=2
        )

        # draw center
        cx, cy = np.mean(poly, axis=0).astype(int)
        cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)

    cv2.imshow("GT Visualization", vis)
    print(f"[INFO] Showing {fname}, GT count = {len(gt_polys)}")
    key = cv2.waitKey(0)

    if key == 27:  # ESC to exit
        break


cv2.destroyAllWindows()
