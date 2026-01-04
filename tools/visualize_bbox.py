# import os
# import cv2
# import numpy as np
# import math

# IMAGE_DIR = "images"
# LABEL_DIR = "labels/train"
# OUTPUT_DIR = "bbox"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# empty_labels = []
# missing_labels = []

# image_files = sorted([
#     f for f in os.listdir(IMAGE_DIR)
#     if f.lower().endswith((".jpg", ".jpeg", ".png"))
# ])

# for img_name in image_files:
#     img_path = os.path.join(IMAGE_DIR, img_name)
#     label_name = os.path.splitext(img_name)[0] + ".txt"
#     label_path = os.path.join(LABEL_DIR, label_name)

#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[WARN] Cannot read image: {img_name}")
#         continue

#     h, w = img.shape[:2]

#     if not os.path.exists(label_path):
#         missing_labels.append(img_name)
#         cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)
#         continue

#     with open(label_path, "r") as f:
#         lines = [line.strip() for line in f if line.strip()]

#     if len(lines) == 0:
#         empty_labels.append(img_name)
#         cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)
#         continue

#     for line in lines:
#         parts = line.split()
#         if len(parts) != 6:
#             continue

#         _, cx, cy, bw, bh, angle = map(float, parts)

#         # YOLO normalized → pixel
#         cx *= w
#         cy *= h
#         bw *= w
#         bh *= h

#         # YOLOv8-OBB angle 是 degree（逆時針）
#         rect = ((cx, cy), (bw, bh), angle)

#         box = cv2.boxPoints(rect)
#         box = box.astype(int)

#         cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

#     cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)

# # Report
# print("Visualization finished.\n")

# print(f"Images with missing label files: {len(missing_labels)}")
# for name in missing_labels:
#     print("  -", name)

# print(f"\nImages with empty label files: {len(empty_labels)}")
# for name in empty_labels:
#     print("  -", name)

import os
import cv2
import numpy as np

IMAGE_DIR = "images/train"
LABEL_DIR = "labels/train"
OUTPUT_DIR = "bbox"

os.makedirs(OUTPUT_DIR, exist_ok=True)

empty_labels = []
missing_labels = []

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot read image: {img_name}")
        continue

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        missing_labels.append(img_name)
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)
        continue

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 0:
        empty_labels.append(img_name)
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)
        continue

    for line in lines:
        parts = list(map(float, line.split()))
        if len(parts) != 9:
            continue

        class_id = int(parts[0])
        coords = parts[1:]

        # normalized → pixel
        pts = np.array([
            [coords[0] * w, coords[1] * h],
            [coords[2] * w, coords[3] * h],
            [coords[4] * w, coords[5] * h],
            [coords[6] * w, coords[7] * h],
        ], dtype=np.int32)

        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)

# Report
print("Visualization finished.\n")

print(f"Images with missing label files: {len(missing_labels)}")
for name in missing_labels:
    print("  -", name)

print(f"\nImages with empty label files: {len(empty_labels)}")
for name in empty_labels:
    print("  -", name)
