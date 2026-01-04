import os
import random
import shutil

# ====== 可調參數 ======
VAL_RATIO = 0.2  # 20% 做 validation
SEED = 42
# =====================

random.seed(SEED)

BASE_DIR = "."

IMG_TRAIN = os.path.join(BASE_DIR, "images", "train")
LBL_TRAIN = os.path.join(BASE_DIR, "labels", "train")

IMG_VAL = os.path.join(BASE_DIR, "images", "val")
LBL_VAL = os.path.join(BASE_DIR, "labels", "val")

os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

images = sorted([
    f for f in os.listdir(IMG_TRAIN)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

num_val = int(len(images) * VAL_RATIO)
val_images = set(random.sample(images, num_val))

print(f"Total images: {len(images)}")
print(f"Validation images: {len(val_images)}")

for img_name in val_images:
    # image
    src_img = os.path.join(IMG_TRAIN, img_name)
    dst_img = os.path.join(IMG_VAL, img_name)
    shutil.move(src_img, dst_img)

    # label
    label_name = os.path.splitext(img_name)[0] + ".txt"
    src_lbl = os.path.join(LBL_TRAIN, label_name)
    dst_lbl = os.path.join(LBL_VAL, label_name)

    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)
    else:
        print(f"[WARN] Missing label for {img_name}")

print("Train / Val split finished.")
