import os

IMAGE_DIR = "../SDXL_Dreambooth/output_images_precise_"
START_INDEX = 271 

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Step 1: 先改成暫時名稱，避免覆寫
temp_names = []
for idx, filename in enumerate(files):
    old_path = os.path.join(IMAGE_DIR, filename)
    temp_name = f"__temp__{idx}.jpg"
    temp_path = os.path.join(IMAGE_DIR, temp_name)
    os.rename(old_path, temp_path)
    temp_names.append(temp_name)

# Step 2: 再改成最終名稱 0.jpg, 1.jpg, ...
for idx, temp_name in enumerate(temp_names):
    temp_path = os.path.join(IMAGE_DIR, temp_name)
    final_index = START_INDEX + idx  
    final_path = os.path.join(IMAGE_DIR, f"{final_index}.jpg") 
    os.rename(temp_path, final_path)

print(f"Safely renamed {len(temp_names)} images.")
