import os
import xml.etree.ElementTree as ET
import math

XML_PATH = "annotations.xml"
OUTPUT_DIR = "labels"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_angle(angle):
    """
    Normalize angle to [-90, 90)
    """
    angle = angle % 360
    if angle >= 180:
        angle -= 360
    if angle < -90:
        angle += 180
    if angle >= 90:
        angle -= 180
    return angle

tree = ET.parse(XML_PATH)
root = tree.getroot()

for image in root.findall("image"):
    img_name = image.get("name")
    img_w = float(image.get("width"))
    img_h = float(image.get("height"))

    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_name)

    lines = []

    for box in image.findall("box"):
        class_id = box.get("label")

        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))

        # Axis-aligned bbox (YOLO still uses this)
        cx = (xtl + xbr) / 2.0 / img_w
        cy = (ytl + ybr) / 2.0 / img_h
        w  = (xbr - xtl) / img_w
        h  = (ybr - ytl) / img_h

        # Rotation
        cvat_angle = float(box.get("rotation", 0.0))
        yolo_angle = normalize_angle(-cvat_angle)

        lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {yolo_angle:.2f}"
        )

    if lines:
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

print("YOLOv8-OBB label conversion finished.")

