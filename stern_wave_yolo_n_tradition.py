# ACV_Final Project
# Author: Stanley Chueh
# Date: 2026-01-08
# Description: Stern Wave Detection using Traditional CV and YOLOv8-OBB
# Usage: python stern_wave_yolo_n_tradition.py

'''
============================================================
Stern Wave Detection & Comparison Framework
------------------------------------------------------------

This script implements and compares two approaches for stern
wave (wake) detection in maritime images:

1) Traditional Computer Vision Pipeline
2) Deep Learning Pipeline (YOLOv8 Oriented Bounding Boxes)

------------------------------------------------------------
PIPELINE OVERVIEW
------------------------------------------------------------

I. Traditional Computer Vision Method (BoatTracker)

   A. Candidate Generation
      - Box_A: Large extended wake (blue sea patterns)
      - Box_B: Fragmented waves / micro-wave structures

   B. Wake Refinement (Traditional)
      1) Merge overlapping Box_B fragments
         - Union-Find merging using rotated rectangle overlap

      2) Local Dominance Suppression
         - For nearby boxes:
             • Keep dominant (largest) wake
             • Suppress smaller boxes ONLY if dominance is strong
         - Controlled by:
             • dist_thresh       (spatial proximity)
             • dominance_thresh (area dominance ratio)

      3) Global Suppression
         - Apply dominance suppression again across all boxes

      4) Hard Overlap Removal
         - Final guarantee:
             • No two output boxes overlap (rotated geometry)

   C. Output
      - Final set of non-overlapping wake bounding boxes

------------------------------------------------------------
II. Deep Learning Method (YOLOv8-OBB)

   - Uses pretrained YOLOv8 model with oriented bounding boxes
   - Directly predicts rotated bounding boxes for wakes
   - No post-processing beyond confidence filtering

------------------------------------------------------------
III. Evaluation

   - Ground truth provided as polygon annotations
   - Metric: Polygon-based Intersection over Union (IoU)
   - Strategy:
       • For each GT object, select best matching prediction
       • Report average IoU over all GT objects

------------------------------------------------------------
IV. Visualization & Comparison

   - Traditional results: green contours
   - YOLO results: red contours
   - Ground truth: blue contours
   - Outputs saved separately for visual comparison

------------------------------------------------------------
'''

import cv2
import numpy as np
from shapely.geometry import Polygon
import os
import re
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =========== Utility Functions ============ #
def parse_gt_file(txt_path):
    """Parses the Ground Truth text file into a list of polygon points."""
    gt_polygons = []
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            coords = re.findall(r'\d+', line)
            if len(coords) >= 8:
                points = []
                for i in range(0, 8, 2):
                    points.append([int(coords[i]), int(coords[i+1])])
                gt_polygons.append(points)
    return gt_polygons

def calculate_iou_poly(pred_points, gt_points):
    """Calculates IoU between two polygons using Shapely."""
    try:
        poly_pred = Polygon(pred_points)
        poly_gt = Polygon(gt_points)
        if not poly_pred.is_valid or not poly_gt.is_valid: return 0.0
        intersection = poly_pred.intersection(poly_gt).area
        union = poly_pred.union(poly_gt).area
        return 0.0 if union == 0 else intersection / union
    except:
        return 0.0

def compute_avg_iou(pred_boxes, gt_polys):
    """
    Computes the Average IoU for a list of predictions against ground truth.
    Strategy: For each GT object, find the best matching Prediction.
    """
    if not gt_polys:
        return 0.0
    
    total_iou = 0.0
    for gt_poly in gt_polys:
        best_iou_for_this_boat = 0
        for pred in pred_boxes:
            iou = calculate_iou_poly(pred, gt_poly)
            if iou > best_iou_for_this_boat:
                best_iou_for_this_boat = iou
        total_iou += best_iou_for_this_boat
    
    return total_iou / len(gt_polys)

def compute_tp_fp_fn(pred_boxes, gt_polys, iou_thresh=0.3):
    matched_gt = set()
    TP = 0
    FP = 0

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_polys):
            iou = calculate_iou_poly(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    FN = len(gt_polys) - len(matched_gt)
    return TP, FP, FN


# ======================================= #

# ==== Traditional Boat Tracker Class ==== #
class BoatTracker:
    def __init__(self):
        pass 
    
    # Box_A
    def get_blue_sea_boxes(self, img, S, V):

        # HSV thresholding(Low saturation, High value) 
        _, m_sat = cv2.threshold(S, 100, 255, cv2.THRESH_BINARY_INV)
        _, m_val = cv2.threshold(V, 140, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(m_sat, m_val)

        # Morphological Closing
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img.shape[0] * img.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 200 or area > img_area * 0.1:
                continue

            # Filter by area and aspect ratio
            if area > 200:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    if ratio > 1.6:
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        if solidity > 0.4:
                            boxes.append(np.int32(cv2.boxPoints(rect)))
        return boxes

    # Box_B
    def get_loop_and_micro_boxes(self, img, gray, S):
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask_macro = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
        mask_micro = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, -5)
        mask_adaptive = cv2.bitwise_or(mask_macro, mask_micro)
        _, m_sat = cv2.threshold(S, 50, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask_adaptive, m_sat)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1) 
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img.shape[0] * img.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 10000:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    if ratio > 1.1:
                        if (width * height) < (img_area * 0.05):
                            boxes.append(np.int32(cv2.boxPoints(rect)))
        return boxes
    
    # Box_B Filter
    def suppress_close_small_boxes(
        self,
        boxes,
        dist_thresh=100,
        dominance_thresh=3.0, #3.0
    ):
        if not boxes:
            return []

        # largest → smallest
        boxes = sorted(boxes, key=cv2.contourArea, reverse=True)
        kept = []

        for b in boxes:
            area_b = cv2.contourArea(b)
            center_b = np.mean(b, axis=0)

            suppress = False

            for k in kept:
                area_k = cv2.contourArea(k)
                center_k = np.mean(k, axis=0)

                dist = np.linalg.norm(center_b - center_k)

                if dist < dist_thresh:
                    dominance_ratio = area_k / area_b

                    if dominance_ratio > dominance_thresh:
                        suppress = True

                    break  # stop checking neighbors

            if not suppress:
                kept.append(b)

        return kept

    # Box_B Merge
    def rotated_overlap(self, box1, box2):
        rect1 = cv2.minAreaRect(box1)
        rect2 = cv2.minAreaRect(box2)
        retval, _ = cv2.rotatedRectangleIntersection(rect1, rect2)
        return retval != cv2.INTERSECT_NONE

    def suppress_overlaps(self, boxes):
        if not boxes:
            return []

        boxes = sorted(boxes, key=cv2.contourArea, reverse=True)
        kept = []

        for b in boxes:
            overlap = False
            for k in kept:
                if self.rotated_overlap(b, k):
                    overlap = True
                    break
            if not overlap:
                kept.append(b)

        return kept

    def merge_overlapping_boxes_unionfind(
        self,
        boxes,
    ):
        """
        Union-Find merge using IoU OR center distance.
        Designed for wake / wave structures.
        """
        n = len(boxes)
        if n == 0:
            return []

        # ---- Step 1: AABB + centers ----
        aabbs = []
        centers = []
        for b in boxes:
            xs = b[:, 0]
            ys = b[:, 1]
            aabbs.append([xs.min(), ys.min(), xs.max(), ys.max()])
            centers.append(np.array([xs.mean(), ys.mean()]))

        # ---- Step 2: Union-Find ----
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # ---- Step 3: Hybrid merge rule ----
        for i in range(n):
            for j in range(i + 1, n):
                if self.rotated_overlap(boxes[i], boxes[j]):
                    union(i, j)

        # ---- Step 4: Collect components ----
        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(boxes[i])

        # ---- Step 5: Merge each component ----
        merged = []
        for group in groups.values():
            all_points = np.vstack(group)
            rect = cv2.minAreaRect(all_points)
            merged.append(np.int32(cv2.boxPoints(rect)))

        return merged

    def run_detection(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        boxes_A = self.get_blue_sea_boxes(img, S, V)
        boxes_B = self.get_loop_and_micro_boxes(img, gray, S)

        # merge fragmented wake boxes only
        boxes_B = self.merge_overlapping_boxes_unionfind(
            boxes_B,
        )

        boxes_B = self.suppress_close_small_boxes(
            boxes_B,
            dist_thresh=100,
            dominance_thresh=2.5,
        )

        # combine all boxes
        all_boxes = boxes_A + boxes_B

        all_boxes = self.suppress_close_small_boxes(
            all_boxes,
            dist_thresh=80, #80
        )

        # FINAL hard constraint: no overlaps allowed
        final_boxes = self.suppress_overlaps(all_boxes)

        return final_boxes

# ==== YOLO Detection Function ==== #

# Load model(warm up)
try:
    yolo_model = YOLO("best_mix_150.pt")

    # ---- YOLO WARM-UP (NOT TIMED) ----
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        _ = yolo_model.predict(
            source=dummy,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None


def run_yolo_detection(img):
    if yolo_model is None: return []
    
    results = yolo_model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    obb_boxes = []

    for r in results:
        if r.obb is None or r.obb.xyxyxyxy is None: continue
        polys = r.obb.xyxyxyxy.cpu().numpy()
        for p in polys:
            if p.shape == (4, 2):
                pts = p.astype(np.int32)
            elif p.shape == (8,):
                pts = np.array([[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]], dtype=np.int32)
            else:
                continue
            obb_boxes.append(pts)
    return obb_boxes

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "Dataset_given", "data", "photo")
    gt_dir  = os.path.join(current_dir, "Dataset_given", "data", "GT")
    
    tracker = BoatTracker()
    test_files = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3']
    
    # Structure to hold results: { 'a1': {'yolo_iou': X, 'yolo_time': Y, 'trad_iou': Z, 'trad_time': W}, ... }
    comparison_data = {}

    print(f"{'Image':<10} | {'Method':<12} | {'Time (s)':<10} | {'Avg IoU':<10}")
    print("-" * 50)

    save_root = "output/"
    trad_dir = os.path.join(save_root, "stern_wave_trad_vis")
    yolo_dir = os.path.join(save_root, "stern_wave_yolo_vis")

    os.makedirs(trad_dir, exist_ok=True)
    os.makedirs(yolo_dir, exist_ok=True)

    for file_id in test_files:
        img_path = os.path.join(img_dir, f"{file_id}.jpg")
        txt_path = os.path.join(gt_dir, f"{file_id}.txt")
        
        if not os.path.exists(img_path):
            print(f"Skipping {file_id}: Image not found.")
            comparison_data[file_id] = None
            continue

        img = cv2.imread(img_path)
        gt_polys = parse_gt_file(txt_path)
        
        # --- 1. Run Traditional Method ---
        start_t = time.time()
        trad_boxes = tracker.run_detection(img)
        time_trad = time.time() - start_t
        iou_trad = compute_avg_iou(trad_boxes, gt_polys)

        # --- 2. Run YOLO Method ---
        start_y = time.time()
        yolo_boxes = run_yolo_detection(img)
        time_yolo = time.time() - start_y
        iou_yolo = compute_avg_iou(yolo_boxes, gt_polys)

        # --- 3. Compute TP, FP, FN ---
        tp_t, fp_t, fn_t = compute_tp_fp_fn(trad_boxes, gt_polys, iou_thresh=0.3)
        tp_y, fp_y, fn_y = compute_tp_fp_fn(yolo_boxes, gt_polys, iou_thresh=0.3)


        # --- Store Results ---
        comparison_data[file_id] = {
            'trad_time': time_trad,
            'trad_iou': iou_trad,
            'yolo_time': time_yolo,
            'yolo_iou': iou_yolo,

            'trad_tp': tp_t,
            'trad_fp': fp_t,
            'trad_fn': fn_t,

            'yolo_tp': tp_y,
            'yolo_fp': fp_y,
            'yolo_fn': fn_y
        }
        
        print(f"{file_id:<10} | {'Trad':<12} | {time_trad:.4f}     | {iou_trad:.4f}")
        print(f"{'':<10} | {'YOLO':<12} | {time_yolo:.4f}     | {iou_yolo:.4f}")

        # --- Save Traditional result ---
        trad_img = img.copy()
        cv2.drawContours(trad_img, trad_boxes, -1, (0, 255, 0), 2)

        # 2. Draw Ground Truth (Blue) - DO THIS BEFORE SAVING
        cv2.drawContours(trad_img, np.array(gt_polys, dtype=np.int32), -1, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(trad_dir, f"{file_id}_trad.jpg"), trad_img)

        # --- Save YOLO result ---
        yolo_img = img.copy()
        cv2.drawContours(yolo_img, yolo_boxes, -1, (0, 0, 255), 2)

        # 2. Draw Ground Truth (Blue) - DO THIS BEFORE SAVING
        cv2.drawContours(yolo_img, np.array(gt_polys, dtype=np.int32), -1, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(yolo_dir, f"{file_id}_yolo.jpg"), yolo_img)


    cv2.destroyAllWindows()

    # === Generate comparison table ==== #
    
    final_data = [
        ["", "1", "2", "3"],         
        ["a", "", "", ""],            
        ["b", "", "", ""],          
        ["c", "", "", ""]           
    ]
    
    # Fill the data into the matrix(images) 
    grid_keys = [
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        ["c1", "c2", "c3"]
    ]
    
    for r_idx, row_keys in enumerate(grid_keys):
        for c_idx, key in enumerate(row_keys):
            data = comparison_data.get(key)
            if data:
                text = (f"T: {data['trad_iou']:.2f} / {data['trad_time']:.3f}s\n"
                        f"Y: {data['yolo_iou']:.2f} / {data['yolo_time']:.3f}s")
            else:
                text = "X"
            
            # +1 because row 0 is headers, col 0 is headers
            final_data[r_idx + 1][c_idx + 1] = text

    # Setup plotting
    fig, ax = plt.subplots(figsize=(12, 6)) # Wider figure for better spacing
    ax.axis("off")

    # Create Table
    # Define widths: First column (labels) is narrower (0.1), others are equal (0.3)
    col_widths = [0.1, 0.3, 0.3, 0.3]
    
    tbl = ax.table(
        cellText=[[""] * 4 for _ in range(4)], 
        loc="center",
        cellLoc="center",
        colWidths=col_widths
    )

    cells = tbl.get_celld()

    cells[(0,1)].get_text().set_text("1")
    cells[(0,2)].get_text().set_text("2")
    cells[(0,3)].get_text().set_text("3")

    cells[(1,0)].get_text().set_text("a")
    cells[(2,0)].get_text().set_text("b")
    cells[(3,0)].get_text().set_text("c")

    cells = tbl.get_celld()
    for (row, col), cell in cells.items():
        if row > 0 and col > 0:
            cell.get_text().set_text("")  # clear data text


    # Manual Styling Loop (The Fix)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    
    cells = tbl.get_celld()

    for (row, col), cell in cells.items():
        cell.set_height(0.2)

        # Header cells
        if row == 0 or col == 0:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#f2f2f2')
            continue
    cells = tbl.get_celld()

    fig.canvas.draw() 

    for r_idx, row_keys in enumerate(grid_keys):
        for c_idx, key in enumerate(row_keys):
            data = comparison_data.get(key)
            if not data:
                continue

            t_iou = data["trad_iou"]
            t_time = data["trad_time"]
            y_iou = data["yolo_iou"]
            y_time = data["yolo_time"]

            # Compare ONLY T vs Y
            t_iou_red  = t_iou > y_iou
            y_iou_red  = y_iou > t_iou

            t_time_red = t_time < y_time
            y_time_red = y_time < t_time

            cell = cells[(r_idx + 1, c_idx + 1)]
            transform = cell.get_transform()
            
            # Traditional line
            ax.text(
                0.40, 0.65,
                f"T: {t_iou:.2f}",
                ha="right",
                va="center",
                fontsize=11,
                color="red" if t_iou_red else "black",
                transform=transform
            )

            ax.text(
                0.50, 0.65,
                "/",
                ha="center",
                va="center",
                fontsize=11,
                color="black",   # ALWAYS BLACK
                transform=transform
            )

            ax.text(
                0.60, 0.65,
                f"{t_time:.3f}s",
                ha="left",
                va="center",
                fontsize=11,
                color="red" if t_time_red else "black",
                transform=transform
            )

            # YOLO line
            ax.text(
                0.40, 0.35,
                f"Y: {y_iou:.2f}",
                ha="right",
                va="center",
                fontsize=11,
                color="red" if y_iou_red else "black",
                transform=transform
            )

            ax.text(
                0.50, 0.35,
                "/",
                ha="center",
                va="center",
                fontsize=11,
                color="black",   # ALWAYS BLACK
                transform=transform
            )

            ax.text(
                0.60, 0.35,
                f"{y_time:.3f}s",
                ha="left",
                va="center",
                fontsize=11,
                color="red" if y_time_red else "black",
                transform=transform
            )

    plt.title("Comparison: Traditional (T) vs YOLO (Y)\nMetric: Avg IoU / Exec Time", pad=20, fontsize=14)

    plt.savefig("stern_wave_comparison_table.png", dpi=300, bbox_inches="tight")
    print("\nTable saved as 'stern_wave_comparison_table.png'")

    # ===== TP / FP / FN BAR CHART ===== #

    image_ids = []
    trad_fp_list = []
    yolo_fp_list = []

    for key in test_files:
        data = comparison_data.get(key)
        if data is None:
            continue

        image_ids.append(key)
        trad_fp_list.append(data['trad_fp'])
        yolo_fp_list.append(data['yolo_fp'])

    x = np.arange(len(image_ids))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, trad_fp_list, width, label='Traditional FP')
    plt.bar(x + width/2, yolo_fp_list, width, label='YOLO FP')

    plt.xticks(x, image_ids)
    plt.ylabel("False Positive Count")
    plt.xlabel("Image ID")
    plt.title("False Positive Comparison per Image(IoU Threshold=0.3)")
    plt.legend()
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig("fp_per_image_comparison.png", dpi=300)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Plot window closed by user.")
