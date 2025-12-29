import cv2
import numpy as np
from shapely.geometry import Polygon
import os
import re

class BoatTracker:
    def __init__(self):
        pass 

    def get_blue_sea_boxes(self, img, S, V):
        """
        Expert A: For a1, a2, b1 (Standard Blue Water)
        Strategy: Catch bright, cyan-ish wakes.
        """
        # Loose Saturation (< 100) allows cyan wakes
        _, m_sat = cv2.threshold(S, 100, 255, cv2.THRESH_BINARY_INV)
        _, m_val = cv2.threshold(V, 140, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(m_sat, m_val)
        
        # Connect gaps to form solid objects
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: Big and Long
            if area > 200:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    
                    # Strict Ratio (> 1.6) rejects circles/loops
                    if ratio > 1.6:
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        if solidity > 0.4:
                            box = cv2.boxPoints(rect)
                            box = np.int32(box)
                            boxes.append(box)
        return boxes

    def get_loop_and_micro_boxes(self, img, gray, S):
        """
        Expert B: For c1, c2 (Loops / Tiny Boats)
        Strategy: Use TWO scales of Adaptive Thresholding.
        """
        # 1. Standard Adaptive (For c2 - Medium chunks)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask_macro = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 51, -10
        )
        
        # 2. Micro Adaptive (For c1 - Tiny dots) -> NEW!
        # Small block size (13) catches tiny details that Block 51 misses
        mask_micro = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 13, -5
        )
        
        # Union the two scales
        mask_adaptive = cv2.bitwise_or(mask_macro, mask_micro)

        # Strict Saturation (< 50) removes blue water noise
        _, m_sat = cv2.threshold(S, 50, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask_adaptive, m_sat)
        
        # Open operation to snap thin loop lines
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1) 
        
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img.shape[0] * img.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: Allow very small fragments (10px) for c1
            if 10 < area < 10000:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    
                    # Ratio > 1.1: Allow chunky fragments
                    if ratio > 1.1:
                        # Safety: Ignore massive boxes
                        if (width * height) < (img_area * 0.05):
                            box = cv2.boxPoints(rect)
                            box = np.int32(box)
                            boxes.append(box)
        return boxes

    def get_sunset_boxes(self, img, gray):
        """
        Expert C: For c3, b2 (Sunset / Glare)
        Strategy: Use BLUE CHANNEL extraction + Top-Hat.
        """
        # Physics Trick: Orange/Red water has LOW Blue. White foam has HIGH Blue.
        B_channel = img[:,:,0] 
        _, mask_blue = cv2.threshold(B_channel, 160, 255, cv2.THRESH_BINARY)
        
        # Top-Hat for local contrast
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, mask_tophat = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
        
        # Union them to get the best of both
        mask = cv2.bitwise_or(mask_blue, mask_tophat)
        
        # Clean up noise
        kernel_clean = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
        
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img.shape[0] * img.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    
                    # Loose ratio (> 0.8)
                    if ratio > 0.8:
                        if (width * height) < (img_area * 0.05):
                            box = cv2.boxPoints(rect)
                            box = np.int32(box)
                            boxes.append(box)
        return boxes

    def find_all_stern_waves(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        S = hsv[:,:,1]
        V = hsv[:,:,2]
        
        # 1. Run All Experts
        boxes_A = self.get_blue_sea_boxes(img, S, V)
        boxes_B = self.get_loop_and_micro_boxes(img, gray, S)
        boxes_C = self.get_sunset_boxes(img, gray)
        
        final_boxes = []
        
        # 2. Smart Merge (Priority: A > B > C)
        
        # Add all 'A' boxes (High Confidence)
        for b in boxes_A:
            final_boxes.append(b)
            
        # Add 'B' boxes only if they don't overlap with 'A'
        for b_B in boxes_B:
            center_B = np.mean(b_B, axis=0)
            is_duplicate = False
            for b_existing in final_boxes:
                center_existing = np.mean(b_existing, axis=0)
                dist = np.linalg.norm(center_B - center_existing)
                if dist < 20: 
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_boxes.append(b_B)
                
        # Add 'C' boxes only if they don't overlap with 'A' or 'B'
        for b_C in boxes_C:
            center_C = np.mean(b_C, axis=0)
            is_duplicate = False
            for b_existing in final_boxes:
                center_existing = np.mean(b_existing, axis=0)
                dist = np.linalg.norm(center_C - center_existing)
                if dist < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_boxes.append(b_C)
                
        return final_boxes, None

    def calculate_iou(self, pred_points, gt_points):
        try:
            poly_pred = Polygon(pred_points)
            poly_gt = Polygon(gt_points)
            if not poly_pred.is_valid or not poly_gt.is_valid: return 0.0
            intersection = poly_pred.intersection(poly_gt).area
            union = poly_pred.union(poly_gt).area
            return 0.0 if union == 0 else intersection / union
        except:
            return 0.0

def parse_gt_file(txt_path):
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

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "Dataset", "data", "photo")
    gt_dir  = os.path.join(current_dir, "Dataset", "data", "GT")
    
    tracker = BoatTracker()
    test_files = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3'] 

    for file_id in test_files:
        img_path = os.path.join(img_dir, f"{file_id}.jpg")
        txt_path = os.path.join(gt_dir, f"{file_id}.txt")
        print(f"\n--- Processing {file_id} ---")
        
        if not os.path.exists(img_path):
            print(f"Skipping {file_id}: Image not found.")
            continue
            
        img = cv2.imread(img_path)
        gt_polys = parse_gt_file(txt_path)
        
        pred_boxes, _ = tracker.find_all_stern_waves(img)
        
        total_iou = 0
        vis_img = img.copy()

        if gt_polys:
            for gt_poly in gt_polys:
                best_iou_for_this_boat = 0
                cv2.drawContours(vis_img, [np.array(gt_poly)], 0, (0, 255, 0), 2)
                for pred in pred_boxes:
                    iou = tracker.calculate_iou(pred, gt_poly)
                    if iou > best_iou_for_this_boat:
                        best_iou_for_this_boat = iou
                total_iou += best_iou_for_this_boat
            avg_iou = total_iou / len(gt_polys)
        else:
            avg_iou = 0.0

        print(f"Average IoU for {file_id}: {avg_iou:.4f}")

        for p in pred_boxes:
            cv2.drawContours(vis_img, [p], 0, (0, 0, 255), 1)
        
        cv2.putText(vis_img, f"Avg IoU: {avg_iou:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(f"Result {file_id}", vis_img)
        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()