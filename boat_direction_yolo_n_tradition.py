# import cv2
# import numpy as np
# import os
# import re
# import math
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# import time

# # =========== Utility Functions ============ #
# def parse_gt_file(txt_path):
#     """Parses GT file into dictionary: {frame_id: [{'p1':(x,y), 'p2':(x,y)}]}"""
#     gt_data = {}
#     current_frame = -1
#     if not os.path.exists(txt_path):
#         return {}
        
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()
        
#     for line in lines:
#         line = line.strip()
#         if not line: continue
        
#         if line.isdigit():
#             current_frame = int(line)
#             gt_data[current_frame] = []
#             continue
            
#         # Parse Object Line
#         match = re.match(r'(\d+)\s*\[(\d+),\s*(\d+)\]\s*\[(\d+),\s*(\d+)\]', line)
#         if match and current_frame != -1:
#             gt_data[current_frame].append({
#                 'p1': (int(match.group(2)), int(match.group(3))),
#                 'p2': (int(match.group(4)), int(match.group(5)))
#             })
#     return gt_data

# def calculate_gt_angle(p1, p2):
#     dx = p2[0] - p1[0]
#     dy = p2[1] - p1[1]
#     return math.degrees(math.atan2(dy, dx))

# def get_angle_diff(a1, a2):
#     diff = abs(a1 - a2) % 360
#     return min(diff, 360 - diff)

# # ======================================= #

# # ==== Traditional Boat Tracker Class ==== #
# class FlowComputer:
#     def __init__(self):
#         self.prev_gray = None
#         self.feature_params = dict(maxCorners=50, qualityLevel=0.1, minDistance=5, blockSize=5)
#         self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#     def get_average_flow(self, img_bgr, mask):
#         gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         if self.prev_gray is None:
#             self.prev_gray = gray
#             return (0, 0)

#         if cv2.countNonZero(mask) == 0:
#             self.prev_gray = gray
#             return (0, 0)
            
#         p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
#         avg_dx, avg_dy = 0, 0
        
#         if p0 is not None:
#             p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
#             if p1 is not None:
#                 good_new = p1[st==1]
#                 good_old = p0[st==1]
#                 if len(good_new) > 0:
#                     vecs = good_new - good_old
#                     mags = np.linalg.norm(vecs, axis=1)
#                     valid_vecs = vecs[mags > 0.5]
#                     if len(valid_vecs) > 0:
#                         avg_vec = np.mean(valid_vecs, axis=0)
#                         avg_dx, avg_dy = avg_vec[0], avg_vec[1]
#         self.prev_gray = gray
#         return (avg_dx, avg_dy)

# class TradBoatDetector:
#     def __init__(self):
#         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

#     def get_blue_sea_boxes(self, img, S, V):
#         # HSV thresholding (Low saturation, High value)
#         _, m_sat = cv2.threshold(S, 100, 255, cv2.THRESH_BINARY_INV)
#         _, m_val = cv2.threshold(V, 140, 255, cv2.THRESH_BINARY)
#         mask = cv2.bitwise_and(m_sat, m_val)

#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#         boxes = []
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         img_area = img.shape[0] * img.shape[1]

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area < 200 or area > img_area * 0.1:
#                 continue

#             rect = cv2.minAreaRect(cnt)
#             width, height = rect[1]
#             if width > 0 and height > 0:
#                 ratio = max(width, height) / min(width, height)
#                 if ratio > 1.6:
#                     hull = cv2.convexHull(cnt)
#                     hull_area = cv2.contourArea(hull)
#                     solidity = area / hull_area if hull_area > 0 else 0
#                     if solidity > 0.4:
#                         boxes.append(np.int32(cv2.boxPoints(rect)))
#         return boxes, mask
    
#     def get_loop_and_micro_boxes(self, img, gray, S):
#         gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         # Macro + Micro Adaptive Thresholding
#         mask_macro = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
#         mask_micro = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, -5)
#         mask_adaptive = cv2.bitwise_or(mask_macro, mask_micro)
        
#         _, m_sat = cv2.threshold(S, 50, 255, cv2.THRESH_BINARY_INV)
#         mask = cv2.bitwise_and(mask_adaptive, m_sat)
        
#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.dilate(mask, kernel, iterations=1) 
        
#         boxes = []
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         img_area = img.shape[0] * img.shape[1]
        
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if 10 < area < 10000:
#                 rect = cv2.minAreaRect(cnt)
#                 width, height = rect[1]
#                 if width > 0 and height > 0:
#                     ratio = max(width, height) / min(width, height)
#                     if ratio > 1.1:
#                         if (width * height) < (img_area * 0.05):
#                             boxes.append(np.int32(cv2.boxPoints(rect)))
#         return boxes, mask

#     def get_motion_boxes(self, img):
#         mask = self.bg_subtractor.apply(img)
#         _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.dilate(mask, kernel, iterations=4)
#         boxes = []
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             if cv2.contourArea(cnt) > 100:
#                 rect = cv2.minAreaRect(cnt)
#                 boxes.append(cv2.boxPoints(rect))
#         return boxes, mask
    
#     def merge_overlapping_boxes_unionfind(self, boxes):
#         n = len(boxes)
#         if n == 0: return []
        
#         parent = list(range(n))
#         def find(x):
#             while parent[x] != x:
#                 parent[x] = parent[parent[x]]; x = parent[x]
#             return x
#         def union(x, y):
#             rx, ry = find(x), find(y)
#             if rx != ry: parent[ry] = rx

#         for i in range(n):
#             for j in range(i + 1, n):
#                 if self.rotated_overlap(boxes[i], boxes[j]):
#                     union(i, j)

#         groups = {}
#         for i in range(n):
#             r = find(i)
#             groups.setdefault(r, []).append(boxes[i])

#         merged = []
#         for group in groups.values():
#             all_points = np.vstack(group)
#             rect = cv2.minAreaRect(all_points)
#             merged.append(np.int32(cv2.boxPoints(rect)))
#         return merged
    
#     def suppress_close_small_boxes(self, boxes, dist_thresh=100, dominance_thresh=3.0):
#         if not boxes: return []
#         boxes = sorted(boxes, key=cv2.contourArea, reverse=True)
#         kept = []
#         for b in boxes:
#             area_b = cv2.contourArea(b)
#             center_b = np.mean(b, axis=0)
#             suppress = False
#             for k in kept:
#                 area_k = cv2.contourArea(k)
#                 center_k = np.mean(k, axis=0)
#                 if np.linalg.norm(center_b - center_k) < dist_thresh:
#                     if (area_k / area_b) > dominance_thresh:
#                         suppress = True; break
#             if not suppress: kept.append(b)
#         return kept
    
#     def suppress_overlaps(self, boxes):
#         if not boxes: return []
#         boxes = sorted(boxes, key=cv2.contourArea, reverse=True)
#         kept = []
#         for b in boxes:
#             overlap = False
#             for k in kept:
#                 if self.rotated_overlap(b, k):
#                     overlap = True; break
#             if not overlap: kept.append(b)
#         return kept

#     def rotated_overlap(self, box1, box2):
#         rect1 = cv2.minAreaRect(box1)
#         rect2 = cv2.minAreaRect(box2)
#         retval, _ = cv2.rotatedRectangleIntersection(rect1, rect2)
#         return retval != cv2.INTERSECT_NONE

#     def detect(self, img):
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         S = hsv[:,:,1]
#         V = hsv[:,:,2]
        
#         # 1. Get Static Candidates (Robust)
#         boxes_A, mask_A = self.get_blue_sea_boxes(img, S, V)
#         boxes_B, mask_B = self.get_loop_and_micro_boxes(img, gray, S)
        
#         # 2. Merge Fragmented Parts (Crucial for B)
#         boxes_B = self.merge_overlapping_boxes_unionfind(boxes_B)
#         boxes_B = self.suppress_close_small_boxes(boxes_B, dist_thresh=100, dominance_thresh=2.5)

#         # 3. Get Motion Candidates (Video only)
#         boxes_C, mask_C = self.get_motion_boxes(img)
        
#         # 4. Combine All
#         all_boxes = boxes_A + boxes_B + boxes_C
        
#         # 5. Global Suppression
#         all_boxes = self.suppress_close_small_boxes(all_boxes, dist_thresh=80)
#         final_boxes = self.suppress_overlaps(all_boxes)
        
#         # Visualization mask (Just for show)
#         combined_mask = cv2.bitwise_or(mask_A, mask_B)
#         combined_mask = cv2.bitwise_or(combined_mask, mask_C)
            
#         final_centers = []
#         final_contours = []
        
#         # Format for Tracker
#         for box in final_boxes:
#             box = np.int32(box)
#             M = cv2.moments(box)
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 final_centers.append((cx, cy))
#                 final_contours.append(box)
                
#         return final_centers, final_contours, combined_mask

# class HybridTracker:
#     def __init__(self):
#         self.tracks = {} 
#         self.next_id = 0
#         self.max_history = 20

#     def update(self, centers, contours):
#         active_ids = []
#         used_dets = set()
#         for tid, data in self.tracks.items():
#             last_pos = data['history'][-1]
#             best_dist = 80.0
#             best_idx = -1
#             for i, c in enumerate(centers):
#                 if i in used_dets: continue
#                 dist = np.hypot(c[0]-last_pos[0], c[1]-last_pos[1])
#                 if dist < best_dist:
#                     best_dist = dist; best_idx = i
#             if best_idx != -1:
#                 self.tracks[tid]['history'].append(centers[best_idx])
#                 self.tracks[tid]['contour'] = contours[best_idx]
#                 if len(self.tracks[tid]['history']) > self.max_history: self.tracks[tid]['history'].pop(0)
#                 used_dets.add(best_idx)
#                 active_ids.append(tid)
#         for i, c in enumerate(centers):
#             if i not in used_dets:
#                 self.tracks[self.next_id] = {'history': [c], 'contour': contours[i]}
#                 active_ids.append(self.next_id); self.next_id += 1
#         return active_ids

#     def get_direction(self, track_id, flow_vec):
#         """ 
#         Calculates direction using PCA Axis aligned with Optical Flow.
#         flow_vec: (dx, dy) form Optical Flow for the whole image/mask
#         """
#         if track_id not in self.tracks: return None, None
#         data = self.tracks[track_id]
#         cnt = data['contour']
#         hist = data['history']
        
#         # Need history or flow to determine direction
#         if len(hist) < 2 and np.linalg.norm(flow_vec) < 0.5:
#             return None, None
            
#         # 1. Calculate PCA Axis (The "Line" of the wake)
#         rect = cv2.minAreaRect(cnt)
#         box = cv2.boxPoints(rect)
#         edge1 = box[1] - box[0]
#         edge2 = box[2] - box[1]
        
#         if np.linalg.norm(edge1) > np.linalg.norm(edge2):
#             main_axis = edge1
#         else:
#             main_axis = edge2
            
#         axis_angle_rad = math.atan2(main_axis[1], main_axis[0])
        
#         # 2. Determine "Flow" Angle (The "Arrow" on the line)
#         # Priority: Optical Flow > Centroid History
#         fx, fy = flow_vec
        
#         # If Flow is too weak, fallback to centroid history
#         if np.hypot(fx, fy) < 0.5 and len(hist) >= 5:
#             p_start = hist[0]
#             p_end = hist[-1]
#             fx = p_end[0] - p_start[0]
#             fy = p_end[1] - p_start[1]
            
#         flow_angle_rad = math.atan2(fy, fx)
        
#         # 3. Align PCA Axis to Flow
#         # We have two choices for PCA: theta, or theta + 180.
#         # Pick the one closer to the Flow direction.
        
#         diff1 = abs(math.degrees(axis_angle_rad - flow_angle_rad)) % 360
#         diff1 = min(diff1, 360 - diff1)
        
#         axis_angle_rev = axis_angle_rad + math.pi
#         diff2 = abs(math.degrees(axis_angle_rev - flow_angle_rad)) % 360
#         diff2 = min(diff2, 360 - diff2)
        
#         final_rad = axis_angle_rad if diff1 < diff2 else axis_angle_rev
#         final_deg = math.degrees(final_rad)
        
#         # For visualization vector
#         curr = hist[-1]
#         p_start = (int(curr[0] - math.cos(final_rad)*30), int(curr[1] - math.sin(final_rad)*30))
#         p_end   = (int(curr[0] + math.cos(final_rad)*30), int(curr[1] + math.sin(final_rad)*30))
        
#         return final_deg, (p_start, p_end)
# # ======================================= #

# # --- YOLO DETECTION FUNCTIONS --- #
# # Load model globally
# try:
#     yolo_model = YOLO("best_mix_150.pt")
# except Exception as e:
#     print(f"Warning: YOLO model not found. {e}")
#     yolo_model = None

# def obb_angle_from_poly(poly):
#     edges = [poly[1]-poly[0], poly[2]-poly[1], poly[3]-poly[2], poly[0]-poly[3]]
#     lengths = [np.hypot(e[0], e[1]) for e in edges]
#     major_edge = edges[np.argmax(lengths)]
#     return math.degrees(math.atan2(major_edge[1], major_edge[0]))

# def yolo_detect_centers(frame):
#     if yolo_model is None: return []
#     results = yolo_model.predict(source=frame, imgsz=640, conf=0.01, verbose=False)
#     detections = []
#     for r in results:
#         if r.obb is None: continue
#         polys = r.obb.xyxyxyxy.cpu().numpy()
#         for poly in polys:
#             cx = int(np.mean(poly[:, 0]))
#             cy = int(np.mean(poly[:, 1]))
#             angle = obb_angle_from_poly(poly)
#             detections.append({"center": (cx, cy), "angle": angle})
#     return detections

# # ==============================================================================
# # RUNNER FUNCTIONS
# # ==============================================================================

# def run_traditional_method(video_path, gt_data):
#     cap = cv2.VideoCapture(video_path)

#     out_dir = "output/boat_direction_trad_vis"
#     os.makedirs(out_dir, exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out_path = os.path.join(out_dir, os.path.basename(video_path))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

#     detector = TradBoatDetector()
#     tracker = HybridTracker()
#     flow_computer = FlowComputer()
#     flow_computer.prev_gray = None
    
#     total_error = 0
#     valid_comparisons = 0
#     frame_idx = 1
#     start_time = time.time()
#     frame_count = 0
#     total_gt_objects = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # -----------------------------
#         # 1. Traditional detection + tracking
#         # -----------------------------
#         centers, contours, mask = detector.detect(frame)
#         flow_vec = flow_computer.get_average_flow(frame, mask)
#         active_ids = tracker.update(centers, contours)

#         # -----------------------------
#         # 2. GT visualization (always draw)
#         # -----------------------------
#         if frame_idx in gt_data and len(gt_data[frame_idx]) > 0:
#             total_gt_objects += len(gt_data[frame_idx])
#             for gt_obj in gt_data[frame_idx]:
#                 cv2.arrowedLine(
#                     frame,
#                     gt_obj["p1"],
#                     gt_obj["p2"],
#                     (0, 255, 0),  # Green = GT
#                     3
#                 )

#             # ⚠️ 評估仍以「第一個 GT」為基準（與你原本一致）
#             gt_obj = gt_data[frame_idx][0]
#             gt_angle = calculate_gt_angle(gt_obj["p1"], gt_obj["p2"])

#             min_dist = 150
#             best_match_err = None
#             best_vec = None

#             # -----------------------------
#             # 3. boad_direction-style direction
#             # -----------------------------
#             for tid in active_ids:
#                 pred_orientation, vec_pts = tracker.get_direction(tid, flow_vec)

#                 if pred_orientation is None:
#                     continue

#                 curr_pos = tracker.tracks[tid]["history"][-1]

#                 dist = np.hypot(
#                     curr_pos[0] - gt_obj["p1"][0],
#                     curr_pos[1] - gt_obj["p1"][1]
#                 )

#                 if dist < min_dist:
#                     min_dist = dist
#                     best_match_err = get_angle_diff(pred_orientation, gt_angle)
#                     best_vec = vec_pts

#             if best_vec is not None:
#                 cv2.arrowedLine(frame, best_vec[0], best_vec[1], (0,255,255), 3)

#             # -----------------------------
#             # 4. Error accumulation
#             # -----------------------------
#             if best_match_err is not None:
#                 total_error += best_match_err
#                 valid_comparisons += 1

#         writer.write(frame)
#         frame_idx += 1

#     cap.release()
#     writer.release()

#     total_time = time.time() - start_time
#     avg_time = total_time / frame_count if frame_count > 0 else 0.0

#     avg_err = total_error / valid_comparisons if valid_comparisons > 0 else 999.0
#     return avg_err, valid_comparisons, avg_time, total_gt_objects

# def run_yolo_method(video_path, gt_data):
#     total_gt_objects = 0
#     cap = cv2.VideoCapture(video_path)

#     out_dir = "output/boat_direction_yolo_vis"
#     os.makedirs(out_dir, exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out_path = os.path.join(out_dir, os.path.basename(video_path))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

#     total_error = 0
#     valid_comparisons = 0
#     frame_idx = 1
#     start_time = time.time() 
#     frame_count = 0          
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break

#         frame_count += 1
        
#         detections = yolo_detect_centers(frame)
        
#         gt_objects = gt_data[frame_idx]

#         if frame_idx in gt_data:
#             total_gt_objects += len(gt_data[frame_idx])

#         for gt_obj in gt_objects:
#             gt_start = gt_obj['p1']
#             gt_angle = calculate_gt_angle(gt_obj['p1'], gt_obj['p2'])

#             best_match_err = None
#             min_dist = 100

#             for det in detections:
#                 curr_pos = det["center"]
#                 pred_orientation = det["angle"]

#                 dist = np.hypot(curr_pos[0] - gt_start[0],
#                                 curr_pos[1] - gt_start[1])

#                 if dist < min_dist:
#                     min_dist = dist

#                     err_direct = get_angle_diff(pred_orientation, gt_angle)
#                     err_flip   = get_angle_diff(pred_orientation + 180, gt_angle)

#                     best_match_err = min(err_direct, err_flip)

#                     # --- visualization polarity ---
#                     vis_angle = pred_orientation
#                     if err_flip < err_direct:
#                         vis_angle = pred_orientation + 180

#                     length = 40
#                     end_pt = (
#                         int(curr_pos[0] + length * math.cos(math.radians(vis_angle))),
#                         int(curr_pos[1] + length * math.sin(math.radians(vis_angle)))
#                     )
#                     cv2.arrowedLine(frame, curr_pos, end_pt, (0, 255, 255), 3)

#             if best_match_err is not None:
#                 total_error += best_match_err
#                 valid_comparisons += 1

#             # draw GT
#             cv2.arrowedLine(frame, gt_obj['p1'], gt_obj['p2'], (0, 255, 0), 3)

#         writer.write(frame)
#         frame_idx += 1

#     cap.release()
#     writer.release()

#     total_time = time.time() - start_time
#     avg_time = total_time / frame_count if frame_count > 0 else 0.0
#     avg_err = (total_error / valid_comparisons) if valid_comparisons > 0 else 999.0
#     return avg_err, valid_comparisons, avg_time, total_gt_objects

# # ==============================================================================
# # MAIN & TABLE GENERATION
# # ==============================================================================

# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     video_dir = os.path.join(current_dir, "Dataset_given", "data", "video")
#     gt_dir    = os.path.join(current_dir, "Dataset_given", "data", "GT")
    
#     # Files grid: d, e, f rows; 1, 2 cols
#     files_grid = [
#         ['d1', 'd2'],
#         ['e1', 'e2'],
#         ['f1', 'f2']
#     ]
    
#     comparison_data = {}

#     print(f"{'Video':<8} | {'Method':<6} | {'AvgErr':<8} | {'Frames'}")
#     print("-" * 40)

#     for row in files_grid:
#         for file_id in row:
#             video_path = os.path.join(video_dir, f"{file_id}.mp4")
#             if not os.path.exists(video_path): video_path = os.path.join(video_dir, f"{file_id}.avi")
#             txt_path = os.path.join(gt_dir, f"{file_id}.txt")
            
#             if not os.path.exists(video_path):
#                 comparison_data[file_id] = None
#                 continue
                
#             gt_data = parse_gt_file(txt_path)
            
#             # Run Trad
#             t_err, t_cnt, t_time, t_total = run_traditional_method(video_path, gt_data)
#             t_rate = (t_cnt / t_total * 100) if t_total > 0 else 0.0
#             print(f"{file_id:<8} | {'Trad':<6} | {t_err:<8.2f} | {t_cnt}")
            
#             # Run YOLO
#             y_err, y_cnt, y_time, y_total = run_yolo_method(video_path, gt_data)
#             y_rate = (y_cnt / y_total * 100) if y_total > 0 else 0.0
#             print(f"{file_id:<8} | {'YOLO':<6} | {y_err:<8.2f} | {y_cnt}")

#             comparison_data[file_id] = {
#                 't_err': t_err, 't_rate': t_rate, 't_time': t_time, 
#                 'y_err': y_err, 'y_rate': y_rate, 'y_time': y_time  
#             }
#     # ==========================================
#     # GENERATE COMPARISON TABLE (ROBUST)
#     # ==========================================
    
#     # 1. Prepare Data Matrix (Headers)
#     # ==========================================
#     # GENERATE COMPARISON TABLE (ROBUST & CLEAN)
#     # ==========================================

#     final_data = [
#         ["", "1", "2"],
#         ["d", "", ""],
#         ["e", "", ""],
#         ["f", "", ""]
#     ]

#     grid_keys = files_grid

#     fig, ax = plt.subplots(
#         figsize=(12, 6),
#         dpi=300,
#         constrained_layout=True
#     )

#     ax.axis("off")

#     tbl = ax.table(
#         cellText=final_data,
#         bbox=[0, 0, 1, 1],
#         loc="center",
#         cellLoc="center"
#     )

#     tbl.auto_set_font_size(False)
#     tbl.set_fontsize(11)

#     cells = tbl.get_celld()

#     # ----------------------------
#     # 1. Style table cells
#     # ----------------------------
#     widths = [0.10, 0.45, 0.45]

#     for (row, col), cell in cells.items():
#         cell.set_height(0.25)
#         cell.set_width(widths[col])

#         if row == 0 or col == 0:
#             cell.set_text_props(weight="bold")
#             cell.set_facecolor("#f2f2f2")
#         else:
#             cell.set_facecolor("white")
#             cell.get_text().set_text("")

#     # IMPORTANT: draw ONCE
#     fig.canvas.draw()

#     # ----------------------------
#     # 2. Draw data text (UPDATED)
#     # ----------------------------
#     for r_idx, row_keys in enumerate(grid_keys):
#         for c_idx, key in enumerate(row_keys):

#             data = comparison_data.get(key)
#             if not data:
#                 continue

#             # Retrieve Time
#             t_err, t_rate, t_time = data["t_err"], data["t_rate"], data["t_time"]
#             y_err, y_rate, y_time = data["y_err"], data["y_rate"], data["y_time"]

#             cell = cells[(r_idx + 1, c_idx + 1)]
#             transform = cell.get_transform()

#             # Comparison logic (Red if better)
#             # Error: Lower is better
#             t_err_red = (t_err != 999.0 and y_err != 999.0 and t_err < y_err)
#             y_err_red = (t_err != 999.0 and y_err != 999.0 and y_err < t_err)

#             # Time: Lower is better
#             t_time_red = (t_time < y_time)
#             y_time_red = (y_time < t_time)

#             t_err_str = f"{t_err:.1f}°" if t_err != 999.0 else "N/A"
#             y_err_str = f"{y_err:.1f}°" if y_err != 999.0 else "N/A"

#             t_rate_red = (t_rate > y_rate)
#             y_rate_red = (y_rate > t_rate)

#             # --- Row 1: Traditional ---
#             # Label
#             ax.text(0.05, 0.65, "T:", ha="left", va="center", fontsize=10, transform=transform)
#             # Error
#             ax.text(0.15, 0.65, t_err_str, ha="left", va="center", fontsize=10,
#                     color="red" if t_err_red else "black", transform=transform)
#             # Rate Text
#             ax.text(0.40, 0.65, f"{t_rate:.1f}%", ha="left", va="center", fontsize=10,
#                     color="red" if t_rate_red else "black", transform=transform)
#             # Time 
#             ax.text(0.65, 0.65, f"{t_time:.3f}s", ha="left", va="center", fontsize=10,
#                     color="red" if t_time_red else "black", transform=transform)

#             # --- Row 2: YOLO ---
#             # Label
#             ax.text(0.05, 0.35, "Y:", ha="left", va="center", fontsize=10, transform=transform)
#             # Error
#             ax.text(0.15, 0.35, y_err_str, ha="left", va="center", fontsize=10,
#                     color="red" if y_err_red else "black", transform=transform)
#             # Rate Text (Changed from "fr" to "%")
#             ax.text(0.40, 0.35, f"{y_rate:.1f}%", ha="left", va="center", fontsize=10,
#                     color="red" if y_rate_red else "black", transform=transform)
#             # Time 
#             ax.text(0.65, 0.35, f"{y_time:.3f}s", ha="left", va="center", fontsize=10,
#                     color="red" if y_time_red else "black", transform=transform)

#     plt.title(
#         "Comparison: Avg Angle Error | Tracking Rate | Avg Time/Frame",
#         fontsize=14,
#         pad=20
#     )

#     print("Saving comparison table as 'boat_direction_comparison.png'...")
#     plt.savefig("boat_direction_comparison.png", dpi=300, bbox_inches="tight")
#     fig.canvas.draw()
import cv2
import numpy as np
import os
import re
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

# =========== Utility Functions ============ #
def parse_gt_file(txt_path):
    """Parses GT file into dictionary: {frame_id: [{'p1':(x,y), 'p2':(x,y)}]}"""
    gt_data = {}
    current_frame = -1
    if not os.path.exists(txt_path):
        return {}
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.isdigit():
            current_frame = int(line)
            gt_data[current_frame] = []
            continue
            
        # Parse Object Line
        match = re.match(r'(\d+)\s*\[(\d+),\s*(\d+)\]\s*\[(\d+),\s*(\d+)\]', line)
        if match and current_frame != -1:
            gt_data[current_frame].append({
                'p1': (int(match.group(2)), int(match.group(3))),
                'p2': (int(match.group(4)), int(match.group(5)))
            })
    return gt_data

def calculate_gt_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def get_angle_diff(a1, a2):
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)

# ======================================= #

# ==== Traditional Boat Tracker Class ==== #
class FlowComputer:
    def __init__(self):
        self.prev_gray = None
        self.feature_params = dict(maxCorners=50, qualityLevel=0.1, minDistance=5, blockSize=5)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def get_average_flow(self, img_bgr, mask):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return (0, 0)

        if cv2.countNonZero(mask) == 0:
            self.prev_gray = gray
            return (0, 0)
            
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
        avg_dx, avg_dy = 0, 0
        
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]
                if len(good_new) > 0:
                    vecs = good_new - good_old
                    mags = np.linalg.norm(vecs, axis=1)
                    valid_vecs = vecs[mags > 0.5]
                    if len(valid_vecs) > 0:
                        avg_vec = np.mean(valid_vecs, axis=0)
                        avg_dx, avg_dy = avg_vec[0], avg_vec[1]
        self.prev_gray = gray
        return (avg_dx, avg_dy)

class TradBoatDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

    def get_color_boxes(self, img, S, V):
        _, m_sat = cv2.threshold(S, 110, 255, cv2.THRESH_BINARY_INV)
        _, m_val = cv2.threshold(V, 130, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(m_sat, m_val)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    if ratio > 1.3: boxes.append(cv2.boxPoints(rect))
        return boxes, mask

    def get_sunset_boxes(self, img, gray):
        B_channel = img[:,:,0] 
        _, mask_blue = cv2.threshold(B_channel, 120, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, mask_tophat = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask_blue, mask_tophat)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 50 < cv2.contourArea(cnt) < 15000:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    if ratio > 1.0: boxes.append(cv2.boxPoints(rect))
        return boxes, mask

    def get_motion_boxes(self, img):
        mask = self.bg_subtractor.apply(img)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=4)
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                rect = cv2.minAreaRect(cnt)
                boxes.append(cv2.boxPoints(rect))
        return boxes, mask

    def detect(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        S = hsv[:,:,1]
        V = hsv[:,:,2]
        
        boxes_A, mask_A = self.get_color_boxes(img, S, V)
        boxes_B, mask_B = self.get_sunset_boxes(img, gray)
        all_boxes = boxes_A + boxes_B
        combined_mask = cv2.bitwise_or(mask_A, mask_B)
        
        if len(all_boxes) == 0:
            boxes_C, mask_C = self.get_motion_boxes(img)
            all_boxes += boxes_C
            combined_mask = cv2.bitwise_or(combined_mask, mask_C)
            
        final_centers = []
        final_contours = []
        for box in all_boxes:
            box = np.int32(box)
            M = cv2.moments(box)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                is_dup = False
                for existing_c in final_centers:
                    if np.hypot(cx-existing_c[0], cy-existing_c[1]) < 20:
                        is_dup = True; break
                if not is_dup:
                    final_centers.append((cx, cy))
                    final_contours.append(box)
        return final_centers, final_contours, combined_mask

class HybridTracker:
    def __init__(self):
        self.tracks = {} 
        self.next_id = 0
        self.max_history = 20

    def update(self, centers, contours):
        active_ids = []
        used_dets = set()
        for tid, data in self.tracks.items():
            last_pos = data['history'][-1]
            best_dist = 80.0
            best_idx = -1
            for i, c in enumerate(centers):
                if i in used_dets: continue
                dist = np.hypot(c[0]-last_pos[0], c[1]-last_pos[1])
                if dist < best_dist:
                    best_dist = dist; best_idx = i
            if best_idx != -1:
                self.tracks[tid]['history'].append(centers[best_idx])
                self.tracks[tid]['contour'] = contours[best_idx]
                if len(self.tracks[tid]['history']) > self.max_history: self.tracks[tid]['history'].pop(0)
                used_dets.add(best_idx)
                active_ids.append(tid)
        for i, c in enumerate(centers):
            if i not in used_dets:
                self.tracks[self.next_id] = {'history': [c], 'contour': contours[i]}
                active_ids.append(self.next_id); self.next_id += 1
        return active_ids

    def get_direction(self, track_id, flow_vec):
        """ 
        Calculates direction using PCA Axis aligned with Optical Flow.
        flow_vec: (dx, dy) form Optical Flow for the whole image/mask
        """
        if track_id not in self.tracks: return None, None
        data = self.tracks[track_id]
        cnt = data['contour']
        hist = data['history']
        
        # Need history or flow to determine direction
        if len(hist) < 2 and np.linalg.norm(flow_vec) < 0.5:
            return None, None
            
        # 1. Calculate PCA Axis (The "Line" of the wake)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]
        
        if np.linalg.norm(edge1) > np.linalg.norm(edge2):
            main_axis = edge1
        else:
            main_axis = edge2
            
        axis_angle_rad = math.atan2(main_axis[1], main_axis[0])
        
        # 2. Determine "Flow" Angle (The "Arrow" on the line)
        # Priority: Optical Flow > Centroid History
        fx, fy = flow_vec
        
        # If Flow is too weak, fallback to centroid history
        if np.hypot(fx, fy) < 0.5 and len(hist) >= 5:
            p_start = hist[0]
            p_end = hist[-1]
            fx = p_end[0] - p_start[0]
            fy = p_end[1] - p_start[1]
            
        flow_angle_rad = math.atan2(fy, fx)
        
        # 3. Align PCA Axis to Flow
        # We have two choices for PCA: theta, or theta + 180.
        # Pick the one closer to the Flow direction.
        
        diff1 = abs(math.degrees(axis_angle_rad - flow_angle_rad)) % 360
        diff1 = min(diff1, 360 - diff1)
        
        axis_angle_rev = axis_angle_rad + math.pi
        diff2 = abs(math.degrees(axis_angle_rev - flow_angle_rad)) % 360
        diff2 = min(diff2, 360 - diff2)
        
        final_rad = axis_angle_rad if diff1 < diff2 else axis_angle_rev
        final_deg = math.degrees(final_rad)
        
        # For visualization vector
        curr = hist[-1]
        p_start = (int(curr[0] - math.cos(final_rad)*30), int(curr[1] - math.sin(final_rad)*30))
        p_end   = (int(curr[0] + math.cos(final_rad)*30), int(curr[1] + math.sin(final_rad)*30))
        
        return final_deg, (p_start, p_end)
# ======================================= #

# --- YOLO DETECTION FUNCTIONS --- #
# Load model globally
try:
    yolo_model = YOLO("best_mix_150.pt")
except Exception as e:
    print(f"Warning: YOLO model not found. {e}")
    yolo_model = None

def obb_angle_from_poly(poly):
    edges = [poly[1]-poly[0], poly[2]-poly[1], poly[3]-poly[2], poly[0]-poly[3]]
    lengths = [np.hypot(e[0], e[1]) for e in edges]
    major_edge = edges[np.argmax(lengths)]
    return math.degrees(math.atan2(major_edge[1], major_edge[0]))

def yolo_detect_centers(frame):
    if yolo_model is None: return []
    results = yolo_model.predict(source=frame, imgsz=640, conf=0.01, verbose=False)
    detections = []
    for r in results:
        if r.obb is None: continue
        polys = r.obb.xyxyxyxy.cpu().numpy()
        for poly in polys:
            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))
            angle = obb_angle_from_poly(poly)
            detections.append({"center": (cx, cy), "angle": angle})
    return detections

# ==============================================================================
# RUNNER FUNCTIONS
# ==============================================================================

def run_traditional_method(video_path, gt_data):
    cap = cv2.VideoCapture(video_path)

    out_dir = "output/boat_direction_trad_vis"
    os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(out_dir, os.path.basename(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    detector = TradBoatDetector()
    tracker = HybridTracker()
    flow_computer = FlowComputer()
    flow_computer.prev_gray = None
    
    total_error = 0
    valid_comparisons = 0
    frame_idx = 1
    start_time = time.time()
    frame_count = 0
    total_gt_objects = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # -----------------------------
        # 1. Traditional detection + tracking
        # -----------------------------
        centers, contours, mask = detector.detect(frame)
        flow_vec = flow_computer.get_average_flow(frame, mask)
        active_ids = tracker.update(centers, contours)

        # -----------------------------
        # 2. GT visualization (always draw)
        # -----------------------------
        if frame_idx in gt_data and len(gt_data[frame_idx]) > 0:
            total_gt_objects += len(gt_data[frame_idx])
            for gt_obj in gt_data[frame_idx]:
                cv2.arrowedLine(
                    frame,
                    gt_obj["p1"],
                    gt_obj["p2"],
                    (0, 255, 0),  # Green = GT
                    3
                )

            # ⚠️ 評估仍以「第一個 GT」為基準（與你原本一致）
            gt_obj = gt_data[frame_idx][0]
            gt_angle = calculate_gt_angle(gt_obj["p1"], gt_obj["p2"])

            min_dist = 150
            best_match_err = None
            best_vec = None

            # -----------------------------
            # 3. boad_direction-style direction
            # -----------------------------
            for tid in active_ids:
                pred_orientation, vec_pts = tracker.get_direction(tid, flow_vec)

                if pred_orientation is None:
                    continue

                curr_pos = tracker.tracks[tid]["history"][-1]

                dist = np.hypot(
                    curr_pos[0] - gt_obj["p1"][0],
                    curr_pos[1] - gt_obj["p1"][1]
                )

                if dist < min_dist:
                    min_dist = dist
                    best_match_err = get_angle_diff(pred_orientation, gt_angle)
                    best_vec = vec_pts

            if best_vec is not None:
                cv2.arrowedLine(frame, best_vec[0], best_vec[1], (0,255,255), 3)

            # -----------------------------
            # 4. Error accumulation
            # -----------------------------
            if best_match_err is not None:
                total_error += best_match_err
                valid_comparisons += 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    avg_time = total_time / frame_count if frame_count > 0 else 0.0

    avg_err = total_error / valid_comparisons if valid_comparisons > 0 else 999.0
    return avg_err, valid_comparisons, avg_time, total_gt_objects

def run_yolo_method(video_path, gt_data):
    total_gt_objects = 0
    cap = cv2.VideoCapture(video_path)

    out_dir = "output/boat_direction_yolo_vis"
    os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(out_dir, os.path.basename(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    total_error = 0
    valid_comparisons = 0
    frame_idx = 1
    start_time = time.time() 
    frame_count = 0          
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        detections = yolo_detect_centers(frame)
        
        gt_objects = gt_data[frame_idx]

        if frame_idx in gt_data:
            total_gt_objects += len(gt_data[frame_idx])

        for gt_obj in gt_objects:
            gt_start = gt_obj['p1']
            gt_angle = calculate_gt_angle(gt_obj['p1'], gt_obj['p2'])

            best_match_err = None
            min_dist = 100

            for det in detections:
                curr_pos = det["center"]
                pred_orientation = det["angle"]

                dist = np.hypot(curr_pos[0] - gt_start[0],
                                curr_pos[1] - gt_start[1])

                if dist < min_dist:
                    min_dist = dist

                    err_direct = get_angle_diff(pred_orientation, gt_angle)
                    err_flip   = get_angle_diff(pred_orientation + 180, gt_angle)

                    best_match_err = min(err_direct, err_flip)

                    # --- visualization polarity ---
                    vis_angle = pred_orientation
                    if err_flip < err_direct:
                        vis_angle = pred_orientation + 180

                    length = 40
                    end_pt = (
                        int(curr_pos[0] + length * math.cos(math.radians(vis_angle))),
                        int(curr_pos[1] + length * math.sin(math.radians(vis_angle)))
                    )
                    cv2.arrowedLine(frame, curr_pos, end_pt, (0, 255, 255), 3)

            if best_match_err is not None:
                total_error += best_match_err
                valid_comparisons += 1

            # draw GT
            cv2.arrowedLine(frame, gt_obj['p1'], gt_obj['p2'], (0, 255, 0), 3)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    avg_time = total_time / frame_count if frame_count > 0 else 0.0
    avg_err = (total_error / valid_comparisons) if valid_comparisons > 0 else 999.0
    return avg_err, valid_comparisons, avg_time, total_gt_objects

# ==============================================================================
# MAIN & TABLE GENERATION
# ==============================================================================

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(current_dir, "Dataset_given", "data", "video")
    gt_dir    = os.path.join(current_dir, "Dataset_given", "data", "GT")
    
    # Files grid: d, e, f rows; 1, 2 cols
    files_grid = [
        ['d1', 'd2'],
        ['e1', 'e2'],
        ['f1', 'f2']
    ]
    
    comparison_data = {}

    print(f"{'Video':<8} | {'Method':<6} | {'AvgErr':<8} | {'Frames'}")
    print("-" * 40)

    for row in files_grid:
        for file_id in row:
            video_path = os.path.join(video_dir, f"{file_id}.mp4")
            if not os.path.exists(video_path): video_path = os.path.join(video_dir, f"{file_id}.avi")
            txt_path = os.path.join(gt_dir, f"{file_id}.txt")
            
            if not os.path.exists(video_path):
                comparison_data[file_id] = None
                continue
                
            gt_data = parse_gt_file(txt_path)
            
            # Run Trad
            t_err, t_cnt, t_time, t_total = run_traditional_method(video_path, gt_data)
            t_rate = (t_cnt / t_total * 100) if t_total > 0 else 0.0
            print(f"{file_id:<8} | {'Trad':<6} | {t_err:<8.2f} | {t_cnt}")
            
            # Run YOLO
            y_err, y_cnt, y_time, y_total = run_yolo_method(video_path, gt_data)
            y_rate = (y_cnt / y_total * 100) if y_total > 0 else 0.0
            print(f"{file_id:<8} | {'YOLO':<6} | {y_err:<8.2f} | {y_cnt}")

            comparison_data[file_id] = {
                't_err': t_err, 't_rate': t_rate, 't_time': t_time, 
                'y_err': y_err, 'y_rate': y_rate, 'y_time': y_time  
            }
    # ==========================================
    # GENERATE COMPARISON TABLE (ROBUST)
    # ==========================================
    
    # 1. Prepare Data Matrix (Headers)
    # ==========================================
    # GENERATE COMPARISON TABLE (ROBUST & CLEAN)
    # ==========================================

    final_data = [
        ["", "1", "2"],
        ["d", "", ""],
        ["e", "", ""],
        ["f", "", ""]
    ]

    grid_keys = files_grid

    fig, ax = plt.subplots(
        figsize=(12, 6),
        dpi=300,
        constrained_layout=True
    )

    ax.axis("off")

    tbl = ax.table(
        cellText=final_data,
        bbox=[0, 0, 1, 1],
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    cells = tbl.get_celld()

    # ----------------------------
    # 1. Style table cells
    # ----------------------------
    widths = [0.10, 0.45, 0.45]

    for (row, col), cell in cells.items():
        cell.set_height(0.25)
        cell.set_width(widths[col])

        if row == 0 or col == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
        else:
            cell.set_facecolor("white")
            cell.get_text().set_text("")

    # IMPORTANT: draw ONCE
    fig.canvas.draw()

    # ----------------------------
    # 2. Draw data text (UPDATED)
    # ----------------------------
    for r_idx, row_keys in enumerate(grid_keys):
        for c_idx, key in enumerate(row_keys):

            data = comparison_data.get(key)
            if not data:
                continue

            # Retrieve Time
            t_err, t_rate, t_time = data["t_err"], data["t_rate"], data["t_time"]
            y_err, y_rate, y_time = data["y_err"], data["y_rate"], data["y_time"]

            cell = cells[(r_idx + 1, c_idx + 1)]
            transform = cell.get_transform()

            # Comparison logic (Red if better)
            # Error: Lower is better
            t_err_red = (t_err != 999.0 and y_err != 999.0 and t_err < y_err)
            y_err_red = (t_err != 999.0 and y_err != 999.0 and y_err < t_err)

            # Time: Lower is better
            t_time_red = (t_time < y_time)
            y_time_red = (y_time < t_time)

            t_err_str = f"{t_err:.1f}°" if t_err != 999.0 else "N/A"
            y_err_str = f"{y_err:.1f}°" if y_err != 999.0 else "N/A"

            t_rate_red = (t_rate > y_rate)
            y_rate_red = (y_rate > t_rate)

            # --- Row 1: Traditional ---
            # Label
            ax.text(0.05, 0.65, "T:", ha="left", va="center", fontsize=10, transform=transform)
            # Error
            ax.text(0.15, 0.65, t_err_str, ha="left", va="center", fontsize=10,
                    color="red" if t_err_red else "black", transform=transform)
            # Rate Text
            ax.text(0.40, 0.65, f"{t_rate:.1f}%", ha="left", va="center", fontsize=10,
                    color="red" if t_rate_red else "black", transform=transform)
            # Time 
            ax.text(0.65, 0.65, f"{t_time:.3f}s", ha="left", va="center", fontsize=10,
                    color="red" if t_time_red else "black", transform=transform)

            # --- Row 2: YOLO ---
            # Label
            ax.text(0.05, 0.35, "Y:", ha="left", va="center", fontsize=10, transform=transform)
            # Error
            ax.text(0.15, 0.35, y_err_str, ha="left", va="center", fontsize=10,
                    color="red" if y_err_red else "black", transform=transform)
            # Rate Text (Changed from "fr" to "%")
            ax.text(0.40, 0.35, f"{y_rate:.1f}%", ha="left", va="center", fontsize=10,
                    color="red" if y_rate_red else "black", transform=transform)
            # Time 
            ax.text(0.65, 0.35, f"{y_time:.3f}s", ha="left", va="center", fontsize=10,
                    color="red" if y_time_red else "black", transform=transform)

    plt.title(
        "Comparison: Avg Angle Error | Tracking Rate | Avg Time/Frame",
        fontsize=14,
        pad=20
    )

    print("Saving comparison table as 'boat_direction_comparison.png'...")
    plt.savefig("boat_direction_comparison.png", dpi=300, bbox_inches="tight")
    fig.canvas.draw()