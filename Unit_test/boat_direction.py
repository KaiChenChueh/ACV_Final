
# import cv2
# import numpy as np
# import os
# import re
# import math

# # ==============================================================================
# # 1. OPTICAL FLOW HELPER (The "Truth" for Direction)
# # ==============================================================================
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

#         # 1. Find points to track inside the boat mask
#         # If mask is empty, return (0,0)
#         if cv2.countNonZero(mask) == 0:
#             self.prev_gray = gray
#             return (0, 0)
            
#         p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
        
#         avg_dx, avg_dy = 0, 0
        
#         if p0 is not None:
#             # 2. Calculate Optical Flow
#             p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            
#             # 3. Select good points
#             if p1 is not None:
#                 good_new = p1[st==1]
#                 good_old = p0[st==1]
                
#                 if len(good_new) > 0:
#                     # Calculate vectors
#                     vecs = good_new - good_old
#                     # Filter out static points (noise)
#                     mags = np.linalg.norm(vecs, axis=1)
#                     valid_vecs = vecs[mags > 0.5]
                    
#                     if len(valid_vecs) > 0:
#                         # Average vector
#                         avg_vec = np.mean(valid_vecs, axis=0)
#                         avg_dx, avg_dy = avg_vec[0], avg_vec[1]

#         self.prev_gray = gray
#         return (avg_dx, avg_dy)

# # ==============================================================================
# # 2. BOAT DETECTOR (Multi-Expert + Motion)
# # ==============================================================================
# class BoatDetector:
#     def __init__(self):
#         # Background Subtractor for "Desperation Mode" (d2, e2)
#         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

#     def get_color_boxes(self, img, S, V):
#         """ Expert A: Standard Blue/Green Water (d1, e1) """
#         _, m_sat = cv2.threshold(S, 110, 255, cv2.THRESH_BINARY_INV)
#         _, m_val = cv2.threshold(V, 130, 255, cv2.THRESH_BINARY)
#         mask = cv2.bitwise_and(m_sat, m_val)
        
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
#         boxes = []
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             if cv2.contourArea(cnt) > 200:
#                 rect = cv2.minAreaRect(cnt)
#                 width, height = rect[1]
#                 if width > 0 and height > 0:
#                     ratio = max(width, height) / min(width, height)
#                     if ratio > 1.3: boxes.append(cv2.boxPoints(rect))
#         return boxes, mask

#     def get_sunset_boxes(self, img, gray):
#         """ Expert B: Sunset/Haze (f1, f2) - Blue Channel Logic """
#         B_channel = img[:,:,0] 
#         _, mask_blue = cv2.threshold(B_channel, 120, 255, cv2.THRESH_BINARY)
        
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
#         tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
#         _, mask_tophat = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
        
#         mask = cv2.bitwise_or(mask_blue, mask_tophat)
        
#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.dilate(mask, kernel, iterations=2)
        
#         boxes = []
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             if 50 < cv2.contourArea(cnt) < 15000:
#                 rect = cv2.minAreaRect(cnt)
#                 width, height = rect[1]
#                 if width > 0 and height > 0:
#                     ratio = max(width, height) / min(width, height)
#                     if ratio > 1.0: boxes.append(cv2.boxPoints(rect))
#         return boxes, mask

#     def get_motion_boxes(self, img):
#         """ Expert C: Motion (Fallback for d2/e2) """
#         mask = self.bg_subtractor.apply(img)
#         # Remove shadows (gray pixels)
#         _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.dilate(mask, kernel, iterations=4) # Heavy dilate to join parts
        
#         boxes = []
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             if cv2.contourArea(cnt) > 100:
#                 rect = cv2.minAreaRect(cnt)
#                 boxes.append(cv2.boxPoints(rect))
#         return boxes, mask

#     def detect(self, img):
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         S = hsv[:,:,1]
#         V = hsv[:,:,2]
        
#         # 1. Run Experts
#         boxes_A, mask_A = self.get_color_boxes(img, S, V)
#         boxes_B, mask_B = self.get_sunset_boxes(img, gray)
        
#         all_boxes = boxes_A + boxes_B
#         combined_mask = cv2.bitwise_or(mask_A, mask_B)
        
#         # 2. Desperation Mode (Motion)
#         # If standard detectors find little/nothing, use Motion
#         if len(all_boxes) == 0:
#             boxes_C, mask_C = self.get_motion_boxes(img)
#             all_boxes += boxes_C
#             combined_mask = cv2.bitwise_or(combined_mask, mask_C)
            
#         # Deduplicate
#         final_centers = []
#         final_contours = []
        
#         for box in all_boxes:
#             box = np.int32(box)
#             M = cv2.moments(box)
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
                
#                 is_dup = False
#                 for existing_c in final_centers:
#                     if np.hypot(cx-existing_c[0], cy-existing_c[1]) < 20:
#                         is_dup = True
#                         break
                
#                 if not is_dup:
#                     final_centers.append((cx, cy))
#                     final_contours.append(box)
                
#         return final_centers, final_contours, combined_mask

# # ==============================================================================
# # 3. HYBRID TRACKER (PCA + Optical Flow Alignment)
# # ==============================================================================
# class HybridTracker:
#     def __init__(self):
#         self.tracks = {} 
#         self.next_id = 0
#         self.max_history = 20

#     def update(self, centers, contours):
#         active_ids = []
#         used_dets = set()

#         # Update existing
#         for tid, data in self.tracks.items():
#             last_pos = data['history'][-1]
#             best_dist = 80.0
#             best_idx = -1
            
#             for i, c in enumerate(centers):
#                 if i in used_dets: continue
#                 dist = np.hypot(c[0]-last_pos[0], c[1]-last_pos[1])
#                 if dist < best_dist:
#                     best_dist = dist
#                     best_idx = i
            
#             if best_idx != -1:
#                 self.tracks[tid]['history'].append(centers[best_idx])
#                 self.tracks[tid]['contour'] = contours[best_idx]
#                 if len(self.tracks[tid]['history']) > self.max_history:
#                     self.tracks[tid]['history'].pop(0)
#                 used_dets.add(best_idx)
#                 active_ids.append(tid)

#         # Create new
#         for i, c in enumerate(centers):
#             if i not in used_dets:
#                 self.tracks[self.next_id] = {'history': [c], 'contour': contours[i]}
#                 active_ids.append(self.next_id)
#                 self.next_id += 1
                
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

# # ==============================================================================
# # 4. MAIN LOOP
# # ==============================================================================
# def parse_gt_file(txt_path):
#     gt_data = {}
#     current_frame = -1
#     if not os.path.exists(txt_path): return {}
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         if not line: continue
#         if line.isdigit():
#             current_frame = int(line)
#             gt_data[current_frame] = []
#             continue
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

# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     video_dir = os.path.join(current_dir, "Dataset", "data", "video")
#     gt_dir    = os.path.join(current_dir, "Dataset", "data", "GT")
    
#     detector = BoatDetector()
#     flow_computer = FlowComputer()
    
#     video_files = ['d1', 'd2', 'e1', 'e2', 'f1', 'f2']
#     print(f"{'Video':<10} | {'Avg Error':<15} | {'Frames Tracked':<15}")
#     print("-" * 45)

#     for file_id in video_files:
#         video_path = os.path.join(video_dir, f"{file_id}.mp4")
#         if not os.path.exists(video_path):
#              video_path = os.path.join(video_dir, f"{file_id}.avi")
        
#         txt_path = os.path.join(gt_dir, f"{file_id}.txt")
#         if not os.path.exists(video_path):
#             print(f"{file_id:<10} | {'Video Not Found':<15} | -")
#             continue

#         gt_data = parse_gt_file(txt_path)
#         cap = cv2.VideoCapture(video_path)
#         tracker = HybridTracker()
        
#         total_error = 0
#         valid_comparisons = 0
#         frame_idx = 1
        
#         # Reset BG Subtractor for each video
#         detector.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
#         flow_computer.prev_gray = None # Reset flow
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret: break
            
#             # 1. Detect
#             centers, contours, mask = detector.detect(frame)
            
#             # 2. Compute Global Optical Flow (for Direction alignment)
#             flow_vec = flow_computer.get_average_flow(frame, mask)
            
#             # 3. Track
#             active_ids = tracker.update(centers, contours)
            
#             # 4. Evaluate
#             if frame_idx in gt_data:
#                 gt_objects = gt_data[frame_idx]
#                 if len(gt_objects) > 0:
#                     gt_obj = gt_objects[0]
#                     gt_angle = calculate_gt_angle(gt_obj['p1'], gt_obj['p2'])
                    
#                     best_match_err = None
#                     min_dist = 150
                    
#                     for tid in active_ids:
#                         # Pass flow_vec to help align PCA
#                         pred_angle, vec_pts = tracker.get_direction(tid, flow_vec)
#                         if pred_angle is not None:
#                             # Match closest to GT start point
#                             # vec_pts is (start, end), we use center/start
#                             curr_pos = tracker.tracks[tid]['history'][-1]
#                             dist = np.hypot(curr_pos[0]-gt_obj['p1'][0], curr_pos[1]-gt_obj['p1'][1])
                            
#                             if dist < min_dist:
#                                 min_dist = dist
#                                 err = get_angle_diff(pred_angle, gt_angle)
#                                 best_match_err = err
                                
#                                 # Vis
#                                 cv2.arrowedLine(frame, vec_pts[0], vec_pts[1], (0, 255, 255), 3)
                    
#                     if best_match_err is not None:
#                         total_error += best_match_err
#                         valid_comparisons += 1
#                         cv2.arrowedLine(frame, gt_obj['p1'], gt_obj['p2'], (0, 255, 0), 2)

#             cv2.imshow(f"Tracking {file_id}", frame)
#             frame_idx += 1
#             if cv2.waitKey(1) & 0xFF == 27: break
        
#         cap.release()
#         cv2.destroyAllWindows()
        
#         if valid_comparisons > 0:
#             avg_err = total_error / valid_comparisons
#             print(f"{file_id:<10} | {avg_err:<15.4f} | {valid_comparisons:<15}")
#         else:
#             print(f"{file_id:<10} | {'No GT Matches':<15} | 0")
import cv2
import numpy as np
import os
import re
import math

# ==============================================================================
# 1. BOAT DETECTOR (The "Multi-Expert" Logic from Image Task)
# ==============================================================================
class BoatDetector:
    def __init__(self):
        pass 

    def get_blue_sea_boxes(self, img, S, V):
        """ Expert A: For standard blue water. """
        _, m_sat = cv2.threshold(S, 105, 255, cv2.THRESH_BINARY_INV)
        _, m_val = cv2.threshold(V, 135, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(m_sat, m_val)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 150:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    if ratio > 1.4:
                        boxes.append(cv2.boxPoints(rect))
        return boxes

    def get_difficult_boxes(self, img, gray, S):
        """ Expert B & C: For sunset/hazy scenes. """
        # Adaptive Threshold
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask_adapt = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 51, -10)
        # Blue Channel for Sunset
        B_channel = img[:,:,0]
        _, mask_blue = cv2.threshold(B_channel, 120, 255, cv2.THRESH_BINARY)
        # Top-Hat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, mask_tophat = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
        
        mask_complex = cv2.bitwise_or(mask_adapt, mask_blue)
        mask_complex = cv2.bitwise_or(mask_complex, mask_tophat)
        
        _, m_sat = cv2.threshold(S, 60, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask_complex, m_sat)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img.shape[0] * img.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < 10000:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    if ratio > 1.0:
                        if (width * height) < (img_area * 0.05):
                            boxes.append(cv2.boxPoints(rect))
        return boxes

    def detect(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        S = hsv[:,:,1]
        V = hsv[:,:,2]
        
        boxes_A = self.get_blue_sea_boxes(img, S, V)
        boxes_B = self.get_difficult_boxes(img, gray, S)
        
        # Merge (Priority to A)
        final_centers = []
        
        # Helper to get center from box
        def get_center(box):
            # --- FIX: Changed np.int0 to np.int32 ---
            box = np.int32(box) 
            M = cv2.moments(box)
            if M["m00"] != 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return (int(np.mean(box[:,0])), int(np.mean(box[:,1])))

        for b in boxes_A:
            final_centers.append(get_center(b))
            
        for b in boxes_B:
            c = get_center(b)
            # Add if not duplicate
            is_dup = False
            for ec in final_centers:
                if np.hypot(c[0]-ec[0], c[1]-ec[1]) < 20:
                    is_dup = True
                    break
            if not is_dup:
                final_centers.append(c)
                
        return final_centers

# ==============================================================================
# 2. OBJECT TRACKER (Assigns IDs frame-to-frame)
# ==============================================================================
class ObjectTracker:
    def __init__(self):
        self.tracks = {} # {id: [history_of_points]}
        self.next_id = 0
        self.max_history = 20
        self.missing_counts = {} # {id: frames_missing}

    def update(self, detected_centers):
        # Match detections to existing tracks
        used_detections = set()
        active_ids = []

        # Simple Greedy Matching
        for track_id, history in self.tracks.items():
            last_pos = history[-1]
            best_dist = 50.0 # Max distance to match
            best_idx = -1
            
            for i, center in enumerate(detected_centers):
                if i in used_detections: continue
                dist = np.hypot(center[0]-last_pos[0], center[1]-last_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                self.tracks[track_id].append(detected_centers[best_idx])
                if len(self.tracks[track_id]) > self.max_history:
                    self.tracks[track_id].pop(0)
                used_detections.add(best_idx)
                self.missing_counts[track_id] = 0
                active_ids.append(track_id)
            else:
                self.missing_counts[track_id] += 1

        # Create new tracks for unmatched detections
        for i, center in enumerate(detected_centers):
            if i not in used_detections:
                self.tracks[self.next_id] = [center]
                self.missing_counts[self.next_id] = 0
                active_ids.append(self.next_id)
                self.next_id += 1
        
        # Remove old tracks
        ids_to_delete = [tid for tid, count in self.missing_counts.items() if count > 5]
        for tid in ids_to_delete:
            del self.tracks[tid]
            del self.missing_counts[tid]
            
        return active_ids

    def get_direction(self, track_id):
        """ Calculates angle (degrees) based on history. """
        if track_id not in self.tracks: return None, None
        hist = self.tracks[track_id]
        if len(hist) < 5: return None, None # Need some history
        
        # Vector from 5 frames ago to now
        p_start = hist[0] # Oldest in buffer
        p_end = hist[-1]  # Current
        
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        
        if np.hypot(dx, dy) < 5: return None, None # Not moving
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return angle_deg, (p_start, p_end)

# ==============================================================================
# 3. UTILS & MAIN
# ==============================================================================
def parse_gt_file(txt_path):
    """
    Parses complex GT format:
    1
    1[596, 358][326, 536]
    2...
    
    Returns dict: { frame_num: [ {'id':1, 'p1':(x,y), 'p2':(x,y)}, ... ] }
    """
    gt_data = {}
    current_frame = -1
    
    if not os.path.exists(txt_path):
        return {}
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Check if line is just a frame number
        if line.isdigit():
            current_frame = int(line)
            gt_data[current_frame] = []
            continue
            
        # Parse Object Line: 1[596, 358][326, 536]
        # Regex to handle potential spaces
        match = re.match(r'(\d+)\s*\[(\d+),\s*(\d+)\]\s*\[(\d+),\s*(\d+)\]', line)
        if match and current_frame != -1:
            obj = {
                'id': int(match.group(1)),
                'p1': (int(match.group(2)), int(match.group(3))), # Start (Tail)
                'p2': (int(match.group(4)), int(match.group(5)))  # End (Head)
            }
            gt_data[current_frame].append(obj)
            
    return gt_data

def calculate_gt_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def get_angle_diff(a1, a2):
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(current_dir, "Dataset", "data", "video")
    gt_dir    = os.path.join(current_dir, "Dataset", "data", "GT")
    
    detector = BoatDetector()
    
    # Process all video files
    video_files = ['d1', 'd2', 'e1', 'e2', 'f1', 'f2']
    
    print(f"{'Video':<10} | {'Avg Error':<15} | {'Frames Tracked':<15}")
    print("-" * 45)

    for file_id in video_files:
        video_path = os.path.join(video_dir, f"{file_id}.mp4")
        if not os.path.exists(video_path):
             video_path = os.path.join(video_dir, f"{file_id}.avi")
             
        txt_path = os.path.join(gt_dir, f"{file_id}.txt")
        
        if not os.path.exists(video_path):
            print(f"{file_id:<10} | {'Video Not Found':<15} | -")
            continue

        gt_data = parse_gt_file(txt_path)
        cap = cv2.VideoCapture(video_path)
        tracker = ObjectTracker()
        
        total_error = 0
        valid_comparisons = 0
        frame_idx = 1 # GT usually starts at 1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Detect
            centers = detector.detect(frame)
            
            # 2. Track
            active_ids = tracker.update(centers)
            
            # 3. Evaluation
            if frame_idx in gt_data:
                gt_objects = gt_data[frame_idx]
                
                # For each GT object, find best matching Tracker object
                for gt_obj in gt_objects:
                    gt_start = gt_obj['p1']
                    gt_angle = calculate_gt_angle(gt_obj['p1'], gt_obj['p2'])
                    
                    best_match_err = None
                    min_dist = 100 # Match radius (pixels)
                    
                    for tid in active_ids:
                        pred_angle, vec_pts = tracker.get_direction(tid)
                        if pred_angle is not None:
                            # Match based on spatial distance (Start point vs Centroid)
                            # Current centroid is vec_pts[1]
                            curr_pos = vec_pts[1]
                            dist = np.hypot(curr_pos[0]-gt_start[0], curr_pos[1]-gt_start[1])
                            
                            if dist < min_dist:
                                min_dist = dist
                                err = get_angle_diff(pred_angle, gt_angle)
                                best_match_err = err
                                
                                # Visual Debug
                                cv2.arrowedLine(frame, vec_pts[0], vec_pts[1], (0, 255, 255), 3) # Yellow=Pred
                    
                    if best_match_err is not None:
                        total_error += best_match_err
                        valid_comparisons += 1
                        
                        # Draw GT
                        cv2.arrowedLine(frame, gt_obj['p1'], gt_obj['p2'], (0, 255, 0), 2) # Green=GT

            cv2.imshow(f"Tracking {file_id}", frame)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        if valid_comparisons > 0:
            avg_err = total_error / valid_comparisons
            print(f"{file_id:<10} | {avg_err:<15.4f} | {valid_comparisons:<15}")
        else:
            print(f"{file_id:<10} | {'No GT Matches':<15} | 0")