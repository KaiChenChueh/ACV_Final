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