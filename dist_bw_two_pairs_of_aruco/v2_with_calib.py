import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
from datetime import datetime
from picamera2 import Picamera2
import threading
import queue

# Real side length of each ArUco marker in mm
MARKER_SIZE_MM = 54.0   # marker size in mm

# Empirical calibration factor for gap in mm (tuned from your test)
GAP_CALIB_FACTOR = 0.924

# Tilt detection threshold (degrees) - warn if tilt exceeds this
TILT_WARNING_THRESHOLD = 5.0  # degrees
TILT_ERROR_THRESHOLD = 10.0   # degrees - significant error expected

# Camera Calibration Data (Globals)
CAMERA_MATRIX = None
DIST_COEFFS = None

def load_calibration(filename="calibration_params.npz"):
    """
    Load camera matrix and distortion coefficients from a file.
    Supports .npz (numpy) and .json formats.
    """
    global CAMERA_MATRIX, DIST_COEFFS
    
    if not os.path.exists(filename):
        print(f"Calibration file {filename} not found.")
        return False

    try:
        if filename.endswith('.npz'):
            with np.load(filename) as X:
                CAMERA_MATRIX = X['camera_matrix']
                DIST_COEFFS = X['dist_coeffs']
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
                CAMERA_MATRIX = np.array(data['cameraMatrix'])
                DIST_COEFFS = np.array(data['distCoeffs'])
        else:
            print(f"Error: Unsupported calibration file format. Only .npz and .json files are supported.")
            return False

        print(f"Loaded calibration from {filename}")
        return True
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return False

def get_marker_pose_3d(corners, marker_size, camera_matrix, dist_coeffs):
    """
    Estimate the 3D pose of a single marker.
    Returns rvec, tvec, and center_3d point.
    """
    # corners shape must be (1, 4, 2) for estimatePoseSingleMarkers
    # if input is (4,2), reshape it
    c = corners.reshape((1, 4, 2))
    
    # Estimate pose
    # Note: estimatePoseSingleMarkers is deprecated in 4.7+ but still common.
    # For newer opencv-contrib-python, use cv2.aruco.estimatePoseSingleMarkers
    # or objPoints logic with solvePnP.
    # This works for standard opencv-contrib setups.
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(c, marker_size, camera_matrix, dist_coeffs)
    
    # Unpack from (1,1,3) array
    if rvec is not None:
        return rvec[0][0], tvec[0][0]
    return None, None

def calculate_3d_distance(tvec1, tvec2):
    """
    Calculate 3D distance and component differences between two 3D points.
    tvec: [x, y, z]
    """
    if tvec1 is None or tvec2 is None:
        return None, None, None, None
    
    delta = tvec2 - tvec1
    dist_3d = np.linalg.norm(delta)
    
    dx = delta[0]
    dy = delta[1]
    dz = delta[2]
    
    return dist_3d, dx, dy, dz



def order_corners(corners):
    """
    Reorder corners to: [top-left, top-right, bottom-right, bottom-left]
    based on image coordinates (y then x).
    corners: (4,2)
    """
    pts = np.array(corners, dtype=np.float32)

    # sort by y (row)
    idx_by_y = np.argsort(pts[:, 1])
    top2 = pts[idx_by_y][:2]
    bottom2 = pts[idx_by_y][2:]

    # sort each pair by x (column)
    top2 = top2[np.argsort(top2[:, 0])]
    bottom2 = bottom2[np.argsort(bottom2[:, 0])]

    tl, tr = top2
    bl, br = bottom2

    # return in TL, TR, BR, BL order
    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_side_lengths_and_ratios(ordered_corners):
    """
    ordered_corners: (4,2) in [tl, tr, br, bl] order
    returns:
        side_lengths_px [top, right, bottom, left]
        side_lengths_mm [top, right, bottom, left]
        mm_per_px_list  [top_ratio, right_ratio, bottom_ratio, left_ratio]
    """
    tl, tr, br, bl = ordered_corners

    top_px = float(np.linalg.norm(tr - tl))
    right_px = float(np.linalg.norm(br - tr))
    bottom_px = float(np.linalg.norm(bl - br))
    left_px = float(np.linalg.norm(tl - bl))

    # mm/px ratios for each side using real marker size
    top_ratio = MARKER_SIZE_MM / top_px
    right_ratio = MARKER_SIZE_MM / right_px
    bottom_ratio = MARKER_SIZE_MM / bottom_px
    left_ratio = MARKER_SIZE_MM / left_px

    side_lengths_px = [top_px, right_px, bottom_px, left_px]
    mm_per_px_list = [top_ratio, right_ratio, bottom_ratio, left_ratio]

    side_lengths_mm = [
        top_px * top_ratio,
        right_px * right_ratio,
        bottom_px * bottom_ratio,
        left_px * left_ratio,
    ]

    return side_lengths_px, side_lengths_mm, mm_per_px_list


# ---------- RAY–SEGMENT INTERSECTION ----------

def intersect_ray_segment(p0, r, a, b):
    """
    Intersect a ray p(t) = p0 + t*r, t >= 0
    with a segment from a to b (0 <= u <= 1).

    Returns (t, intersection_point) if they intersect with t >= 0 and 0<=u<=1,
    else None.
    """
    p0 = np.asarray(p0, dtype=float)
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    s = b - a  # segment direction

    denom = s[0] * (-r[1]) - s[1] * (-r[0])
    if abs(denom) < 1e-9:
        return None  # parallel or degenerate

    dx = p0[0] - a[0]
    dy = p0[1] - a[1]

    u = (dx * (-r[1]) - dy * (-r[0])) / denom
    t = (s[0] * dy - s[1] * dx) / denom

    if u < 0.0 or u > 1.0 or t < 0.0:
        return None

    intersection_point = p0 + t * r
    return t, intersection_point


def calculate_marker_rotation(ordered_corners):
    """
    Calculate the rotation angle of the marker in degrees.
    Returns rotation angle (0-360 degrees) where 0 is horizontal.
    """
    tl, tr, br, bl = ordered_corners
    
    # Calculate rotation from top edge (TL to TR)
    top_edge = tr - tl
    # Calculate angle from horizontal (x-axis)
    angle_rad = np.arctan2(top_edge[1], top_edge[0])
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to 0-360 degrees
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg


def detect_marker_tilt(ordered_corners):
    """
    Detect tilt angle of marker by analyzing corner geometry.
    Returns tilt_angle (degrees), tilt_correction_factor, and tilt_status.
    """
    tl, tr, br, bl = ordered_corners
    
    # Calculate side lengths
    top_side = np.linalg.norm(tr - tl)
    bottom_side = np.linalg.norm(br - bl)
    left_side = np.linalg.norm(bl - tl)
    right_side = np.linalg.norm(br - tr)
    
    # Calculate angles at corners
    # Top-left angle
    vec1 = tr - tl
    vec2 = bl - tl
    angle_tl = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
    angle_tl_deg = np.degrees(angle_tl)
    
    # Top-right angle
    vec1 = tl - tr
    vec2 = br - tr
    angle_tr = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
    angle_tr_deg = np.degrees(angle_tr)
    
    # Calculate deviation from 90 degrees (perfect square)
    avg_angle = (angle_tl_deg + angle_tr_deg) / 2.0
    angle_deviation = abs(90.0 - avg_angle)
    
    # Calculate aspect ratio distortion
    # For a square marker, top/bottom and left/right should be equal
    horizontal_ratio = abs(top_side - bottom_side) / max(top_side, bottom_side) if max(top_side, bottom_side) > 0 else 0
    vertical_ratio = abs(left_side - right_side) / max(left_side, right_side) if max(left_side, right_side) > 0 else 0
    
    # Combined tilt metric
    tilt_angle = max(angle_deviation, horizontal_ratio * 90, vertical_ratio * 90)
    
    # Calculate correction factor based on perspective
    # When tilted, measurements are shorter than actual
    # Correction: 1 / cos(tilt_angle_in_radians)
    tilt_rad = np.radians(tilt_angle)
    if tilt_angle < 0.5:  # Less than 0.5 degrees - no correction needed
        correction_factor = 1.0
        tilt_status = "OK"
    elif tilt_angle < TILT_WARNING_THRESHOLD:
        correction_factor = 1.0 / np.cos(tilt_rad) if np.cos(tilt_rad) > 0 else 1.0
        tilt_status = "Minor"
    elif tilt_angle < TILT_ERROR_THRESHOLD:
        correction_factor = 1.0 / np.cos(tilt_rad) if np.cos(tilt_rad) > 0 else 1.0
        tilt_status = "Warning"
    else:
        correction_factor = 1.0 / np.cos(tilt_rad) if np.cos(tilt_rad) > 0 else 1.0
        tilt_status = "Error"
    
    return tilt_angle, correction_factor, tilt_status


def compute_perp_gap_for_point(start_point, n_unit, seg_a, seg_b, mm_per_px, calib_factor, tilt_correction=1.0):
    """
    Shoot a ray from start_point along n_unit to intersect the segment (seg_a -> seg_b).
    Returns (gap_px, gap_mm_raw, gap_mm_calib, hit_point) or (None, None, None, None) if no hit.
    """
    res = intersect_ray_segment(start_point, n_unit, seg_a, seg_b)
    if res is None:
        return None, None, None, None

    t_min, hit_point = res
    gap_px = float(t_min)  # since n_unit is unit length
    gap_mm_raw = gap_px * mm_per_px
    gap_mm_calib = gap_mm_raw * calib_factor * tilt_correction  # Apply tilt correction
    return gap_px, gap_mm_raw, gap_mm_calib, hit_point


def process_pair(frame, corners_list, idx_left, idx_right,
    label_left, label_right,
                 text_y_offset, pair_name,
                 camera_matrix=None, dist_coeffs=None):
    """
    Process one pair of markers:
    - compute mm/px (average over 8 ratios of this pair only)
    - shoot 3 straight horizontal lines (to the right) from right edge of left marker
    - intersect with LEFT edge of right marker
    - draw lines and text for this pair
    - IF CALIBRATED: compute 3D pose and distance

    Returns (final_gap_mm, avg_mm_per_px, gaps_info, corner_info, tilt_info, pose_info) 
    or (None, None, None, None, None, None) on failure.
    """
    # reorder corners of each marker into TL, TR, BR, BL
    left_corners_raw = corners_list[idx_left].reshape((4, 2))
    right_corners_raw = corners_list[idx_right].reshape((4, 2))

    left_corners = order_corners(left_corners_raw)
    right_corners = order_corners(right_corners_raw)
    
    # Calculate rotation angles for both markers
    left_rotation = calculate_marker_rotation(left_corners)
    right_rotation = calculate_marker_rotation(right_corners)
    
    # Detect tilt for both markers
    left_tilt_angle, left_tilt_correction, left_tilt_status = detect_marker_tilt(left_corners)
    right_tilt_angle, right_tilt_correction, right_tilt_status = detect_marker_tilt(right_corners)
    
    # Use average tilt correction for the pair
    avg_tilt_angle = (left_tilt_angle + right_tilt_angle) / 2.0
    avg_tilt_correction = (left_tilt_correction + right_tilt_correction) / 2.0
    tilt_status = "Error" if max(left_tilt_status, right_tilt_status) == "Error" else \
                  "Minor" if max(left_tilt_status, right_tilt_status) == "Minor" else "OK"
    
    tilt_info = {
        "angle": avg_tilt_angle,
        "correction_factor": avg_tilt_correction,
        "status": tilt_status,
        "left_angle": left_tilt_angle,
        "right_angle": right_tilt_angle,
        "left_rotation": left_rotation,
        "right_rotation": right_rotation,
        "avg_rotation": (left_rotation + right_rotation) / 2.0
    }

    # 3D POSE ESTIMATION
    pose_info = None
    if camera_matrix is not None and dist_coeffs is not None:
        l_rvec, l_tvec = get_marker_pose_3d(left_corners, MARKER_SIZE_MM, camera_matrix, dist_coeffs)
        r_rvec, r_tvec = get_marker_pose_3d(right_corners, MARKER_SIZE_MM, camera_matrix, dist_coeffs)
        
        if l_tvec is not None and r_tvec is not None:
            # Draw axis for 3D visualization
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, l_rvec, l_tvec, 30)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, r_rvec, r_tvec, 30)
            
            dist_3d, dx, dy, dz = calculate_3d_distance(l_tvec, r_tvec)
            pose_info = {
                "l_tvec": l_tvec,
                "r_tvec": r_tvec,
                "dist_3d": dist_3d,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "l_z": l_tvec[2],
                "r_z": r_tvec[2]
            }

    # centers for label positioning
    left_center = np.mean(left_corners, axis=0).astype(int)
    right_center = np.mean(right_corners, axis=0).astype(int)

    # label ABOVE each marker, using top-most y of that marker
    left_top_y = int(np.min(left_corners[:, 1]))
    right_top_y = int(np.min(right_corners[:, 1]))

    cv2.putText(
        frame,
        label_left,
        (left_center[0] - 50, left_top_y - 10),   # above left marker
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        label_right,
        (right_center[0] - 50, right_top_y - 10),  # above right marker
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    # compute side lengths & mm/px for both markers (only this pair)
    _, _, left_ratios = compute_side_lengths_and_ratios(left_corners)
    _, _, right_ratios = compute_side_lengths_and_ratios(right_corners)

    # use only these 8 ratios for this pair
    all_ratios = left_ratios + right_ratios   # 8 values for this pair
    avg_mm_per_px = float(np.mean(all_ratios))

    # unpack in TL, TR, BR, BL order
    l_tl, l_tr, l_br, l_bl = left_corners
    r_tl, r_tr, r_br, r_bl = right_corners

    # ========== DRAW CORNER POSITIONS ==========
    # Draw corners and labels for left marker
    left_corner_points = [l_tl, l_tr, l_br, l_bl]
    left_corner_labels = ["TL", "TR", "BR", "BL"]
    left_corner_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 128)]
    
    left_corner_info = {}
    for i, (corner, label, color) in enumerate(zip(left_corner_points, left_corner_labels, left_corner_colors)):
        corner_int = (int(corner[0]), int(corner[1]))
        left_corner_info[label] = corner_int
        # Draw circle at corner
        cv2.circle(frame, corner_int, 6, color, -1)
        cv2.circle(frame, corner_int, 8, (255, 255, 255), 2)
        # Draw label with coordinates
        label_text = f"{label}: ({corner_int[0]}, {corner_int[1]})"
        # Position label offset from corner
        offset_x = -60 if i in [0, 3] else 10
        offset_y = -15 if i in [0, 1] else 20
        text_pos = (corner_int[0] + offset_x, corner_int[1] + offset_y)
        cv2.putText(frame, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw corners and labels for right marker
    right_corner_points = [r_tl, r_tr, r_br, r_bl]
    right_corner_labels = ["TL", "TR", "BR", "BL"]
    right_corner_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 128)]
    
    right_corner_info = {}
    for i, (corner, label, color) in enumerate(zip(right_corner_points, right_corner_labels, right_corner_colors)):
        corner_int = (int(corner[0]), int(corner[1]))
        right_corner_info[label] = corner_int
        # Draw circle at corner
        cv2.circle(frame, corner_int, 6, color, -1)
        cv2.circle(frame, corner_int, 8, (255, 255, 255), 2)
        # Draw label with coordinates
        label_text = f"{label}: ({corner_int[0]}, {corner_int[1]})"
        # Position label offset from corner
        offset_x = -60 if i in [0, 3] else 10
        offset_y = -15 if i in [0, 1] else 20
        text_pos = (corner_int[0] + offset_x, corner_int[1] + offset_y)
        cv2.putText(frame, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ========== STRAIGHT-LINE GAP LOGIC (TOP, MID, BOTTOM) ==========

    # Inner right edge of LEFT marker: between l_tr (top) and l_br (bottom)
    left_top_point = l_tr
    left_bottom_point = l_br
    left_mid_point = (l_tr + l_br) / 2.0

    # Inner LEFT edge of RIGHT marker: between r_tl (top) and r_bl (bottom)
    right_left_top = r_tl
    right_left_bottom = r_bl

    # FIXED ray direction: straight to the right along image X axis
    n_unit = np.array([1.0, 0.0], dtype=float)

    # compute gaps for TOP, MID, BOTTOM from left edge
    gaps_info = {}
    named_points = {
        "Top": left_top_point,
        "Mid": left_mid_point,
        "Bottom": left_bottom_point,
    }

    valid_gaps_px = []
    valid_gaps_mm_calib = []
    hit_points = {}

    for name, pt in named_points.items():
        gap_px, gap_mm_raw, gap_mm_calib, hit_pt = compute_perp_gap_for_point(
            pt,
            n_unit,
            right_left_top,
            right_left_bottom,
            avg_mm_per_px,
            GAP_CALIB_FACTOR,
            avg_tilt_correction,  # Apply tilt correction
        )
        gaps_info[name] = (gap_px, gap_mm_raw, gap_mm_calib)
        hit_points[name] = hit_pt

        if gap_px is not None:
            valid_gaps_px.append(gap_px)
            valid_gaps_mm_calib.append(gap_mm_calib)

    if len(valid_gaps_px) == 0:
        return None, avg_mm_per_px, None, {"left": left_corner_info, "right": right_corner_info}, tilt_info, pose_info

    # Draw the three straight horizontal lines and circles where valid
    for name, pt in named_points.items():
        gap_px, _, _ = gaps_info[name]
        hit_pt = hit_points[name]
        if gap_px is not None and hit_pt is not None:
            cv2.line(
                frame,
                (int(pt[0]), int(pt[1])),
                (int(hit_pt[0]), int(hit_pt[1])),
                (0, 255, 0),
                2,
            )
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            cv2.circle(frame, (int(hit_pt[0]), int(hit_pt[1])), 4, (255, 0, 0), -1)

    # average over valid lines
    avg_gap_px = float(np.mean(valid_gaps_px))
    avg_gap_mm_calib = float(np.mean(valid_gaps_mm_calib))
    final_gap_mm = avg_gap_mm_calib

    corner_info = {
        "left": left_corner_info,
        "right": right_corner_info
    }

    return final_gap_mm, avg_mm_per_px, gaps_info, corner_info, tilt_info, pose_info


class GapMeasurementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ArUco Gap Measurement System")
        # Larger default window so bottom buttons and frames stay visible
        self.root.geometry("1800x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Create output folder
        self.output_folder = "gap_measurement_images"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.image_count = 0
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.info_queue = queue.Queue(maxsize=2)

        # Measurement mode state
        self.measurement_mode = False
        self.first_measurement = None
        self.first_image_path = None
        
        # Store last known values for continuous display
        self.last_top_info = {
            "gap_mm": None,
            "ratio": None,
            "gaps": {},
            "status": "Waiting..."
        }
        self.last_bottom_info = {
            "gap_mm": None,
            "ratio": None,
            "gaps": {},
            "status": "Waiting..."
        }
        self.last_top_corners = {}
        self.last_bottom_corners = {}
        # Placeholder for start/stop movement summary
        self.last_movement_results = {
            "top": {"distance": None, "points": {"Top": None, "Mid": None, "Bottom": None}, "method": "--"},
            "bottom": {"distance": None, "points": {"Top": None, "Mid": None, "Bottom": None}, "method": "--"}
        }
        
        # Setup camera and detector
        self.setup_camera()
        
        # Setup GUI
        self.setup_gui()
        
        # Load calibration
        load_calibration()
        
        # Start video thread
        self.start_video_thread()
        
    def setup_camera(self):
        """Initialize camera and ArUco detector"""
        # Use 4X4_250 to match high marker IDs (e.g., 100+)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.params = cv2.aruco.DetectorParameters()
        self.params.adaptiveThreshWinSizeMin = 3
        self.params.adaptiveThreshWinSizeMax = 23
        self.params.adaptiveThreshWinSizeStep = 10
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 5
        self.params.cornerRefinementMaxIterations = 30
        self.params.cornerRefinementMinAccuracy = 0.1
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        
        self.picam2 = Picamera2()
        # Preview configuration for live view
        # Use a 1080p stream so we cover a larger area with normal-quality detail
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (1920, 1080)}
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Still configuration for 4K captures (create but don't configure yet)
        self.still_config = self.picam2.create_still_configuration(
            main={"format": "RGB888", "size": (3840, 2160)}  # 4K resolution
        )
        
    def setup_gui(self):
        """Setup the GUI components with modern layout"""
        # Main container with side-by-side layout
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side: Small video window
        left_panel = tk.Frame(main_container, bg='#2b2b2b')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        video_label_frame = tk.LabelFrame(left_panel, text="Live Feed", bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
        video_label_frame.pack(padx=5, pady=5)

        self.video_label = tk.Label(video_label_frame, bg='black', width=960, height=540)
        self.video_label.pack(padx=2, pady=2)

        # Results display area below video feed
        results_frame = tk.LabelFrame(left_panel, text="Measurement Results", bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
        results_frame.pack(padx=5, pady=5, fill=tk.X)

        self.results_status_label = tk.Label(
            results_frame,
            text="Waiting for measurement (Start/Stop) to compute movement",
            font=('Arial', 9),
            bg='#2b2b2b',
            fg='white',
            anchor='w',
            justify=tk.LEFT
        )
        self.results_status_label.pack(fill=tk.X, padx=6, pady=(4, 6))

        self.movement_cards = {}
        self.movement_cards["top"] = self._create_movement_card(results_frame, "TOP MOVEMENT", '#1976d2')
        self.movement_cards["bottom"] = self._create_movement_card(results_frame, "BOTTOM MOVEMENT", '#d32f2f')
        self._reset_movement_cards()
        
        # Right side: Information display
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Dashboard (Main measurements)
        dashboard_tab = ttk.Frame(notebook)
        notebook.add(dashboard_tab, text="📊 Dashboard")
        self.setup_dashboard_tab(dashboard_tab)
        
        # Tab 2: Detailed Measurements
        measurements_tab = ttk.Frame(notebook)
        notebook.add(measurements_tab, text="📏 Measurements")
        
        self.measurements_text = scrolledtext.ScrolledText(
            measurements_tab, 
            wrap=tk.WORD, 
            font=('Courier', 10),
            bg='#f5f5f5'
        )
        self.measurements_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 3: Corner Positions
        corners_tab = ttk.Frame(notebook)
        notebook.add(corners_tab, text="📍 Corners")
        
        self.corners_text = scrolledtext.ScrolledText(
            corners_tab, 
            wrap=tk.WORD, 
            font=('Courier', 10),
            bg='#f5f5f5'
        )
        self.corners_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 4: System Info
        details_tab = ttk.Frame(notebook)
        notebook.add(details_tab, text="⚙️ System")
        
        self.details_text = scrolledtext.ScrolledText(
            details_tab, 
            wrap=tk.WORD, 
            font=('Courier', 9),
            bg='#f5f5f5'
        )
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons at bottom
        button_frame = tk.Frame(self.root, bg='#e0e0e0')
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = tk.Button(
            button_frame,
            text="Start Measurement",
            command=self.start_measurement,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(
            button_frame,
            text="Stop Measurement",
            command=self.stop_measurement,
            bg='#FF9800',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(
            button_frame,
            text="Save Image (S)",
            command=self.save_image,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.quit_button = tk.Button(
            button_frame,
            text="Quit (Q)",
            command=self.quit_app,
            bg='#f44336',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        self.quit_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            button_frame, 
            text="Status: Running", 
            fg='green',
            font=('Arial', 10)
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Bind keyboard shortcuts
        self.root.bind('<KeyPress-s>', lambda e: self.save_image())
        self.root.bind('<KeyPress-S>', lambda e: self.save_image())
        self.root.bind('<KeyPress-q>', lambda e: self.quit_app())
        self.root.bind('<KeyPress-Q>', lambda e: self.quit_app())
        self.root.focus_set()
        
    def setup_dashboard_tab(self, parent):
        """Setup modern dashboard with cards for measurements"""
        # Top pair card
        top_card = tk.LabelFrame(parent, text="TOP PAIR", font=('Arial', 12, 'bold'), 
                                 bg='#ffffff', fg='#1976d2', padx=10, pady=10)
        top_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Status and main value
        top_status_frame = tk.Frame(top_card, bg='#ffffff')
        top_status_frame.pack(fill=tk.X, pady=5)
        
        self.top_status_label = tk.Label(top_status_frame, text="Status: ○ Waiting...", 
                                         font=('Arial', 10), bg='#ffffff', fg='#666666')
        self.top_status_label.pack(side=tk.LEFT)
        
        self.top_gap_label = tk.Label(top_status_frame, text="Gap: -- mm", 
                                     font=('Arial', 16, 'bold'), bg='#ffffff', fg='#1976d2')
        self.top_gap_label.pack(side=tk.RIGHT)
        
        # Individual measurements frame
        top_measurements_frame = tk.Frame(top_card, bg='#ffffff')
        top_measurements_frame.pack(fill=tk.X, pady=5)
        
        self.top_measurements_labels = {}
        for i, name in enumerate(["Top", "Mid", "Bottom"]):
            frame = tk.Frame(top_measurements_frame, bg='#f5f5f5', relief=tk.RAISED, bd=1)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
            top_measurements_frame.columnconfigure(i, weight=1)
            
            label_name = tk.Label(frame, text=name, font=('Arial', 9, 'bold'), bg='#f5f5f5')
            label_name.pack(pady=2)
            
            label_value = tk.Label(frame, text="-- mm", font=('Arial', 11), bg='#f5f5f5', fg='#333333')
            label_value.pack()
            self.top_measurements_labels[name] = label_value
        
        # Ratio and tilt display
        top_info_frame = tk.Frame(top_card, bg='#ffffff')
        top_info_frame.pack(fill=tk.X, pady=2)
        
        self.top_ratio_label = tk.Label(top_info_frame, text="mm/px: --", 
                                        font=('Arial', 9), bg='#ffffff', fg='#666666')
        self.top_ratio_label.pack(side=tk.LEFT)
        
        self.top_tilt_label = tk.Label(top_info_frame, text="", 
                                      font=('Arial', 9, 'bold'), bg='#ffffff', fg='#4caf50')
        self.top_tilt_label.pack(side=tk.RIGHT, padx=5)
        
        # Rotation display frame
        top_rotation_frame = tk.Frame(top_card, bg='#ffffff')
        top_rotation_frame.pack(fill=tk.X, pady=2)
        
        rotation_title = tk.Label(top_rotation_frame, text="Rotation:", 
                                 font=('Arial', 9, 'bold'), bg='#ffffff', fg='#666666')
        rotation_title.pack(side=tk.LEFT, padx=(0, 5))
        
        self.top_left_rotation_label = tk.Label(top_rotation_frame, text="Left: --°", 
                                                font=('Arial', 9), bg='#ffffff', fg='#1976d2')
        self.top_left_rotation_label.pack(side=tk.LEFT, padx=5)
        
        self.top_right_rotation_label = tk.Label(top_rotation_frame, text="Right: --°", 
                                                 font=('Arial', 9), bg='#ffffff', fg='#1976d2')
        self.top_right_rotation_label.pack(side=tk.LEFT, padx=5)
        
        self.top_avg_rotation_label = tk.Label(top_rotation_frame, text="Avg: --°", 
                                               font=('Arial', 9), bg='#ffffff', fg='#666666')
        self.top_avg_rotation_label.pack(side=tk.LEFT, padx=5)
        
        # Alignment guidance
        self.top_alignment_label = tk.Label(top_card, text="", 
                                           font=('Arial', 8, 'italic'), bg='#ffffff', fg='#4caf50')
        self.top_alignment_label.pack(pady=2)

        # 3D Data Frame
        top_3d_frame = tk.Frame(top_card, bg='#ffffff')
        top_3d_frame.pack(fill=tk.X, pady=5)
        
        self.top_3d_dist_label = tk.Label(top_3d_frame, text="3D Dist: --", font=('Arial', 9), bg='#ffffff', fg='#9c27b0')
        self.top_3d_dist_label.pack(side=tk.LEFT, padx=5)
        
        self.top_lz_label = tk.Label(top_3d_frame, text="L.Z: --", font=('Arial', 9), bg='#ffffff', fg='#9c27b0')
        self.top_lz_label.pack(side=tk.LEFT, padx=5)
        
        self.top_rz_label = tk.Label(top_3d_frame, text="R.Z: --", font=('Arial', 9), bg='#ffffff', fg='#9c27b0')
        self.top_rz_label.pack(side=tk.LEFT, padx=5)
        
        # Bottom pair card
        bottom_card = tk.LabelFrame(parent, text="BOTTOM PAIR", font=('Arial', 12, 'bold'),
                                    bg='#ffffff', fg='#d32f2f', padx=10, pady=10)
        bottom_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Status and main value
        bottom_status_frame = tk.Frame(bottom_card, bg='#ffffff')
        bottom_status_frame.pack(fill=tk.X, pady=5)
        
        self.bottom_status_label = tk.Label(bottom_status_frame, text="Status: ○ Waiting...",
                                           font=('Arial', 10), bg='#ffffff', fg='#666666')
        self.bottom_status_label.pack(side=tk.LEFT)
        
        self.bottom_gap_label = tk.Label(bottom_status_frame, text="Gap: -- mm",
                                         font=('Arial', 16, 'bold'), bg='#ffffff', fg='#d32f2f')
        self.bottom_gap_label.pack(side=tk.RIGHT)
        
        # Individual measurements frame
        bottom_measurements_frame = tk.Frame(bottom_card, bg='#ffffff')
        bottom_measurements_frame.pack(fill=tk.X, pady=5)
        
        self.bottom_measurements_labels = {}
        for i, name in enumerate(["Top", "Mid", "Bottom"]):
            frame = tk.Frame(bottom_measurements_frame, bg='#f5f5f5', relief=tk.RAISED, bd=1)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
            bottom_measurements_frame.columnconfigure(i, weight=1)
            
            label_name = tk.Label(frame, text=name, font=('Arial', 9, 'bold'), bg='#f5f5f5')
            label_name.pack(pady=2)
            
            label_value = tk.Label(frame, text="-- mm", font=('Arial', 11), bg='#f5f5f5', fg='#333333')
            label_value.pack()
            self.bottom_measurements_labels[name] = label_value
        
        # Ratio and tilt display
        bottom_info_frame = tk.Frame(bottom_card, bg='#ffffff')
        bottom_info_frame.pack(fill=tk.X, pady=2)
        
        self.bottom_ratio_label = tk.Label(bottom_info_frame, text="mm/px: --",
                                          font=('Arial', 9), bg='#ffffff', fg='#666666')
        self.bottom_ratio_label.pack(side=tk.LEFT)
        
        self.bottom_tilt_label = tk.Label(bottom_info_frame, text="",
                                         font=('Arial', 9, 'bold'), bg='#ffffff', fg='#4caf50')
        self.bottom_tilt_label.pack(side=tk.RIGHT, padx=5)
        
        # Rotation display frame
        bottom_rotation_frame = tk.Frame(bottom_card, bg='#ffffff')
        bottom_rotation_frame.pack(fill=tk.X, pady=2)
        
        rotation_title = tk.Label(bottom_rotation_frame, text="Rotation:", 
                                 font=('Arial', 9, 'bold'), bg='#ffffff', fg='#666666')
        rotation_title.pack(side=tk.LEFT, padx=(0, 5))
        
        self.bottom_left_rotation_label = tk.Label(bottom_rotation_frame, text="Left: --°",
                                                   font=('Arial', 9), bg='#ffffff', fg='#d32f2f')
        self.bottom_left_rotation_label.pack(side=tk.LEFT, padx=5)
        
        self.bottom_right_rotation_label = tk.Label(bottom_rotation_frame, text="Right: --°",
                                                    font=('Arial', 9), bg='#ffffff', fg='#d32f2f')
        self.bottom_right_rotation_label.pack(side=tk.LEFT, padx=5)
        
        self.bottom_avg_rotation_label = tk.Label(bottom_rotation_frame, text="Avg: --°",
                                                  font=('Arial', 9), bg='#ffffff', fg='#666666')
        self.bottom_avg_rotation_label.pack(side=tk.LEFT, padx=5)
        
        # Alignment guidance
        self.bottom_alignment_label = tk.Label(bottom_card, text="",
                                               font=('Arial', 8, 'italic'), bg='#ffffff', fg='#4caf50')
        self.bottom_alignment_label.pack(pady=2)

        # 3D Data Frame
        bottom_3d_frame = tk.Frame(bottom_card, bg='#ffffff')
        bottom_3d_frame.pack(fill=tk.X, pady=5)
        
        self.bottom_3d_dist_label = tk.Label(bottom_3d_frame, text="3D Dist: --", font=('Arial', 9), bg='#ffffff', fg='#9c27b0')
        self.bottom_3d_dist_label.pack(side=tk.LEFT, padx=5)
        
        self.bottom_lz_label = tk.Label(bottom_3d_frame, text="L.Z: --", font=('Arial', 9), bg='#ffffff', fg='#9c27b0')
        self.bottom_lz_label.pack(side=tk.LEFT, padx=5)
        
        self.bottom_rz_label = tk.Label(bottom_3d_frame, text="R.Z: --", font=('Arial', 9), bg='#ffffff', fg='#9c27b0')
        self.bottom_rz_label.pack(side=tk.LEFT, padx=5)
        
    def _create_movement_card(self, parent, title, accent_color):
        """Create a compact card (mirrors right-side UI) for movement results"""
        card = tk.Frame(parent, bg='#ffffff', relief=tk.RIDGE, bd=1)
        card.pack(fill=tk.X, expand=True, padx=5, pady=4)

        header = tk.Frame(card, bg='#ffffff')
        header.pack(fill=tk.X, pady=(2, 0))

        title_label = tk.Label(header, text=title, font=('Arial', 11, 'bold'), bg='#ffffff', fg=accent_color)
        title_label.pack(side=tk.LEFT)

        status_label = tk.Label(header, text="Status: Waiting", font=('Arial', 9), bg='#ffffff', fg='#666666')
        status_label.pack(side=tk.RIGHT)

        distance_label = tk.Label(card, text="Distance: -- mm", font=('Arial', 13, 'bold'), bg='#ffffff', fg=accent_color)
        distance_label.pack(fill=tk.X, padx=4, pady=(2, 6))

        points_frame = tk.Frame(card, bg='#ffffff')
        points_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        points_frame.columnconfigure(0, weight=1)
        points_frame.columnconfigure(1, weight=1)
        points_frame.columnconfigure(2, weight=1)

        point_labels = {}
        for idx, name in enumerate(["Top", "Mid", "Bottom"]):
            pf = tk.Frame(points_frame, bg='#f5f5f5', relief=tk.RAISED, bd=1)
            pf.grid(row=0, column=idx, padx=3, pady=2, sticky='ew')

            tk.Label(pf, text=name, font=('Arial', 9, 'bold'), bg='#f5f5f5').pack(pady=(3, 1))
            val_label = tk.Label(pf, text="-- mm", font=('Arial', 11), bg='#f5f5f5', fg='#333333')
            val_label.pack(pady=(0, 3))
            point_labels[name] = val_label

        method_label = tk.Label(card, text="Method: --", font=('Arial', 8), bg='#ffffff', fg='#888888', anchor='w', justify=tk.LEFT)
        method_label.pack(fill=tk.X, padx=4, pady=(0, 4))

        return {
            "card": card,
            "status": status_label,
            "distance": distance_label,
            "points": point_labels,
            "method": method_label,
            "accent": accent_color
        }

    def _reset_movement_cards(self, status_text=None):
        """Reset movement cards to a waiting state"""
        if status_text:
            self.results_status_label.config(text=status_text, fg='white')
        for card_key, card in self.movement_cards.items():
            card["status"].config(text="Status: Waiting", fg='#666666')
            card["distance"].config(text="Distance: -- mm", fg=card["accent"])
            for lbl in card["points"].values():
                lbl.config(text="-- mm", fg='#999999')
            card["method"].config(text="Method: --", fg='#888888')
        self.last_movement_results = {
            "top": {"distance": None, "points": {"Top": None, "Mid": None, "Bottom": None}, "method": "--"},
            "bottom": {"distance": None, "points": {"Top": None, "Mid": None, "Bottom": None}, "method": "--"}
        }

    def _update_movement_cards(self, movement_data, method_text="gap_difference"):
        """Populate movement cards with computed distances"""
        for key in ["top", "bottom"]:
            card = self.movement_cards.get(key)
            pair_data = movement_data.get(key, {}) if movement_data else {}
            distance = pair_data.get("distance")
            points = pair_data.get("points", {})
            method = pair_data.get("method", method_text)
            method_display = "3D pose" if method == "3D_pose" else "Gap difference" if method == "gap_difference" else method

            if card:
                if distance is not None:
                    card["distance"].config(text=f"Distance: {distance:+.2f} mm", fg=card["accent"])
                    card["status"].config(text="Status: Completed", fg='#4caf50')
                else:
                    card["distance"].config(text="Distance: -- mm", fg=card["accent"])
                    card["status"].config(text="Status: Waiting", fg='#666666')

                for name, lbl in card["points"].items():
                    delta = points.get(name)
                    if delta is not None:
                        lbl.config(text=f"{delta:+.2f} mm", fg='#333333')
                    else:
                        lbl.config(text="-- mm", fg='#999999')

                card["method"].config(text=f"Method: {method_display}", fg='#888888')

    def start_video_thread(self):
        """Start the video capture thread"""
        self.running = True
        self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
        self.video_thread.start()
        self.update_video()
        self.update_info()
        
    def video_capture_loop(self):
        """Video capture loop running in separate thread"""
        while self.running:
            try:
                frame_rgb = self.picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                corners_list, ids, _ = self.detector.detectMarkers(frame)
                
                # Initialize gap variables
                gap_top_mm = None
                gap_bottom_mm = None
                top_info = {}
                bottom_info = {}
                top_corners = {}
                bottom_corners = {}
                top_tilt_info = None
                bottom_tilt_info = None
                
                if ids is not None and len(corners_list) >= 2:
                    cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)
                    
                    # Compute marker centers
                    centers = []
                    for i, c in enumerate(corners_list):
                        pts = c.reshape((4, 2))
                        center = np.mean(pts, axis=0)
                        centers.append((i, center[0], center[1]))
                    
                    centers_sorted_by_y = sorted(centers, key=lambda t: t[2])
                    
                    # Process TOP pair
                    if len(centers_sorted_by_y) >= 2:
                        top_two = centers_sorted_by_y[:2]
                        top_two_sorted_by_x = sorted(top_two, key=lambda t: t[1])
                        top_left_idx = top_two_sorted_by_x[0][0]
                        top_right_idx = top_two_sorted_by_x[1][0]
                        
                        gap_top_mm, ratio_top, gaps_info_top, corners_info_top, tilt_info_top, pose_info_top = process_pair(
                            frame,
                            corners_list,
                            top_left_idx,
                            top_right_idx,
                            label_left="LEFT TOP1",
                            label_right="LEFT TOP2",
                            text_y_offset=30,
                            pair_name="TOP PAIR",
                            camera_matrix=CAMERA_MATRIX,
                            dist_coeffs=DIST_COEFFS
                        )
                        top_info = {
                            "gap_mm": gap_top_mm,
                            "ratio": ratio_top,
                            "gaps": gaps_info_top,
                            "tilt": tilt_info_top,
                            "pose": pose_info_top
                        }
                        top_corners = corners_info_top if corners_info_top else {}
                    
                    # Process BOTTOM pair
                    if len(centers_sorted_by_y) >= 4:
                        bottom_two = centers_sorted_by_y[-2:]
                        bottom_two_sorted_by_x = sorted(bottom_two, key=lambda t: t[1])
                        bottom_left_idx = bottom_two_sorted_by_x[0][0]
                        bottom_right_idx = bottom_two_sorted_by_x[1][0]
                        
                        gap_bottom_mm, ratio_bottom, gaps_info_bottom, corners_info_bottom, tilt_info_bottom, pose_info_bottom = process_pair(
                            frame,
                            corners_list,
                            bottom_left_idx,
                            bottom_right_idx,
                            label_left="LEFT BOTTOM1",
                            label_right="LEFT BOTTOM2",
                            text_y_offset=230,
                            pair_name="BOTTOM PAIR",
                            camera_matrix=CAMERA_MATRIX,
                            dist_coeffs=DIST_COEFFS
                        )
                        bottom_info = {
                            "gap_mm": gap_bottom_mm,
                            "ratio": ratio_bottom,
                            "gaps": gaps_info_bottom,
                            "tilt": tilt_info_bottom,
                            "pose": pose_info_bottom
                        }
                        bottom_corners = corners_info_bottom if corners_info_bottom else {}
                        bottom_tilt_info = tilt_info_bottom
                
                # Resize frame for on-screen display (slightly downscaled 1080p)
                display_frame = cv2.resize(frame, (960, 540))
                
                # Convert to RGB for tkinter
                frame_rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Put frame and info in queues
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_rgb_display)
                
                info_data = {
                    "top": top_info,
                    "bottom": bottom_info,
                    "top_corners": top_corners,
                    "bottom_corners": bottom_corners,
                    "frame": frame  # Keep original for saving
                }
                if not self.info_queue.full():
                    self.info_queue.put(info_data)
                    
            except Exception as e:
                print(f"Error in video loop: {e}")
                
    def update_video(self):
        """Update video display"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=img_tk)
                self.video_label.image = img_tk
        except Exception as e:
            print(f"Error updating video: {e}")
        
        if self.running:
            self.root.after(33, self.update_video)  # ~30 FPS
            
    def update_info(self):
        """Update information display"""
        try:
            if not self.info_queue.empty():
                info = self.info_queue.get()
                self.current_info = info
                self.update_dashboard(info)
                self.update_measurements_tab(info)
                self.update_corners_tab(info)
                self.update_details_tab(info)
        except Exception as e:
            print(f"Error updating info: {e}")
        
        if self.running:
            self.root.after(100, self.update_info)  # Update info 10 times per second
    
    def _generate_alignment_guidance(self, tilt_angle, tilt_status, rotation_diff, left_rotation, right_rotation):
        """Generate alignment guidance text based on tilt and rotation"""
        if tilt_status == "Error":
            return {"text": "⚠️ Severe tilt detected - adjust camera angle", "color": "#f44336"}
        elif tilt_status == "Warning":
            return {"text": "⚠️ Moderate tilt - measurements may be affected", "color": "#ff9800"}
        elif rotation_diff > 5:
            return {"text": f"Markers misaligned ({rotation_diff:.1f}° difference)", "color": "#ffc107"}
        elif tilt_status == "Minor":
            return {"text": "Minor tilt detected - acceptable", "color": "#ffc107"}
        else:
            return {"text": "✓ Alignment good", "color": "#4caf50"}
            
    def update_dashboard(self, info):
        """Update dashboard with live values"""
        # Update top pair
        if info["top"].get("gap_mm") is not None:
            self.last_top_info = {
                "gap_mm": info["top"]["gap_mm"],
                "ratio": info["top"]["ratio"],
                "gaps": info["top"].get("gaps", {}),
                "status": "Active",
                "tilt": info["top"].get("tilt")
            }
            self.top_status_label.config(text="Status: ● Active", fg='#4caf50')
            self.top_gap_label.config(text=f"Gap: {info['top']['gap_mm']:.2f} mm", fg='#1976d2')
            self.top_ratio_label.config(text=f"mm/px: {info['top']['ratio']:.5f}")
            
            # Update tilt warning and rotation
            if info["top"].get("tilt"):
                tilt = info["top"]["tilt"]
                tilt_angle = tilt.get("angle", 0)
                tilt_status = tilt.get("status", "OK")
                left_rotation = tilt.get("left_rotation", 0)
                right_rotation = tilt.get("right_rotation", 0)
                avg_rotation = tilt.get("avg_rotation", 0)
                
                # Update rotation displays
                self.top_left_rotation_label.config(text=f"Left: {left_rotation:.1f}°")
                self.top_right_rotation_label.config(text=f"Right: {right_rotation:.1f}°")
                self.top_avg_rotation_label.config(text=f"Avg: {avg_rotation:.1f}°")
                
                # Calculate rotation difference
                rotation_diff = abs(left_rotation - right_rotation)
                if rotation_diff > 180:
                    rotation_diff = 360 - rotation_diff
                
                # Generate alignment guidance
                guidance = self._generate_alignment_guidance(tilt_angle, tilt_status, rotation_diff, left_rotation, right_rotation)
                self.top_alignment_label.config(text=guidance["text"], fg=guidance["color"])
                
                if tilt_status == "Error":
                    self.top_tilt_label.config(text=f"⚠️ Tilt: {tilt_angle:.1f}°", fg='#f44336')
                elif tilt_status == "Warning":
                    self.top_tilt_label.config(text=f"⚠️ Tilt: {tilt_angle:.1f}°", fg='#ff9800')
                elif tilt_status == "Minor":
                    self.top_tilt_label.config(text=f"Tilt: {tilt_angle:.1f}°", fg='#ffc107')
                else:
                    self.top_tilt_label.config(text="", fg='#4caf50')
            else:
                self.top_tilt_label.config(text="")
                self.top_left_rotation_label.config(text="Left: --°")
                self.top_right_rotation_label.config(text="Right: --°")
                self.top_avg_rotation_label.config(text="Avg: --°")
                self.top_alignment_label.config(text="")
            
            # Update individual measurements
            if info["top"].get("gaps"):
                for name in ["Top", "Mid", "Bottom"]:
                    gap_data = info["top"]["gaps"].get(name)
                    if gap_data and gap_data[0] is not None:
                        _, _, gap_mm_calib = gap_data
                        self.top_measurements_labels[name].config(text=f"{gap_mm_calib:.2f} mm", fg='#333333')
                    else:
                        self.top_measurements_labels[name].config(text="-- mm", fg='#999999')

            # Update 3D info
            if info["top"].get("pose"):
                pose = info["top"]["pose"]
                dist_3d = pose.get("dist_3d")
                l_z = pose.get("l_z")
                r_z = pose.get("r_z")
                
                if dist_3d is not None:
                    self.top_3d_dist_label.config(text=f"3D Dist: {dist_3d:.2f} mm")
                else:
                    self.top_3d_dist_label.config(text="3D Dist: --")
                    
                if l_z is not None:
                    self.top_lz_label.config(text=f"L.Z: {l_z:.1f}mm")
                else:
                    self.top_lz_label.config(text="L.Z: --")
                    
                if r_z is not None:
                    self.top_rz_label.config(text=f"R.Z: {r_z:.1f}mm")
                else:
                    self.top_rz_label.config(text="R.Z: --")
            else:
                self.top_3d_dist_label.config(text="3D Dist: --")
                self.top_lz_label.config(text="L.Z: --")
                self.top_rz_label.config(text="R.Z: --")
        else:
            if self.last_top_info["gap_mm"] is not None:
                self.top_status_label.config(text="Status: ○ Last known", fg='#ff9800')
                self.top_gap_label.config(text=f"Gap: {self.last_top_info['gap_mm']:.2f} mm", fg='#666666')
                self.top_ratio_label.config(text=f"mm/px: {self.last_top_info['ratio']:.5f}")
            else:
                self.top_status_label.config(text="Status: ○ Waiting...", fg='#999999')
                self.top_gap_label.config(text="Gap: -- mm", fg='#999999')
                self.top_ratio_label.config(text="mm/px: --")
                self.top_rotation_label.config(text="Rotation: --°")
                self.top_3d_dist_label.config(text="3D Dist: --")
                self.top_lz_label.config(text="L.Z: --")
                self.top_rz_label.config(text="R.Z: --")
        
        # Update bottom pair
        if info["bottom"].get("gap_mm") is not None:
            self.last_bottom_info = {
                "gap_mm": info["bottom"]["gap_mm"],
                "ratio": info["bottom"]["ratio"],
                "gaps": info["bottom"].get("gaps", {}),
                "status": "Active",
                "tilt": info["bottom"].get("tilt")
            }
            self.bottom_status_label.config(text="Status: ● Active", fg='#4caf50')
            self.bottom_gap_label.config(text=f"Gap: {info['bottom']['gap_mm']:.2f} mm", fg='#d32f2f')
            self.bottom_ratio_label.config(text=f"mm/px: {info['bottom']['ratio']:.5f}")
            
            # Update tilt warning and rotation
            if info["bottom"].get("tilt"):
                tilt = info["bottom"]["tilt"]
                tilt_angle = tilt.get("angle", 0)
                tilt_status = tilt.get("status", "OK")
                left_rotation = tilt.get("left_rotation", 0)
                right_rotation = tilt.get("right_rotation", 0)
                avg_rotation = tilt.get("avg_rotation", 0)
                
                # Update rotation displays
                self.bottom_left_rotation_label.config(text=f"Left: {left_rotation:.1f}°")
                self.bottom_right_rotation_label.config(text=f"Right: {right_rotation:.1f}°")
                self.bottom_avg_rotation_label.config(text=f"Avg: {avg_rotation:.1f}°")
                
                # Calculate rotation difference
                rotation_diff = abs(left_rotation - right_rotation)
                if rotation_diff > 180:
                    rotation_diff = 360 - rotation_diff
                
                # Generate alignment guidance
                guidance = self._generate_alignment_guidance(tilt_angle, tilt_status, rotation_diff, left_rotation, right_rotation)
                self.bottom_alignment_label.config(text=guidance["text"], fg=guidance["color"])
                
                if tilt_status == "Error":
                    self.bottom_tilt_label.config(text=f"⚠️ Tilt: {tilt_angle:.1f}°", fg='#f44336')
                elif tilt_status == "Warning":
                    self.bottom_tilt_label.config(text=f"⚠️ Tilt: {tilt_angle:.1f}°", fg='#ff9800')
                elif tilt_status == "Minor":
                    self.bottom_tilt_label.config(text=f"Tilt: {tilt_angle:.1f}°", fg='#ffc107')
                else:
                    self.bottom_tilt_label.config(text="", fg='#4caf50')
            else:
                self.bottom_tilt_label.config(text="")
                self.bottom_left_rotation_label.config(text="Left: --°")
                self.bottom_right_rotation_label.config(text="Right: --°")
                self.bottom_avg_rotation_label.config(text="Avg: --°")
                self.bottom_alignment_label.config(text="")
            
            # Update individual measurements
            if info["bottom"].get("gaps"):
                for name in ["Top", "Mid", "Bottom"]:
                    gap_data = info["bottom"]["gaps"].get(name)
                    if gap_data and gap_data[0] is not None:
                        _, _, gap_mm_calib = gap_data
                        self.bottom_measurements_labels[name].config(text=f"{gap_mm_calib:.2f} mm", fg='#333333')
                    else:
                        self.bottom_measurements_labels[name].config(text="-- mm", fg='#999999')

            # Update 3D info
            if info["bottom"].get("pose"):
                pose = info["bottom"]["pose"]
                dist_3d = pose.get("dist_3d")
                l_z = pose.get("l_z")
                r_z = pose.get("r_z")
                
                if dist_3d is not None:
                    self.bottom_3d_dist_label.config(text=f"3D Dist: {dist_3d:.2f} mm")
                else:
                    self.bottom_3d_dist_label.config(text="3D Dist: --")
                    
                if l_z is not None:
                    self.bottom_lz_label.config(text=f"L.Z: {l_z:.1f}mm")
                else:
                    self.bottom_lz_label.config(text="L.Z: --")
                    
                if r_z is not None:
                    self.bottom_rz_label.config(text=f"R.Z: {r_z:.1f}mm")
                else:
                    self.bottom_rz_label.config(text="R.Z: --")
            else:
                self.bottom_3d_dist_label.config(text="3D Dist: --")
                self.bottom_lz_label.config(text="L.Z: --")
                self.bottom_rz_label.config(text="R.Z: --")
        else:
            if self.last_bottom_info["gap_mm"] is not None:
                self.bottom_status_label.config(text="Status: ○ Last known", fg='#ff9800')
                self.bottom_gap_label.config(text=f"Gap: {self.last_bottom_info['gap_mm']:.2f} mm", fg='#666666')
                self.bottom_ratio_label.config(text=f"mm/px: {self.last_bottom_info['ratio']:.5f}")
            else:
                self.bottom_status_label.config(text="Status: ○ Waiting...", fg='#999999')
                self.bottom_gap_label.config(text="Gap: -- mm", fg='#999999')
                self.bottom_ratio_label.config(text="mm/px: --")
                self.bottom_3d_dist_label.config(text="3D Dist: --")
                self.bottom_lz_label.config(text="L.Z: --")
                self.bottom_rz_label.config(text="R.Z: --")
                self.bottom_left_rotation_label.config(text="Left: --°")
                self.bottom_right_rotation_label.config(text="Right: --°")
                self.bottom_avg_rotation_label.config(text="Avg: --°")
            
    def update_measurements_tab(self, info):
        """Update measurements tab - only update values, keep structure"""
        # Update stored values if new data is available
        if info["top"].get("gap_mm") is not None:
            self.last_top_info = {
                "gap_mm": info["top"]["gap_mm"],
                "ratio": info["top"]["ratio"],
                "gaps": info["top"].get("gaps", {}),
                "status": "Active"
            }
        else:
            self.last_top_info["status"] = "No markers"
            
        if info["bottom"].get("gap_mm") is not None:
            self.last_bottom_info = {
                "gap_mm": info["bottom"]["gap_mm"],
                "ratio": info["bottom"]["ratio"],
                "gaps": info["bottom"].get("gaps", {}),
                "status": "Active"
            }
        else:
            self.last_bottom_info["status"] = "No markers"
        
        # Build text with current/last known values
        text = "=" * 60 + "\n"
        text += "GAP MEASUREMENTS\n"
        text += "=" * 60 + "\n\n"
        
        # Top Pair
        text += "TOP PAIR:\n"
        text += "-" * 60 + "\n"
        if self.last_top_info["gap_mm"] is not None:
            status_indicator = "●" if self.last_top_info["status"] == "Active" else "○"
            text += f"Status: {status_indicator} {self.last_top_info['status']}\n"
            text += f"Average Gap: {self.last_top_info['gap_mm']:.2f} mm\n"
            text += f"mm/px Ratio: {self.last_top_info['ratio']:.5f}\n"
            if self.last_top_info.get("gaps"):
                text += "\nIndividual Measurements:\n"
                for name in ["Top", "Mid", "Bottom"]:
                    gap_data = self.last_top_info["gaps"].get(name)
                    if gap_data and gap_data[0] is not None:
                        gap_px, gap_mm_raw, gap_mm_calib = gap_data
                        text += f"  {name:6s}: {gap_px:6.1f} px | "
                        text += f"Raw: {gap_mm_raw:6.2f} mm | "
                        text += f"Calib: {gap_mm_calib:6.2f} mm\n"
                    else:
                        text += f"  {name:6s}: --\n"
        else:
            text += "Status: ○ Waiting for markers...\n"
            text += "Average Gap: --\n"
            text += "mm/px Ratio: --\n"
        
        text += "\n" + "=" * 60 + "\n\n"
        
        # Bottom Pair
        text += "BOTTOM PAIR:\n"
        text += "-" * 60 + "\n"
        if self.last_bottom_info["gap_mm"] is not None:
            status_indicator = "●" if self.last_bottom_info["status"] == "Active" else "○"
            text += f"Status: {status_indicator} {self.last_bottom_info['status']}\n"
            text += f"Average Gap: {self.last_bottom_info['gap_mm']:.2f} mm\n"
            text += f"mm/px Ratio: {self.last_bottom_info['ratio']:.5f}\n"
            if self.last_bottom_info.get("gaps"):
                text += "\nIndividual Measurements:\n"
                for name in ["Top", "Mid", "Bottom"]:
                    gap_data = self.last_bottom_info["gaps"].get(name)
                    if gap_data and gap_data[0] is not None:
                        gap_px, gap_mm_raw, gap_mm_calib = gap_data
                        text += f"  {name:6s}: {gap_px:6.1f} px | "
                        text += f"Raw: {gap_mm_raw:6.2f} mm | "
                        text += f"Calib: {gap_mm_calib:6.2f} mm\n"
                    else:
                        text += f"  {name:6s}: --\n"
        else:
            text += "Status: ○ Waiting for markers...\n"
            text += "Average Gap: --\n"
            text += "mm/px Ratio: --\n"
        
        # Only update the text, don't clear everything
        self.measurements_text.delete(1.0, tk.END)
        self.measurements_text.insert(1.0, text)
        
    def update_corners_tab(self, info):
        """Update corner positions tab - keep last known values"""
        # Update stored corners if new data is available
        if info.get("top_corners"):
            self.last_top_corners = info["top_corners"]
        if info.get("bottom_corners"):
            self.last_bottom_corners = info["bottom_corners"]
        
        text = "=" * 60 + "\n"
        text += "CORNER POSITIONS (pixels)\n"
        text += "=" * 60 + "\n\n"
        
        # Top Pair Corners
        text += "TOP PAIR:\n"
        text += "-" * 60 + "\n"
        if self.last_top_corners:
            if "left" in self.last_top_corners:
                status = "● Active" if info.get("top_corners") else "○ Last known"
                text += f"LEFT TOP1 Marker ({status}):\n"
                for label in ["TL", "TR", "BR", "BL"]:
                    corner = self.last_top_corners["left"].get(label)
                    if corner:
                        text += f"  {label}: ({corner[0]:4d}, {corner[1]:4d})\n"
                    else:
                        text += f"  {label}: (----, ----)\n"
            
            if "right" in self.last_top_corners:
                status = "● Active" if info.get("top_corners") else "○ Last known"
                text += f"\nLEFT TOP2 Marker ({status}):\n"
                for label in ["TL", "TR", "BR", "BL"]:
                    corner = self.last_top_corners["right"].get(label)
                    if corner:
                        text += f"  {label}: ({corner[0]:4d}, {corner[1]:4d})\n"
                    else:
                        text += f"  {label}: (----, ----)\n"
        else:
            text += "LEFT TOP1 Marker: Waiting...\n"
            text += "LEFT TOP2 Marker: Waiting...\n"
        
        text += "\n" + "=" * 60 + "\n\n"
        
        # Bottom Pair Corners
        text += "BOTTOM PAIR:\n"
        text += "-" * 60 + "\n"
        if self.last_bottom_corners:
            if "left" in self.last_bottom_corners:
                status = "● Active" if info.get("bottom_corners") else "○ Last known"
                text += f"LEFT BOTTOM1 Marker ({status}):\n"
                for label in ["TL", "TR", "BR", "BL"]:
                    corner = self.last_bottom_corners["left"].get(label)
                    if corner:
                        text += f"  {label}: ({corner[0]:4d}, {corner[1]:4d})\n"
                    else:
                        text += f"  {label}: (----, ----)\n"
            
            if "right" in self.last_bottom_corners:
                status = "● Active" if info.get("bottom_corners") else "○ Last known"
                text += f"\nLEFT BOTTOM2 Marker ({status}):\n"
                for label in ["TL", "TR", "BR", "BL"]:
                    corner = self.last_bottom_corners["right"].get(label)
                    if corner:
                        text += f"  {label}: ({corner[0]:4d}, {corner[1]:4d})\n"
                    else:
                        text += f"  {label}: (----, ----)\n"
        else:
            text += "LEFT BOTTOM1 Marker: Waiting...\n"
            text += "LEFT BOTTOM2 Marker: Waiting...\n"
        
        # Only update the text
        self.corners_text.delete(1.0, tk.END)
        self.corners_text.insert(1.0, text)
        
    def update_details_tab(self, info):
        """Update detailed information tab - keep structure, update values"""
        text = "=" * 60 + "\n"
        text += "DETAILED INFORMATION\n"
        text += "=" * 60 + "\n\n"
        
        text += f"Calibration Factor: {GAP_CALIB_FACTOR:.4f}\n"
        text += f"Marker Size: {MARKER_SIZE_MM} mm\n"
        text += f"Output Folder: {self.output_folder}\n"
        text += f"Images Saved: {self.image_count}\n\n"
        
        text += "-" * 60 + "\n"
        text += "TOP PAIR DETAILS:\n"
        text += "-" * 60 + "\n"
        if self.last_top_info["gap_mm"] is not None:
            status_indicator = "●" if self.last_top_info["status"] == "Active" else "○"
            text += f"Status: {status_indicator} {self.last_top_info['status']}\n"
            text += f"Average Gap: {self.last_top_info['gap_mm']:.4f} mm\n"
            text += f"mm/px Ratio: {self.last_top_info['ratio']:.6f}\n"
            text += f"Calibration Applied: Yes\n"
        else:
            text += "Status: ○ Waiting for markers...\n"
            text += "Average Gap: --\n"
            text += "mm/px Ratio: --\n"
            text += "Calibration Applied: --\n"
        
        text += "\n" + "-" * 60 + "\n"
        text += "BOTTOM PAIR DETAILS:\n"
        text += "-" * 60 + "\n"
        if self.last_bottom_info["gap_mm"] is not None:
            status_indicator = "●" if self.last_bottom_info["status"] == "Active" else "○"
            text += f"Status: {status_indicator} {self.last_bottom_info['status']}\n"
            text += f"Average Gap: {self.last_bottom_info['gap_mm']:.4f} mm\n"
            text += f"mm/px Ratio: {self.last_bottom_info['ratio']:.6f}\n"
            text += f"Calibration Applied: Yes\n"
        else:
            text += "Status: ○ Waiting for markers...\n"
            text += "Average Gap: --\n"
            text += "mm/px Ratio: --\n"
            text += "Calibration Applied: --\n"
        
        # Only update the text
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, text)
        
    def save_image(self):
        """Save current frame with measurements and all values"""
        if hasattr(self, 'current_info') and self.current_info:
            try:
                self.image_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                frame = self.current_info.get("frame")
                if frame is not None:
                    filename_parts = [f"img_{self.image_count:04d}", timestamp]
                    if self.current_info["top"].get("gap_mm") is not None:
                        filename_parts.append(f"top_{self.current_info['top']['gap_mm']:.2f}mm")
                    if self.current_info["bottom"].get("gap_mm") is not None:
                        filename_parts.append(f"bot_{self.current_info['bottom']['gap_mm']:.2f}mm")
                    
                    base_filename = "_".join(filename_parts)
                    image_filepath = os.path.join(self.output_folder, base_filename + ".jpg")
                    data_filepath = os.path.join(self.output_folder, base_filename + ".json")
                    text_filepath = os.path.join(self.output_folder, base_filename + ".txt")
                    
                    # Save image
                    cv2.imwrite(image_filepath, frame)
                    
                    # Prepare measurement data
                    measurement_data = {
                        "timestamp": timestamp,
                        "image_filename": base_filename + ".jpg",
                        "image_count": self.image_count,
                        "calibration_factor": GAP_CALIB_FACTOR,
                        "marker_size_mm": MARKER_SIZE_MM,
                        "top_pair": self._extract_pair_data(self.current_info["top"], self.last_top_corners),
                        "bottom_pair": self._extract_pair_data(self.current_info["bottom"], self.last_bottom_corners)
                    }
                    
                    # Save JSON data
                    with open(data_filepath, 'w') as f:
                        json.dump(measurement_data, f, indent=2)
                    
                    # Save human-readable text file with GUI values
                    self._save_gui_values_text(text_filepath, timestamp)
                    
                    if os.path.exists(image_filepath) and os.path.exists(data_filepath) and os.path.exists(text_filepath):
                        self.status_label.config(text=f"Saved: {base_filename}.jpg + .json + .txt", fg='green')
                        self.root.after(2000, lambda: self.status_label.config(text="Status: Running", fg='green'))
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}", fg='red')
                
    def _extract_pair_data(self, pair_info, corners_info):
        """Extract measurement data for a pair"""

        def _f(val):
            try:
                return float(val) if val is not None else None
            except Exception:
                return None

        data = {
            "status": "Active" if pair_info.get("gap_mm") is not None else "No markers",
            "average_gap_mm": _f(pair_info.get("gap_mm")),
            "mm_per_px_ratio": _f(pair_info.get("ratio")),
            "individual_measurements": {},
            "corner_positions": {},
            "tilt_information": {}
        }
        
        # Add tilt information if available
        if pair_info.get("tilt"):
            tilt = pair_info["tilt"]
            data["tilt_information"] = {
                "tilt_angle_degrees": round(tilt.get("angle", 0), 2),
                "correction_factor": round(tilt.get("correction_factor", 1.0), 4),
                "tilt_status": tilt.get("status", "OK"),
                "left_marker_tilt": round(tilt.get("left_angle", 0), 2),
                "right_marker_tilt": round(tilt.get("right_angle", 0), 2),
                "left_marker_rotation": round(tilt.get("left_rotation", 0), 2),
                "right_marker_rotation": round(tilt.get("right_rotation", 0), 2),
                "average_rotation": round(tilt.get("avg_rotation", 0), 2)
            }
        
        # Extract individual gap measurements
        if pair_info.get("gaps"):
            for name in ["Top", "Mid", "Bottom"]:
                gap_data = pair_info["gaps"].get(name)
                if gap_data and gap_data[0] is not None:
                    gap_px, gap_mm_raw, gap_mm_calib = gap_data
                    data["individual_measurements"][name.lower()] = {
                        "gap_pixels": _f(round(gap_px, 2)),
                        "gap_mm_raw": _f(round(gap_mm_raw, 3)),
                        "gap_mm_calibrated": _f(round(gap_mm_calib, 3))
                    }
                else:
                    data["individual_measurements"][name.lower()] = None
        
        # Extract corner positions
        if corners_info:
            if "left" in corners_info:
                data["corner_positions"]["left_marker"] = {}
                for label in ["TL", "TR", "BR", "BL"]:
                    corner = corners_info["left"].get(label)
                    if corner:
                        data["corner_positions"]["left_marker"][label.lower()] = {
                            "x": int(corner[0]),
                            "y": int(corner[1])
                        }
                    else:
                        data["corner_positions"]["left_marker"][label.lower()] = None
            
            if "right" in corners_info:
                data["corner_positions"]["right_marker"] = {}
                for label in ["TL", "TR", "BR", "BL"]:
                    corner = corners_info["right"].get(label)
                    if corner:
                        data["corner_positions"]["right_marker"][label.lower()] = {
                            "x": int(corner[0]),
                            "y": int(corner[1])
                        }
                    else:
                        data["corner_positions"]["right_marker"][label.lower()] = None
        
        return data
    
    def _save_gui_values_text(self, filepath, timestamp):
        """Save GUI display values in human-readable text format"""
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("ARUCO GAP MEASUREMENT VALUES\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Image Count: {self.image_count}\n")
            f.write(f"Calibration Factor: {GAP_CALIB_FACTOR:.4f}\n")
            f.write(f"Marker Size: {MARKER_SIZE_MM} mm\n")
            f.write("\n" + "=" * 70 + "\n\n")
            
            # Top Pair
            f.write("TOP PAIR\n")
            f.write("-" * 70 + "\n")
            if self.last_top_info["gap_mm"] is not None:
                status = "● Active" if self.last_top_info["status"] == "Active" else "○ Last known"
                f.write(f"Status: {status}\n")
                f.write(f"Average Gap: {self.last_top_info['gap_mm']:.2f} mm\n")
                f.write(f"mm/px Ratio: {self.last_top_info['ratio']:.5f}\n")
                
                # Add tilt information if available
                if hasattr(self, 'current_info') and self.current_info.get("top", {}).get("tilt"):
                    tilt = self.current_info["top"]["tilt"]
                    f.write(f"Tilt Angle: {tilt.get('angle', 0):.2f}°\n")
                    f.write(f"Tilt Status: {tilt.get('status', 'OK')}\n")
                    f.write(f"Tilt Correction Factor: {tilt.get('correction_factor', 1.0):.4f}\n")
                    f.write(f"Rotation Angle: {tilt.get('avg_rotation', 0):.2f}°\n")
                    f.write(f"  Left Marker: {tilt.get('left_rotation', 0):.2f}°\n")
                    f.write(f"  Right Marker: {tilt.get('right_rotation', 0):.2f}°\n")
                
                f.write("\n")
                
                f.write("Individual Measurements:\n")
                if self.last_top_info.get("gaps"):
                    for name in ["Top", "Mid", "Bottom"]:
                        gap_data = self.last_top_info["gaps"].get(name)
                        if gap_data and gap_data[0] is not None:
                            _, _, gap_mm_calib = gap_data
                            f.write(f"  {name:6s}: {gap_mm_calib:.2f} mm\n")
                        else:
                            f.write(f"  {name:6s}: -- mm\n")
            else:
                f.write("Status: ○ Waiting for markers...\n")
                f.write("Average Gap: -- mm\n")
                f.write("mm/px Ratio: --\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
            
            # Bottom Pair
            f.write("BOTTOM PAIR\n")
            f.write("-" * 70 + "\n")
            if self.last_bottom_info["gap_mm"] is not None:
                status = "● Active" if self.last_bottom_info["status"] == "Active" else "○ Last known"
                f.write(f"Status: {status}\n")
                f.write(f"Average Gap: {self.last_bottom_info['gap_mm']:.2f} mm\n")
                f.write(f"mm/px Ratio: {self.last_bottom_info['ratio']:.5f}\n")
                
                # Add tilt information if available
                if hasattr(self, 'current_info') and self.current_info.get("bottom", {}).get("tilt"):
                    tilt = self.current_info["bottom"]["tilt"]
                    f.write(f"Tilt Angle: {tilt.get('angle', 0):.2f}°\n")
                    f.write(f"Tilt Status: {tilt.get('status', 'OK')}\n")
                    f.write(f"Tilt Correction Factor: {tilt.get('correction_factor', 1.0):.4f}\n")
                    f.write(f"Rotation Angle: {tilt.get('avg_rotation', 0):.2f}°\n")
                    f.write(f"  Left Marker: {tilt.get('left_rotation', 0):.2f}°\n")
                    f.write(f"  Right Marker: {tilt.get('right_rotation', 0):.2f}°\n")
                
                f.write("\n")
                
                f.write("Individual Measurements:\n")
                if self.last_bottom_info.get("gaps"):
                    for name in ["Top", "Mid", "Bottom"]:
                        gap_data = self.last_bottom_info["gaps"].get(name)
                        if gap_data and gap_data[0] is not None:
                            _, _, gap_mm_calib = gap_data
                            f.write(f"  {name:6s}: {gap_mm_calib:.2f} mm\n")
                        else:
                            f.write(f"  {name:6s}: -- mm\n")
            else:
                f.write("Status: ○ Waiting for markers...\n")
                f.write("Average Gap: -- mm\n")
                f.write("mm/px Ratio: --\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
            
            # Corner Positions
            f.write("CORNER POSITIONS (pixels)\n")
            f.write("-" * 70 + "\n\n")
            
            # Top Pair Corners
            f.write("TOP PAIR:\n")
            if self.last_top_corners:
                if "left" in self.last_top_corners:
                    f.write("LEFT TOP1 Marker:\n")
                    for label in ["TL", "TR", "BR", "BL"]:
                        corner = self.last_top_corners["left"].get(label)
                        if corner:
                            f.write(f"  {label}: ({corner[0]:4d}, {corner[1]:4d})\n")
                        else:
                            f.write(f"  {label}: (----, ----)\n")
                
                if "right" in self.last_top_corners:
                    f.write("\nLEFT TOP2 Marker:\n")
                    for label in ["TL", "TR", "BR", "BL"]:
                        corner = self.last_top_corners["right"].get(label)
                        if corner:
                            f.write(f"  {label}: ({corner[0]:4d}, {corner[1]:4d})\n")
                        else:
                            f.write(f"  {label}: (----, ----)\n")
            else:
                f.write("No corner data available\n")
            
            f.write("\n" + "-" * 70 + "\n\n")
            
            # Bottom Pair Corners
            f.write("BOTTOM PAIR:\n")
            if self.last_bottom_corners:
                if "left" in self.last_bottom_corners:
                    f.write("LEFT BOTTOM1 Marker:\n")
                    for label in ["TL", "TR", "BR", "BL"]:
                        corner = self.last_bottom_corners["left"].get(label)
                        if corner:
                            f.write(f"  {label}: ({corner[0]:4d}, {corner[1]:4d})\n")
                        else:
                            f.write(f"  {label}: (----, ----)\n")
                
                if "right" in self.last_bottom_corners:
                    f.write("\nLEFT BOTTOM2 Marker:\n")
                    for label in ["TL", "TR", "BR", "BL"]:
                        corner = self.last_bottom_corners["right"].get(label)
                        if corner:
                            f.write(f"  {label}: ({corner[0]:4d}, {corner[1]:4d})\n")
                        else:
                            f.write(f"  {label}: (----, ----)\n")
            else:
                f.write("No corner data available\n")
            
            f.write("\n" + "=" * 70 + "\n")
                
    def start_measurement(self):
        """Start measurement by capturing first 4K photo"""
        try:
            # Stop current camera operation
            self.picam2.stop()

            # Configure for 4K capture
            self.picam2.configure(self.still_config)
            self.picam2.start()

            # Capture 4K image
            frame_4k = self.picam2.capture_array()

            # Stop and reconfigure for preview
            self.picam2.stop()
            preview_config = self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (1280, 720)}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()

            # Convert to BGR for OpenCV processing
            frame_bgr = cv2.cvtColor(frame_4k, cv2.COLOR_RGB2BGR)

            # Process the frame to get measurements
            measurement_data = self._process_measurement_frame(frame_bgr)

            if measurement_data:
                self.first_measurement = measurement_data
                self.measurement_mode = True

                # Save the first image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"measurement_start_{timestamp}.jpg"
                self.first_image_path = os.path.join(self.output_folder, filename)
                cv2.imwrite(self.first_image_path, frame_bgr)

                # Update UI
                self.start_button.config(state=tk.DISABLED, bg='#666666')
                self.stop_button.config(state=tk.NORMAL, bg='#FF9800')
                self.status_label.config(text="Status: Measurement started - Move object and press Stop", fg='blue')
                self._reset_movement_cards(status_text="Start captured. Move markers then press Stop.")

                print(f"First measurement captured: {filename}")
            else:
                self.status_label.config(text="Error: Could not detect markers for first measurement", fg='red')
                self.results_status_label.config(text="Error: Could not detect markers for first measurement", fg='red')

        except Exception as e:
            self.status_label.config(text=f"Error starting measurement: {str(e)}", fg='red')
            self.results_status_label.config(text=f"Error starting measurement: {str(e)}", fg='red')
            print(f"Error in start_measurement: {e}")

    def stop_measurement(self):
        """Stop measurement by capturing second 4K photo and calculating distance"""
        try:
            # Stop current camera operation
            self.picam2.stop()

            # Configure for 4K capture
            self.picam2.configure(self.still_config)
            self.picam2.start()

            # Capture 4K image
            frame_4k = self.picam2.capture_array()

            # Stop and reconfigure for preview
            self.picam2.stop()
            preview_config = self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (1280, 720)}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()

            # Convert to BGR for OpenCV processing
            frame_bgr = cv2.cvtColor(frame_4k, cv2.COLOR_RGB2BGR)

            # Process the frame to get measurements
            second_measurement = self._process_measurement_frame(frame_bgr)

            if second_measurement and self.first_measurement:
                # Calculate distance moved
                distance_moved = self._calculate_distance_moved(self.first_measurement, second_measurement)

                # Save the second image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"measurement_stop_{timestamp}.jpg"
                second_image_path = os.path.join(self.output_folder, filename)
                cv2.imwrite(second_image_path, frame_bgr)

                # Save comparison data
                self._save_measurement_comparison(self.first_measurement, second_measurement, distance_moved, timestamp)

                # Update UI
                self.start_button.config(state=tk.NORMAL, bg='#2196F3')
                self.stop_button.config(state=tk.DISABLED, bg='#666666')
                self.measurement_mode = False

                # Display result
                if distance_moved:
                    self.status_label.config(
                        text=f"Distance moved: {distance_moved['total_distance']:.2f}mm (X:{distance_moved['dx']:.2f}, Y:{distance_moved['dy']:.2f}, Z:{distance_moved['dz']:.2f})",
                        fg='green'
                    )
                    # Display detailed results in GUI
                    self._display_measurement_results(self.first_measurement, second_measurement, distance_moved, timestamp)
                else:
                    self.status_label.config(text="Measurement complete - check saved files", fg='green')

                print(f"Second measurement captured: {filename}")
                print(f"Distance moved: {distance_moved}")

            else:
                self.status_label.config(text="Error: Could not detect markers for second measurement", fg='red')
                self.results_status_label.config(text="Error: Could not detect markers for second measurement", fg='red')

        except Exception as e:
            self.status_label.config(text=f"Error stopping measurement: {str(e)}", fg='red')
            self.results_status_label.config(text=f"Error stopping measurement: {str(e)}", fg='red')
            print(f"Error in stop_measurement: {e}")

    def _process_measurement_frame(self, frame):
        """Process a frame to extract measurement data"""
        try:
            corners_list, ids, _ = self.detector.detectMarkers(frame)

            measurement_data = {
                "top": {},
                "bottom": {},
                "corners": {},
                "frame": frame
            }

            if ids is not None and len(corners_list) >= 2:
                cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

                # Compute marker centers
                centers = []
                for i, c in enumerate(corners_list):
                    pts = c.reshape((4, 2))
                    center = np.mean(pts, axis=0)
                    centers.append((i, center[0], center[1]))

                centers_sorted_by_y = sorted(centers, key=lambda t: t[2])

                # Process TOP pair
                if len(centers_sorted_by_y) >= 2:
                    top_two = centers_sorted_by_y[:2]
                    top_two_sorted_by_x = sorted(top_two, key=lambda t: t[1])
                    top_left_idx = top_two_sorted_by_x[0][0]
                    top_right_idx = top_two_sorted_by_x[1][0]

                    gap_top_mm, ratio_top, gaps_info_top, corners_info_top, tilt_info_top, pose_info_top = process_pair(
                        frame,
                        corners_list,
                        top_left_idx,
                        top_right_idx,
                        label_left="LEFT TOP1",
                        label_right="LEFT TOP2",
                        text_y_offset=30,
                        pair_name="TOP PAIR",
                        camera_matrix=CAMERA_MATRIX,
                        dist_coeffs=DIST_COEFFS
                    )
                    measurement_data["top"] = {
                        "gap_mm": gap_top_mm,
                        "ratio": ratio_top,
                        "gaps": gaps_info_top,
                        "tilt": tilt_info_top,
                        "pose": pose_info_top
                    }
                    measurement_data["corners"]["top"] = corners_info_top

                # Process BOTTOM pair
                if len(centers_sorted_by_y) >= 4:
                    bottom_two = centers_sorted_by_y[-2:]
                    bottom_two_sorted_by_x = sorted(bottom_two, key=lambda t: t[1])
                    bottom_left_idx = bottom_two_sorted_by_x[0][0]
                    bottom_right_idx = bottom_two_sorted_by_x[1][0]

                    gap_bottom_mm, ratio_bottom, gaps_info_bottom, corners_info_bottom, tilt_info_bottom, pose_info_bottom = process_pair(
                        frame,
                        corners_list,
                        bottom_left_idx,
                        bottom_right_idx,
                        label_left="LEFT BOTTOM1",
                        label_right="LEFT BOTTOM2",
                        text_y_offset=230,
                        pair_name="BOTTOM PAIR",
                        camera_matrix=CAMERA_MATRIX,
                        dist_coeffs=DIST_COEFFS
                    )
                    measurement_data["bottom"] = {
                        "gap_mm": gap_bottom_mm,
                        "ratio": ratio_bottom,
                        "gaps": gaps_info_bottom,
                        "tilt": tilt_info_bottom,
                        "pose": pose_info_bottom
                    }
                    measurement_data["corners"]["bottom"] = corners_info_bottom

            return measurement_data

        except Exception as e:
            print(f"Error processing measurement frame: {e}")
            return None

    def _calculate_distance_moved(self, first_measurement, second_measurement):
        """Calculate distance moved between two measurements"""
        try:
            distance_data = {}

            # Try to use 3D pose data first (most accurate)
            if (first_measurement["top"].get("pose") and second_measurement["top"].get("pose") and
                first_measurement["top"]["pose"].get("l_tvec") is not None and
                second_measurement["top"]["pose"].get("l_tvec") is not None):

                # Use left marker of top pair for distance calculation
                first_pos = first_measurement["top"]["pose"]["l_tvec"]
                second_pos = second_measurement["top"]["pose"]["l_tvec"]

                dx = second_pos[0] - first_pos[0]
                dy = second_pos[1] - first_pos[1]
                dz = second_pos[2] - first_pos[2]
                total_distance = np.sqrt(dx**2 + dy**2 + dz**2)

                distance_data = {
                    "method": "3D_pose",
                    "total_distance": total_distance,
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "first_pos": first_pos,
                    "second_pos": second_pos
                }

            # Fallback to gap measurements if 3D pose not available
            elif (first_measurement["top"].get("gap_mm") and second_measurement["top"].get("gap_mm")):
                gap_change = second_measurement["top"]["gap_mm"] - first_measurement["top"]["gap_mm"]

                distance_data = {
                    "method": "gap_difference",
                    "total_distance": abs(gap_change),
                    "dx": gap_change,  # Approximate as movement in X direction
                    "dy": 0.0,
                    "dz": 0.0,
                    "gap_change": gap_change
                }

            return distance_data

        except Exception as e:
            print(f"Error calculating distance moved: {e}")
            return None

    def _save_measurement_comparison(self, first_measurement, second_measurement, distance_moved, timestamp):
        """Save comparison data between start and stop measurements"""
        try:
            filename = f"measurement_comparison_{timestamp}.txt"
            filepath = os.path.join(self.output_folder, filename)

            with open(filepath, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("ARUCO MARKER DISTANCE MEASUREMENT COMPARISON\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"First Image: measurement_start_{timestamp}.jpg\n")
                f.write(f"Second Image: measurement_stop_{timestamp}.jpg\n")
                f.write("\n")

                # Distance moved summary
                if distance_moved:
                    f.write("DISTANCE MOVED:\n")
                    f.write("-" * 40 + "\n")
                    if distance_moved["method"] == "3D_pose":
                        f.write(f"Method: 3D Pose Tracking\n")
                        f.write(f"Total Distance: {distance_moved['total_distance']:.2f} mm\n")
                        f.write(f"X Movement: {distance_moved['dx']:.2f} mm\n")
                        f.write(f"Y Movement: {distance_moved['dy']:.2f} mm\n")
                        f.write(f"Z Movement: {distance_moved['dz']:.2f} mm\n")
                    elif distance_moved["method"] == "gap_difference":
                        f.write(f"Method: Gap Difference\n")
                        f.write(f"Gap Change: {distance_moved['gap_change']:.2f} mm\n")
                    f.write("\n")
                else:
                    f.write("DISTANCE CALCULATION: Failed\n\n")

                # First measurement
                f.write("FIRST MEASUREMENT (START):\n")
                f.write("-" * 40 + "\n")
                self._write_measurement_data(f, first_measurement, "START")

                f.write("\n" + "=" * 80 + "\n\n")

                # Second measurement
                f.write("SECOND MEASUREMENT (STOP):\n")
                f.write("-" * 40 + "\n")
                self._write_measurement_data(f, second_measurement, "STOP")

                f.write("\n" + "=" * 80 + "\n")

            print(f"Comparison data saved: {filename}")

        except Exception as e:
            print(f"Error saving measurement comparison: {e}")

    def _compute_point_deltas(self, first_pair, second_pair):
        """Compute per-point movement (Top/Mid/Bottom) between two measurements"""
        deltas = {}
        for name in ["Top", "Mid", "Bottom"]:
            start = first_pair.get("gaps", {}).get(name) if first_pair else None
            end = second_pair.get("gaps", {}).get(name) if second_pair else None
            if start and end and start[2] is not None and end[2] is not None:
                deltas[name] = end[2] - start[2]
            else:
                deltas[name] = None
        return deltas

    def _display_measurement_results(self, first_measurement, second_measurement, distance_moved, timestamp):
        """Display movement results in the card UI under the live feed"""
        try:
            top_point_deltas = self._compute_point_deltas(
                first_measurement.get("top", {}),
                second_measurement.get("top", {})
            )
            bottom_point_deltas = self._compute_point_deltas(
                first_measurement.get("bottom", {}),
                second_measurement.get("bottom", {})
            )

            top_distance = None
            bottom_distance = None
            if first_measurement["top"].get("gap_mm") is not None and second_measurement["top"].get("gap_mm") is not None:
                top_distance = second_measurement["top"]["gap_mm"] - first_measurement["top"]["gap_mm"]

            if first_measurement["bottom"].get("gap_mm") is not None and second_measurement["bottom"].get("gap_mm") is not None:
                bottom_distance = second_measurement["bottom"]["gap_mm"] - first_measurement["bottom"]["gap_mm"]

            method_label = distance_moved["method"] if distance_moved and distance_moved.get("method") else "gap_difference"

            movement_data = {
                "top": {"distance": top_distance, "points": top_point_deltas, "method": method_label},
                "bottom": {"distance": bottom_distance, "points": bottom_point_deltas, "method": method_label}
            }

            self.last_movement_results = movement_data
            self._update_movement_cards(movement_data, method_label)

            status_text = "Movement calculated"
            if top_distance is not None or bottom_distance is not None:
                status_text = f"Movement calculated ({'3D pose' if method_label == '3D_pose' else 'Gap difference'})"
            self.results_status_label.config(text=status_text, fg='white')

        except Exception as e:
            print(f"Error displaying measurement results: {e}")
            self.results_status_label.config(text=f"Error displaying results: {str(e)}", fg='red')

    def _write_measurement_data(self, f, measurement, label):
        """Write measurement data to file"""
        # Top pair
        if measurement["top"].get("gap_mm"):
            f.write(f"Top Pair Gap ({label}): {measurement['top']['gap_mm']:.2f} mm\n")
            f.write(f"Top Pair Ratio ({label}): {measurement['top']['ratio']:.5f}\n")

            if measurement["top"].get("pose") and measurement["top"]["pose"].get("l_tvec") is not None:
                tvec = measurement["top"]["pose"]["l_tvec"]
                f.write(f"Top Left Marker Position ({label}): X={tvec[0]:.2f}, Y={tvec[1]:.2f}, Z={tvec[2]:.2f} mm\n")

        # Bottom pair
        if measurement["bottom"].get("gap_mm"):
            f.write(f"Bottom Pair Gap ({label}): {measurement['bottom']['gap_mm']:.2f} mm\n")
            f.write(f"Bottom Pair Ratio ({label}): {measurement['bottom']['ratio']:.5f}\n")

            if measurement["bottom"].get("pose") and measurement["bottom"]["pose"].get("l_tvec") is not None:
                tvec = measurement["bottom"]["pose"]["l_tvec"]
                f.write(f"Bottom Left Marker Position ({label}): X={tvec[0]:.2f}, Y={tvec[1]:.2f}, Z={tvec[2]:.2f} mm\n")

    def quit_app(self):
        """Quit application"""
        self.running = False
        self.picam2.stop()
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GapMeasurementGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

