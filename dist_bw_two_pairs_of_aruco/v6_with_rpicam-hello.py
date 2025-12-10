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
import subprocess
import time

# Real side length of each ArUco marker in mm
MARKER_SIZE_MM = 54.0   # marker size in mm

# Empirical calibration factor for gap in mm (tuned from your test)
GAP_CALIB_FACTOR = 0.9845

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
    # gets the rotation vector, translation vector, and the rejections of the markers
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

class RPiCamArucoDistance:
    def __init__(self, root):
        self.root = root
        self.root.title("RPi Cam ArUco Distance Measurement")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')

        # Create output folder
        self.output_folder = "rpicam_aruco_captures"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.info_queue = queue.Queue(maxsize=2)

        # Measurement state
        self.first_measurement = None
        self.second_measurement = None
        self.distance_moved = None
        self.first_image_path = None
        self.second_image_path = None

        # Camera parameters
        self.camera_params = {
            # Start neutral: auto exposure/awb, brightness 0, gain 1
            'exposure_time': 0,      # ignored while ae_mode is 'auto'
            'analogue_gain': 1.0,
            'brightness': 0.0,
            'contrast': 1.0,
            'saturation': 1.0,
            'sharpness': 1.0,
            'awb_mode': 'auto',
            'ae_mode': 'auto',
            'focus_mode': 'auto'
        }

        # ArUco detection parameters
        self.aruco_params = {
            'adaptiveThreshWinSizeMin': 3,
            'adaptiveThreshWinSizeMax': 23,
            'adaptiveThreshWinSizeStep': 10,
            'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
            'cornerRefinementWinSize': 5,
            'cornerRefinementMaxIterations': 30,
            'cornerRefinementMinAccuracy': 0.1
        }

        # Initialize GUI variables (will be used by setup_gui)
        self.thresh_min_var = tk.IntVar(value=self.aruco_params['adaptiveThreshWinSizeMin'])
        self.thresh_max_var = tk.IntVar(value=self.aruco_params['adaptiveThreshWinSizeMax'])
        self.corner_acc_var = tk.DoubleVar(value=self.aruco_params['cornerRefinementMinAccuracy'])
        self.exposure_var = tk.DoubleVar(value=self.camera_params['exposure_time'])
        self.gain_var = tk.DoubleVar(value=self.camera_params['analogue_gain'])
        self.brightness_var = tk.DoubleVar(value=self.camera_params['brightness'])
        self.contrast_var = tk.DoubleVar(value=self.camera_params['contrast'])
        self.awb_var = tk.StringVar(value=self.camera_params['awb_mode'])

        # Setup camera and detector
        self.setup_camera()

        # Setup GUI
        self.setup_gui()

        # Load calibration
        load_calibration()

        # Start video thread
        self.start_video_thread()

        # Bind keyboard events
        self.root.bind('<KeyPress-s>', lambda e: self.capture_first_image())
        self.root.bind('<KeyPress-S>', lambda e: self.capture_first_image())
        self.root.bind('<KeyPress-p>', lambda e: self.capture_second_image())
        self.root.bind('<KeyPress-P>', lambda e: self.capture_second_image())
        self.root.focus_set()

    def setup_camera(self):
        """Initialize camera and ArUco detector"""
        # Use 4X4_250 to match high marker IDs (e.g., 100+)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.params = cv2.aruco.DetectorParameters()
        self.update_aruco_params()  # This creates the detector

        self.picam2 = Picamera2()
        # Preview configuration for live view
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (1280, 720)}
        )
        self.picam2.configure(config)
        self.apply_camera_params()
        self.picam2.start()

    def restart_video_thread(self):
        """Restart the live video capture loop after camera reconfiguration"""
        try:
            # Stop current loop
            self.running = False
            time.sleep(0.2)
            # Start new loop
            self.running = True
            self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.video_thread.start()
            self.update_video()
            self.update_info()
            print("Live feed restarted")
        except Exception as e:
            print(f"Error restarting live feed: {e}")

    def update_aruco_params(self):
        """Update ArUco detector parameters from current settings"""
        self.params.adaptiveThreshWinSizeMin = self.aruco_params['adaptiveThreshWinSizeMin']
        self.params.adaptiveThreshWinSizeMax = self.aruco_params['adaptiveThreshWinSizeMax']
        self.params.adaptiveThreshWinSizeStep = self.aruco_params['adaptiveThreshWinSizeStep']
        self.params.cornerRefinementMethod = self.aruco_params['cornerRefinementMethod']
        self.params.cornerRefinementWinSize = self.aruco_params['cornerRefinementWinSize']
        self.params.cornerRefinementMaxIterations = self.aruco_params['cornerRefinementMaxIterations']
        self.params.cornerRefinementMinAccuracy = self.aruco_params['cornerRefinementMinAccuracy']

        # Recreate detector with new parameters
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)

    def apply_camera_params(self):
        """Apply current camera parameters to Picamera2"""
        try:
            if hasattr(self.picam2, 'set_controls'):
                controls = {}

                # Set exposure if not auto
                if self.camera_params['ae_mode'] != 'auto':
                    controls['ExposureTime'] = self.camera_params['exposure_time']

                # Set gain
                controls['AnalogueGain'] = self.camera_params['analogue_gain']

                # Set brightness, contrast, saturation, sharpness
                controls['Brightness'] = self.camera_params['brightness']
                controls['Contrast'] = self.camera_params['contrast']
                controls['Saturation'] = self.camera_params['saturation']
                controls['Sharpness'] = self.camera_params['sharpness']

                # Set auto white balance mode
                if self.camera_params['awb_mode'] == 'auto':
                    controls['ColourGains'] = (1.0, 1.0)  # Let AWB handle it
                elif self.camera_params['awb_mode'] == 'indoor':
                    controls['ColourGains'] = (1.5, 2.5)
                elif self.camera_params['awb_mode'] == 'outdoor':
                    controls['ColourGains'] = (2.0, 1.2)

                if controls:
                    self.picam2.set_controls(controls)
                    print(f"Applied camera controls: {controls}")

        except Exception as e:
            print(f"Error applying camera parameters: {e}")

    def update_exposure(self, value):
        """Update exposure time"""
        self.camera_params['exposure_time'] = float(value)
        self.apply_camera_params()

    def update_gain(self, value):
        """Update analogue gain"""
        self.camera_params['analogue_gain'] = float(value)
        self.apply_camera_params()

    def update_brightness(self, value):
        """Update brightness"""
        self.camera_params['brightness'] = float(value)
        self.apply_camera_params()

    def update_contrast(self, value):
        """Update contrast"""
        self.camera_params['contrast'] = float(value)
        self.apply_camera_params()

    def update_awb(self, event=None):
        """Update white balance mode"""
        self.camera_params['awb_mode'] = self.awb_var.get()
        self.apply_camera_params()

    def update_aruco_params(self):
        """Update ArUco detection parameters and recreate detector"""
        # Update parameters from GUI variables
        self.aruco_params['adaptiveThreshWinSizeMin'] = self.thresh_min_var.get()
        self.aruco_params['adaptiveThreshWinSizeMax'] = self.thresh_max_var.get()
        self.aruco_params['cornerRefinementMinAccuracy'] = self.corner_acc_var.get()

        # Apply parameters to detector
        self.params.adaptiveThreshWinSizeMin = self.aruco_params['adaptiveThreshWinSizeMin']
        self.params.adaptiveThreshWinSizeMax = self.aruco_params['adaptiveThreshWinSizeMax']
        self.params.adaptiveThreshWinSizeStep = self.aruco_params['adaptiveThreshWinSizeStep']
        self.params.cornerRefinementMethod = self.aruco_params['cornerRefinementMethod']
        self.params.cornerRefinementWinSize = self.aruco_params['cornerRefinementWinSize']
        self.params.cornerRefinementMaxIterations = self.aruco_params['cornerRefinementMaxIterations']
        self.params.cornerRefinementMinAccuracy = self.aruco_params['cornerRefinementMinAccuracy']

        # Recreate detector with new parameters
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        print(f"ArUco detector updated with parameters: {self.aruco_params}")

    def setup_gui(self):
        """Setup the GUI components"""
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left side: Live video window
        left_panel = tk.Frame(main_container, bg='#eeeeee')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))

        video_label_frame = tk.LabelFrame(left_panel, text="Live Feed (ArUco Detection)", bg='#eeeeee', fg='#222222', font=('Arial', 11, 'bold'))
        video_label_frame.pack(padx=5, pady=5)

        self.video_label = tk.Label(video_label_frame, bg='black', width=640, height=360, bd=1, relief=tk.SOLID)
        self.video_label.pack(padx=2, pady=2)

        # Instructions panel
        instructions_frame = tk.LabelFrame(left_panel, text="Instructions", bg='#eeeeee', fg='#222222', font=('Arial', 11, 'bold'))
        instructions_frame.pack(padx=5, pady=5, fill=tk.X)

        instructions_text = tk.Text(
            instructions_frame,
            height=7,
            wrap=tk.WORD,
            font=('Courier', 10, 'bold'),
            bg='#ffffff',
            fg='#111111',
            padx=5,
            pady=5
        )
        instructions_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        instructions_text.insert(tk.END, "Setup: Position 4 ArUco markers in two pairs (top/bottom)\n")
        instructions_text.insert(tk.END, "Adjust camera parameters as needed for optimal detection\n\n")
        instructions_text.insert(tk.END, "Press 'S' key: Capture first 4K image\n")
        instructions_text.insert(tk.END, "Press 'P' key: Capture second 4K image\n")
        instructions_text.insert(tk.END, "Images will be saved and distance moved calculated\n")
        instructions_text.insert(tk.END, "Results shown in terminal and GUI")
        instructions_text.config(state=tk.DISABLED)

        # Camera Parameters panel
        params_frame = tk.LabelFrame(left_panel, text="Camera Parameters", bg='#eeeeee', fg='#222222', font=('Arial', 11, 'bold'))
        params_frame.pack(padx=5, pady=5, fill=tk.X)

        # Create a scrollable frame for parameters
        params_canvas = tk.Canvas(params_frame, bg='#f7f7f7', height=200, highlightthickness=0)
        params_scrollbar = tk.Scrollbar(params_frame, orient="vertical", command=params_canvas.yview)
        params_scrollable_frame = tk.Frame(params_canvas, bg='#f7f7f7')

        params_scrollable_frame.bind(
            "<Configure>",
            lambda e: params_canvas.configure(scrollregion=params_canvas.bbox("all"))
        )

        params_canvas.create_window((0, 0), window=params_scrollable_frame, anchor="nw")
        params_canvas.configure(yscrollcommand=params_scrollbar.set)

        # Exposure controls
        exposure_frame = tk.Frame(params_scrollable_frame, bg='#f7f7f7')
        exposure_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(exposure_frame, text="Exposure (auto start):", fg='#222222', bg='#f7f7f7').pack(side=tk.LEFT)
        exposure_scale = tk.Scale(exposure_frame, from_=1000, to=100000, resolution=1000,
                                orient=tk.HORIZONTAL, variable=self.exposure_var,
                                bg='#f7f7f7', fg='#222222', troughcolor='#888888',
                                command=self.update_exposure)
        exposure_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Gain controls
        gain_frame = tk.Frame(params_scrollable_frame, bg='#f7f7f7')
        gain_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(gain_frame, text="Gain:", fg='#222222', bg='#f7f7f7').pack(side=tk.LEFT)
        gain_scale = tk.Scale(gain_frame, from_=1.0, to=16.0, resolution=0.1,
                            orient=tk.HORIZONTAL, variable=self.gain_var,
                            bg='#f7f7f7', fg='#222222', troughcolor='#888888',
                            command=self.update_gain)
        gain_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Brightness controls
        brightness_frame = tk.Frame(params_scrollable_frame, bg='#f7f7f7')
        brightness_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(brightness_frame, text="Brightness:", fg='#222222', bg='#f7f7f7').pack(side=tk.LEFT)
        brightness_scale = tk.Scale(brightness_frame, from_=-1.0, to=1.0, resolution=0.1,
                                  orient=tk.HORIZONTAL, variable=self.brightness_var,
                                  bg='#f7f7f7', fg='#222222', troughcolor='#888888',
                                  command=self.update_brightness)
        brightness_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Contrast controls
        contrast_frame = tk.Frame(params_scrollable_frame, bg='#f7f7f7')
        contrast_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(contrast_frame, text="Contrast:", fg='#222222', bg='#f7f7f7').pack(side=tk.LEFT)
        contrast_scale = tk.Scale(contrast_frame, from_=0.0, to=2.0, resolution=0.1,
                                orient=tk.HORIZONTAL, variable=self.contrast_var,
                                bg='#f7f7f7', fg='#222222', troughcolor='#888888',
                                command=self.update_contrast)
        contrast_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # AWB Mode
        awb_frame = tk.Frame(params_scrollable_frame, bg='#f7f7f7')
        awb_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(awb_frame, text="AWB Mode:", fg='#222222', bg='#f7f7f7').pack(side=tk.LEFT)
        awb_combo = ttk.Combobox(awb_frame, textvariable=self.awb_var,
                                values=['auto', 'indoor', 'outdoor'],
                                state='readonly', width=10)
        awb_combo.pack(side=tk.LEFT, padx=5)
        awb_combo.bind('<<ComboboxSelected>>', self.update_awb)

        # ArUco Parameters section
        aruco_frame = tk.LabelFrame(params_scrollable_frame, text="ArUco Detection", bg='#e9e9e9', fg='#222222')
        aruco_frame.pack(fill=tk.X, padx=5, pady=5)

        # Threshold min
        thresh_min_frame = tk.Frame(aruco_frame, bg='#e9e9e9')
        thresh_min_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(thresh_min_frame, text="Thresh Min:", fg='#222222', bg='#e9e9e9').pack(side=tk.LEFT)
        thresh_min_spin = tk.Spinbox(thresh_min_frame, from_=3, to=23, textvariable=self.thresh_min_var,
                                   bg='#ffffff', fg='#222222', width=5,
                                   command=self.update_aruco_params)
        thresh_min_spin.pack(side=tk.LEFT, padx=5)

        # Threshold max
        thresh_max_frame = tk.Frame(aruco_frame, bg='#e9e9e9')
        thresh_max_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(thresh_max_frame, text="Thresh Max:", fg='#222222', bg='#e9e9e9').pack(side=tk.LEFT)
        thresh_max_spin = tk.Spinbox(thresh_max_frame, from_=23, to=100, textvariable=self.thresh_max_var,
                                   bg='#ffffff', fg='#222222', width=5,
                                   command=self.update_aruco_params)
        thresh_max_spin.pack(side=tk.LEFT, padx=5)

        # Corner refinement accuracy
        corner_acc_frame = tk.Frame(aruco_frame, bg='#e9e9e9')
        corner_acc_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(corner_acc_frame, text="Corner Accuracy:", fg='#222222', bg='#e9e9e9').pack(side=tk.LEFT)
        corner_acc_spin = tk.Spinbox(corner_acc_frame, from_=0.01, to=1.0, increment=0.01,
                                   textvariable=self.corner_acc_var, format="%.2f",
                                   bg='#ffffff', fg='#222222', width=5,
                                   command=self.update_aruco_params)
        corner_acc_spin.pack(side=tk.LEFT, padx=5)

        params_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Results display
        results_frame = tk.LabelFrame(left_panel, text="Results", bg='#eeeeee', fg='#222222', font=('Arial', 11, 'bold'))
        results_frame.pack(padx=5, pady=5, fill=tk.X)

        self.results_text = tk.Text(
            results_frame,
            height=4,
            wrap=tk.WORD,
            font=('Courier', 10, 'bold'),
            bg='#ffffff',
            fg='#0b5e0b',
            padx=5,
            pady=5
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.results_text.insert(tk.END, "Waiting for captures...\n")
        self.results_text.config(state=tk.DISABLED)

        # Right side: Information display
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create notebook for tabs
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Current Measurement
        current_tab = ttk.Frame(notebook)
        notebook.add(current_tab, text="📊 Current")

        self.current_text = scrolledtext.ScrolledText(
            current_tab,
            wrap=tk.WORD,
            font=('Courier', 10),
            bg='#f5f5f5'
        )
        self.current_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 2: Distance Moved
        distance_tab = ttk.Frame(notebook)
        notebook.add(distance_tab, text="📏 Distance")

        self.distance_text = scrolledtext.ScrolledText(
            distance_tab,
            wrap=tk.WORD,
            font=('Courier', 10),
            bg='#f5f5f5'
        )
        self.distance_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Status: Ready - Press 'S' to start measurement",
            fg='blue',
            font=('Arial', 10, 'bold')
        )
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

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

                # Detect ArUco markers for live preview
                corners_list, ids, _ = self.detector.detectMarkers(frame)

                if ids is not None and len(corners_list) >= 2:
                    cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

                    # Quick processing to show detected pairs
                    centers = []
                    for i, c in enumerate(corners_list):
                        pts = c.reshape((4, 2))
                        center = np.mean(pts, axis=0)
                        centers.append((i, center[0], center[1]))

                    centers_sorted_by_y = sorted(centers, key=lambda t: t[2])

                    # Process TOP pair if available
                    if len(centers_sorted_by_y) >= 2:
                        top_two = centers_sorted_by_y[:2]
                        top_two_sorted_by_x = sorted(top_two, key=lambda t: t[1])
                        top_left_idx = top_two_sorted_by_x[0][0]
                        top_right_idx = top_two_sorted_by_x[1][0]

                        # Draw simple labels
                        left_center = np.mean(corners_list[top_left_idx].reshape((4, 2)), axis=0).astype(int)
                        right_center = np.mean(corners_list[top_right_idx].reshape((4, 2)), axis=0).astype(int)

                        cv2.putText(frame, "TOP LEFT", (left_center[0] - 50, left_center[1] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame, "TOP RIGHT", (right_center[0] - 50, right_center[1] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Process BOTTOM pair if available
                    if len(centers_sorted_by_y) >= 4:
                        bottom_two = centers_sorted_by_y[-2:]
                        bottom_two_sorted_by_x = sorted(bottom_two, key=lambda t: t[1])
                        bottom_left_idx = bottom_two_sorted_by_x[0][0]
                        bottom_right_idx = bottom_two_sorted_by_x[1][0]

                        # Draw simple labels
                        left_center = np.mean(corners_list[bottom_left_idx].reshape((4, 2)), axis=0).astype(int)
                        right_center = np.mean(corners_list[bottom_right_idx].reshape((4, 2)), axis=0).astype(int)

                        cv2.putText(frame, "BOTTOM LEFT", (left_center[0] - 60, left_center[1] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame, "BOTTOM RIGHT", (right_center[0] - 70, right_center[1] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Resize frame for smaller display
                display_frame = cv2.resize(frame, (640, 360))

                # Convert to RGB for tkinter
                frame_rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                # Put frame in queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_rgb_display)

                # Put info in queue (just basic marker count for now)
                info_data = {
                    "marker_count": len(corners_list) if ids is not None else 0,
                    "frame": frame
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
            self.root.after(33, self.update_video)

    def update_info(self):
        """Update information display"""
        try:
            if not self.info_queue.empty():
                info = self.info_queue.get()
                self.update_current_tab(info)
        except Exception as e:
            print(f"Error updating info: {e}")

        if self.running:
            self.root.after(100, self.update_info)

    def update_current_tab(self, info):
        """Update current measurement tab"""
        text = "=" * 50 + "\n"
        text += "LIVE FEED STATUS\n"
        text += "=" * 50 + "\n\n"
        text += f"Markers Detected: {info['marker_count']}\n"
        if info['marker_count'] >= 4:
            text += "✓ Four markers detected - ready for dual pair measurement\n"
        elif info['marker_count'] >= 2:
            text += "⚠️ Two markers detected - can measure single pair\n"
        else:
            text += "❌ Need at least 2 markers for measurement\n\n"

        if self.first_measurement:
            text += "FIRST CAPTURE: COMPLETED\n"
            if self.first_measurement["top"].get("gap_mm"):
                text += f"Top Gap: {self.first_measurement['top']['gap_mm']:.2f} mm\n"
            if self.first_measurement["bottom"].get("gap_mm"):
                text += f"Bottom Gap: {self.first_measurement['bottom']['gap_mm']:.2f} mm\n"
        else:
            text += "FIRST CAPTURE: PENDING (Press 'S')\n"

        text += "\n"

        if self.second_measurement:
            text += "SECOND CAPTURE: COMPLETED\n"
            if self.second_measurement["top"].get("gap_mm"):
                text += f"Top Gap: {self.second_measurement['top']['gap_mm']:.2f} mm\n"
            if self.second_measurement["bottom"].get("gap_mm"):
                text += f"Bottom Gap: {self.second_measurement['bottom']['gap_mm']:.2f} mm\n"
        else:
            text += "SECOND CAPTURE: PENDING (Press 'P')\n"

        if self.distance_moved:
            text += "\nDISTANCE MOVED:\n"
            if self.distance_moved["method"] == "3D_pose":
                text += f"Method: 3D Pose Tracking\n"
                text += f"Total Distance: {self.distance_moved['total_distance']:.2f} mm\n"
                text += f"X Movement: {self.distance_moved['dx']:.2f} mm\n"
                text += f"Y Movement: {self.distance_moved['dy']:.2f} mm\n"
                text += f"Z Movement: {self.distance_moved['dz']:.2f} mm\n"
            elif self.distance_moved["method"] == "gap_difference":
                text += f"Method: Gap Difference\n"
                text += f"Gap Change: {self.distance_moved['gap_change']:.2f} mm\n"

        self.current_text.delete(1.0, tk.END)
        self.current_text.insert(1.0, text)

    def capture_image_with_rpicam(self, filename):
        """Capture a 4K image using rpicam-still command"""
        try:
            filepath = os.path.join(self.output_folder, filename)

            # Stop the live feed thread and camera temporarily
            print("Stopping live camera feed...")
            self.running = False
            time.sleep(0.2)
            self.picam2.stop()
            time.sleep(2)  # Wait for camera to be released

            # Try to close the camera manager completely
            try:
                self.picam2.close()
                print("Camera closed")
            except:
                pass

            time.sleep(1)

            # rpicam-still command for 4K capture with parameters
            cmd = [
                "rpicam-still",
                "--width", "3840",
                "--height", "2160",
                "--quality", "95",
                "--output", filepath,
                "--timeout", "3000",  # 3 second timeout for capture
                "--nopreview"  # No preview window
            ]

            # Add camera parameter overrides
            if self.camera_params['ae_mode'] != 'auto':
                cmd.extend(["--exposure", str(self.camera_params['exposure_time'])])
            else:
                cmd.extend(["--exposure", "normal"])

            if self.camera_params['awb_mode'] != 'auto':
                if self.camera_params['awb_mode'] == 'indoor':
                    cmd.extend(["--awb", "indoor"])
                elif self.camera_params['awb_mode'] == 'outdoor':
                    cmd.extend(["--awb", "sunlight"])
            else:
                cmd.extend(["--awb", "auto"])

            # Add gain if not default
            if abs(self.camera_params['analogue_gain'] - 1.0) > 0.1:
                cmd.extend(["--gain", str(self.camera_params['analogue_gain'])])

            # Add brightness if not default
            if abs(self.camera_params['brightness']) > 0.01:
                cmd.extend(["--brightness", str(self.camera_params['brightness'])])

            # Add contrast if not default
            if abs(self.camera_params['contrast'] - 1.0) > 0.01:
                cmd.extend(["--contrast", str(self.camera_params['contrast'])])

            print(f"Capturing 4K image: {filename}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                print(f"Successfully captured: {filename}")

                # Restart the Picamera2 live feed by recreating the instance
                print("Restarting live camera feed...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"format": "RGB888", "size": (1280, 720)}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(0.5)  # Brief wait for camera to stabilize

                # Restart video thread so live feed resumes
                self.restart_video_thread()

                return filepath
            else:
                print(f"Error capturing image: {result.stderr}")

                # Try to restart the camera even on failure
                try:
                    print("Attempting to restart live camera feed...")
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(
                        main={"format": "RGB888", "size": (1280, 720)}
                    )
                    self.picam2.configure(config)
                    self.picam2.start()
                    time.sleep(0.5)
                    self.restart_video_thread()
                except Exception as restart_error:
                    print(f"Failed to restart camera: {restart_error}")

                return None

        except subprocess.TimeoutExpired:
            print("Image capture timeout")

            # Try to restart the camera
            try:
                print("Attempting to restart live camera feed after timeout...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"format": "RGB888", "size": (1280, 720)}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(0.5)
                self.restart_video_thread()
            except Exception as restart_error:
                print(f"Failed to restart camera: {restart_error}")

            return None
        except Exception as e:
            print(f"Error in capture_image_with_rpicam: {e}")

            # Try to restart the camera
            try:
                print("Attempting to restart live camera feed after error...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"format": "RGB888", "size": (1280, 720)}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(0.5)
                self.restart_video_thread()
            except Exception as restart_error:
                print(f"Failed to restart camera: {restart_error}")

            return None

    def process_captured_image(self, image_path):
        """Process a captured image for ArUco detection"""
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None

            # Read the captured image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not read image: {image_path}")
                return None

            print(f"Processing image: {os.path.basename(image_path)}") # base name returns the file name alone without the path

            # Detect ArUco markers
            corners_list, ids, _ = self.detector.detectMarkers(frame) # gets the corners, ids, and rejections of the markers

            measurement_data = {
                "top": {},
                "bottom": {},
                "corners": {},
                "image_path": image_path
            }

            if ids is not None and len(corners_list) >= 2:
                cv2.aruco.drawDetectedMarkers(frame, corners_list, ids) # draws the markers on the frame

                # Compute marker centers
                centers = [] # list of tuples (index, x, y) for each marker center
                for i, c in enumerate(corners_list): # i is idx and c is corners
                    pts = c.reshape((4, 2)) # reshape the corners into a 4x2 array
                    center = np.mean(pts, axis=0)# not sure what this does
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

                # Save processed image with annotations
                processed_filename = os.path.basename(image_path).replace('.jpg', '_processed.jpg')
                processed_path = os.path.join(self.output_folder, processed_filename)
                cv2.imwrite(processed_path, frame)
                print(f"Saved processed image: {processed_filename}")

                return measurement_data
            else:
                print("No markers detected in captured image")
                return None

        except Exception as e:
            print(f"Error processing captured image: {e}")
            return None

    def capture_first_image(self):
        """Capture first image with 'S' key"""
        if self.first_measurement is not None:
            print("First measurement already exists. Reset by capturing second image first.")
            return

        print("\n=== CAPTURING FIRST IMAGE ===")
        self.status_label.config(text="Status: Capturing first image...", fg='orange')

        # Close the current window
        self.root.withdraw()
        print("Window closed for capture...")

        # Capture image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"first_capture_{timestamp}.jpg"
        image_path = self.capture_image_with_rpicam(filename)

        if image_path:
            # Process the image
            measurement_data = self.process_captured_image(image_path)

            if measurement_data:
                self.first_measurement = measurement_data
                self.first_image_path = image_path

                # Display results
                self.display_measurement_results("FIRST", measurement_data)

                print("✓ First capture completed successfully")
                print(f"  Image: {filename}")
                if measurement_data["top"].get("gap_mm"):
                    print(".2f")
                if measurement_data["bottom"].get("gap_mm"):
                    print(".2f")

                # Wait a moment and confirm camera is ready
                print("Waiting for camera to stabilize...")
                time.sleep(3)  # Extra wait to ensure camera is fully ready

                # Show the window again
                self.root.deiconify()
                self.root.lift()
                self.root.focus_force()

                self.status_label.config(text="Status: First image captured - Camera ready, press 'P' for second image", fg='green')
                print("✓ Camera restarted and ready for second capture")
                print("✓ Window reopened - you can now move the markers/object and press 'P' to capture the second image")

                # Open image in file viewer (if available)
                try:
                    subprocess.run(["xdg-open", image_path], check=False)
                except:
                    pass
            else:
                self.status_label.config(text="Status: Failed to process first image", fg='red')
        else:
            self.status_label.config(text="Status: Failed to capture first image", fg='red')

    def capture_second_image(self):
        """Capture second image with 'P' key"""
        if self.first_measurement is None:
            print("Please capture first image first (press 'S')")
            return

        print("\n=== CAPTURING SECOND IMAGE ===")
        self.status_label.config(text="Status: Capturing second image...", fg='orange')

        # Capture image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"second_capture_{timestamp}.jpg"
        image_path = self.capture_image_with_rpicam(filename)

        if image_path:
            # Process the image
            measurement_data = self.process_captured_image(image_path)

            if measurement_data:
                self.second_measurement = measurement_data
                self.second_image_path = image_path

                # Calculate distance moved
                self.distance_moved = self.calculate_distance_moved(self.first_measurement, measurement_data)

                # Display results
                self.display_measurement_results("SECOND", measurement_data)
                self.display_distance_results()

                print("✓ Second capture completed successfully")
                print(f"  Image: {filename}")

                # Display distance moved prominently
                if self.distance_moved:
                    if self.distance_moved["method"] == "3D_pose":
                        print(f"DISTANCE MOVED: {self.distance_moved['total_distance']:.2f} mm")
                        print(f"  X Movement: {self.distance_moved['dx']:.2f} mm")
                        print(f"  Y Movement: {self.distance_moved['dy']:.2f} mm")
                        print(f"  Z Movement: {self.distance_moved['dz']:.2f} mm")
                    elif self.distance_moved["method"] == "gap_difference":
                        print(f"Gap Change: {self.distance_moved['gap_change']:.2f} mm")

                if self.distance_moved:
                    self.status_label.config(text=f"DISTANCE MOVED: {self.distance_moved['total_distance']:.2f} mm", fg='green')
                else:
                    self.status_label.config(text="Status: Second image captured", fg='green')

                # Open image in file viewer (if available)
                try:
                    subprocess.run(["xdg-open", image_path], check=False)
                except:
                    pass
            else:
                self.status_label.config(text="Status: Failed to process second image", fg='red')
        else:
            self.status_label.config(text="Status: Failed to capture second image", fg='red')

    def calculate_distance_moved(self, first_measurement, second_measurement):
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

    def display_measurement_results(self, label, measurement_data):
        """Display measurement results in GUI"""
        # Update results text area
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        if label == "FIRST":
            self.results_text.insert(tk.END, "FIRST CAPTURE RESULTS:\n")
        else:
            self.results_text.insert(tk.END, "SECOND CAPTURE RESULTS:\n")

        self.results_text.insert(tk.END, "=" * 25 + "\n")

        if measurement_data["top"].get("gap_mm"):
            self.results_text.insert(tk.END, ".2f")
        if measurement_data["bottom"].get("gap_mm"):
            self.results_text.insert(tk.END, ".2f")

        if self.distance_moved and label == "SECOND":
            self.results_text.insert(tk.END, "\nDISTANCE MOVED:\n")
            if self.distance_moved["method"] == "3D_pose":
                self.results_text.insert(tk.END, ".2f")
            elif self.distance_moved["method"] == "gap_difference":
                self.results_text.insert(tk.END, ".2f")

        self.results_text.config(state=tk.DISABLED)

    def display_distance_results(self):
        """Display distance moved results in distance tab"""
        if not self.distance_moved:
            return

        text = "=" * 60 + "\n"
        text += "DISTANCE MOVED CALCULATION\n"
        text += "=" * 60 + "\n\n"

        if self.distance_moved["method"] == "3D_pose":
            text += "Method: 3D Pose Tracking (Most Accurate)\n\n"
            text += "MOVEMENT COMPONENTS:\n"
            text += "-" * 40 + "\n"
            text += f"Total Distance: {self.distance_moved['total_distance']:.2f} mm\n"
            text += f"X Movement: {self.distance_moved['dx']:.2f} mm\n"
            text += f"Y Movement: {self.distance_moved['dy']:.2f} mm\n"
            text += f"Z Movement: {self.distance_moved['dz']:.2f} mm\n"
            text += "\n"
            text += "3D POSITION DATA:\n"
            text += "-" * 40 + "\n"
            text += "First Position:  "
            text += f"X={self.distance_moved['first_pos'][0]:.1f}, Y={self.distance_moved['first_pos'][1]:.1f}, Z={self.distance_moved['first_pos'][2]:.1f}\n"
            text += "Second Position:"
            text += f"X={self.distance_moved['second_pos'][0]:.1f}, Y={self.distance_moved['second_pos'][1]:.1f}, Z={self.distance_moved['second_pos'][2]:.1f}\n"
        elif self.distance_moved["method"] == "gap_difference":
            text += "Method: Gap Difference (Approximate)\n\n"
            text += f"Gap Change: {self.distance_moved['gap_change']:.2f} mm\n"
            text += f"Approximate Distance: {self.distance_moved['total_distance']:.2f} mm\n"
        text += "\n" + "=" * 60 + "\n"

        self.distance_text.delete(1.0, tk.END)
        self.distance_text.insert(1.0, text)

    def quit_app(self):
        """Quit application"""
        self.running = False
        self.picam2.stop()
        self.root.quit()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = RPiCamArucoDistance(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    