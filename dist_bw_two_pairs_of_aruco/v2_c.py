import cv2
import numpy as np
import os
from datetime import datetime
from picamera2 import Picamera2

# Real side length of each ArUco marker in mm
MARKER_SIZE_MM = 54.0   # marker size in mm

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


def detect_marker_perspective(corners):
    """
    Detect perspective distortion in marker by checking if it's a square.
    Returns correction factor (1.0 = no correction needed, >1.0 = needs correction).
    """
    tl, tr, br, bl = corners
    
    # Calculate side lengths
    top_side = np.linalg.norm(tr - tl)
    right_side = np.linalg.norm(br - tr)
    bottom_side = np.linalg.norm(bl - br)
    left_side = np.linalg.norm(tl - bl)
    
    # For a square, all sides should be equal
    # Calculate aspect ratio distortion
    avg_horizontal = (top_side + bottom_side) / 2.0
    avg_vertical = (left_side + right_side) / 2.0
    
    if avg_horizontal > 0 and avg_vertical > 0:
        aspect_ratio = max(avg_horizontal, avg_vertical) / min(avg_horizontal, avg_vertical)
        # If aspect ratio is close to 1, no correction needed
        # If significantly different, apply small correction
        if aspect_ratio > 1.05:  # More than 5% distortion
            # Correction factor based on perspective
            correction = np.sqrt(aspect_ratio)
            return min(correction, 1.02)  # Cap at 2% correction
    return 1.0


def compute_perp_gap_for_point(start_point, n_unit, seg_a, seg_b, mm_per_px, perspective_correction=1.0):
    """
    Shoot a ray from start_point along n_unit to intersect the segment (seg_a -> seg_b).
    Returns (gap_px, gap_mm, hit_point) or (None, None, None) if no hit.
    Uses perspective correction for better accuracy.
    """
    res = intersect_ray_segment(start_point, n_unit, seg_a, seg_b)
    if res is None:
        return None, None, None

    t_min, hit_point = res
    gap_px = float(t_min)  # since n_unit is unit length
    gap_mm = gap_px * mm_per_px * perspective_correction
    return gap_px, gap_mm, hit_point


def process_pair(frame, corners_list, idx_left, idx_right,
                 label_left, label_right,
                 text_y_offset, pair_name):
    """
    Process one pair of markers:
    - compute mm/px (average over 8 ratios of this pair only)
    - shoot 3 straight horizontal lines (to the right) from right edge of left marker
    - intersect with LEFT edge of right marker
    - draw lines and text for this pair

    Returns (final_gap_mm, avg_mm_per_px) or (None, None) on failure.
    """

    # reorder corners of each marker into TL, TR, BR, BL
    left_corners_raw = corners_list[idx_left].reshape((4, 2))
    right_corners_raw = corners_list[idx_right].reshape((4, 2))

    left_corners = order_corners(left_corners_raw)
    right_corners = order_corners(right_corners_raw)

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
    left_sides_px, _, left_ratios = compute_side_lengths_and_ratios(left_corners)
    right_sides_px, _, right_ratios = compute_side_lengths_and_ratios(right_corners)

    # Weighted average: weight by side length (longer sides are more reliable)
    all_ratios = left_ratios + right_ratios   # 8 values for this pair
    all_weights = left_sides_px + right_sides_px  # 8 weights corresponding to ratios
    
    # Normalize weights
    total_weight = sum(all_weights)
    if total_weight > 0:
        weighted_sum = sum(r * w for r, w in zip(all_ratios, all_weights))
        avg_mm_per_px = float(weighted_sum / total_weight)
    else:
        avg_mm_per_px = float(np.mean(all_ratios))
    
    # Calculate perspective correction for both markers
    left_perspective = detect_marker_perspective(left_corners)
    right_perspective = detect_marker_perspective(right_corners)
    avg_perspective_correction = (left_perspective + right_perspective) / 2.0

    # unpack in TL, TR, BR, BL order
    l_tl, l_tr, l_br, l_bl = left_corners
    r_tl, r_tr, r_br, r_bl = right_corners

    # ========== DRAW CORNER POSITIONS ==========
    # Draw corners and labels for left marker
    left_corner_points = [l_tl, l_tr, l_br, l_bl]
    left_corner_labels = ["TL", "TR", "BR", "BL"]
    left_corner_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 128)]  # Magenta, Cyan, Yellow, Purple
    
    for i, (corner, label, color) in enumerate(zip(left_corner_points, left_corner_labels, left_corner_colors)):
        corner_int = (int(corner[0]), int(corner[1]))
        # Draw circle at corner
        cv2.circle(frame, corner_int, 6, color, -1)
        cv2.circle(frame, corner_int, 8, (255, 255, 255), 2)
        # Draw label with coordinates
        label_text = f"{label}: ({corner_int[0]}, {corner_int[1]})"
        # Position label offset from corner
        offset_x = -60 if i in [0, 3] else 10  # Left side for TL and BL
        offset_y = -15 if i in [0, 1] else 20   # Above for TL and TR
        text_pos = (corner_int[0] + offset_x, corner_int[1] + offset_y)
        cv2.putText(frame, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw corners and labels for right marker
    right_corner_points = [r_tl, r_tr, r_br, r_bl]
    right_corner_labels = ["TL", "TR", "BR", "BL"]
    right_corner_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 128)]  # Magenta, Cyan, Yellow, Purple
    
    for i, (corner, label, color) in enumerate(zip(right_corner_points, right_corner_labels, right_corner_colors)):
        corner_int = (int(corner[0]), int(corner[1]))
        # Draw circle at corner
        cv2.circle(frame, corner_int, 6, color, -1)
        cv2.circle(frame, corner_int, 8, (255, 255, 255), 2)
        # Draw label with coordinates
        label_text = f"{label}: ({corner_int[0]}, {corner_int[1]})"
        # Position label offset from corner
        offset_x = -60 if i in [0, 3] else 10  # Left side for TL and BL
        offset_y = -15 if i in [0, 1] else 20   # Above for TL and TR
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
    valid_gaps_mm = []
    hit_points = {}

    for name, pt in named_points.items():
        gap_px, gap_mm, hit_pt = compute_perp_gap_for_point(
            pt,
            n_unit,
            right_left_top,
            right_left_bottom,
            avg_mm_per_px,
            avg_perspective_correction,
        )
        gaps_info[name] = (gap_px, gap_mm)
        hit_points[name] = hit_pt

        if gap_px is not None:
            valid_gaps_px.append(gap_px)
            valid_gaps_mm.append(gap_mm)

    if len(valid_gaps_px) == 0:
        # text to indicate no intersection for this pair
        cv2.putText(
            frame,
            f"{pair_name}: no intersection",
            (20, text_y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        return None, avg_mm_per_px

    # Draw the three straight horizontal lines and circles where valid
    for name, pt in named_points.items():
        gap_px, _ = gaps_info[name]
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
    avg_gap_mm = float(np.mean(valid_gaps_mm))
    final_gap_mm = avg_gap_mm

    # Text overlay for this pair
    y = text_y_offset
    cv2.putText(
        frame,
        f"{pair_name} mm/px (weighted): {avg_mm_per_px:.4f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )
    y += 22
    cv2.putText(
        frame,
        f"{pair_name} perspective corr: {avg_perspective_correction:.4f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    y += 25
    for name in ["Top", "Mid", "Bottom"]:
        gap_px, gap_mm = gaps_info[name]
        if gap_px is not None:
            cv2.putText(
                frame,
                f"{pair_name} {name}: {gap_px:.1f}px | {gap_mm:.2f}mm",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                f"{pair_name} {name}: no intersection",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        y += 18

    cv2.putText(
        frame,
        f"{pair_name} AVG GAP: {avg_gap_px:.1f}px | {final_gap_mm:.2f}mm",
        (20, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
    )

    return final_gap_mm, avg_mm_per_px


def main():
    # Create output folder for saving images
    output_folder = "gap_measurement_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Using output folder: {output_folder}")
    
    # ---- ArUco setup ----
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    # Enable subpixel corner refinement for better accuracy
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # ---- Picamera2 setup ----
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()

    print("Press 'q' to quit, 's' to save current image with distances.")
    image_count = 0

    while True:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        corners_list, ids, _ = detector.detectMarkers(frame)
        
        # Initialize gap variables for this frame
        gap_top_mm = None
        gap_bottom_mm = None

        if ids is not None and len(corners_list) >= 2:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

            # compute (marker_index, x, y) for each marker center
            centers = []
            for i, c in enumerate(corners_list):
                pts = c.reshape((4, 2))
                center = np.mean(pts, axis=0)
                centers.append((i, center[0], center[1]))  # (index, x, y)

            # sort markers by y (top to bottom)
            centers_sorted_by_y = sorted(centers, key=lambda t: t[2])

            # --- TOP pair (LEFT TOP1 & LEFT TOP2) from top two markers by y ---
            if len(centers_sorted_by_y) >= 2:
                top_two = centers_sorted_by_y[:2]
                top_two_sorted_by_x = sorted(top_two, key=lambda t: t[1])  # sort by x
                top_left_idx = top_two_sorted_by_x[0][0]
                top_right_idx = top_two_sorted_by_x[1][0]

                gap_top_mm, ratio_top = process_pair(
                    frame,
                    corners_list,
                    top_left_idx,
                    top_right_idx,
                    label_left="LEFT TOP1",
                    label_right="LEFT TOP2",
                    text_y_offset=30,
                    pair_name="TOP PAIR",
                )
            else:
                gap_top_mm, ratio_top = None, None

            # --- BOTTOM pair (LEFT BOTTOM1 & LEFT BOTTOM2) from bottom two markers by y ---
            if len(centers_sorted_by_y) >= 4:
                bottom_two = centers_sorted_by_y[-2:]
                bottom_two_sorted_by_x = sorted(bottom_two, key=lambda t: t[1])
                bottom_left_idx = bottom_two_sorted_by_x[0][0]
                bottom_right_idx = bottom_two_sorted_by_x[1][0]

                gap_bottom_mm, ratio_bottom = process_pair(
                    frame,
                    corners_list,
                    bottom_left_idx,
                    bottom_right_idx,
                    label_left="LEFT BOTTOM1",
                    label_right="LEFT BOTTOM2",
                    text_y_offset=230,
                    pair_name="BOTTOM PAIR",
                )
            else:
                gap_bottom_mm, ratio_bottom = None, None

        cv2.imshow("Top & Bottom ArUco Gap Measurement", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # Save image with distance information in filename
            image_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename with distance information
            filename_parts = [f"img_{image_count:04d}", timestamp]
            if gap_top_mm is not None:
                filename_parts.append(f"top_{gap_top_mm:.2f}mm")
            if gap_bottom_mm is not None:
                filename_parts.append(f"bot_{gap_bottom_mm:.2f}mm")
            
            filename = "_".join(filename_parts) + ".jpg"
            filepath = os.path.join(output_folder, filename)
            
            # Save the image
            cv2.imwrite(filepath, frame)
            
            if os.path.exists(filepath):
                print(f"\n[{image_count}] Saved: {filename}")
                if gap_top_mm is not None:
                    print(f"     TOP PAIR gap: {gap_top_mm:.2f}mm")
                if gap_bottom_mm is not None:
                    print(f"     BOTTOM PAIR gap: {gap_bottom_mm:.2f}mm")
            else:
                print(f"\nERROR: Failed to save {filename}")

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
