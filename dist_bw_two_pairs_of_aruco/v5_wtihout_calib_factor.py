import cv2
import numpy as np
import os
import sys
import time
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


def compute_ray_gap_for_point(start_point, n_unit, seg_a, seg_b, mm_per_px):
    """
    Shoot a ray from start_point along n_unit to intersect the segment (seg_a -> seg_b).
    Returns (gap_px, gap_mm, hit_point) or (None, None, None) if no hit.
    """
    res = intersect_ray_segment(start_point, n_unit, seg_a, seg_b)
    if res is None:
        return None, None, None

    t_min, hit_point = res
    gap_px = float(t_min)  # since n_unit is unit length
    gap_mm = gap_px * mm_per_px
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
        (left_center[0] - 60, left_top_y - 20),   # a bit bigger offset for 4K
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        label_right,
        (right_center[0] - 60, right_top_y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
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

    # ====== STRAIGHT-LINE GAP LOGIC (TOP, MID, BOTTOM) ======

    # Inner right edge of LEFT marker: between l_tr (top) and l_br (bottom)
    left_top_point = l_tr
    left_bottom_point = l_br
    left_mid_point = (l_tr + l_br) / 2.0

    # Inner LEFT edge of RIGHT marker: between r_tl (top) and r_bl (bottom)
    right_left_top = r_tl
    right_left_bottom = r_bl

    # FIXED ray direction: straight to the right along image X axis
    n_unit = np.array([1.0, 0.0], dtype=float)

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
        gap_px, gap_mm, hit_pt = compute_ray_gap_for_point(
            pt,
            n_unit,
            right_left_top,
            right_left_bottom,
            avg_mm_per_px,
        )
        gaps_info[name] = (gap_px, gap_mm)
        hit_points[name] = hit_pt

        if gap_px is not None:
            valid_gaps_px.append(gap_px)
            valid_gaps_mm.append(gap_mm)

    if len(valid_gaps_px) == 0:
        cv2.putText(
            frame,
            f"{pair_name}: no intersection",
            (40, text_y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
                3,
            )
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(hit_pt[0]), int(hit_pt[1])), 5, (255, 0, 0), -1)

    # average over valid lines
    avg_gap_px = float(np.mean(valid_gaps_px))
    avg_gap_mm = float(np.mean(valid_gaps_mm))
    final_gap_mm = avg_gap_mm

    # Text overlay for this pair (scaled up a bit for 4K)
    y = text_y_offset
    cv2.putText(
        frame,
        f"{pair_name} mm/px (avg 8): {avg_mm_per_px:.5f}",
        (40, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    y += 35
    for name in ["Top", "Mid", "Bottom"]:
        gap_px, gap_mm = gaps_info[name]
        if gap_px is not None:
            cv2.putText(
                frame,
                f"{pair_name} {name}: {gap_px:.1f}px | {gap_mm:.2f}mm",
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                f"{pair_name} {name}: no intersection",
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        y += 28

    """ cv2.putText(
        frame,
        f"{pair_name} AVG GAP: {avg_gap_px:.1f}px | {final_gap_mm:.2f}mm",
        (40, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 200, 255),
        2,
    ) """

    return final_gap_mm, avg_mm_per_px


def check_display_available():
    """Check if display is available for OpenCV windows"""
    # Check if DISPLAY environment variable is set (X11) or Wayland
    if 'DISPLAY' not in os.environ and 'WAYLAND_DISPLAY' not in os.environ:
        return False
    # Try to create a test window (will fail silently if no display)
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.namedWindow('__test__', cv2.WINDOW_NORMAL)
        cv2.imshow('__test__', test_img)
        cv2.waitKey(1)
        cv2.destroyWindow('__test__')
        return True
    except Exception:
        return False


def main():
    # Check if display is available
    has_display = check_display_available()
    if not has_display:
        print("No display available - running in headless mode")
        print("Measurements will be printed to console")
        print("Press Ctrl+C to quit.\n")
    else:
        print("Display available - showing video window")
        print("Press 'q' to quit.\n")
    
    # ---- ArUco setup ----
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # ---- Picamera2 setup ----
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()

    while True:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        corners_list, ids, _ = detector.detectMarkers(frame)

        top_gap_mm = None
        bottom_gap_mm = None

        if ids is not None and len(corners_list) >= 2:
            if has_display:
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

                top_gap_mm, _ = process_pair(
                    frame,
                    corners_list,
                    top_left_idx,
                    top_right_idx,
                    label_left="LEFT TOP1",
                    label_right="LEFT TOP2",
                    text_y_offset=80,
                    pair_name="TOP PAIR",
                )

            # --- BOTTOM pair (LEFT BOTTOM1 & LEFT BOTTOM2) from bottom two markers by y ---
            if len(centers_sorted_by_y) >= 4:
                bottom_two = centers_sorted_by_y[-2:]
                bottom_two_sorted_by_x = sorted(bottom_two, key=lambda t: t[1])
                bottom_left_idx = bottom_two_sorted_by_x[0][0]
                bottom_right_idx = bottom_two_sorted_by_x[1][0]

                bottom_gap_mm, _ = process_pair(
                    frame,
                    corners_list,
                    bottom_left_idx,
                    bottom_right_idx,
                    label_left="LEFT BOTTOM1",
                    label_right="LEFT BOTTOM2",
                    text_y_offset=420,
                    pair_name="BOTTOM PAIR",
                )

        # Print measurements in headless mode
        if not has_display:
            if top_gap_mm is not None or bottom_gap_mm is not None:
                print("\r" + " " * 80, end="")  # Clear line
                msg = "TOP: "
                msg += f"{top_gap_mm:.2f}mm" if top_gap_mm is not None else "N/A"
                msg += " | BOTTOM: "
                msg += f"{bottom_gap_mm:.2f}mm" if bottom_gap_mm is not None else "N/A"
                print(f"\r{msg}", end="", flush=True)

        # Show window only if display is available
        if has_display:
            cv2.imshow("Top & Bottom ArUco Gap Measurement - 4K", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # In headless mode, just sleep a bit to avoid 100% CPU
            time.sleep(0.033)  # ~30 FPS

    picam2.stop()
    if has_display:
        cv2.destroyAllWindows()
    else:
        print("\n")  # New line after headless output


if __name__ == "__main__":
    main()
