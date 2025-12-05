import cv2
import numpy as np
from picamera2 import Picamera2

# Real side length of each ArUco marker in mm
MARKER_SIZE_MM = 54.0   # your marker size

# Updated empirical calibration factor for gap in mm
GAP_CALIB_FACTOR = 0.9845  # tuned from your 100mm test


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

    top_px = np.linalg.norm(tr - tl)
    right_px = np.linalg.norm(br - tr)
    bottom_px = np.linalg.norm(bl - br)
    left_px = np.linalg.norm(tl - bl)

    # mm/px ratios for each side using real marker size
    top_ratio = MARKER_SIZE_MM / top_px
    right_ratio = MARKER_SIZE_MM / right_px
    bottom_ratio = MARKER_SIZE_MM / bottom_px
    left_ratio = MARKER_SIZE_MM / left_px

    side_lengths_px = [top_px, right_px, bottom_px, left_px]
    mm_per_px_list = [top_ratio, right_ratio, bottom_ratio, left_ratio]

    # corresponding lengths in mm (should be ~MARKER_SIZE_MM each)
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
    # Solve: a + u*s = p0 + t*r  (2x2 system)
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


def compute_perp_gap_for_point(start_point, n_unit, seg_a, seg_b, mm_per_px, calib_factor):
    """
    Shoot a perpendicular ray from start_point along n_unit to intersect the segment (seg_a -> seg_b).
    Returns (gap_px, gap_mm_raw, gap_mm_calib, hit_point) or (None, None, None, None) if no hit.
    """
    res = intersect_ray_segment(start_point, n_unit, seg_a, seg_b)
    if res is None:
        return None, None, None, None

    t_min, hit_point = res
    gap_px = float(t_min)  # since n_unit is unit length
    gap_mm_raw = gap_px * mm_per_px
    gap_mm_calib = gap_mm_raw * calib_factor
    return gap_px, gap_mm_raw, gap_mm_calib, hit_point


def main():
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

    print("Press 'q' to quit.")

    while True:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        corners_list, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(ids) >= 2:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

            # compute centers to decide left vs right marker
            centers = []
            for c in corners_list:
                pts = c.reshape((4, 2))
                center = np.mean(pts, axis=0)
                centers.append(center)

            indices_sorted = sorted(range(len(centers)), key=lambda i: centers[i][0])
            left_idx = indices_sorted[0]
            right_idx = indices_sorted[-1]

            # reorder corners of each marker into TL, TR, BR, BL
            left_corners_raw = corners_list[left_idx].reshape((4, 2))
            right_corners_raw = corners_list[right_idx].reshape((4, 2))

            left_corners = order_corners(left_corners_raw)
            right_corners = order_corners(right_corners_raw)

            # label markers on image
            left_center = np.mean(left_corners, axis=0).astype(int)
            right_center = np.mean(right_corners, axis=0).astype(int)

            cv2.putText(
                frame,
                "LEFT",
                (left_center[0] - 30, left_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "RIGHT",
                (right_center[0] - 30, right_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # compute side lengths & mm/px for both markers
            left_lengths_px, left_lengths_mm, left_ratios = compute_side_lengths_and_ratios(
                left_corners
            )
            right_lengths_px, right_lengths_mm, right_ratios = compute_side_lengths_and_ratios(
                right_corners
            )

            # --- NEW: use all 8 side ratios (4 from LEFT + 4 from RIGHT)
            all_ratios = left_ratios + right_ratios   # list of 8 values
            avg_mm_per_px = float(np.mean(all_ratios))

            # unpack in TL, TR, BR, BL order
            l_tl, l_tr, l_br, l_bl = left_corners
            r_tl, r_tr, r_br, r_bl = right_corners

            # ========== PERPENDICULAR GAP LOGIC (TOP, MID, BOTTOM) ==========

            # Right edge of LEFT marker: between l_tr (top) and l_br (bottom)
            left_top_point = l_tr
            left_bottom_point = l_br
            left_mid_point = (l_tr + l_br) / 2.0

            # LEFT side of RIGHT marker (inner-facing edge):
            # between its TL and BL corners
            right_left_top = r_tl
            right_left_bottom = r_bl

            # center of RIGHT marker (for direction)
            right_mid_point = np.mean(right_corners, axis=0)

            # direction vector of left inner edge (downwards)
            edge_vec = left_bottom_point - left_top_point  # from top to bottom

            # perpendicular direction
            n = np.array([edge_vec[1], -edge_vec[0]], dtype=float)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-9:
                # edge degenerate; skip this frame
                cv2.imshow("Two ArUco Markers - Perpendicular Gap Measurement", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            n_unit = n / n_norm

            # choose direction that points towards RIGHT marker, using midpoint
            vec_to_right = right_mid_point - left_mid_point
            if np.dot(n_unit, vec_to_right) < 0:
                n_unit = -n_unit  # flip to face right marker

            # compute perpendicular gaps for TOP, MID, BOTTOM from left edge
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
                )
                gaps_info[name] = (gap_px, gap_mm_raw, gap_mm_calib)
                hit_points[name] = hit_pt

                if gap_px is not None:
                    valid_gaps_px.append(gap_px)
                    valid_gaps_mm_calib.append(gap_mm_calib)

            # Draw lines & text only if we have at least one valid intersection
            if len(valid_gaps_px) > 0:
                # Draw the three perpendicular lines and circles where valid
                for name, pt in named_points.items():
                    gap_px, _, _ = gaps_info[name]
                    hit_pt = hit_points[name]
                    if gap_px is not None and hit_pt is not None:
                        # line from LEFT marker to LEFT edge of RIGHT marker
                        cv2.line(
                            frame,
                            (int(pt[0]), int(pt[1])),
                            (int(hit_pt[0]), int(hit_pt[1])),
                            (0, 255, 0),
                            2,
                        )
                        # points
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        cv2.circle(frame, (int(hit_pt[0]), int(hit_pt[1])), 4, (255, 0, 0), -1)

                # average over valid lines
                avg_gap_px = float(np.mean(valid_gaps_px))
                avg_gap_mm_calib = float(np.mean(valid_gaps_mm_calib))
                final_gap_mm = avg_gap_mm_calib

                # Text overlay
                info_y = min(left_center[1], right_center[1]) - 100
                if info_y < 20:
                    info_y = 20

                cv2.putText(
                    frame,
                    f"mm/px (avg 8 sides): {avg_mm_per_px:.4f}",
                    (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Calib factor: {GAP_CALIB_FACTOR:.4f}",
                    (20, info_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                line_y = info_y + 45
                for name in ["Top", "Mid", "Bottom"]:
                    gap_px, gap_mm_raw, gap_mm_calib = gaps_info[name]
                    if gap_px is not None:
                        cv2.putText(
                            frame,
                            f"{name} gap: {gap_px:.1f}px | raw {gap_mm_raw:.2f}mm -> calib {gap_mm_calib:.2f}mm",
                            (20, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                        )
                        line_y += 25
                    else:
                        cv2.putText(
                            frame,
                            f"{name} gap: no intersection",
                            (20, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                        )
                        line_y += 25

                cv2.putText(
                    frame,
                    f"Avg gap: {avg_gap_px:.1f}px | {final_gap_mm:.2f}mm",
                    (20, line_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    2,
                )

                # Console logs
                print("========== FRAME ==========")
                side_names = ["Top", "Right", "Bottom", "Left"]
                print("LEFT MARKER (side lengths / ratios):")
                for i, name in enumerate(side_names):
                    print(
                        f"  {name}: {left_lengths_px[i]:.2f}px, "
                        f"{left_lengths_mm[i]:.2f}mm, "
                        f"ratio={left_ratios[i]:.6f} mm/px"
                    )
                print("\nRIGHT MARKER (side lengths / ratios):")
                for i, name in enumerate(side_names):
                    print(
                        f"  {name}: {right_lengths_px[i]:.2f}px, "
                        f"{right_lengths_mm[i]:.2f}mm, "
                        f"ratio={right_ratios[i]:.6f} mm/px"
                    )

                print("\nAll 8 mm/px ratios:", all_ratios)
                print("Avg mm/px (8 sides):", avg_mm_per_px)

                print("\nPerpendicular gaps (RIGHT edge of LEFT → LEFT edge of RIGHT):")
                for name in ["Top", "Mid", "Bottom"]:
                    gap_px, gap_mm_raw, gap_mm_calib = gaps_info[name]
                    if gap_px is not None:
                        print(
                            f"  {name}: {gap_px:.2f}px | raw {gap_mm_raw:.2f}mm -> calib {gap_mm_calib:.2f}mm"
                        )
                    else:
                        print(f"  {name}: no intersection")

                print(f"\nAverage gap (calib): {final_gap_mm:.2f}mm")
                print(f"Final gap: {final_gap_mm:.2f}mm")
                print("===========================\n")

            else:
                # No intersection found at all (ray from none of the 3 points hit the LEFT edge of right marker)
                cv2.putText(
                    frame,
                    "Perp rays did not hit LEFT edge of RIGHT marker",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Two ArUco Markers - Perpendicular Gap Measurement", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
