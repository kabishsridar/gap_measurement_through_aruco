import cv2
import numpy as np
from picamera2 import Picamera2

# Real side length of each ArUco marker in mm
MARKER_SIZE_MM = 100.0

# Empirical calibration factor for gap in mm (from your tests)
# This scales the measured gap so it matches your real gap more closely.
GAP_CALIB_FACTOR = 0.955


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

    # corresponding lengths in mm (should be ~100mm each)
    side_lengths_mm = [
        top_px * top_ratio,
        right_px * right_ratio,
        bottom_px * bottom_ratio,
        left_px * left_ratio,
    ]

    return side_lengths_px, side_lengths_mm, mm_per_px_list


def main():
    # ---- ArUco setup ----
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
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

            # --- TOP-based calibration: use only top sides of LEFT and RIGHT
            # side order: [top, right, bottom, left]
            left_top_ratio = left_ratios[0]   # top side of LEFT marker
            right_top_ratio = right_ratios[0] # top side of RIGHT marker

            avg_top_mm_per_px = (left_top_ratio + right_top_ratio) / 2.0

            # unpack in TL, TR, BR, BL order
            l_tl, l_tr, l_br, l_bl = left_corners
            r_tl, r_tr, r_br, r_bl = right_corners

            # ---- INNER edges ----
            # Right edge of LEFT marker: between l_tr (top) and l_br (bottom)
            left_top_point = l_tr
            left_bottom_point = l_br
            left_mid_point = (l_tr + l_br) / 2.0

            # Left edge of RIGHT marker: between r_tl (top) and r_bl (bottom)
            right_top_point = r_tl
            right_bottom_point = r_bl
            right_mid_point = (r_tl + r_bl) / 2.0

            # distances in px between inner facing edges
            d_top_px = np.linalg.norm(right_top_point - left_top_point)
            d_mid_px = np.linalg.norm(right_mid_point - left_mid_point)
            d_bottom_px = np.linalg.norm(right_bottom_point - left_bottom_point)

            # convert each to mm using avg_top_mm_per_px
            top_gap_mm_raw = d_top_px * avg_top_mm_per_px
            mid_gap_mm_raw = d_mid_px * avg_top_mm_per_px
            bottom_gap_mm_raw = d_bottom_px * avg_top_mm_per_px

            # apply empirical calibration factor
            top_gap_mm = top_gap_mm_raw * GAP_CALIB_FACTOR
            mid_gap_mm = mid_gap_mm_raw * GAP_CALIB_FACTOR
            bottom_gap_mm = bottom_gap_mm_raw * GAP_CALIB_FACTOR

            avg_gap_px = (d_top_px + d_mid_px + d_bottom_px) / 3.0
            avg_gap_mm = (top_gap_mm + mid_gap_mm + bottom_gap_mm) / 3.0

            gap_without_margins_mm = avg_gap_mm + 10.0  # still add 10 mm margin

            # draw three inner lines
            cv2.line(
                frame,
                (int(left_top_point[0]), int(left_top_point[1])),
                (int(right_top_point[0]), int(right_top_point[1])),
                (0, 255, 0),
                2,
            )

            cv2.line(
                frame,
                (int(left_mid_point[0]), int(left_mid_point[1])),
                (int(right_mid_point[0]), int(right_mid_point[1])),
                (255, 0, 0),
                2,
            )

            cv2.line(
                frame,
                (int(left_bottom_point[0]), int(left_bottom_point[1])),
                (int(right_bottom_point[0]), int(right_bottom_point[1])),
                (0, 0, 255),
                2,
            )

            # small markers to visualize used edge points
            for p in [
                left_top_point,
                left_mid_point,
                left_bottom_point,
                right_top_point,
                right_mid_point,
                right_bottom_point,
            ]:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)

            # show text on image
            info_y = min(left_center[1], right_center[1]) - 80
            if info_y < 20:
                info_y = 20

            cv2.putText(
                frame,
                f"TOP-based mm/px (avg top L+R): {avg_top_mm_per_px:.4f}",
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
            cv2.putText(
                frame,
                f"Top gap: {d_top_px:.1f}px | {top_gap_mm:.2f}mm",
                (20, info_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Mid gap: {d_mid_px:.1f}px | {mid_gap_mm:.2f}mm",
                (20, info_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Bottom gap: {d_bottom_px:.1f}px | {bottom_gap_mm:.2f}mm",
                (20, info_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Avg gap: {avg_gap_px:.1f}px | {avg_gap_mm:.2f}mm  (+10 => {gap_without_margins_mm:.2f}mm)",
                (20, info_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
            )

            # --------- CONSOLE LOGS: FULL DETAIL ----------
            side_names = ["Top", "Right", "Bottom", "Left"]

            print("========== FRAME ==========")
            print("LEFT MARKER (size of four sides):")
            for i, name in enumerate(side_names):
                print(
                    f"  {name}: {left_lengths_px[i]:.2f}px, "
                    f"{left_lengths_mm[i]:.2f}mm, "
                    f"ratio={left_ratios[i]:.6f} mm/px"
                )

            print("\nRIGHT MARKER (size of four sides):")
            for i, name in enumerate(side_names):
                print(
                    f"  {name}: {right_lengths_px[i]:.2f}px, "
                    f"{right_lengths_mm[i]:.2f}mm, "
                    f"ratio={right_ratios[i]:.6f} mm/px"
                )

            print("\nTOP RATIOS USED FOR GAP:")
            print(f"  Left marker TOP side ratio : {left_top_ratio:.6f} mm/px")
            print(f"  Right marker TOP side ratio: {right_top_ratio:.6f} mm/px")
            print(f"  Average TOP mm/px (raw)    : {avg_top_mm_per_px:.6f} mm/px")
            print(f"  GAP_CALIB_FACTOR (applied) : {GAP_CALIB_FACTOR:.6f}")

            print("\nGAPS BETWEEN INNER EDGES (Right edge of LEFT ↔ Left edge of RIGHT):")
            print(
                f"  Top   gap: {d_top_px:.2f}px  | raw {top_gap_mm_raw:.2f}mm -> calib {top_gap_mm:.2f}mm"
            )
            print(
                f"  Mid   gap: {d_mid_px:.2f}px  | raw {mid_gap_mm_raw:.2f}mm -> calib {mid_gap_mm:.2f}mm"
            )
            print(
                f"  Bottom gap: {d_bottom_px:.2f}px  | raw {bottom_gap_mm_raw:.2f}mm -> calib {bottom_gap_mm:.2f}mm"
            )

            print("\nFINAL RESULT (average of 3 lines, calibrated):")
            print(f"  Average gap: {avg_gap_px:.2f}px  | {avg_gap_mm:.2f}mm")
            print(f"  Final gap (avg + 10mm): {gap_without_margins_mm:.2f}mm")
            print("===========================\n")

        cv2.imshow("Two ArUco Markers - Inner Gap Measurement", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
