import cv2
import numpy as np
from picamera2 import Picamera2

# ----- Calibration Factor (Auto-refined from your test) -----
GAP_CALIB_FACTOR = 0.957729   # precise correction based on your 59.83mm reading

# Real side length of each ArUco marker in mm
MARKER_SIZE_MM = 100.0

def order_corners(corners):
    pts = np.array(corners, dtype=np.float32)
    idx_by_y = np.argsort(pts[:, 1])
    top2 = pts[idx_by_y][:2]
    bottom2 = pts[idx_by_y][2:]
    top2 = top2[np.argsort(top2[:, 0])]
    bottom2 = bottom2[np.argsort(bottom2[:, 0])]
    tl, tr = top2
    bl, br = bottom2
    return np.array([tl, tr, br, bl], dtype=np.float32)

def compute_side_lengths_and_ratios(ordered_corners):
    tl, tr, br, bl = ordered_corners
    side_lengths_px = [
        np.linalg.norm(tr - tl),   # top
        np.linalg.norm(br - tr),   # right
        np.linalg.norm(bl - br),   # bottom
        np.linalg.norm(tl - bl)    # left
    ]
    mm_per_px_list = [MARKER_SIZE_MM / px for px in side_lengths_px]
    side_lengths_mm = [side_lengths_px[i] * mm_per_px_list[i] for i in range(4)]
    return side_lengths_px, side_lengths_mm, mm_per_px_list

def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
    picam2.configure(config)
    picam2.start()

    print("Press 'q' to quit.\n")

    while True:
        frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        corners_list, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(ids) >= 2:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

            centers = [np.mean(c.reshape((4,2)), axis=0) for c in corners_list]
            left_idx, right_idx = sorted(range(len(centers)), key=lambda i: centers[i][0])[0], \
                                  sorted(range(len(centers)), key=lambda i: centers[i][0])[-1]

            left = order_corners(corners_list[left_idx].reshape((4, 2)))
            right = order_corners(corners_list[right_idx].reshape((4, 2)))

            left_px, left_mm, left_ratio = compute_side_lengths_and_ratios(left)
            right_px, right_mm, right_ratio = compute_side_lengths_and_ratios(right)

            # New calibration source: ONLY top edge from both markers
            avg_top_mm_per_px = (left_ratio[0] + right_ratio[0]) / 2.0

            # Inner edge reference points
            l_tl, l_tr, l_br, _ = left
            r_tl, _, r_br, r_bl = right

            left_top, left_bottom = l_tr, l_br
            right_top, right_bottom = r_tl, r_bl
            left_mid = (l_tr + l_br) / 2
            right_mid = (r_tl + r_bl) / 2

            # Distances in px
            d_top_px = np.linalg.norm(right_top - left_top)
            d_mid_px = np.linalg.norm(right_mid - left_mid)
            d_bottom_px = np.linalg.norm(right_bottom - left_bottom)

            # Raw measurement in mm using optical scaling
            top_raw = d_top_px * avg_top_mm_per_px
            mid_raw = d_mid_px * avg_top_mm_per_px
            bottom_raw = d_bottom_px * avg_top_mm_per_px

            # Apply final calibration factor
            top_mm = top_raw * GAP_CALIB_FACTOR
            mid_mm = mid_raw * GAP_CALIB_FACTOR
            bottom_mm = bottom_raw * GAP_CALIB_FACTOR

            avg_mm = (top_mm + mid_mm + bottom_mm) / 3.0

            # Display high precision values
            y = 40
            text = [
                f"mm/px (top-based avg): {avg_top_mm_per_px:.6f}",
                f"Calibration factor: {GAP_CALIB_FACTOR:.6f}",
                f"Top:   {d_top_px:.3f}px | {top_mm:.6f}mm",
                f"Mid:   {d_mid_px:.3f}px | {mid_mm:.6f}mm",
                f"Bottom:{d_bottom_px:.3f}px | {bottom_mm:.6f}mm",
                f"FINAL GAP: {avg_mm:.6f} mm"
            ]

            for line in text:
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                y += 25

            print("\n===== FRAME RESULT =====")
            for l in text: print(l)
            print("========================\n")

        cv2.imshow("Gap Measurement", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
