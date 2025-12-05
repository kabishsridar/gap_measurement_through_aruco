import cv2
import numpy as np
from picamera2 import Picamera2

MARKER_SIZE_MM = 100.0  # real side length of each ArUco marker in mm

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
    returns side_lengths_px [top, right, bottom, left]
            mm_per_px_list  [top_ratio, right_ratio, bottom_ratio, left_ratio]
    """
    tl, tr, br, bl = ordered_corners

    top_px = np.linalg.norm(tr - tl)
    right_px = np.linalg.norm(br - tr)
    bottom_px = np.linalg.norm(bl - br)
    left_px = np.linalg.norm(tl - bl)

    top_ratio = MARKER_SIZE_MM / top_px
    right_ratio = MARKER_SIZE_MM / right_px
    bottom_ratio = MARKER_SIZE_MM / bottom_px
    left_ratio = MARKER_SIZE_MM / left_px

    side_lengths_px = [top_px, right_px, bottom_px, left_px]
    mm_per_px_list = [top_ratio, right_ratio, bottom_ratio, left_ratio]

    return side_lengths_px, mm_per_px_list

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

            # label markers
            left_center = np.mean(left_corners, axis=0).astype(int)
            right_center = np.mean(right_corners, axis=0).astype(int)

            cv2.putText(frame, "LEFT", (left_center[0] - 30, left_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "RIGHT", (right_center[0] - 30, right_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # compute side lengths & mm/px for both markers
            left_lengths_px, left_ratios = compute_side_lengths_and_ratios(left_corners)
            right_lengths_px, right_ratios = compute_side_lengths_and_ratios(right_corners)

            all_ratios = left_ratios + right_ratios
            avg_mm_per_px = float(np.mean(all_ratios))

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

            avg_gap_px = (d_top_px + d_mid_px + d_bottom_px) / 3.0
            avg_gap_mm = avg_gap_px * avg_mm_per_px
            gap_without_margins_mm = avg_gap_mm + 10.0  # your +10 mm

            # draw three inner lines
            cv2.line(frame,
                     (int(left_top_point[0]), int(left_top_point[1])),
                     (int(right_top_point[0]), int(right_top_point[1])),
                     (0, 255, 0), 2)

            cv2.line(frame,
                     (int(left_mid_point[0]), int(left_mid_point[1])),
                     (int(right_mid_point[0]), int(right_mid_point[1])),
                     (255, 0, 0), 2)

            cv2.line(frame,
                     (int(left_bottom_point[0]), int(left_bottom_point[1])),
                     (int(right_bottom_point[0]), int(right_bottom_point[1])),
                     (0, 0, 255), 2)

            # small markers to visualize used edge points
            for p in [left_top_point, left_mid_point, left_bottom_point,
                      right_top_point, right_mid_point, right_bottom_point]:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)

            # show text
            info_y = min(left_center[1], right_center[1]) - 60
            if info_y < 20:
                info_y = 20

            cv2.putText(frame,
                        f"avg mm/px: {avg_mm_per_px:.4f}",
                        (20, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame,
                        f"Top gap px/mm    : {d_top_px:.2f}px, {d_top_px*avg_mm_per_px:.2f}mm",
                        (20, info_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Mid gap px/mm    : {d_mid_px:.2f}px, {d_mid_px*avg_mm_per_px:.2f}mm",
                        (20, info_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Bottom gap px/mm : {d_bottom_px:.2f}px, {d_bottom_px*avg_mm_per_px:.2f}mm",
                        (20, info_y + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Avg gap: {avg_gap_mm:.2f} mm",
                        (20, info_y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # console logs
            print("---- FRAME ----")
            print(f"avg mm/px        : {avg_mm_per_px:.6f}")
            print(f"Top gap px/mm    : {d_top_px:.2f}px, {d_top_px*avg_mm_per_px:.2f}mm")
            print(f"Mid gap px/mm    : {d_mid_px:.2f}px, {d_mid_px*avg_mm_per_px:.2f}mm")
            print(f"Bottom gap px/mm : {d_bottom_px:.2f}px, {d_bottom_px*avg_mm_per_px:.2f}mm")
            print(f"Avg gap          : {avg_gap_mm:.2f} mm")
            print("----------------")

        cv2.imshow("Two ArUco Markers - Inner Gap Measurement", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
