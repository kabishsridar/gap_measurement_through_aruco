import cv2
import numpy as np
from picamera2 import Picamera2

def main():
    # Real known size of the marker per side
    MARKER_SIZE_MM = 100  # mm

    # ---- ArUco Setup ----
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # ---- Picamera2 Setup ----
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()

    print("Press 'q' to exit.")

    while True:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        corners_list, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

            for corners, marker_id in zip(corners_list, ids):
                pts = corners.reshape((4, 2))
                tl, tr, br, bl = pts

                # Pixel lengths per side
                top_px = np.linalg.norm(tr - tl)
                right_px = np.linalg.norm(br - tr)
                bottom_px = np.linalg.norm(bl - br)
                left_px = np.linalg.norm(tl - bl)

                # mm/px ratio per side
                top_ratio = MARKER_SIZE_MM / top_px
                right_ratio = MARKER_SIZE_MM / right_px
                bottom_ratio = MARKER_SIZE_MM / bottom_px
                left_ratio = MARKER_SIZE_MM / left_px

                # Convert each side to mm
                top_mm = top_px * top_ratio
                right_mm = right_px * right_ratio
                bottom_mm = bottom_px * bottom_ratio
                left_mm = left_px * left_ratio

                # Midpoints of edges (where text will be displayed)
                mid_top = ((tl + tr) / 2).astype(int)
                mid_right = ((tr + br) / 2).astype(int)
                mid_bottom = ((br + bl) / 2).astype(int)
                mid_left = ((bl + tl) / 2).astype(int)

                font = cv2.FONT_HERSHEY_SIMPLEX

                # ---- Draw text near each edge ----
                cv2.putText(frame,
                            f"{top_px:.1f}px | {top_mm:.2f}mm | {top_ratio:.4f}mm/px",
                            (mid_top[0] - 160, mid_top[1] - 10),
                            font, 0.45, (0, 255, 0), 2)

                cv2.putText(frame,
                            f"{right_px:.1f}px | {right_mm:.2f}mm | {right_ratio:.4f}mm/px",
                            (mid_right[0] + 10, mid_right[1]),
                            font, 0.45, (0, 255, 0), 2)

                cv2.putText(frame,
                            f"{bottom_px:.1f}px | {bottom_mm:.2f}mm | {bottom_ratio:.4f}mm/px",
                            (mid_bottom[0] - 160, mid_bottom[1] + 20),
                            font, 0.45, (0, 255, 0), 2)

                cv2.putText(frame,
                            f"{left_px:.1f}px | {left_mm:.2f}mm | {left_ratio:.4f}mm/px",
                            (mid_left[0] - 210, mid_left[1]),
                            font, 0.45, (0, 255, 0), 2)

        cv2.imshow("Aruco Size + Calibration (px/mm/ratio)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
