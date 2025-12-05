import cv2
import numpy as np
from picamera2 import Picamera2

def main():
    # ---------- ArUco dictionary & detector (OpenCV 4.7+ style) ----------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    # ---------- Picamera2 setup ----------
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(preview_config)
    picam2.start()

    print("Press 'q' to quit.")

    while True:
        # Capture frame from Picamera2
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Detect ArUco markers
        corners_list, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # Draw marker borders and IDs
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

            for marker_corners, marker_id in zip(corners_list, ids):
                # corners: shape (1, 4, 2) -> (4, 2)
                corners = marker_corners.reshape((4, 2))

                # Corner order: 0=TL, 1=TR, 2=BR, 3=BL
                tl, tr, br, bl = corners

                # ---------- Compute side lengths in pixels ----------
                side_top = np.linalg.norm(tr - tl)       # top edge
                side_right = np.linalg.norm(br - tr)     # right edge
                side_bottom = np.linalg.norm(bl - br)    # bottom edge
                side_left = np.linalg.norm(tl - bl)      # left edge

                # Optional: print in terminal
                print(f"Marker ID {int(marker_id)}")
                print(f"  Top    : {side_top:.2f}px")
                print(f"  Right  : {side_right:.2f}px")
                print(f"  Bottom : {side_bottom:.2f}px")
                print(f"  Left   : {side_left:.2f}px")
                print("-" * 40)

                # ---------- Midpoints of each side ----------
                mid_top = ((tl + tr) / 2).astype(int)
                mid_right = ((tr + br) / 2).astype(int)
                mid_bottom = ((br + bl) / 2).astype(int)
                mid_left = ((bl + tl) / 2).astype(int)

                # ---------- Draw lengths at each side position ----------
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 2

                # Top side text (slightly above the edge)
                cv2.putText(
                    frame,
                    f"{side_top:.1f}px",
                    (mid_top[0] - 30, mid_top[1] - 10),
                    font, scale, (0, 255, 0), thickness, cv2.LINE_AA
                )

                # Right side text (slightly to the right)
                cv2.putText(
                    frame,
                    f"{side_right:.1f}px",
                    (mid_right[0] + 10, mid_right[1] + 5),
                    font, scale, (0, 255, 0), thickness, cv2.LINE_AA
                )

                # Bottom side text (slightly below the edge)
                cv2.putText(
                    frame,
                    f"{side_bottom:.1f}px",
                    (mid_bottom[0] - 30, mid_bottom[1] + 20),
                    font, scale, (0, 255, 0), thickness, cv2.LINE_AA
                )

                # Left side text (slightly to the left)
                cv2.putText(
                    frame,
                    f"{side_left:.1f}px",
                    (mid_left[0] - 80, mid_left[1] + 5),
                    font, scale, (0, 255, 0), thickness, cv2.LINE_AA
                )

                # ---------- (Optional) draw and label corners ----------
                for i, c in enumerate(corners):
                    cx, cy = int(c[0]), int(c[1])
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(
                        frame, str(i),
                        (cx + 5, cy - 5),
                        font, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                    )

        # Show the result
        cv2.imshow("ArUco Side Lengths (px) on Marker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
