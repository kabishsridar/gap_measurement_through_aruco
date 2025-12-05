import cv2
import numpy as np
from picamera2 import Picamera2

# ---------- SETTINGS ----------
MARKER_LENGTH = 10.0      # cm (side length of ArUco marker)
SMOOTHING_ALPHA = 0.25    # 0..1, lower = smoother

# ---------- LOAD CAMERA CALIBRATION ----------
calib = np.load("/home/kabish/gap_measuement_dec/calibrated_data/MultiMatrix_rpi.npz")
camera_matrix = calib["camMatrix"]
dist_coeffs = calib["distCoef"]

# ---------- ARUCO SETUP ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# 3D marker model (cm) – TL, TR, BR, BL in marker coordinates
half = MARKER_LENGTH / 2.0
object_points = np.array([
    [-half,  half, 0.0],  # top-left
    [ half,  half, 0.0],  # top-right
    [ half, -half, 0.0],  # bottom-right
    [-half, -half, 0.0],  # bottom-left
], dtype=np.float32)

# Use IPPE if available, else fallback
if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
    SOLVE_FLAG = cv2.SOLVEPNP_IPPE_SQUARE
else:
    SOLVE_FLAG = cv2.SOLVEPNP_ITERATIVE

# ---------- CAMERA SETUP ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

cv2.namedWindow("Aruco Z-Depth", cv2.WINDOW_AUTOSIZE)

smoothed_depth_cm = None

try:
    while True:
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            continue

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_corners, marker_id in zip(corners, ids):
                pts2d = marker_corners[0].astype(np.float32)  # shape (4,2)

                # ---- Pose estimation ----
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    pts2d,
                    camera_matrix,
                    dist_coeffs,
                    flags=SOLVE_FLAG
                )
                if not success:
                    continue

                # t = (X, Y, Z) in camera coordinates, units are cm
                t = tvec.reshape(3)

                # ---------- TRUE DEPTH ALONG CAMERA Z-AXIS ----------
                # Per your idea: imagine a line along the camera Z-axis and
                # drop a perpendicular from the marker center to that axis.
                # The intersection has Z coordinate = t[2].
                true_depth_cm = float(abs(t[2]))

                # Marker pixel width (TL -> TR) for cm/px conversion
                pixel_width = float(np.linalg.norm(pts2d[0] - pts2d[1]))
                if pixel_width <= 0:
                    continue

                cm_per_pixel = MARKER_LENGTH / pixel_width
                true_depth_px = true_depth_cm / cm_per_pixel

                # ---------- SMOOTHING ----------
                if smoothed_depth_cm is None:
                    smoothed_depth_cm = true_depth_cm
                else:
                    smoothed_depth_cm = (
                        SMOOTHING_ALPHA * true_depth_cm
                        + (1.0 - SMOOTHING_ALPHA) * smoothed_depth_cm
                    )
                smoothed_depth_px = smoothed_depth_cm / cm_per_pixel

                # Draw axes for debugging (optional)
                try:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH)
                except cv2.error:
                    pass

                # ---------- DISPLAY + VISUALIZATION ----------
                pts_int = pts2d.astype(int)
                TL, TR, BR, BL = pts_int

                # Center of top edge (approx center in x, y for text/line)
                cx = int((TL[0] + TR[0]) / 2)
                cy = int((TL[1] + BL[1]) / 2)

                # Optional: draw horizontal line along image x-axis at center
                cv2.line(
                    frame,
                    (0, cy),
                    (frame.shape[1] - 1, cy),
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )
                # Mark center point
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

                # Position text slightly above marker
                top_y = int(min(TL[1], TR[1]))
                line_h = 24
                y2 = top_y - 8
                y1 = y2 - line_h

                text1 = f"Depth Z (cm): {smoothed_depth_cm:.2f}"
                text2 = f"Depth Z (px): {smoothed_depth_px:.1f}"

                cv2.putText(
                    frame, text1, (cx - 160, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, text2, (cx - 160, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA
                )

        cv2.imshow("Aruco Z-Depth", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
