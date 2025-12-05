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

cv2.namedWindow("Aruco True Depth", cv2.WINDOW_AUTOSIZE)

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

                t = tvec.reshape(3)          # X, Y, Z of marker center in camera coords (cm)
                R, _ = cv2.Rodrigues(rvec)   # rotation matrix
                normal = R[:, 2]            # marker plane normal in camera coords (unit vector)

                # Guard: if marker normal is almost perpendicular to camera Z, skip (degenerate)
                if abs(normal[2]) < 1e-3:
                    continue

                # ---------- TRUE DEPTH (perpendicular to camera) ----------
                # Distance along marker normal:
                plane_cm = float(abs(np.dot(normal, t)))       # along marker normal
                # Corrected depth along camera Z-axis (optical axis):
                true_depth_cm = plane_cm / abs(normal[2])

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

                # ---------- DISPLAY ABOVE MARKER ----------
                pts_int = pts2d.astype(int)
                TL, TR = pts_int[0], pts_int[1]

                cx = int((TL[0] + TR[0]) / 2)
                top_y = int(min(TL[1], TR[1]))
                line_h = 24

                # bottom line just above marker
                y2 = top_y - 8
                y1 = y2 - line_h

                text1 = f"Depth(cm): {smoothed_depth_cm:.2f}"
                text2 = f"Depth(px): {smoothed_depth_px:.1f}"

                cv2.putText(
                    frame, text1, (cx - 140, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, text2, (cx - 140, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA
                )

        cv2.imshow("Aruco True Depth", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
