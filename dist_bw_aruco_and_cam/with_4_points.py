import cv2
import numpy as np
from picamera2 import Picamera2

# ---------- SETTINGS ----------
MARKER_LENGTH = 10.0      # cm (side length of ArUco)
SMOOTHING_ALPHA = 0.25    # 0..1, lower = smoother

# ---------- LOAD CAMERA CALIBRATION ----------
calib = np.load("/home/kabish/gap_measuement_dec/calibrated_data/MultiMatrix_rpi.npz")
camera_matrix = calib["camMatrix"]
dist_coeffs = calib["distCoef"]

# ---------- ARUCO SETUP ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# 3D marker model (cm) – TL, TR, BR, BL
half = MARKER_LENGTH / 2.0
object_points = np.array([
    [-half,  half, 0.0],  # top-left
    [ half,  half, 0.0],  # top-right
    [ half, -half, 0.0],  # bottom-right
    [-half, -half, 0.0],  # bottom-left
], dtype=np.float32)

# Use IPPE if available, otherwise fall back
SOLVE_FLAG = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE

# ---------- CAMERA SETUP ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

cv2.namedWindow("Perfect Distance - ArUco", cv2.WINDOW_AUTOSIZE)

smoothed_plane_cm = None

try:
    while True:
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            continue

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_corners, marker_id in zip(corners, ids):
                pts2d = marker_corners[0].astype(np.float32)  # (4,2)

                # ---- Pose estimation ----
                success, rvec, tvec = cv2.solvePnP(
                    object_points, pts2d, camera_matrix, dist_coeffs,
                    flags=SOLVE_FLAG
                )
                if not success:
                    continue

                # t: marker center in camera coordinates (cm)
                t = tvec.reshape(3)
                R, _ = cv2.Rodrigues(rvec)

                # --------- PERFECT DISTANCE (HEIGHT) ----------
                # normal of marker plane in camera frame (marker Z-axis)
                normal = R[:, 2]
                # perpendicular distance from camera to marker plane, in cm
                plane_cm = float(abs(np.dot(normal, t)))

                # pixel width of marker (for cm <-> px scale)
                pixel_width = float(np.linalg.norm(pts2d[0] - pts2d[1]))  # TL->TR in pixels
                if pixel_width <= 0:
                    continue

                cm_per_pixel = MARKER_LENGTH / pixel_width
                plane_px = plane_cm / cm_per_pixel

                # --------- SMOOTHING ----------
                if smoothed_plane_cm is None:
                    smoothed_plane_cm = plane_cm
                else:
                    smoothed_plane_cm = (
                        SMOOTHING_ALPHA * plane_cm
                        + (1.0 - SMOOTHING_ALPHA) * smoothed_plane_cm
                    )
                smoothed_plane_px = smoothed_plane_cm / cm_per_pixel

                # Draw axes for visual debugging (optional)
                try:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH)
                except:
                    pass

                # --------- DISPLAY ABOVE MARKER ----------
                pts_int = pts2d.astype(int)
                TL, TR = pts_int[0], pts_int[1]

                cx = int((TL[0] + TR[0]) / 2)
                top_y = int(min(TL[1], TR[1]))
                line_h = 22

                # bottom line just above marker
                y2 = top_y - 8
                y1 = y2 - line_h

                # Height in cm (smoothed) and px (smoothed)
                text1 = f"Height(cm): {smoothed_plane_cm:.2f}"
                text2 = f"Height(px): {smoothed_plane_px:.1f}"

                cv2.putText(
                    frame, text1, (cx - 140, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, text2, (cx - 140, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                )

        cv2.imshow("Perfect Distance - ArUco", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
