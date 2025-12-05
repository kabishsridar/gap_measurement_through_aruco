import cv2 as cv
import numpy as np
from picamera2 import Picamera2

length = 10.0 # marker len in cm

calib = np.load("/home/kabish/gap_measuement_dec/calibrated_data/MultiMatrix_rpi.npz")
camera_matrix = calib["camMatrix"]
dist_coeffs = calib["distCoef"]

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

half = length / 2.0
object_points = np.array([
    [-half,  half, 0.0],  # top-left
    [ half,  half, 0.0],  # top-right
    [ half, -half, 0.0],  # bottom-right
    [-half, -half, 0.0],  # bottom-left
], dtype=np.float32)

if hasattr(cv, "SOLVEPNP_IPPE_SQUARE"):
    SOLVE_FLAG = cv.SOLVEPNP_IPPE_SQUARE
else:
    SOLVE_FLAG = cv.SOLVEPNP_ITERATIVE

cv.namedWindow('straight line z axis', cv.WINDOW_AUTOSIZE)

while True:
    frame = picam2.capture_array()
    if frame is None or frame.size == 0:
        continue

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and len(corners) > 0:
        cv.aruco.drawDetectMarkers(frame, corners, ids)
        for marker_corners, marker_id in zip(corners, ids):
                pts2d = marker_corners[0].astype(np.float32)  # shape (4,2)

                # ---- Pose estimation ----
                success, rvec, tvec = cv.solvePnP(
                    object_points,
                    pts2d,
                    camera_matrix,
                    dist_coeffs,
                    flags=SOLVE_FLAG
                )
                if not success:
                    continue

                t = tvec.reshape(3)
                true_depth_cm = float(abs(t[2]))
