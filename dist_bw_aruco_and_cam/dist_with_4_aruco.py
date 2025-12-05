import cv2
import numpy as np
from picamera2 import Picamera2

# --- SETTINGS ---
MARKER_LENGTH = 5.0  # cm
MARKER_IDS = [0, 1, 2, 3]  # IDs of the markers in the square

# --- LOAD CAMERA CALIBRATION ---
calib = np.load("/home/kabish/gap_measuement_dec/calibrated_data/MultiMatrix_rpi.npz")
camera_matrix = calib["camMatrix"]
dist_coeffs = calib["distCoef"]

# --- INIT ARUCO DETECTOR ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# --- INIT PICAMERA2 ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# 3D coordinates of marker corners relative to marker center
s = MARKER_LENGTH / 2
marker_corners_3d = np.array([
    [-s,  s, 0],
    [ s,  s, 0],
    [ s, -s, 0],
    [-s, -s, 0]
], dtype=np.float32)

def best_fit_plane(points):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[2, :]
    return normal, centroid

try:
    while True:
        frame = picam2.capture_array()
        corners, ids, rejected = detector.detectMarkers(frame)

        all_3d_points = []

        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                if marker_id in MARKER_IDS:
                    retval, rvec, tvec = cv2.solvePnP(
                        marker_corners_3d,
                        corner.reshape(-1,2),
                        camera_matrix,
                        dist_coeffs
                    )
                    if retval:
                        # Compute marker center distance
                        distance_marker = np.linalg.norm(tvec)
                        
                        # Draw marker and axes
                        cv2.aruco.drawDetectedMarkers(frame, [corner])
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH)

                        # Display distance below the marker
                        bottom_center = np.mean(corner.reshape(4,2)[[2,3]], axis=0).astype(int)
                        cv2.putText(frame, f"{distance_marker:.2f} cm",
                                    (bottom_center[0]-40, bottom_center[1]+20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0,255,0), 2, cv2.LINE_AA)

                        # Append 3D corners for plane fitting
                        R, _ = cv2.Rodrigues(rvec)
                        for pt in marker_corners_3d:
                            pt_cam = R @ pt.reshape(3,1) + tvec
                            all_3d_points.append(pt_cam.flatten())

        # Compute exact perpendicular distance using all corners
        if len(all_3d_points) >= 3:
            all_3d_points = np.array(all_3d_points)
            normal, centroid = best_fit_plane(all_3d_points)
            distance_plane = np.abs(np.dot(normal, -centroid)) / np.linalg.norm(normal)
            cv2.putText(frame, f"Exact Distance: {distance_plane:.2f} cm",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ArUco Distances", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
