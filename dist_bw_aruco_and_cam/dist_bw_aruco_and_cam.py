import cv2
import numpy as np
from picamera2 import Picamera2

# --- SETTINGS ---
MARKER_LENGTH = 10.0  # cm, physical ArUco marker size

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

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # Estimate pose of each detected marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, camera_matrix, dist_coeffs
            )

            for rvec, tvec, corner in zip(rvecs, tvecs, corners):
                # Compute distance (Euclidean norm)
                distance = np.linalg.norm(tvec[0])

                # Draw marker and axis
                cv2.aruco.drawDetectedMarkers(frame, [corner])
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH)

                # Display distance
                cv2.putText(frame, f"Distance: {distance:.2f} cm",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show video
        cv2.imshow("Aruco Distance - Picamera2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
