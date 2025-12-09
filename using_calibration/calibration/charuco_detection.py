#importing all necessary modules
import os
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

# function to check whether the path exists, if not, it creates the directory
def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
# function that counts the number of images in the directory specified
def next_count(output_dir: str, prefix: str) -> int:
    existing = [
        f for f in os.listdir(output_dir)
        if f.startswith(prefix) and f.lower().endswith((".png", ".jpg", ".jpeg")) # true if this file is in extension png, jpg, jpeg only
    ] # stores all the images in this list
    nums = [] # list that stores the number of images in the directory
    for fname in existing:
        stem = fname.split(".")[0] # splits the file name and extension seperately (after the .)
        tail = stem.replace(prefix, "").strip("_") # removes the prefix and underscore
        if tail.isdigit(): # if the tail is a digit, append it to the list
            nums.append(int(tail))
    return max(nums) if nums else 0 # returns the number of images

# --- 1. Define Board Parameters ---
# Board: 9 columns x 12 rows of squares
# initiating the values to the variables
SQUARES_HORIZONTALLY = 9   # columns of squares (9 cols of markers)
SQUARES_VERTICALLY = 12      # rows of squares (12 rows of markers)

# Physical sizes (in meters) — taken from the printed footer (Square 18 mm, Marker 13 mm)
SQUARE_LENGTH = 0.018       # 18 mm in meters
MARKER_LENGTH = 0.0130      # 13 mm in meters

# Define the ArUco dictionary and ChArUco board
ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) # sets the aruco dictionary
board = cv2.aruco.CharucoBoard((SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), 
                               SQUARE_LENGTH, 
                               MARKER_LENGTH, 
                               ARUCO_DICTIONARY)

# Saving configuration
SAVE_DIR = "charuco_images_calibration"
SAVE_PREFIX = "charuco"

# Create the ArUco Detector with improved parameters for better detection
# sets the parameters for corner refinement
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3 # thresholding window size
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.cornerRefinementMaxIterations = 30
aruco_params.cornerRefinementMinAccuracy = 0.1
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICTIONARY, aruco_params) # creates the aruco detector object

# --- 2. Initialize Picamera2 ---
def run_live_charuco_detection(): # function to run the live detection
    print("Initializing Picamera2...")
    picam2 = Picamera2() # creating the picam2 object
    
    # Configure the camera stream (use a suitable resolution for processing)
    # The 'main' stream is used for processing.
    camera_config = picam2.create_video_configuration(main={"size": (1280, 720)}) # initiating variable to configuring to 1280 , 720p
    picam2.configure(camera_config) # configuring the camera
    
    # Optional: Set focus mode for better image quality if using autofocus lenses
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) # auto focus mode
    
    picam2.start()
    print("Camera started. Press 'q' to exit.")

    # Prepare save directory and filename counter
    ensure_dir(SAVE_DIR)
    save_count = next_count(SAVE_DIR, SAVE_PREFIX)


    try:
        while True:
            # 3. Capture Frame from Picamera2
            # Use 'main' stream and convert to numpy array in BGR format
            # Picamera2 captures as RGB, so we explicitly convert to BGR for OpenCV display
            frame_rgb = picam2.capture_array("main") # captures the frame from the camera
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # converts the frame to bgr format

            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converts the frame to gray color for detection

            # SOLUTION: Image preprocessing to improve detection
            # Apply histogram equalization for better contrast
            gray_enhanced = cv2.equalizeHist(gray) # histogram is applied
            # Apply slight Gaussian blur to reduce noise
            gray_enhanced = cv2.GaussianBlur(gray_enhanced, (3, 3), 0) # applied guassian blur

            # 4. Detect ArUco Markers first
            markerCorners, markerIds, _ = aruco_detector.detectMarkers(gray_enhanced)
            
            # 5. Interpolate ChArUco corners from detected markers
            charucoCorners = None
            charucoIds = None
            if markerIds is not None and len(markerIds) > 0:
                # Draw detected ArUco markers
                cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds) # draws the markers
                
                # Analyze detection before interpolation
                marker_ids_list = markerIds.flatten().tolist() # creates a marker ids list
                try:
                    expected_ids = board.getIds().flatten().tolist()
                    detected_set = set(marker_ids_list)
                    expected_set = set(expected_ids)
                    matching_ids = detected_set & expected_set # creates a detected set and expected set and checks how much ids match
                    missing_ids = expected_set - detected_set
                    extra_ids = detected_set - expected_set
                    matching_count = len(matching_ids)
                    
                except Exception as e:
                    print(f"  Error analyzing marker IDs: {e}") # returns the error
                    matching_count = 0
                
                # Interpolate ChArUco corners from detected markers
                # Multiple solutions to overcome interpolation errors:
                
                # SOLUTION 1: Filter markers to only use those matching the board
                retval = None
                charucoCorners = None
                charucoIds = None
                solution_used = "None"
                
                try:
                    expected_ids = board.getIds().flatten().tolist()
                    expected_set = set(expected_ids)
                    marker_ids_flat = markerIds.flatten()
                    
                    # Filter markers to only include those with matching IDs
                    filtered_corners = []
                    filtered_ids_list = []
                    for i in range(len(markerCorners)):
                        marker_id = marker_ids_flat[i]
                        if marker_id in expected_set:
                            filtered_corners.append(markerCorners[i])
                            filtered_ids_list.append(int(marker_id))
                    
                    if len(filtered_ids_list) >= 4:  # Need at least 4 markers
                        filtered_ids_array = np.array(filtered_ids_list, dtype=np.int32).reshape(-1, 1)
                        
                        # Try interpolation with filtered markers (only matching IDs)
                        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                            markerCorners=filtered_corners,
                            markerIds=filtered_ids_array,
                            image=gray,
                            board=board
                        )
                        if charucoCorners is not None and len(charucoCorners) > 0:
                            solution_used = "Filtered matching markers"
                except Exception as e:
                    pass

                # SOLUTION 2: Try using enhanced image
                if charucoCorners is None or len(charucoCorners) == 0:
                    try:
                        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                            markerCorners=filtered_corners if len(filtered_ids_list) >= 4 else markerCorners,
                            markerIds=filtered_ids_array if len(filtered_ids_list) >= 4 else markerIds,
                            image=gray_enhanced,  # Use enhanced image
                            board=board
                        )
                        if charucoCorners is not None and len(charucoCorners) > 0:
                            solution_used = "Enhanced image preprocessing"
                    except Exception as e:
                        pass

                # SOLUTION 3: Try using CharucoDetector.detectBoard() with original image
                if charucoCorners is None or len(charucoCorners) == 0:
                    try:
                        charuco_detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)
                        charucoCorners, charucoIds, markerCorners2, markerIds2 = charuco_detector.detectBoard(gray)
                        
                        if charucoCorners is not None and len(charucoCorners) > 0:
                            solution_used = "CharucoDetector.detectBoard() (original)"
                            retval = len(charucoCorners)
                    except Exception as e:
                        pass

                # SOLUTION 4: Try using CharucoDetector.detectBoard() with enhanced image
                if charucoCorners is None or len(charucoCorners) == 0:
                    try:
                        charuco_detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)
                        charucoCorners, charucoIds, markerCorners2, markerIds2 = charuco_detector.detectBoard(gray_enhanced)
                        
                        if charucoCorners is not None and len(charucoCorners) > 0:
                            solution_used = "CharucoDetector.detectBoard() (enhanced)"
                            retval = len(charucoCorners)
                    except Exception as e:
                        pass

                # SOLUTION 5: Try with slightly different board size parameters (common issue)
                # Sometimes printed boards have slightly different sizes due to printer scaling
                if charucoCorners is None or len(charucoCorners) == 0:
                    try:
                        # Try with 5% larger sizes (common printer scaling issue)
                        board_larger = cv2.aruco.CharucoBoard(
                            (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
                            SQUARE_LENGTH * 1.05,  # 5% larger
                            MARKER_LENGTH * 1.05,
                            ARUCO_DICTIONARY
                        )
                        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                            markerCorners=filtered_corners if len(filtered_ids_list) >= 4 else markerCorners,
                            markerIds=filtered_ids_array if len(filtered_ids_list) >= 4 else markerIds,
                            image=gray,
                            board=board_larger
                        )
                        if charucoCorners is not None and len(charucoCorners) > 0:
                            solution_used = "Board with 5% larger sizes"
                    except Exception as e:
                        pass

                # SOLUTION 6: Try with slightly smaller board size parameters
                if charucoCorners is None or len(charucoCorners) == 0:
                    try:
                        # Try with 5% smaller sizes
                        board_smaller = cv2.aruco.CharucoBoard(
                            (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
                            SQUARE_LENGTH * 0.95,  # 5% smaller
                            MARKER_LENGTH * 0.95,
                            ARUCO_DICTIONARY
                        )
                        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                            markerCorners=filtered_corners if len(filtered_ids_list) >= 4 else markerCorners,
                            markerIds=filtered_ids_array if len(filtered_ids_list) >= 4 else markerIds,
                            image=gray,
                            board=board_smaller
                        )
                        if charucoCorners is not None and len(charucoCorners) > 0:
                            solution_used = "Board with 5% smaller sizes"
                    except Exception as e:
                        pass

                # SOLUTION 7: Fallback to original method with all markers
                if charucoCorners is None or len(charucoCorners) == 0:
                    try:
                        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                            markerCorners=markerCorners,
                            markerIds=markerIds,
                            image=gray,
                            board=board
                        )
                        if charucoCorners is not None and len(charucoCorners) > 0:
                            solution_used = "Original method (all markers)"
                    except Exception as e:
                        pass
                # Analyze interpolation result (compact)
            
            # 6. Process and Draw Results
            frame_drawn = frame.copy()
            if charucoIds is not None and len(charucoIds) > 0:
                # Draw the detected ChArUco corners on the frame
                frame_drawn = cv2.aruco.drawDetectedCornersCharuco(
                    frame_drawn, charucoCorners, charucoIds, (0, 255, 0))

                # Display the number of corners and markers found
                markers_count = len(markerIds) if markerIds is not None else 0
                cv2.putText(
                    frame_drawn,
                    f"Markers: {markers_count} | Corners: {len(charucoIds)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                markers_count = len(markerIds) if markerIds is not None else 0
                if markers_count > 0:
                    cv2.putText(
                        frame_drawn,
                        f"Markers: {markers_count} | ChArUco corners not found",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame_drawn,
                        "ChArUco Not Detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # 7. Display the output
            cv2.imshow("ChArUco Detector (Picamera2)", frame_drawn)

            # Handle key inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('s'), ord('S')):
                # Save current frame for calibration
                save_count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{SAVE_PREFIX}_{save_count:03d}_{ts}.jpg"
                path = os.path.join(SAVE_DIR, fname)
                if cv2.imwrite(path, frame):
                    print(f"Saved {fname} to {SAVE_DIR}")
                else:
                    print(f"ERROR: Could not save {fname}")

    finally:
        # 8. Cleanup
        print("Stopping camera and cleaning up.")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_charuco_detection()