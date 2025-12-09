#!/usr/bin/env python3
"""
Live ChArUco detection with PiCamera2

Board:
- SquaresX (columns): 20
- SquaresY (rows):    14
- Square size:        40 mm
- Marker size:        30 mm
Dictionary:           DICT_4X4_250

Press 's' or SPACE to save image, 'q' or ESC to quit.
"""

import cv2
import numpy as np
import os
from datetime import datetime
from picamera2 import Picamera2

# ----------------- BOARD CONFIG -----------------
# Number of chessboard squares (NOT markers)
SQUARES_X = 20          # columns
SQUARES_Y = 14          # rows

# Physical sizes (in mm – used for calibration later)
SQUARE_SIZE_MM = 40.0   # square side length
MARKER_SIZE_MM = 30.0   # aruco marker side length

# Use the same dictionary you printed the board with
DICT_TYPE = cv2.aruco.DICT_4X4_250


def create_charuco_board():
    """
    Create a CharucoBoard object that matches your printed board.
    Units for squareLength and markerLength can be arbitrary, but
    we pass them in mm so they directly match the real board.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        size=(SQUARES_X, SQUARES_Y),
        squareLength=SQUARE_SIZE_MM,
        markerLength=MARKER_SIZE_MM,
        dictionary=aruco_dict
    )
    return board, aruco_dict


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def next_count(output_dir: str, prefix: str) -> int:
    """Find the next image number based on existing files."""
    existing = [
        f for f in os.listdir(output_dir)
        if f.startswith(prefix) and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    nums = []
    for fname in existing:
        stem = fname.split(".")[0]
        tail = stem.replace(prefix, "").strip("_")
        if tail.isdigit():
            nums.append(int(tail))
    return max(nums) if nums else 0


def main():
    board, aruco_dict = create_charuco_board()

    # Detector parameters (you can tweak if needed)
    detector_params = cv2.aruco.DetectorParameters()
    # Example tweaks if detection is weak:
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 30
    detector_params.cornerRefinementMinAccuracy = 0.1
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    # Setup output directory
    output_dir = "charuco_images_a3"
    prefix = "charuco_a3"
    ensure_dir(output_dir)
    count = next_count(output_dir, prefix)

    # --------------- PiCamera2 SETUP ---------------
    picam2 = Picamera2()

    # Configure preview stream (adjust size if you want)
    preview_config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(preview_config)
    picam2.start()

    cv2.namedWindow("ChArUco Detection", cv2.WINDOW_NORMAL)

    print("Starting ChArUco detection. Press 's' or SPACE to save, 'q' or ESC to exit.")

    try:
        while True:
            # Capture frame as RGB
            frame_rgb = picam2.capture_array()

            # For OpenCV display, convert to BGR
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            # --------- STEP 1: Detect ArUco markers ---------
            corners, ids, rejected = detector.detectMarkers(gray)

            # Draw detected markers (like in the video)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

                # --------- STEP 2: Interpolate ChArUco corners ---------
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=board
                )

                # Draw ChArUco corners (sub-pixel chessboard corners)
                if retval is not None and retval > 0 and charuco_corners is not None:
                    # Draw corners using OpenCV function (red)
                    cv2.aruco.drawDetectedCornersCharuco(
                        frame_bgr,
                        charuco_corners,
                        charuco_ids,
                        (0, 0, 255)  # Red color
                    )
                    
                    # Add prominent green circles for better visibility
                    for pt in charuco_corners:
                        cx, cy = int(pt[0][0]), int(pt[0][1])
                        # Draw outer circle (thicker, more visible)
                        cv2.circle(frame_bgr, (cx, cy), 6, (0, 255, 0), 2)
                        # Draw inner filled circle
                        cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 0), -1)
                        # Draw center point
                        cv2.circle(frame_bgr, (cx, cy), 1, (255, 255, 255), -1)

                    # Display corner count with background for better visibility
                    text = f"ChArUco corners: {len(charuco_corners)}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_x, text_y = 10, 35
                    # Draw background rectangle
                    cv2.rectangle(frame_bgr, 
                                (text_x - 5, text_y - text_size[1] - 5),
                                (text_x + text_size[0] + 5, text_y + 5),
                                (0, 0, 0), -1)
                    # Draw text
                    cv2.putText(
                        frame_bgr,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                cv2.putText(
                    frame_bgr,
                    "No markers detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Show result
            cv2.imshow("ChArUco Detection", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s') or key == ord(' '):  # 's' or SPACE to save
                count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{prefix}_{count:03d}_{ts}.jpg"
                path = os.path.join(output_dir, fname)
                if cv2.imwrite(path, frame_bgr):
                    print(f"Saved {fname}")
                else:
                    print(f"ERROR: Could not save {fname}")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"Finished. Saved {count} images to {output_dir}")


if __name__ == "__main__":
    main()
