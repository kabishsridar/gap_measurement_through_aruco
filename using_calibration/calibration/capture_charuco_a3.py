"""
Charuco capture for A3 board (20 cols x 14 rows, 20mm squares, 15mm markers).
Uses ArUco DICT_4X4_250. Saves images on SPACE/'s', quits on q/ESC.
Shows live detection overlay (markers + Charuco corners).
"""

import argparse
import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2

DICT_TYPE = cv2.aruco.DICT_4X4_250
CHARUCO_ROWS = 14
CHARUCO_COLS = 20
SQUARE_SIZE_MM = 20.0
MARKER_SIZE_MM = 15.0

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def next_count(output_dir: str, prefix: str) -> int:
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


def start_picamera(width: int | None, height: int | None) -> Picamera2:
    cam = Picamera2()
    main_cfg = {"format": "RGB888"}
    if width and height:
        main_cfg["size"] = (width, height)
    config = cam.create_video_configuration(main=main_cfg)
    cam.configure(config)
    cam.start()
    time.sleep(0.2)  # small settle
    return cam


def main():
    parser = argparse.ArgumentParser(
        description="Capture Charuco images for calibration (A3, 20x14, 4X4_250)"
    )
    parser.add_argument("-d", "--dir", default="charuco_images_a3",
                        help="directory to save images")
    parser.add_argument("-p", "--prefix", default="charuco",
                        help="image filename prefix")
    parser.add_argument("--width", type=int, default=1280,
                        help="capture width (default: 640)")
    parser.add_argument("--height", type=int, default=720,
                        help="capture height (default: 480)")

    args = parser.parse_args()

    ensure_dir(args.dir)

    cam = start_picamera(args.width, args.height)
    w = args.width or 0
    h = args.height or 0
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        SQUARE_SIZE_MM,
        MARKER_SIZE_MM,
        dictionary
    )
    # Use improved detector parameters for better detection
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 23
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 30
    detector_params.cornerRefinementMinAccuracy = 0.1
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    print(f"Camera ready at {w} x {h}, dict=4X4_250 (picamera2)")
    print(f"Board: {CHARUCO_COLS} cols x {CHARUCO_ROWS} rows, squares {SQUARE_SIZE_MM}mm, markers {MARKER_SIZE_MM}mm")
    print("Controls: SPACE/'s' to save, q/ESC to quit")

    count = next_count(args.dir, args.prefix)
    while True:
        frame = cam.capture_array()
        if frame is None:
            print("WARN: failed to read frame, retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        charuco_corners = None
        charuco_ids = None
        
        if ids is not None and len(ids) > 0:
            # Draw detected ArUco markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Interpolate Charuco corners from detected markers
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )
            
            # Draw Charuco corners if found
            if charuco_corners is not None and len(charuco_corners) > 0:
                # Draw corners using OpenCV function (red)
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 0, 255))
                # Add green circles for better visibility
                for pt in charuco_corners:
                    cx, cy = int(pt[0][0]), int(pt[0][1])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        overlay = f"{args.prefix} #{count} | {w}x{h} | markers {len(ids) if ids is not None else 0} | corners {len(charuco_corners) if charuco_corners is not None else 0}"
        display = frame.copy()
        cv2.putText(display, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Charuco Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord(" "), ord("s")):
            count += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{args.prefix}_{count:03d}_{ts}.jpg"
            path = os.path.join(args.dir, fname)
            if cv2.imwrite(path, frame):
                print(f"Saved {fname}")
            else:
                print(f"ERROR: Could not save {fname}")

    cam.stop()
    cv2.destroyAllWindows()
    print(f"Finished. Saved {count} images to {args.dir}")


if __name__ == "__main__":
    main()
