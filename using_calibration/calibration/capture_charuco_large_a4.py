"""
Charuco capture for large A4 board (14 cols x 19 rows of squares = 13 cols x 18 rows of markers).
Uses ArUco DICT_4X4_250. Saves images on SPACE/'s', quits on q/ESC.
Shows live detection overlay (markers + Charuco corners).
Board: 15mm squares, 12mm markers. Board size: 210mm x 285mm (fits A4: 210mm x 297mm).
"""

import argparse
import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2

DICT_TYPE = cv2.aruco.DICT_4X4_250
# Board configuration: Detecting 12 markers in 4 cols x 3 rows pattern
# This means 5 cols x 4 rows of squares (gives 4 cols x 3 rows of markers = 12 markers)
# Note: Printed board label "4x6" doesn't match actual visible board
CHARUCO_COLS = 5  # cols of squares (gives 4 cols of markers)
CHARUCO_ROWS = 4  # rows of squares (gives 3 rows of markers)
SQUARE_SIZE_MM = 40.0
MARKER_SIZE_MM = 30.0


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
        description="Capture Charuco images for calibration (Large A4, 14x19, 4X4_250)"
    )
    parser.add_argument("-d", "--dir", default="charuco_images_large_a4",
                        help="directory to save images")
    parser.add_argument("-p", "--prefix", default="charuco_large_a4",
                        help="image filename prefix")
    parser.add_argument("--width", type=int, default=1280,
                        help="capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="capture height (default: 720)")

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
    print(f"Board: {CHARUCO_COLS} cols x {CHARUCO_ROWS} rows of squares")
    print(f"  (This gives {CHARUCO_COLS-1} cols x {CHARUCO_ROWS-1} rows of markers)")
    print(f"  Square size: {SQUARE_SIZE_MM}mm, Marker size: {MARKER_SIZE_MM}mm")
    print(f"  Board size: {CHARUCO_COLS * SQUARE_SIZE_MM}mm x {CHARUCO_ROWS * SQUARE_SIZE_MM}mm (A4: 210mm x 297mm)")
    # Get expected marker IDs from the board
    try:
        expected_ids = board.getIds().flatten().tolist()
        print(f"  Board expects marker IDs: {sorted(expected_ids)}")
    except:
        print(f"  (Could not retrieve expected marker IDs from board)")
    print("Controls: SPACE/'s' to save, q/ESC to quit")

    count = next_count(args.dir, args.prefix)
    debug_frame_count = 0
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
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Debug: print marker IDs occasionally
            debug_frame_count += 1
            if debug_frame_count % 30 == 0:  # Every 30 frames
                marker_ids_list = ids.flatten().tolist() if ids is not None else []
                try:
                    expected_ids = board.getIds().flatten().tolist()
                    print(f"Detected {len(ids)} markers with IDs: {sorted(marker_ids_list)}")
                    print(f"  Board expects marker IDs: {sorted(expected_ids)}")
                    detected_set = set(marker_ids_list)
                    expected_set = set(expected_ids)
                    matching = detected_set & expected_set
                    missing = expected_set - detected_set
                    extra = detected_set - expected_set
                    print(f"  Matching IDs: {sorted(matching)} ({len(matching)}/{len(expected_ids)})")
                    if missing:
                        print(f"  Missing IDs: {sorted(missing)}")
                    if extra:
                        print(f"  Extra IDs (not in board): {sorted(extra)}")
                except:
                    print(f"Detected {len(ids)} markers with IDs: {sorted(marker_ids_list)}")
                print(f"  Board expects: {CHARUCO_COLS} cols x {CHARUCO_ROWS} rows of squares")
                print(f"  (This gives {CHARUCO_COLS-1} cols x {CHARUCO_ROWS-1} rows = {(CHARUCO_COLS-1)*(CHARUCO_ROWS-1)} markers)")
            
            # Interpolate ChArUco corners from detected markers
            # Note: This requires detected marker IDs to match the board's expected marker IDs
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )
            if debug_frame_count % 30 == 0:
                if charuco_corners is not None and len(charuco_corners) > 0:
                    print(f"  ✓ Successfully interpolated {len(charuco_corners)} ChArUco corners!")
                else:
                    print(f"  ✗ interpolateCornersCharuco returned 0 corners (retval={retval})")
                    print(f"  Possible issues:")
                    print(f"    1. Detected marker IDs don't match board's expected IDs")
                    print(f"    2. Not enough markers detected (need at least 4-6 for interpolation)")
                    print(f"    3. Markers not in correct grid pattern")
                    print(f"  Try: Ensure all markers are visible and board matches generation parameters")
            
            # Draw Charuco corners if found
            if charuco_corners is not None and len(charuco_corners) > 0:
                # Draw corners using OpenCV function (red)
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 0, 255))
                # Add green circles for better visibility
                for pt in charuco_corners:
                    cx, cy = int(pt[0][0]), int(pt[0][1])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
                if debug_frame_count % 30 == 0:
                    print(f"  Successfully detected {len(charuco_corners)} ChArUco corners!")

        overlay = f"{args.prefix} #{count} | {w}x{h} | markers {len(ids) if ids is not None else 0} | corners {len(charuco_corners) if charuco_corners is not None else 0}"
        display = frame.copy()
        cv2.putText(display, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Charuco Capture Large A4", display)

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

