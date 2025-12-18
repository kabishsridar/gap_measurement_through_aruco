# ============================
# Chessboard Image Capture
# Raspberry Pi + PiCamera2
# ============================

import cv2 as cv
import os
import time
from picamera2 import Picamera2

# ----------------------------
# USER SETTINGS
# ----------------------------

# IMPORTANT:
# Inner corners count (NOT squares!)
# Example:
# 10x7 squares  -> (9,6)
# 9x6 squares   -> (8,5)
CHESSBOARD_SIZE = (9, 6)

SAVE_DIR = "images_rpi"
MIN_IMAGES = 20   # Recommended 20–30 images

# ----------------------------
# SETUP CAMERA
# ----------------------------

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720)},
    lores={"size": (640, 480)},
    display="main"
)
picam2.configure(config)
picam2.start()

time.sleep(0.5)

cv.namedWindow("Frame", cv.WINDOW_NORMAL)

# ----------------------------
# DIRECTORY CHECK
# ----------------------------

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f'Created directory: {SAVE_DIR}')
else:
    print(f'Directory exists: {SAVE_DIR}')

# ----------------------------
# TERMINATION CRITERIA
# ----------------------------

criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ----------------------------
# CHESSBOARD DETECTION FUNCTION
# ----------------------------

def detect_chessboard(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    flags = (
        cv.CALIB_CB_ADAPTIVE_THRESH +
        cv.CALIB_CB_NORMALIZE_IMAGE +
        cv.CALIB_CB_FAST_CHECK
    )

    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)

    if ret:
        corners = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        cv.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)

    return frame, ret

# ----------------------------
# MAIN LOOP
# ----------------------------

img_count = 0

print("\nControls:")
print("  s → save image (only if detected)")
print("  q → quit\n")

while True:
    frame = picam2.capture_array("main")
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    display = frame.copy()

    display, found = detect_chessboard(display)

    status = "DETECTED" if found else "NOT DETECTED"
    color = (0, 255, 0) if found else (0, 0, 255)

    cv.putText(
        display,
        f"Chessboard: {status}",
        (30, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv.putText(
        display,
        f"Saved Images: {img_count}",
        (30, 80),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv.imshow("Frame", display)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s') and found:
        filename = f"{SAVE_DIR}/image_{img_count:03d}.png"
        cv.imwrite(filename, frame)
        print(f"Saved: {filename}")
        img_count += 1

picam2.stop()
cv.destroyAllWindows()

print(f"\nTotal images saved: {img_count}")

if img_count < MIN_IMAGES:
    print("⚠️ Warning: Capture at least 20 images for accurate calibration.")
else:
    print("✅ Enough images captured for calibration.")
