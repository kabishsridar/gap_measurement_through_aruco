import cv2
import numpy as np
import os
import glob

# ---------------- USER SETTINGS ----------------
IMAGES_DIR = "charuco_images_a3"      # folder containing captured images
OUTPUT_FILE = "charuco_a3_camera_params.npz"  # output calibration file
DICT_TYPE = cv2.aruco.DICT_4X4_250    # MUST match capture_images.py

# Your A3 Charuco board specifications:
SQUARE_SIZE_MM = 20.0                 # checker (outer black square) size in mm
MARKER_SIZE_MM = 15.0                  # inner ArUco marker size in mm
CHARUCO_COLS = 18                      # squares horizontally (width)
CHARUCO_ROWS = 12                      # squares vertically (height)
# ------------------------------------------------

print("=" * 60)
print("Charuco Camera Calibration (A3 Board)")
print("=" * 60)
print(f"Board: {CHARUCO_COLS}x{CHARUCO_ROWS} squares")
print(f"Square size: {SQUARE_SIZE_MM}mm, Marker size: {MARKER_SIZE_MM}mm")
print(f"Images directory: {IMAGES_DIR}")
print("=" * 60)

# Create Charuco board
dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
board = cv2.aruco.CharucoBoard(
    size=(CHARUCO_COLS, CHARUCO_ROWS),
    squareLength=SQUARE_SIZE_MM,
    markerLength=MARKER_SIZE_MM,
    dictionary=dictionary
)

# Setup detector
detector_params = cv2.aruco.DetectorParameters()
# Use improved parameters for better detection
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 23
detector_params.adaptiveThreshWinSizeStep = 10
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.cornerRefinementWinSize = 5
detector_params.cornerRefinementMaxIterations = 30
detector_params.cornerRefinementMinAccuracy = 0.1
aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

# Find all image files
image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")) + 
                     glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
                     glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")))

if len(image_files) == 0:
    print(f"\nERROR: No images found in '{IMAGES_DIR}' directory!")
    print("Please run capture_images.py first to capture calibration images.")
    exit(1)

print(f"\nFound {len(image_files)} images to process...\n")

# Storage for calibration data
all_charuco_corners = []
all_charuco_ids = []
image_size = None

# Process each image
successful_images = 0
failed_images = 0

for idx, img_path in enumerate(image_files, 1):
    print(f"Processing [{idx}/{len(image_files)}]: {os.path.basename(img_path)}", end=" ... ")
    
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("FAILED (cannot read image)")
        failed_images += 1
        continue
    
    # Get image size (should be consistent across all images)
    if image_size is None:
        image_size = (img.shape[1], img.shape[0])  # (width, height)
    elif (img.shape[1], img.shape[0]) != image_size:
        print(f"WARNING: Image size mismatch! Expected {image_size}, got {(img.shape[1], img.shape[0])}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
    if ids is None or len(ids) == 0:
        print("FAILED (no markers detected)")
        failed_images += 1
        continue
    
    # Interpolate Charuco corners from detected markers
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )
    
    if retval is None or charuco_corners is None or len(charuco_corners) < 4:
        print(f"FAILED (only {len(charuco_corners) if charuco_corners is not None else 0} corners detected)")
        failed_images += 1
        continue
    
    # Store successful detection
    all_charuco_corners.append(charuco_corners)
    all_charuco_ids.append(charuco_ids)
    successful_images += 1
    print(f"OK ({len(charuco_corners)} corners)")

print("\n" + "=" * 60)
print(f"Successfully processed: {successful_images} images")
print(f"Failed: {failed_images} images")
print("=" * 60)

if successful_images < 3:
    print("\nERROR: Need at least 3 successful images for calibration!")
    print("Please capture more images with better corner detection.")
    exit(1)

# Perform calibration
print("\nPerforming camera calibration...")
print("This may take a moment...\n")

# Calibrate camera
# Note: retval is the reprojection error returned by OpenCV
reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None,
    flags=cv2.CALIB_FIX_ASPECT_RATIO  # Fix aspect ratio for better results
)

if camera_matrix is None or dist_coeffs is None:
    print("ERROR: Calibration failed!")
    exit(1)

# Calculate detailed reprojection error for verification
print("Calculating detailed reprojection error...")
total_error = 0
total_points = 0

# Get all 3D object points from the board
obj_points = board.getChessboardCorners()

for i in range(len(all_charuco_corners)):
    if all_charuco_corners[i] is None or len(all_charuco_corners[i]) == 0:
        continue
    
    # Get 3D points corresponding to detected corner IDs
    detected_ids = all_charuco_ids[i].flatten()
    obj_pts = obj_points[detected_ids]
    
    # Project 3D points to 2D using the estimated pose
    image_points, _ = cv2.projectPoints(
        obj_pts,
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )
    
    # Reshape for comparison
    detected_corners = all_charuco_corners[i].reshape(-1, 2)
    projected_corners = image_points.reshape(-1, 2)
    
    # Calculate mean squared error
    error = np.mean(np.linalg.norm(detected_corners - projected_corners, axis=1))
    total_error += error
    total_points += len(detected_corners)

mean_error = total_error / total_points if total_points > 0 else reprojection_error

# Display results
print("\n" + "=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)
print(f"Reprojection error (OpenCV): {reprojection_error:.4f} pixels")
print(f"Reprojection error (calculated): {mean_error:.4f} pixels")
print(f"\nCamera Matrix:")
print(camera_matrix)
print(f"\nDistortion Coefficients:")
print(dist_coeffs)
print(f"\nImage Size: {image_size[0]} x {image_size[1]}")
print("=" * 60)

# Save calibration data
print(f"\nSaving calibration to: {OUTPUT_FILE}")
np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    img_width=image_size[0],
    img_height=image_size[1],
    square_size_mm=SQUARE_SIZE_MM,
    marker_size_mm=MARKER_SIZE_MM,
    charuco_cols=CHARUCO_COLS,
    charuco_rows=CHARUCO_ROWS,
    dict_type=DICT_TYPE,
    reprojection_error=reprojection_error,
    num_images=successful_images
)

print("✓ Calibration saved successfully!")
print("\nYou can now use this calibration file for distance measurements.")
print(f"Load it with: np.load('{OUTPUT_FILE}')")

