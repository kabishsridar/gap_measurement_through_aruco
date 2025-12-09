import numpy as np
import cv2
import os

def create_dummy_calibration(filename="calibration_params.npz"):
    """
    Creates a dummy camera calibration file for testing 3D ArUco logic.
    WARNING: These values are APPROXIMATE and will not give perfect real-world accuracy.
    """
    # Approximate camera matrix for a 1280x720 camera
    # fx, fy are focal lengths in pixels. Usually width/2 * 1.5 approx for standard lens.
    width = 1280
    height = 720
    fx = width
    fy = width
    cx = width / 2
    cy = height / 2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Assume zero distortion for dummy
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    print(f"Generating dummy calibration data...")
    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Dist Coefficients:\n{dist_coeffs}")

    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"\nSaved to {filename}")

if __name__ == "__main__":
    create_dummy_calibration()
