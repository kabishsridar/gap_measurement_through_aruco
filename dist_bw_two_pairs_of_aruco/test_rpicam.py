#!/usr/bin/env python3
"""
Test script to verify rpicam integration works
"""
import subprocess
import os
import time
from datetime import datetime

def test_rpicam_capture():
    """Test capturing an image with rpicam-still"""
    output_dir = "rpicam_aruco_captures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_capture_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)

    # rpicam-still command for test capture
    cmd = [
        "rpicam-still",
        "--width", "1920",  # Smaller for testing
        "--height", "1080",
        "--quality", "95",
        "--output", filepath,
        "--timeout", "2000",  # 2 second timeout
        "--nopreview"
    ]

    print(f"Testing rpicam capture to: {filepath}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"✓ Capture successful: {filename} ({size} bytes)")
                return True
            else:
                print("✗ File not created")
                return False
        else:
            print(f"✗ Capture failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Capture timeout")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing rpicam integration...")
    success = test_rpicam_capture()
    if success:
        print("✓ rpicam integration test passed")
    else:
        print("✗ rpicam integration test failed")
