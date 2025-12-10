#!/usr/bin/env python3
"""
Test script to verify camera stop/start functionality
"""
import time
import subprocess
import os
from picamera2 import Picamera2

def test_camera_control():
    """Test starting/stopping Picamera2 and capturing with rpicam-still"""
    try:
        print("Creating Picamera2 instance...")
        picam2 = Picamera2()

        print("Configuring for preview...")
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (1280, 720)}
        )
        picam2.configure(config)

        print("Starting camera...")
        picam2.start()
        time.sleep(2)  # Let it stabilize

        print("Capturing test frame with Picamera2...")
        frame = picam2.capture_array()
        print(f"✓ Picamera2 capture successful: {frame.shape}")

        print("Stopping Picamera2...")
        picam2.stop()
        time.sleep(3)  # Wait longer for release

        # Try to close the camera manager completely
        try:
            picam2.close()
            print("Camera closed")
        except:
            pass

        time.sleep(1)

        print("Testing rpicam-still capture...")
        output_dir = "rpicam_aruco_captures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, "test_control.jpg")
        cmd = [
            "rpicam-still",
            "--width", "1920",
            "--height", "1080",
            "--quality", "95",
            "--output", filepath,
            "--timeout", "2000",
            "--nopreview"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✓ rpicam-still capture successful")
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"✓ File created: {size} bytes")
            else:
                print("✗ File not found")
                return False
        else:
            print(f"✗ rpicam-still failed: {result.stderr}")
            return False

        print("Restarting Picamera2...")
        # Recreate Picamera2 instance
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (1280, 720)}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)

        print("Testing Picamera2 again...")
        frame2 = picam2.capture_array()
        print(f"✓ Picamera2 restart successful: {frame2.shape}")

        print("Cleaning up...")
        picam2.stop()

        # Clean up test file
        if os.path.exists(filepath):
            os.remove(filepath)

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing camera control logic...")
    success = test_camera_control()
    if success:
        print("✓ Camera control test PASSED")
    else:
        print("✗ Camera control test FAILED")
