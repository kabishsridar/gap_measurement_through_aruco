import cv2 as cv # importing the required modules
import os
from picamera2 import Picamera2
import time

Chess_Board_Dimensions = (9, 6) # assigning the dimensions of the chess board

n = 0  # image counter

# checks images dir is exist or not
image_path = "calibration_images"

Dir_Check = os.path.isdir(image_path) # checks whether the path is presented

if not Dir_Check:  # if directory does not exist, a new one is created
    os.makedirs(image_path)
    print(f'"{image_path}" Directory is created')
else:
    print(f'"{image_path}" Directory already exists.')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# the termination criteria 
# (will terminate when the iteration reaches 30 or the movement of corner is less than 0.001 px)

def detect_checker_board(image, grayImage, criteria, boardDimension): # function to detect the board
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension) # gets the position of corners of chessboard
    if ret == True: # if read
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret) # draws circle in the edge

    return image, ret

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720)},
    lores={"size": (640, 480)},
    display="lores"
)
picam2.configure(config)
picam2.start()
time.sleep(1)

# cv.namedWindow("Omac Distance Measurement", cv.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array("main")  # Correct stream call
    copyFrame = frame.copy()
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert from PiCamera2 default RGB to BGR
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(
        frame, gray, criteria, Chess_Board_Dimensions
    )

    cv.putText(
        frame,
        f"saved_img : {n}",
        (30, 40),
        cv.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.imshow("frame", frame)
    cv.imshow("copyFrame", copyFrame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s") and board_detected:
        cv.imwrite(f"{image_path}/image{n}.png", copyFrame)
        print(f"saved image number {n}")
        n += 1


picam2.stop()
cv.destroyAllWindows()

print("Total saved Images:", n)