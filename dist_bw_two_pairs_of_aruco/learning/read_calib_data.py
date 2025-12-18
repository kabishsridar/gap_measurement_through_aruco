import numpy as np
data = np.load('/home/omac/gap_measurement_through_aruco/dist_bw_two_pairs_of_aruco/learning/calib_data_rpi/MultiMatrix.npz')

print(f"dist coef: {data['distCoef']}\nCamMatrix : {data['camMatrix']}")